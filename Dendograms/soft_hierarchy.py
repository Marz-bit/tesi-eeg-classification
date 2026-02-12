# soft_hierarchy.py
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

import json
import numpy as np

import matplotlib
matplotlib.use("Agg")  # cluster-safe
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform, pdist

from bionic_apps.databases import Databases
from bionic_apps.handlers import HDF5Dataset
from bionic_apps.ai.classifier import init_classifier, ClassifierType


DB_NAME = Databases.CYBATHLON2028_INIT

#DB già esistente (quello embed)
DB_FILE = Path("/home/mcaroti/tmp/embed_hierarchy_CYBATHLON2028_INIT.hdf5")

#output pipeline B
OUTDIR = Path(f"/home/mcaroti/tmp/soft_hierarchy_{DB_NAME.name}")

# ✅ directory dove salvi TUTTO sulla flat EEGNet
RUN_DIR = Path(f"/home/mcaroti/tmp/eegnet_flat_run_{DB_NAME.name}")

CLASSIFIER_LAYER_NAME: Optional[str] = "dense"


#plotting
DPI = 150
SHOW_PLOTS = False  # su cluster: False

def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


def _l2_normalize(x: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (n + eps)


def _load_label_classes(run_dir: Path) -> np.ndarray:
    f = run_dir / "label_classes.npy"
    if not f.exists():
        raise RuntimeError(
            f"[SOFT] Manca {f}. Serve per associare correttamente i pesi ai nomi delle classi."
        )
    classes = np.load(f, allow_pickle=True)
    print(f"[SOFT] label_classes.npy: {list(classes)}")
    return np.array(classes, dtype=object)


def _find_weights_file(run_dir: Path, meta: Optional[Dict[str, Any]]) -> Path:
    """
    Preferisce meta['weights_file'] se presente, altrimenti usa RUN_DIR/models/EEGNet.weights.h5,
    altrimenti prende il .h5 più recente sotto run_dir.
    """
    if meta is not None:
        wf = meta.get("weights_file", None)
        if wf:
            wf = Path(wf)
            if wf.exists():
                print(f"[SOFT] weights_file da meta.json: {wf}")
                return wf

    cand = run_dir / "models" / "EEGNet.weights.h5"
    if cand.exists():
        print(f"[SOFT] weights_file default: {cand}")
        return cand

    h5s = sorted(run_dir.glob("**/*.h5"), key=lambda p: p.stat().st_mtime, reverse=True)
    if h5s:
        print(f"[SOFT] weights_file fallback: {h5s[0]}")
        return h5s[0]

    raise FileNotFoundError(f"[SOFT] Nessun file .h5 trovato in {run_dir} (o sottocartelle).")


def _get_fs_and_input_shape(meta: Optional[Dict[str, Any]], db: HDF5Dataset, train_idx: Optional[np.ndarray] = None):
    """
    Usa meta.json se possibile. Fallback: legge dal DB.
    """
    fs = None
    input_shape = None

    if meta is not None:
        if "fs" in meta:
            fs = float(meta["fs"])
        if "input_shape" in meta:
            # meta salva tuple -> in json diventa list
            input_shape = tuple(meta["input_shape"])

    if fs is None:
        fs = db.get_fs()

    if input_shape is None:
        # se non ho train_idx, prendo 0
        idx = int(train_idx[0]) if (train_idx is not None and len(train_idx) > 0) else 0
        input_shape = db.get_data(idx).shape

    return fs, input_shape


def _load_external_split(run_dir: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Se presenti, carica train_idx/val_idx (non indispensabili per Pipeline B, ma utili per consistenza).
    """
    train_f = run_dir / "train_idx.npy"
    val_f = run_dir / "val_idx.npy"
    if train_f.exists() and val_f.exists():
        train_idx = np.load(train_f)
        val_idx = np.load(val_f)
        print(f"[SOFT] split ricaricato: train={len(train_idx)} val={len(val_idx)}")
        return train_idx, val_idx
    print("[SOFT] split non trovato (ok, non è necessario per Pipeline B).")
    return None, None


def _get_keras_model_from_clf(clf):
    """
    Nel tuo framework spesso il modello sta in clf._model.
    """
    if hasattr(clf, "_model"):
        return clf._model
    if hasattr(clf, "model"):
        return clf.model
    return None


def _infer_classifier_layer_name(model, n_classes: int) -> str:
    print("[SOFT] Keras layers (name -> class):")
    for i, layer in enumerate(model.layers):
        print(f"  [{i:02d}] {layer.name} -> {layer.__class__.__name__}")

    for layer in reversed(model.layers):
        cls = layer.__class__.__name__.lower()
        if "dropout" in cls or "activation" in cls:
            continue
        try:
            out_shape = getattr(layer, "output_shape", None)
            ws = layer.get_weights()
            if (
                isinstance(out_shape, tuple) and len(out_shape) == 2 and out_shape[1] == n_classes
                and ws and np.asarray(ws[0]).shape[-1] == n_classes
            ):
                print(f"[SOFT] classifier layer auto: {layer.name}")
                return layer.name
        except Exception:
            pass

    #fallback: ultimo layer con pesi
    for layer in reversed(model.layers):
        try:
            ws = layer.get_weights()
            if ws:
                print(f"[SOFT] classifier layer fallback: {layer.name}")
                return layer.name
        except Exception:
            pass

    raise RuntimeError("[SOFT] Non trovo un layer di classificazione con pesi.")


def extract_last_layer_class_weights(model, n_classes: int, layer_name: Optional[str] = None) -> Tuple[np.ndarray, str]:
    if layer_name is None:
        layer_name = _infer_classifier_layer_name(model, n_classes)

    layer = model.get_layer(layer_name)
    ws = layer.get_weights()
    if not ws:
        raise RuntimeError(f"[SOFT] Layer '{layer_name}' non ha pesi. Imposta CLASSIFIER_LAYER_NAME manualmente.")

    W = np.asarray(ws[0])  #kernel

    #Dense: (d, K) -> (K, d)
    if W.ndim == 2:
        if W.shape[1] != n_classes:
            raise RuntimeError(f"[SOFT] Shape pesi inattesa {W.shape}, atteso (d,{n_classes}).")
        Wc = W.T
    else:
        #conv 1x1 o simili: (...,K) -> (K,d)
        if W.shape[-1] != n_classes:
            raise RuntimeError(f"[SOFT] Shape pesi inattesa {W.shape}, ultimo asse != n_classes.")
        Wc = W.reshape(-1, n_classes).T

    print(f"[SOFT] W estratta da layer='{layer_name}' -> {Wc.shape} (K,d)")
    return Wc.astype(np.float64), layer_name


def build_similarity_and_distance_from_weights(W: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    W_hat = _l2_normalize(W, axis=1)
    S = W_hat @ W_hat.T
    np.fill_diagonal(S, 1.0)

    condensed = pdist(W_hat, metric="cosine")  # = 1 - cos
    D = squareform(condensed)
    np.fill_diagonal(D, 0.0)

    print("[SOFT] costruite S (cosine sim) e D (1-cos).")
    return S, D, W_hat


def plot_dendrogram(D: np.ndarray, class_labels: List, savepath: Path):
    condensed = squareform(D, checks=False)
    Z = linkage(condensed, method="average")

    plt.figure(figsize=(9, 5))
    dendrogram(
        Z,
        labels=[str(c) for c in class_labels],
        orientation="top",
        distance_sort="ascending"
    )
    plt.title("Dendrogramma classi (Pipeline B: last-layer weights)")
    plt.ylabel("D(c,d) = 1 - cos(ŵ_c, ŵ_d)")
    plt.tight_layout()

    savepath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(savepath), dpi=DPI)
    print(f"[SOFT] dendrogramma salvato in {savepath}")

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()

    return Z


def save_soft_artifacts(outdir: Path,
                        W: np.ndarray,
                        W_hat: np.ndarray,
                        S: np.ndarray,
                        D: np.ndarray,
                        class_labels: List,
                        layer_name: str,
                        weights_file: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir / "weights_W.npy", W)
    np.save(outdir / "weights_W_hat.npy", W_hat)
    np.save(outdir / "similarity_S.npy", S)
    np.save(outdir / "distance_D.npy", D)
    np.save(outdir / "class_labels.npy", np.array(class_labels, dtype=object))
    np.save(outdir / "classifier_layer_name.npy", np.array([layer_name], dtype=object))

    meta_soft = {
        "db_name": DB_NAME.name,
        "db_file": str(DB_FILE),
        "run_dir": str(RUN_DIR),
        "weights_file_used": str(weights_file),
        "classifier_layer_name": layer_name,
        "distance": "cosine (1 - cos)",
        "n_classes": int(len(class_labels)),
        "d_weight": int(W.shape[1]),
    }
    with open(outdir / "meta_soft.json", "w") as f:
        json.dump(meta_soft, f, indent=2)

    print(f"[SOFT] artefatti salvati in {outdir}")


def main():
    if not DB_FILE.exists():
        raise RuntimeError(f"[SOFT] DB_FILE non esiste: {DB_FILE}")
    if not RUN_DIR.exists():
        raise RuntimeError(f"[SOFT] RUN_DIR non esiste: {RUN_DIR}")

    OUTDIR.mkdir(parents=True, exist_ok=True)

    meta = _load_json(RUN_DIR / "meta.json")
    class_labels = _load_label_classes(RUN_DIR)
    n_classes = len(class_labels)


    db = HDF5Dataset(DB_FILE)
    try:
        train_idx, val_idx = _load_external_split(RUN_DIR)
        fs, input_shape = _get_fs_and_input_shape(meta, db, train_idx=train_idx)
        print(f"[SOFT] fs={fs} input_shape={input_shape}")
    finally:
        db.close()

    clf = init_classifier(
        ClassifierType.EEG_NET,
        input_shape,
        n_classes,
        fs=fs,
        save_path=str(RUN_DIR)
    )

    weights_file = _find_weights_file(RUN_DIR, meta)

    model = _get_keras_model_from_clf(clf)
    if model is None:
        raise RuntimeError("[SOFT] Non trovo modello Keras in clf (_model/model).")

    try:
        model.load_weights(str(weights_file))
        print(f"[SOFT] pesi caricati: {weights_file}")
    except Exception as e:
        raise RuntimeError(
            f"[SOFT] Errore nel load_weights({weights_file}). "
            "Verifica che i pesi siano compatibili con l'architettura creata da init_classifier."
        ) from e

    W, layer_name = extract_last_layer_class_weights(
        model,
        n_classes=n_classes,
        layer_name=CLASSIFIER_LAYER_NAME
    )

    S, D, W_hat = build_similarity_and_distance_from_weights(W)

    plot_dendrogram(
        D,
        class_labels=list(class_labels),
        savepath=OUTDIR / f"dendrogram_soft_weight_hierarchy_{DB_NAME.name}.png"
    )

    save_soft_artifacts(
        outdir=OUTDIR,
        W=W,
        W_hat=W_hat,
        S=S,
        D=D,
        class_labels=list(class_labels),
        layer_name=layer_name,
        weights_file=weights_file
    )


if __name__ == "__main__":
    main()
