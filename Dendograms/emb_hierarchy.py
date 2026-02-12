# embed_hierarchy_cybathlon.py
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform, pdist

from bionic_apps.databases import Databases
from bionic_apps.feature_extraction import FeatureType
from bionic_apps.preprocess import generate_db
from bionic_apps.preprocess.io import DataLoader, SubjectHandle
from bionic_apps.handlers import HDF5Dataset
from bionic_apps.ai.classifier import init_classifier, ClassifierType
from bionic_apps.offline_analyses import _fit_clf

DB_NAME = Databases.CYBATHLON2028_INIT
FEATURE_TYPE = FeatureType.RAW

EPOCH_TMIN, EPOCH_TMAX = 0.5, 5.0
WINDOW_LEN, WINDOW_STEP = 1.0, 0.1

FILTER_PARAMS = dict(order=5, l_freq=0.2, h_freq=45)
FEATURE_KWARGS = dict(norm=True)

SUBJECT_HANDLE = SubjectHandle.INDEPENDENT_DAYS
USE_DROP_SUBJECT_LIST = True
BALANCE_DATA = True
AUGMENT_DATA = False

DB_FILE = Path(f"tmp/embed_hierarchy_{DB_NAME.name}.hdf5")
OUTDIR = Path(f"tmp/embed_hierarchy_{DB_NAME.name}")
RUN_DIR = Path(f"tmp/eegnet_flat_run_{DB_NAME.name}")

EPOCHS = 150
PATIENCE = 20
GIVE_UP = 40
BATCH_SIZE = 32
INTERNAL_VAL_SPLIT = 0.2
EXTERNAL_VAL_SPLIT = 0.2
RNG = 42

EMBED_LAYER_NAME: Optional[str] = "flatten"


def _ensure_database():
    DB_FILE.parent.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(
        use_drop_subject_list=USE_DROP_SUBJECT_LIST,
        subject_handle=SUBJECT_HANDLE
    )

    generate_db(
        db_name=DB_NAME,
        db_filename=DB_FILE,
        loader=loader,
        feature_type=FEATURE_TYPE,
        epoch_tmin=EPOCH_TMIN,
        epoch_tmax=EPOCH_TMAX,
        window_length=WINDOW_LEN,
        window_step=WINDOW_STEP,
        ch_selection=None,
        feature_kwargs=FEATURE_KWARGS,
        filter_params=FILTER_PARAMS,
        do_artefact_rejection=False,
        balance_data=BALANCE_DATA,
        binarize_labels=False,
        fast_load=True,
        subjects='all',
        augment_data=AUGMENT_DATA,
        mode='auto',
        n_jobs=-3
    )


def _l2_normalize(x: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (n + eps)


def extract_penultimate_embeddings(clf, X: np.ndarray, layer_name: str = "flatten") -> np.ndarray:
    import keras

    if hasattr(clf, "_model"):
        model = clf._model
    elif isinstance(clf, keras.Model):
        model = clf
    else:
        raise RuntimeError("Non trovo il modello Keras. Nel tuo framework dovrebbe essere clf._model (BaseNet).")

    try:
        out_tensor = model.get_layer(layer_name).output
    except Exception as e:
        avail = [l.name for l in model.layers]
        raise RuntimeError(f"Layer '{layer_name}' non trovato. Ultimi 20 layer: {avail[-20:]}") from e

    feat_model = keras.Model(inputs=model.input, outputs=out_tensor)
    emb = feat_model.predict(X, batch_size=BATCH_SIZE, verbose=0)
    emb = np.asarray(emb)

    if emb.ndim != 2:
        raise RuntimeError(f"Embedding non 2D: shape={emb.shape}. Scegli un layer diverso.")

    print(f"[HIER-EMB] embedding estratte da layer='{layer_name}' -> {emb.shape}")
    return emb


def get_or_make_split(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split ESTERNO train/val:
    - se esiste in RUN_DIR → ricarico (split identico tra run)
    - altrimenti → creo e salvo
    """
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    train_file = RUN_DIR / "train_idx.npy"
    val_file = RUN_DIR / "val_idx.npy"

    if train_file.exists() and val_file.exists():
        train_idx = np.load(train_file)
        val_idx = np.load(val_file)

        # sanity check: indici validi rispetto al DB corrente
        n = len(y)
        ok = (
            train_idx.ndim == 1 and val_idx.ndim == 1 and
            train_idx.size > 0 and val_idx.size > 0 and
            train_idx.min() >= 0 and val_idx.min() >= 0 and
            train_idx.max() < n and val_idx.max() < n
        )
        if not ok:
            print("[HIER-EMB] Split salvato non compatibile col DB corrente → rigenero e sovrascrivo.")
        else:
            print(f"[HIER-EMB] Split ricaricato da file: train={len(train_idx)} val={len(val_idx)}")
            return train_idx, val_idx

    # se non esiste o non valido → crea e salva
    idx_all = np.arange(len(y))
    train_idx, val_idx = train_test_split(
        idx_all,
        test_size=EXTERNAL_VAL_SPLIT,
        random_state=RNG,
        stratify=y
    )
    np.save(train_file, train_idx)
    np.save(val_file, val_idx)
    print(f"[HIER-EMB] Split creato e salvato: train={len(train_idx)} val={len(val_idx)}")
    return train_idx, val_idx


def train_flat_eegnet_and_get_val_embeddings() -> Tuple[np.ndarray, np.ndarray, LabelEncoder, List]:
    print("[HIER-EMB] Generazione / caricamento DB...")
    _ensure_database()

    db = HDF5Dataset(DB_FILE)
    try:
        y_raw = db.get_y()
        label_encoder = LabelEncoder().fit(y_raw)
        y = label_encoder.transform(y_raw)
        n_classes = len(label_encoder.classes_)

        print(f"[HIER-EMB] n finestre: {len(y)}")
        print(f"[HIER-EMB] classi: {list(label_encoder.classes_)}")

        # ✅ split esterno persistente
        train_idx, val_idx = get_or_make_split(y)

        input_shape = db.get_data(train_idx[0]).shape
        fs = db.get_fs()

        clf = init_classifier(
            ClassifierType.EEG_NET,
            input_shape,
            n_classes,
            fs=fs,
            save_path=str(RUN_DIR)  # pesi → tmp/eegnet_flat_run_.../models/
        )

        ep_ind = db.get_epoch_group()

        print("[HIER-EMB] training EEGNet (flat 9-classi)...")
        _fit_clf(
            clf, db, y,
            train_ind=train_idx,
            ep_ind=ep_ind,
            shuffle=True,
            epochs=EPOCHS,
            validation_split=INTERNAL_VAL_SPLIT,
            batch_size=BATCH_SIZE,
            patience=PATIENCE,
            give_up=GIVE_UP,
            weight_classes=False,
            verbose='auto'
        )

        #salva artefatti utili per riuso futuro (modello/pesi/meta)
        import json
        import time

        RUN_DIR.mkdir(parents=True, exist_ok=True)

        #salva label mapping (utile se in futuro vuoi ricostruire label encoder identico)
        np.save(RUN_DIR / "label_classes.npy", np.array(label_encoder.classes_, dtype=object))

        model_file = RUN_DIR / f"EEGNet_flat_{DB_NAME.name}.keras"
        clf.save(model_file)

        weights_file = clf.save_weights()

        with open(RUN_DIR / "model_summary.txt", "w") as f:
            clf._model.summary(print_fn=lambda s: f.write(s + "\n"))

        meta = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "db_name": DB_NAME.name,
            "db_file": str(DB_FILE),
            "feature_type": str(FEATURE_TYPE),
            "epoch_tmin": EPOCH_TMIN,
            "epoch_tmax": EPOCH_TMAX,
            "window_len": WINDOW_LEN,
            "window_step": WINDOW_STEP,
            "filter_params": FILTER_PARAMS,
            "feature_kwargs": FEATURE_KWARGS,
            "balance_data": BALANCE_DATA,
            "augment_data": AUGMENT_DATA,
            "rng": RNG,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "internal_val_split": INTERNAL_VAL_SPLIT,
            "external_val_split": EXTERNAL_VAL_SPLIT,
            "embed_layer_name": EMBED_LAYER_NAME,
            "model_file": str(model_file),
            "weights_file": str(weights_file),
            "fs": float(fs),
            "input_shape": tuple(input_shape),
        }
        with open(RUN_DIR / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"[HIER-EMB] Run salvata in: {RUN_DIR}")
        print(f"[HIER-EMB] Modello: {model_file}")
        print(f"[HIER-EMB] Pesi: {weights_file}")

        #embedding sul validation esterno (sempre lo stesso, grazie a val_idx persistente)
        x_val = db.get_data(val_idx)
        y_val = y[val_idx]

        print("[HIER-EMB] estrazione embedding su validation esterno...")
        emb_val = extract_penultimate_embeddings(clf, x_val, layer_name=EMBED_LAYER_NAME)

    finally:
        db.close()

    return y_val, emb_val, label_encoder, list(label_encoder.classes_)


def build_class_prototypes(y_enc: np.ndarray, emb: np.ndarray, n_classes: int) -> np.ndarray:
    emb = emb.astype(np.float64)
    emb_hat = _l2_normalize(emb, axis=1)

    prototypes = np.zeros((n_classes, emb_hat.shape[1]), dtype=np.float64)
    for c in range(n_classes):
        mask = (y_enc == c)
        if not np.any(mask):
            raise RuntimeError(f"Nessun campione trovato per la classe {c} (controlla split/labels).")
        prototypes[c] = emb_hat[mask].mean(axis=0)

    prototypes = _l2_normalize(prototypes, axis=1)
    print(f"[HIER-EMB] prototipi costruiti: {prototypes.shape} (K x D)")
    return prototypes


def build_cosine_distance_from_prototypes(prototypes: np.ndarray) -> np.ndarray:
    condensed = pdist(prototypes, metric="cosine")
    D = squareform(condensed)
    np.fill_diagonal(D, 0.0)
    print("[HIER-EMB] matrice distanza D costruita (cosine distance).")
    return D


def plot_dendrogram(D: np.ndarray, class_labels: List, savepath: Optional[Path] = None):
    condensed = squareform(D, checks=False)
    Z = linkage(condensed, method="average")

    plt.figure(figsize=(9, 5))
    dendrogram(
        Z,
        labels=[str(c) for c in class_labels],
        orientation="top",
        distance_sort="ascending"
    )
    plt.title("Dendrogramma classi (Embedding/Prototype-based hierarchy)")
    plt.ylabel("D(c,d) = 1 - cos(μc, μd)")
    plt.tight_layout()

    if savepath is not None:
        savepath = Path(savepath)
        savepath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(savepath), dpi=150)
        print(f"[HIER-EMB] dendrogramma salvato in {savepath}")

    plt.show()
    return Z


def save_hierarchy_artifacts(prototypes: np.ndarray, D: np.ndarray, class_labels: List, outdir: Path):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir / "prototypes_mu.npy", prototypes)
    np.save(outdir / "distance_D.npy", D)
    np.save(outdir / "class_labels.npy", np.array(class_labels, dtype=object))
    print(f"[HIER-EMB] artefatti salvati in {outdir}")


def main():
    y_val, emb_val, le, class_labels = train_flat_eegnet_and_get_val_embeddings()

    prototypes = build_class_prototypes(
        y_enc=y_val,
        emb=emb_val,
        n_classes=len(class_labels)
    )

    D = build_cosine_distance_from_prototypes(prototypes)

    plot_dendrogram(
        D,
        class_labels,
        savepath=OUTDIR / f"dendrogram_embedding_hierarchy_{DB_NAME.name}.png"
    )

    save_hierarchy_artifacts(prototypes, D, class_labels, outdir=OUTDIR)


if __name__ == "__main__":
    main()
