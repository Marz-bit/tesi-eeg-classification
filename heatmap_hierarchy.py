from pathlib import Path
from typing import Dict, List, Tuple, Optional, Hashable

import numpy as np
import matplotlib.pyplot as plt

from skimage.metrics import structural_similarity as ssim
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram

from bionic_apps.databases import Databases
from bionic_apps.feature_extraction import FeatureType
from bionic_apps.preprocess import generate_db
from bionic_apps.preprocess.io import DataLoader, SubjectHandle
from bionic_apps.handlers import HDF5Dataset

import os
import imageio.v2 as imageio

import json
from pathlib import Path


#Configurazione

DB_NAME = Databases.CYBATHLON2028_INIT
FEATURE_TYPE = FeatureType.HEAT_MAP

#PHYSIONET PARAMS
#EPOCH_TMIN, EPOCH_TMAX = 0.0, 4.0
#WINDOW_LEN, WINDOW_STEP = 1.0, 0.1

#CYBATHLON2024_INIT PARAMS
EPOCH_TMIN, EPOCH_TMAX = 0.5, 5.0
WINDOW_LEN, WINDOW_STEP = 1.0, 0.1

MAX_PER_CLASS = None

FILTER_PARAMS = dict(order=5, l_freq=1, h_freq=45)
FEATURE_KWARGS = dict(norm=True)

DB_FILE = Path("tmp/CYBATHLON2028_INIT_None.hdf5")


def _ensure_database():
    DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(use_drop_subject_list=True,
                        subject_handle=SubjectHandle.INDEPENDENT_DAYS)
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
        do_artefact_rejection=True,
        balance_data=True,
        binarize_labels=False,
        fast_load=True,
        subjects='all',
        augment_data=False,
        mode='auto',
        n_jobs=-3
    )

#Carica tutte le immagini (X), le label (y) e il soggetto (subj) dal file HDF5.
def _load_all_data():
    db = HDF5Dataset(DB_FILE)

    y = db.get_y()
    subj = db.get_subject_group()
    X = db.get_data(np.arange(y.shape[0]))

    db.close()

    # converto le label a str
    try:
        y = y.astype(str)
    except Exception:
        y = np.array([str(v) for v in y], dtype=str)

    return X, np.array(y), np.array(subj)



#NORMALIZZA GLOBALMENTE OGNI CANALE (alpha, beta, theta) usando min e max GLOBALI su tutto il dataset.
#Mantiene la scala relativa tra le bande

def _scale_01_global_per_channel(X: np.ndarray) -> np.ndarray:
    X = X.astype(np.float32, copy=False)
    # min e max su (N,H,W) separati per canale
    x_min = X.min(axis=(0, 1, 2), keepdims=True)
    x_max = X.max(axis=(0, 1, 2), keepdims=True)
    denom = np.maximum(x_max - x_min, 1e-6)
    return (X - x_min) / denom

#DISTANZA SSIM PER CANALE, distanza basata su SSIM calcolato separatamente sui 3 canali (alpha, beta, theta)
#e poi mediato. Ritorna 1 - SSIM_media.

def _ssim_distance(a: np.ndarray, b: np.ndarray) -> float:
    ssim_vals = []
    for c in range(3):
        s = ssim(
            a[..., c],
            b[..., c],
            data_range=1.0,
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False
        )
        ssim_vals.append(s)
    ssim_mean = float(np.mean(ssim_vals))
    return 1.0 - ssim_mean

#MEDOIDI DELLE CLASSI, dato un insieme di indici (tutti della stessa classe), trova l'immagine
#che ha la distanza media più bassa da tutte le altre → il medoide.

def _medoid_index_of_set(X: np.ndarray, idxs: np.ndarray) -> int:
    if idxs.size == 1:
        return int(idxs[0])

    m = idxs.size
    X_sub = X[idxs]
    D = np.zeros((m, m), dtype=np.float32)
    for i in range(m):
        for j in range(i + 1, m):
            d = _ssim_distance(X_sub[i], X_sub[j])
            D[i, j] = D[j, i] = d
            if (i + 1) % 50 == 0 or (i + 1) == m:
                print(f"[MEDOID] Riga {i + 1}/{m} completata")

    mean_d = D.mean(axis=1)
    local_medoid = int(np.argmin(mean_d))
    print(f"[MEDOID] Medoid trovato: indice globale {local_medoid}")
    return int(idxs[local_medoid])

#Ritorna un dizionario: {label_classe: indice_globale_medoide}
def _compute_class_medoids(
    X: np.ndarray,
    y: np.ndarray,
    *,
    per_image_scaling: bool = True,
    max_per_class: Optional[int] = None, #None,
    rng: int = 42
) -> Dict[Hashable, int]:
    print("[MEDOIDS] Inizio _compute_class_medoids")
    print(f"[MEDOIDS] X.shape = {X.shape}, num classi = {len(np.unique(y))}")

    if per_image_scaling:
        print("[MEDOIDS] Scaling per immagine...")
        X = _scale_01_global_per_channel(X)
    print("[MEDOIDS] Scaling per immagine completato.")

    r = np.random.default_rng(rng)
    class_medoids: Dict[Hashable, int] = {}

    for c in np.unique(y):
        idxs = np.where(y == c)[0]
        # se una classe ha tantissime finestre, ne prendo solo max_per_class
        if max_per_class is not None and idxs.size > max_per_class:
            print(f"[MEDOIDS] Classe {c}: riduco da {idxs.size} a {max_per_class} finestre (sampling)")
            idxs = r.choice(idxs, size=max_per_class, replace=False)
        medoid_idx_global = _medoid_index_of_set(X, idxs)
        print(f"[MEDOIDS] Classe {c}: medoid globale = {medoid_idx_global}")

        class_medoids[c] = medoid_idx_global
        print("[MEDOIDS] Fine _compute_class_medoids")

    return class_medoids


def save_medoids_images(X, y, class_medoids, outdir="tmp/medoids"):
    os.makedirs(outdir, exist_ok=True)

    for cls, idx in class_medoids.items():
        img = X[idx]                     # immagine (H, W, 3) del medoid della classe cls
        img_uint8 = (img * 255).astype(np.uint8)   # da [0,1] float → [0,255] uint8 per salvarla come PNG

        filepath = os.path.join(outdir, f"medoid_class_{cls}.png")
        imageio.imwrite(filepath, img_uint8)
        print(f"[SAVE MEDOID] Salvato medoid classe {cls}: {filepath}")

#MATRICE DI DISTANZA TRA CLASSI, costruisce una matrice KxK dove K = numero di classi.
#Ogni cella (i,j) è la distanza 1-SSIM tra i medoidi delle due classi.

def _build_class_distance_matrix_medoids(
    X: np.ndarray,
    class_medoids: Dict[Hashable, int],
    class_order: Optional[List[Hashable]] = None
) -> Tuple[np.ndarray, List[Hashable]]:

    if class_order is None:
        try:
            class_order = sorted(class_medoids.keys(), key=str)
        except Exception:
            class_order = list(class_medoids.keys())

    K = len(class_order)
    D = np.zeros((K, K), dtype=np.float32)
    for i in range(K):
        for j in range(i + 1, K):
            ia = class_medoids[class_order[i]]
            jb = class_medoids[class_order[j]]
            d = _ssim_distance(X[ia], X[jb])
            D[i, j] = D[j, i] = d
    return D, class_order

def save_medoids_distances(D_class: np.ndarray,
                           class_order: List[Hashable],
                           outdir: str):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    #Salvo la matrice completa (K x K)
    np.save(outdir / "D_class.npy", D_class)
    print(f"[SAVE DIST] Salvata matrice D_class in {outdir / 'D_class.npy'}")

    #Salvo anche una versione JSON con coppie (classe_i, classe_j) -> distanza
    pairwise = {}
    K = len(class_order)

    for i in range(K):
        ci = str(class_order[i])
        for j in range(i + 1, K):
            cj = str(class_order[j])
            dist = float(D_class[i, j])
            key = f"{ci}__{cj}"     # es: "0__1", "right__left", ecc.
            pairwise[key] = dist

    data = {
        "class_order": [str(c) for c in class_order],
        "pairwise_distances": pairwise
    }

    json_path = outdir / "D_class_pairs.json"
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"[SAVE DIST] Salvate distanze pairwise in {json_path}")


#PLOT DEL DENDROGRAMMA
def _plot_class_dendrogram(
    D_class: np.ndarray,
    class_order: List[Hashable],
    class_names: Optional[Dict[Hashable, str]] = None,
    method: str = "average",
    title: str = "Dendrogramma classi (SSIM via medoidi)",
    figsize: Tuple[int, int] = (9, 5),
    savepath: Optional[str] = None
):
    print("[PLOT] Inizio plot dendrogramma")
    assert D_class.shape[0] == D_class.shape[1], "D deve essere quadrata"
    # SciPy vuole il formato “condensed”
    condensed = squareform(D_class, checks=False)
    print("[PLOT] Linkage...")
    Z = linkage(condensed, method=method)

    labels = [
        (class_names[c] if class_names and c in class_names else str(c))
        for c in class_order
    ]

    plt.figure(figsize=figsize)
    dendrogram(Z, labels=labels, orientation="top", distance_sort="ascending")
    plt.title(title)
    plt.ylabel("Distanza (1 - SSIM)")
    plt.tight_layout()
    if savepath:
        print(f"[PLOT] Salvo figura in: {savepath}")
        plt.savefig(savepath, dpi=150)
    plt.show()
    print("[PLOT] Plot completato")

#MAIN

def run_class_dendrogram():
    print("[STEP 0] Inizio run_class_dendrogram")
    print("[STEP 1] Generazione/caricamento DB...")
    #assicura il DB
    _ensure_database()
    print("[STEP 1] DB pronto.")
    #carica dati
    print("[STEP 2] Carico tutti i dati da HDF5...")
    X, y, subj = _load_all_data()
    print(f"[STEP 2] Dati caricati: X.shape = {X.shape}, y.shape = {y.shape}, subj.shape = {subj.shape}")

    #stampo quante finestre per classe (solo info)
    vals, cnts = np.unique(y, return_counts=True)
    print("\n=== CLASSI PRESENTI NEL DATABASE ===")
    for v, c in zip(vals, cnts):
        print(f"{v}: {c} finestre")
    print("====================================\n")

    #normalizza (globale per canale → preserva alpha/beta/theta)
    print("[STEP 3] Inizio normalizzazione globale per canale...")
    X_scaled = _scale_01_global_per_channel(X)
    print("[STEP 3] Normalizzazione completata.")

    #trova il medoide per ogni classe
    print("[STEP 4] Calcolo dei medoids per classe...")
    class_medoids = _compute_class_medoids(
        X_scaled,
        y,
        per_image_scaling=False,  #perché abbiamo già scalato sopra
        max_per_class=MAX_PER_CLASS, #none #se una classe ha più di 300 immagini (finestre), ne prendo solo 300 casuali per trovare il suo medoide
        rng=42
    )
    print(f"[STEP 4] Medoids calcolati per {len(class_medoids)} classi.")

    run_tag = f"max{MAX_PER_CLASS}"
    medoid_dir = f"tmp/medoids_{run_tag}"

    print("[STEP 4b] Salvataggio immagini medoid...")
    save_medoids_images(X_scaled, y, class_medoids, outdir=medoid_dir)
    print("[STEP 4b] Medoid salvati.")


    #costruisci la matrice di distanza tra classi
    print("[STEP 5] Costruzione matrice di distanza tra classi...")
    D_class, class_order = _build_class_distance_matrix_medoids(
        X_scaled,
        class_medoids
    )
    print("[STEP 5] Matrice di distanza costruita. Shape:", D_class.shape)

    #Salvo le distanze tra i medoid
    run_tag = f"max{MAX_PER_CLASS}"
    dist_dir = f"tmp/dendro_data_{run_tag}"
    print("[STEP 5b] Salvataggio distanze tra medoid...")
    save_medoids_distances(D_class, class_order, outdir=dist_dir)
    print("[STEP 5b] Distanze tra medoid salvate.")

    #(opzionale) nomi più leggibili
    class_names = {
        # "0": "left hand",
        # "1": "right hand",
        # ...
    }

    #plot
    print("[STEP 6] Plot e salvataggio del dendrogramma...")
    _plot_class_dendrogram(
        D_class,
        class_order,
        class_names=class_names,
        method="average",
        title="Dendrogramma classi (SSIM via medoidi)",
        figsize=(9, 5),
        savepath = f"tmp/dendrogram_{DB_NAME}_max{MAX_PER_CLASS}.png"
    )

    print("[STEP 7] Dendrogramma salvato. Fine run_class_dendrogram.")


if __name__ == "__main__":
    run_class_dendrogram()


