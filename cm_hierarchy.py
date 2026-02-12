# cm_hierarchy_cybathlon.py

from pathlib import Path
from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

from bionic_apps.databases import Databases
from bionic_apps.feature_extraction import FeatureType
from bionic_apps.preprocess import generate_db
from bionic_apps.preprocess.io import DataLoader, SubjectHandle
from bionic_apps.handlers import HDF5Dataset
from bionic_apps.ai.classifier import init_classifier, ClassifierType
from bionic_apps.offline_analyses import _fit_clf

from bionic_apps.databases.eeg.defaults import (
    BOTH_LEGS, REST, RIGHT_HAND, SPATIAL, ROTATION, WORD, SUBTRACT, MUSIC, LEFT_HAND
)

# Sottoinsieme di classi da usare per la rete flat e per il dendrogramma
#SUBSET_LABELS = {MUSIC, WORD, SUBTRACT, REST, ROTATION, BOTH_LEGS}

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

DB_FILE = Path(f"tmp/cm_hierarchy_{DB_NAME.name}.hdf5")
OUTDIR = Path(f"tmp/cm_hierarchy_{DB_NAME.name}")


#Iperparametri EEGNet (consistenti con test.py)
EPOCHS = 150
PATIENCE = 20
GIVE_UP = 40
BATCH_SIZE = 32
INTERNAL_VAL_SPLIT = 0.2  #per early stopping interno
EXTERNAL_VAL_SPLIT = 0.2  #quello per la confusion matrix
RNG = 42


#Generazione DB
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
        do_artefact_rejection=False,  #False see if it changes
        balance_data=BALANCE_DATA,
        binarize_labels=False,   #9 classi
        fast_load=False,
        subjects='all',
        augment_data=AUGMENT_DATA,
        mode='auto',
        n_jobs=-3
    )



# #Train flat EEGNet + pred val: genera DB, allena EEGNet 9-classi su TUTTE le finestre usando
#_fit_clf (quindi con validation interna per early stopping), usa uno split ESTERNO train/val
#(20%) per costruire la CM
def train_flat_eegnet_and_get_val_predictions() -> Tuple[np.ndarray, np.ndarray, LabelEncoder, List]:

    print("[HIER] Generazione / caricamento DB...")
    _ensure_database()

    db = HDF5Dataset(DB_FILE)
    try:
        y_raw = db.get_y()                     #label originali (stringhe o int)
        label_encoder = LabelEncoder().fit(y_raw)
        y = label_encoder.transform(y_raw) #classi modificate in [0,8]
        n_classes = len(label_encoder.classes_)

        print(f"[HIER] n finestre: {len(y)}")
        print(f"[HIER] classi: {label_encoder.classes_}")

        #split esterno per la CM (20% validation)
        idx_all = np.arange(len(y))
        train_idx, val_idx = train_test_split(
            idx_all,
            test_size=EXTERNAL_VAL_SPLIT,
            random_state=RNG,
            stratify=y  #garantisce che la distribuzione delle classi sia bilanciata tra train e val
        )
        print(f"[HIER] train windows: {len(train_idx)}, val windows: {len(val_idx)}")

        #inizializzo EEGNet come nel resto del framework
        input_shape = db.get_data(train_idx[0]).shape
        fs = db.get_fs()
        clf = init_classifier(
            ClassifierType.EEG_NET,
            input_shape,
            n_classes,
            fs=fs
        )

        #ep_group per BalancedKFold interno (_fit_clf)
        ep_ind = db.get_epoch_group()   #ndica a quale epoca appartiene ogni finestra

        print("[HIER] training EEGNet (flat 9-classi)...")
        _fit_clf(
            clf, db, y,
            train_ind=train_idx,    #traino solo sul train
            ep_ind=ep_ind,
            shuffle=True,
            epochs=EPOCHS,
            validation_split=INTERNAL_VAL_SPLIT,   #SOLO per early stopping interno
            batch_size=BATCH_SIZE,
            patience=PATIENCE,
            give_up=GIVE_UP,
            weight_classes=False,
            verbose='auto'
        )

        # predizioni sul validation ESTERNO → confusion matrix
        x_val = db.get_data(val_idx)
        y_val = y[val_idx]  #vere classi
        y_pred_val = clf.predict(x_val)  #classi predette

    finally:
        db.close()

    return y_val, y_pred_val, label_encoder, list(label_encoder.classes_)

# def train_flat_eegnet_and_get_val_predictions() -> Tuple[np.ndarray, np.ndarray, LabelEncoder, List]:
#     """
#     Allena una EEGNet flat SOLO sulle classi in SUBSET_LABELS
#     e restituisce le predizioni sulla validation esterna per costruire la CM.
#     """
    #
    # print("[HIER] Generazione / caricamento DB...")
    # _ensure_database()
    #
    # db = HDF5Dataset(DB_FILE)
    # try:
    #     # Label originali (stringhe o int, come definite nel db)
    #     y_raw = db.get_y()
    #
    #     # ---- FILTRO: tengo solo le finestre appartenenti alle 6 classi che ti interessano ----
    #     mask = np.isin(y_raw, list(SUBSET_LABELS))
    #     idx_all = np.where(mask)[0]          # indici delle finestre valide
    #     y_raw_sub = y_raw[mask]             # solo le label nelle 6 classi
    #
    #     print(f"[HIER] n finestre TOTALI: {len(y_raw)}")
    #     print(f"[HIER] n finestre SOTTOINSIEME (6 classi): {len(y_raw_sub)}")
    #
    #     # Encoder SOLO sulle 6 classi
    #     label_encoder = LabelEncoder().fit(y_raw_sub)
    #     y_encoded_sub = label_encoder.transform(y_raw_sub)   # valori 0..5
    #     n_classes = len(label_encoder.classes_)
    #
    #     print(f"[HIER] classi (subset): {label_encoder.classes_}")
    #
    #     # Creo un array y_full della stessa lunghezza di y_raw, ma uso solo le posizioni mask=True
    #     y_full = np.zeros_like(y_raw, dtype=int)
    #     y_full[mask] = y_encoded_sub
    #
    #     # ---- Split esterno train/val solo sulle finestre del sottoinsieme ----
    #     train_idx, val_idx = train_test_split(
    #         idx_all,
    #         test_size=EXTERNAL_VAL_SPLIT,
    #         random_state=RNG,
    #         stratify=y_encoded_sub   # stratify deve avere stessa lunghezza di idx_all
    #     )
    #
    #     print(f"[HIER] train windows (subset): {len(train_idx)}, val windows (subset): {len(val_idx)}")
    #
    #     # Inizializza EEGNet a 6 classi
    #     input_shape = db.get_data(train_idx[0]).shape
    #     fs = db.get_fs()
    #     clf = init_classifier(
    #         ClassifierType.EEG_NET,
    #         input_shape,
    #         n_classes,
    #         fs=fs
    #     )
    #
    #     # ep_group per BalancedKFold interno (_fit_clf)
    #     ep_ind = db.get_epoch_group()
    #
    #     print("[HIER] training EEGNet (flat 6-classi, subset)...")
    #     _fit_clf(
    #         clf, db, y_full,
    #         train_ind=train_idx,    # allena solo sulle finestre delle 6 classi
    #         ep_ind=ep_ind,
    #         shuffle=True,
    #         epochs=EPOCHS,
    #         validation_split=INTERNAL_VAL_SPLIT,   # per early stopping interno
    #         batch_size=BATCH_SIZE,
    #         patience=PATIENCE,
    #         give_up=GIVE_UP,
    #         weight_classes=False,
    #         verbose='auto'
    #     )
    #
    #     # Predizioni sulla validation esterna (solo subset)
    #     x_val = db.get_data(val_idx)
    #     y_val = y_full[val_idx]           # vere classi codificate 0..5
    #     y_pred_val = clf.predict(x_val)   # classi predette 0..5
    #
    # finally:
    #     db.close()
    #
    # # label_encoder.classes_ contiene le 6 label originali (MUSIC, WORD, SUBTRACT, REST, ROTATION, BOTH_LEGS)
    # return y_val, y_pred_val, label_encoder, list(label_encoder.classes_)

#CM normalizzata, P, D.
#CM righe = classi predette, colonne = classi vere. Normalizzata per colonna

def build_normalized_cm(y_true_enc, y_pred_enc, n_classes):

    cm_true_pred = confusion_matrix(
        y_true_enc,
        y_pred_enc,
        labels=np.arange(n_classes)
    )
    #sklearn: righe = true, colonne = pred
    #noi vogliamo righe = pred, colonne = true → trasposta
    cm_counts = cm_true_pred.T

    col_sums = cm_counts.sum(axis=0, keepdims=True) #tot campioni appartenenti a class
    col_sums[col_sums == 0] = 1.0
    cm_norm = cm_counts / col_sums #divido per 1 ogni casella della colonna, Pij probabilità condizionata della classe j di essere classificata come classe i

    print("[HIER] CM normalizzata costruita.")
    print("       somma per colonna:", cm_norm.sum(axis=0))
    return cm_norm, cm_counts

#P, righe = pred, colonne = true
#se CMjj=1 allora la rete classifica quella clase alla prefezione ma se CMjj =/ 1 e CMij>0
#allora la classe j è scambiata con la classe i
def build_penalty_matrix(cm_norm):

    K = cm_norm.shape[0]
    P = np.zeros_like(cm_norm, dtype=np.float64)

    for i in range(K):
        for j in range(K):
            if i == j:
                continue    #per ogni CM_ij con i != j:
            val = cm_norm[i, j] #=CM_ij (probabilità)
            if val <= 0.0:
                continue
            P[i, j] += val  #aggiungi CM_ij a P_ij, P_ji, P_ii, P_jj
            P[j, i] += val
            P[i, i] += val
            P[j, j] += val

    print("[HIER] Penalty matrix P costruita.")
    return P #la matrice è simmetrica e tutti i valori sono ≥ 0

#S, matrice di similarità, simmetrica, diagonale = 1

def build_pearson_distance(P):
    """
    Similarità sim(i,j) = Pearson(p_i, p_j) tra le COLONNE di P,

    Distanza D(i,j) = 1 - sim(i,j).
    """
    K = P.shape[0]
    S = np.eye(K, dtype=np.float64)

    for i in range(K):
        for j in range(i + 1, K):   #per ogni coppia di classi i,j prendi il profilo di confusione
            pi = P[:, i]    #colonne classe j
            pj = P[:, j]    #colonne classe i

            if np.allclose(pi, pi[0]) or np.allclose(pj, pj[0]):
                corr = 0.0  #se un vettore è costante (varianza zero), metti corr = 0 per evitare NaN
            else:
                corr = np.corrcoef(pi, pj)[0, 1] #altrimenti np.corrcoef ti dà un numero tra -1 e 1

            S[i, j] = S[j, i] = corr    #Similarità sim(i,j) = Pearson(p_i, p_j) tra i due profili di confusione

    D = 1.0 - S     #Distanza D(i,j) = 1 - sim(i,j)
    np.fill_diagonal(D, 0.0)    #metti diagonale a 0
    print("[HIER] Matrice di similarità (Pearson) e distanza D costruite.")
    return D, S



#Dendrogramma + salvataggi


def plot_dendrogram(D, class_labels, savepath=None):
    assert D.shape[0] == D.shape[1], "D deve essere quadrata"
    condensed = squareform(D, checks=False) #trasformo in un vettore la traingolare superiore
    Z = linkage(condensed, method="average") #fa il clustering gerarchico agglomerativo usando la distanza D

    plt.figure(figsize=(9, 5))
    dendrogram(
        Z,
        labels=[str(c) for c in class_labels],
        orientation="top",
        distance_sort="ascending" #asse Y = distanza D al momento del merge.
    )
    plt.title("Dendrogramma classi (CM-based hierarchy)")
    plt.ylabel("D(i,j) = 1 - Pearson(p_i, p_j)")
    plt.tight_layout()
    if savepath is not None:
        savepath = Path(savepath)
        savepath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(savepath), dpi=150)
        print(f"[HIER] Dendrogramma salvato in {savepath}")
    plt.show()
    return Z

def save_hierarchy_matrices(cm_norm, cm_counts, P, D, class_labels, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir / "cm_norm.npy", cm_norm)
    np.save(outdir / "cm_counts.npy", cm_counts)
    np.save(outdir / "penalty_P.npy", P)
    np.save(outdir / "distance_D.npy", D)
    np.save(outdir / "class_labels.npy", np.array(class_labels, dtype=object))
    print(f"[HIER] Matrici salvate in {outdir}")


#MAIN: tutti gli step

def main():
    # Step 1 – flat EEGNet + pred su validation esterno
    y_val, y_pred_val, le, class_labels = train_flat_eegnet_and_get_val_predictions()

    # Step 2 – CM normalizzata (colonna = classe vera)
    cm_norm, cm_counts = build_normalized_cm(
        y_true_enc=y_val,
        y_pred_enc=y_pred_val,
        n_classes=len(class_labels)
    )

    # Step 3 – Penalty Matrix
    P = build_penalty_matrix(cm_norm)

    # Step 4 – Similarità Pearson + distanza
    D, S = build_pearson_distance(P)

    # Step 5 – Dendrogramma
    plot_dendrogram(D, class_labels, savepath=OUTDIR / f"dendrogram_cm_hierarchy_{DB_NAME.name}.png")

    # Salvataggio matrici
    save_hierarchy_matrices(cm_norm, cm_counts, P, D, class_labels, outdir=OUTDIR)


if __name__ == "__main__":
    main()
