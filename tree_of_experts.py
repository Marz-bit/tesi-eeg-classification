# bionic_apps/ai/tree_of_experts.py
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from typing import Dict, List, Optional, Sequence

import keras

from bionic_apps.ai.interface import BaseNet
from bionic_apps.ai.eeg_nets import EEGNet

from bionic_apps.databases.eeg.defaults import (
    BOTH_LEGS, REST, RIGHT_HAND, SPATIAL, ROTATION, WORD, SUBTRACT, MUSIC, LEFT_HAND
)

# #SPLIT ORIGINARIO
# #mapping nodo-nome cartella su disco (come nel tuo CM_HIER_FALSEART_1-F)
# NODE_FOLDERS = {
#     "N1": "COGNITIVO vs. MOTORIO POSTURALE",
#     "N2": "SUBTRACT vs. PAROLE",
#     "N3": "MUSIC vs. WORD",
#     "N4": "POSTURALE vs. MOTORIO",
#     "N5": "REST vs. POSITION",
#     "N6": "ROTATION vs. SPATIAL",
#     "N7": "BOTH LEGS vs. UPPER",
#     "N8": "LEFT HAND vs. RIGHT HAND",
# }

# #HEATMAPS
# NODE_FOLDERS = {
#     "N1": "0_BLR_MWSRHLHS",
#     "N2": "5_R_BL",
#     "N3": "3_RHLHR_MWSS",
#     "N4": "7_RH_RLH",
#     "N5": "4_LH_R",
#     "N6": "6_MSW_S",
#     "N7": "2_M_SW",
#     "N8": "1_SPAT_W",
# }

# SOFT
# #mapping nodo-nome cartella su disco (come nel tuo CM_HIER_FALSEART_1-F)
# NODE_FOLDERS = {
#     "N1": "COGNITIVO vs. MOTORIO POSTURALE",
#     "N2": "MUSIC vs. MENTALE",
#     "N3": "WORD vs. SUBTRACT",
#     "N4": "POSTURALE vs. MOTORIO",
#     "N5": "REST vs. POSITION",
#     "N6": "ROTATION vs. SPATIAL",
#     "N7": "BOTH LEGS vs. UPPER",
#     "N8": "LEFT HAND vs. RIGHT HAND",
# }

# EMB
#mapping nodo-nome cartella su disco (come nel tuo CM_HIER_FALSEART_1-F)
NODE_FOLDERS = {
    "N1": "COGNITIVO vs. MOTORIO POSTURALE",
    "N2": "MUSIC vs. MENTALE",
    "N3": "WORD vs. SUBTRACT",
    "N4": "POSTURALE vs. MOTORIO",
    "N5": "SPATIAL vs. RR",
    "N6": "REST vs. ROTATION",
    "N7": "BOTH LEGS vs. UPPER",
    "N8": "LEFT HAND vs. RIGHT HAND",
}

# #mi serve perchè s_n sarà left_set U right_set e m_n=1 se e solo se label appartiene a S_n
# #mi serve anche per definire y_n=0 se classe vera è in left_set e y_n=1 se classe vera è in right_set
# NODE_SPLITS = {
#     "N1": ({SUBTRACT, MUSIC, WORD},
#            {RIGHT_HAND, LEFT_HAND, ROTATION, BOTH_LEGS, REST, SPATIAL}),
#
#     "N2": ({SUBTRACT},
#            {MUSIC, WORD}),
#
#     "N3": ({MUSIC},
#            {WORD}),
#
#     "N4": ({REST, ROTATION, SPATIAL},
#            {BOTH_LEGS, LEFT_HAND, RIGHT_HAND}),
#
#     "N5": ({REST},
#            {ROTATION, SPATIAL}),
#
#     "N6": ({ROTATION},
#            {SPATIAL}),
#
#     "N7": ({BOTH_LEGS},
#            {LEFT_HAND, RIGHT_HAND}),
#
#     "N8": ({LEFT_HAND},
#            {RIGHT_HAND}),
# }


# #HEATMAPS SPLITS
# NODE_SPLITS = {
#     "N1": ({BOTH_LEGS, REST},
#            {MUSIC, WORD, SUBTRACT, RIGHT_HAND, LEFT_HAND, ROTATION , SPATIAL}),
#
#     "N2": ({REST},
#            {BOTH_LEGS}),
#
#     "N3": ({RIGHT_HAND, LEFT_HAND, ROTATION},
#            {MUSIC, WORD, SPATIAL, SUBTRACT}),
#
#     "N4": ({RIGHT_HAND},
#            {ROTATION, LEFT_HAND}),
#
#     "N5": ({LEFT_HAND},
#            {ROTATION}),
#
#     "N6": ({MUSIC, SPATIAL, WORD},
#            {SUBTRACT}),
#
#     "N7": ({MUSIC},
#            {SPATIAL, WORD}),
#
#     "N8": ({SPATIAL},
#            {WORD}),
# }


# #SOFT
# NODE_SPLITS = {
#     "N1": ({SUBTRACT, MUSIC, WORD},
#            {RIGHT_HAND, LEFT_HAND, ROTATION, BOTH_LEGS, REST, SPATIAL}),
#
#     "N2": ({MUSIC},
#            {SUBTRACT, WORD}),
#
#     "N3": ({WORD},
#            {SUBTRACT}),
#
#     "N4": ({REST, ROTATION, SPATIAL},
#            {BOTH_LEGS, LEFT_HAND, RIGHT_HAND}),
#
#     "N5": ({REST},
#            {ROTATION, SPATIAL}),
#
#     "N6": ({ROTATION},
#            {SPATIAL}),
#
#     "N7": ({BOTH_LEGS},
#            {LEFT_HAND, RIGHT_HAND}),
#
#     "N8": ({LEFT_HAND},
#            {RIGHT_HAND}),
# }


# EMB
NODE_SPLITS = {
    "N1": ({SUBTRACT, MUSIC, WORD},
           {RIGHT_HAND, LEFT_HAND, ROTATION, BOTH_LEGS, REST, SPATIAL}),

    "N2": ({MUSIC},
           {SUBTRACT, WORD}),

    "N3": ({WORD},
           {SUBTRACT}),

    "N4": ({REST, ROTATION, SPATIAL},
           {BOTH_LEGS, LEFT_HAND, RIGHT_HAND}),

    "N5": ({SPATIAL},
           {ROTATION, REST}),

    "N6": ({REST},
           {ROTATION}),

    "N7": ({BOTH_LEGS},
           {LEFT_HAND, RIGHT_HAND}),

    "N8": ({LEFT_HAND},
           {RIGHT_HAND}),
}

 #Set delle 9 classi finali ammesse come foglie
LEAF_SET = {MUSIC, WORD, SUBTRACT, REST, SPATIAL, ROTATION, RIGHT_HAND, LEFT_HAND, BOTH_LEGS}

#Ogni expert produce una softmax a 2 classi (B,2). Dove B è batch size ossia quante
#finestre sto presentando e 2 il numero di classi. Queste funzioni estraggono:
#p0 = P(classe0 | x) (first_set)
#p1 = P(classe1 | x) (second_set)

def _p1(two_class_softmax):
    # (B,2) -> P(class=1) (B,)
    return keras.ops.take(two_class_softmax, 1, axis=-1)


def _p0(two_class_softmax):
    # (B,2) -> P(class=0) (B,)
    return keras.ops.take(two_class_softmax, 0, axis=-1)

#Serve solo a trasformare (B,) in (B,1) prima della concatenazione finale
def _expand1(x):
    return keras.ops.expand_dims(x, axis=-1)

#per ogni esempio del batch ottengo un vettore di 9 numeri, a riga out[i, :] è la distribuzione sulle
#9 classi del campione i-esimo del batch.
class TreeOfExpertsEEGNet(BaseNet):
    """
    Tree of Experts basato su 8 EEGNet binari (pre-addestrati).
    Output: softmax (B, 9) sulle foglie finali.

    load_weights(weight_file) accetta un JSON:
      {"N1": "/path/...weights.h5", ..., "N8": "/path/...weights.h5"}
    """

    def __init__(
        self,
        input_shape,
        classes: int,
        *,
        fs: int,
        leaf_labels: Optional[Sequence[str]] = None,
        save_path: str = "tf_log/",
        learning_rate: float = 1e-3,
        eegnet_kwargs: Optional[dict] = None,
        node_loss_weight: float = 1,
        # node_weights: Optional[dict] = None,
    ):
        assert fs is not None, "fs è richiesto per EEGNet"
        self._fs = fs
        self._eegnet_kwargs = eegnet_kwargs or {}

        #ordine foglie = ordine output
        if leaf_labels is None:
            #fallback, meglio passarlo da label_encoder.classes_
            leaf_labels = [MUSIC, WORD, SUBTRACT, REST, SPATIAL, ROTATION, RIGHT_HAND, LEFT_HAND, BOTH_LEGS]

        self._leaf_labels = list(leaf_labels)
        missing = set(self._leaf_labels) - LEAF_SET
        if missing:
            raise ValueError(f"leaf_labels contiene label non previste dal tree: {missing}")

        #precalcola (una volta sola) le tabelle per ottenere m_n e y_n da y_true
        self._node_order = list(NODE_FOLDERS.keys())  # ["N1"... "N8"] #Serve per avere un ordine fisso dei nodi (N1..N8)
        self._subtree_idx = {} #quali colonne (foglie) della (B,9) appartengono al sottoalbero S_n
        self._right_idx = {} #quali colonne appartengono al ramo destro del nodo

        self._node_loss_weight = float(node_loss_weight) #peso di più i nodi più difficili


        for n in self._node_order:
            left_set, right_set = NODE_SPLITS[n]
            subtree = set(left_set) | set(right_set) #S_n

            self._subtree_idx[n] = [self._leaf_labels.index(l) for l in subtree]
            self._right_idx[n] = [self._leaf_labels.index(l) for l in right_set]

        # lookup tables per ottenere m_n e y_n a partire da y_true (classe 0..8)
        n_classes = len(self._leaf_labels)
        n_nodes = len(self._node_order)

        mask_tab = np.zeros((n_classes, n_nodes), dtype="float32")
        targ_tab = np.zeros((n_classes, n_nodes), dtype="float32")

        for c, lab in enumerate(self._leaf_labels):
            for j, n in enumerate(self._node_order):
                left_set, right_set = NODE_SPLITS[n]
                if (lab in left_set) or (lab in right_set):
                    mask_tab[c, j] = 1.0 #m_n, se la label c appartiene al sottoalbero di nodo j
                    targ_tab[c, j] = 1.0 if (lab in right_set) else 0.0 #y_n, se la label c è nel right_set del nodo j, altrimenti 0

        self._mask_tab = keras.ops.array(mask_tab)  # (9,8)
        self._targ_tab = keras.ops.array(targ_tab)  # (9,8)

        #Crea il contenitore degli expert
        self._experts: Dict[str, keras.Model] = {}
        super().__init__(input_shape, classes, save_path=save_path, learning_rate=learning_rate)

    #E' come costruisco ogni nodo, ogni nodo è una EEG NET binaria, è la shape della rete binaria non contiene ancora i pesi
    def _make_expert_model(self, node_name: str) -> keras.Model:
        nn = EEGNet(self._input_shape, 2, fs=self._fs, save_path=str(self._save_path), **self._eegnet_kwargs)
        return nn._model  #modello keras

    def _build_graph(self):
        inp = keras.layers.Input(shape=self._input_shape, name="eeg") #lo stesso input alimenta tutti gli experts

        #costruiamo gli 8 expert e prendiamo le loro uscite softmax (B,2)
        node_out = {}
        for n in NODE_FOLDERS.keys():
            m = self._make_expert_model(n) #metto i pesi nell'architettura costruendo il modello keras da utilizzare
            m._name = f"{n}_expert"
            self._experts[n] = m #mettop i pesi di self._experts[n] in m che costruisce l'archietettura dei binari EEGNET
            node_out[n] = m(inp) #output è vettore binarrio uscito da rete binaria

        #convenzione: class0 = first_set, class1 = second_set
        #quindi prendiamo p0 e p1 per ogni nodo
        p = {}
        for n, out in node_out.items():
            p[(n, 0)] = _p0(out)
            p[(n, 1)] = _p1(out)

        eps = 1e-7

        def _clip01(x):
            return keras.ops.clip(x, eps, 1.0 - eps)

        def _log(x):
            return keras.ops.log(_clip01(x))

        logp = {}
        for n in self._node_order:
            logp[(n, 0)] = keras.layers.Lambda(lambda x: _log(x), name=f"logp_{n}_0")(p[(n, 0)])
            logp[(n, 1)] = keras.layers.Lambda(lambda x: _log(x), name=f"logp_{n}_1")(p[(n, 1)])
        #FINE NUOVO

        #node right probabilities (B,8) = p_n = p right usato nella BCE per nodo
        node_cols = []
        for n in self._node_order:
            node_cols.append(keras.layers.Lambda(_expand1, name=f"p_right_{n}")(p[(n, 1)]))
        self._p_right_nodes = keras.layers.Concatenate(axis=-1, name="p_right_nodes")(node_cols)

        #PRIMO SPLIT
        #alias nodi per leggibilità (rispecchiano i tuoi split)
        #N1: COGN (0) vs MOTORPOST (1)
        #N2: SUBTRACT (0) vs PAROLE (1)
        #N3: MUSIC (0) vs WORD (1)
        #N4: POSTURALE (0) vs MOTORIO (1)
        #N5: REST (0) vs POSITION (1)
        #N6: ROTATION (0) vs SPATIAL (1)
        #N7: BOTH_LEGS (0) vs UPPER (1)
        #N8: LEFT (0) vs RIGHT (1)

        #MOLTIPLICAZIONE
        # leaves = {
        #     SUBTRACT:   p[("N1", 0)] * p[("N2", 0)],
        #     MUSIC:      p[("N1", 0)] * p[("N2", 1)] * p[("N3", 0)],
        #     WORD:       p[("N1", 0)] * p[("N2", 1)] * p[("N3", 1)],
        #
        #     REST:       p[("N1", 1)] * p[("N4", 0)] * p[("N5", 0)],
        #     ROTATION:   p[("N1", 1)] * p[("N4", 0)] * p[("N5", 1)] * p[("N6", 0)],
        #     SPATIAL:    p[("N1", 1)] * p[("N4", 0)] * p[("N5", 1)] * p[("N6", 1)],
        #
        #     BOTH_LEGS:  p[("N1", 1)] * p[("N4", 1)] * p[("N7", 0)],
        #     LEFT_HAND:  p[("N1", 1)] * p[("N4", 1)] * p[("N7", 1)] * p[("N8", 0)],
        #     RIGHT_HAND: p[("N1", 1)] * p[("N4", 1)] * p[("N7", 1)] * p[("N8", 1)],
        # }

        #SOMMA
        # # foglie in log-space: somma dei log invece del prodotto delle prob
        # leaf_log = {
        #     SUBTRACT: logp[("N1", 0)] + logp[("N2", 0)],
        #     MUSIC: logp[("N1", 0)] + logp[("N2", 1)] + logp[("N3", 0)],
        #     WORD: logp[("N1", 0)] + logp[("N2", 1)] + logp[("N3", 1)],
        #
        #     REST: logp[("N1", 1)] + logp[("N4", 0)] + logp[("N5", 0)],
        #     ROTATION: logp[("N1", 1)] + logp[("N4", 0)] + logp[("N5", 1)] + logp[("N6", 0)],
        #     SPATIAL: logp[("N1", 1)] + logp[("N4", 0)] + logp[("N5", 1)] + logp[("N6", 1)],
        #
        #     BOTH_LEGS: logp[("N1", 1)] + logp[("N4", 1)] + logp[("N7", 0)],
        #     LEFT_HAND: logp[("N1", 1)] + logp[("N4", 1)] + logp[("N7", 1)] + logp[("N8", 0)],
        #     RIGHT_HAND: logp[("N1", 1)] + logp[("N4", 1)] + logp[("N7", 1)] + logp[("N8", 1)],
        # }

        # #HEATMAPS
        # # foglie in log-space: somma dei log invece del prodotto delle prob
        # leaf_log = {
        #     SUBTRACT: logp[("N1", 1)] + logp[("N3", 1)] +  logp[("N6", 1)],
        #     MUSIC: logp[("N1", 1)] + logp[("N3", 1)] + logp[("N6", 0)] +  logp[("N7", 0)],
        #     WORD: logp[("N1", 1)] + logp[("N3", 1)] + logp[("N6", 0)] +  logp[("N7", 1)] +  logp[("N8", 1)],
        #     SPATIAL: logp[("N1", 1)] + logp[("N3", 1)] + logp[("N6", 0)] +  logp[("N7", 1)] +  logp[("N8", 0)],
        #
        #     REST: logp[("N1", 0)] + logp[("N2", 0)],
        #     BOTH_LEGS: logp[("N1", 0)] + logp[("N2", 1)],
        #
        #     LEFT_HAND: logp[("N1", 1)] + logp[("N3", 0)] + logp[("N4", 1)] + logp[("N5", 0)],
        #     RIGHT_HAND: logp[("N1", 1)] + logp[("N3", 0)] + logp[("N4", 0)],
        #     ROTATION: logp[("N1", 1)] + logp[("N3", 0)] + logp[("N4", 1)] + logp[("N5", 1)],
        # }


        # #SOFT
        # # foglie in log-space: somma dei log invece del prodotto delle prob
        # leaf_log = {
        #     SUBTRACT: logp[("N1", 0)] + logp[("N2", 1)] + logp[("N3", 1)],
        #     MUSIC: logp[("N1", 0)] + logp[("N2", 0)],
        #     WORD: logp[("N1", 0)] + logp[("N2", 1)] + logp[("N3", 0)],
        #
        #     REST: logp[("N1", 1)] + logp[("N4", 0)] + logp[("N5", 0)],
        #     ROTATION: logp[("N1", 1)] + logp[("N4", 0)] + logp[("N5", 1)] + logp[("N6", 0)],
        #     SPATIAL: logp[("N1", 1)] + logp[("N4", 0)] + logp[("N5", 1)] + logp[("N6", 1)],
        #
        #     BOTH_LEGS: logp[("N1", 1)] + logp[("N4", 1)] + logp[("N7", 0)],
        #     LEFT_HAND: logp[("N1", 1)] + logp[("N4", 1)] + logp[("N7", 1)] + logp[("N8", 0)],
        #     RIGHT_HAND: logp[("N1", 1)] + logp[("N4", 1)] + logp[("N7", 1)] + logp[("N8", 1)],
        # }

        #EMB
        # foglie in log-space: somma dei log invece del prodotto delle prob
        leaf_log = {
            SUBTRACT: logp[("N1", 0)] + logp[("N2", 1)] + logp[("N3", 1)],
            MUSIC: logp[("N1", 0)] + logp[("N2", 0)],
            WORD: logp[("N1", 0)] + logp[("N2", 1)] + logp[("N3", 0)],

            REST: logp[("N1", 1)] + logp[("N4", 0)] + logp[("N5", 1)] + logp[("N6", 0)],
            ROTATION: logp[("N1", 1)] + logp[("N4", 0)] + logp[("N5", 1)] + logp[("N6", 1)],
            SPATIAL: logp[("N1", 1)] + logp[("N4", 0)] + logp[("N5", 0)],

            BOTH_LEGS: logp[("N1", 1)] + logp[("N4", 1)] + logp[("N7", 0)],
            LEFT_HAND: logp[("N1", 1)] + logp[("N4", 1)] + logp[("N7", 1)] + logp[("N8", 0)],
            RIGHT_HAND: logp[("N1", 1)] + logp[("N4", 1)] + logp[("N7", 1)] + logp[("N8", 1)],
        }

        #PER LEAF LOSS
        #def _log_expand(x):
            #return keras.ops.expand_dims(keras.ops.log(x + 1e-7), axis=-1)

        #PER MOLTIPLICAZIONE
        # log_cols = []
        # for lab in self._leaf_labels:
        #     log_cols.append(keras.layers.Lambda(_log_expand, name=f"logleaf_{lab}")(leaves[lab]))
        #
        # logits = keras.layers.Concatenate(axis=-1, name="leaf_logits")(log_cols)
        # out = keras.layers.Softmax(axis=-1, name="leaf_probs")(logits)
        # return inp, out

        #PER USARE SUM
        # Stack dei logits secondo self._leaf_labels (ordine coerente col label_encoder)
        log_cols = []
        for lab in self._leaf_labels:
            log_cols.append(keras.layers.Lambda(_expand1, name=f"leaflog_{lab}")(leaf_log[lab]))

        logits = keras.layers.Concatenate(axis=-1, name="leaf_logits")(log_cols)
        out = keras.layers.Softmax(axis=-1, name="leaf_probs")(logits)
        return inp, out
        #FINE NUOVO


    def _masked_local_node_loss(self, y_true, leaf_probs):
        """
        leaf_probs: (B,9) = output finale del tree
        y_true: (B,) con class index 0..8 coerente con self._leaf_labels
        """
        eps = 1e-7

        y_true = keras.ops.reshape(y_true, (-1,))
        y_true = keras.ops.cast(y_true, "int32")

        # (B,8) mask e target binario per nodo
        m = keras.ops.take(self._mask_tab, y_true, axis=0)
        y = keras.ops.take(self._targ_tab, y_true, axis=0)

        # (B,8) p_n ricavati da leaf_probs
        p_list = []
        for n in self._node_order:
            #sub= somma delle leaf_probs sulle foglie del sottoalbero → massa del nodo
            sub = keras.ops.sum(keras.ops.take(leaf_probs, self._subtree_idx[n], axis=-1), axis=-1)

            #rig = somma delle leaf_probs sulle foglie del ramo destro
            rig = keras.ops.sum(keras.ops.take(leaf_probs, self._right_idx[n], axis=-1), axis=-1)
            p = rig / (sub + eps)  # p_right del nodo
            p_list.append(p)

        p = keras.ops.stack(p_list, axis=-1)
        p = keras.ops.clip(p, eps, 1.0 - eps)

        #BCE per nodo (B,8)
        bce = -(y * keras.ops.log(p) + (1.0 - y) * keras.ops.log(1.0 - p))

        #masko: L_n = m_n * BCE
        node_loss = m * bce

        #normalizzazione per-sample: sum(L_n) / (sum(m_n)+eps)
        num = keras.ops.sum(node_loss, axis=-1)
        den = keras.ops.sum(m, axis=-1) + eps
        return num / den

    def _leaf_loss(self, y_true, leaf_probs):
        # y_true: (B,) int 0..8 coerente con self._leaf_labels
        y_true = keras.ops.reshape(y_true, (-1,))
        y_true = keras.ops.cast(y_true, "int32")

        # evita log(0)
        eps = 1e-7
        leaf_probs = keras.ops.clip(leaf_probs, eps, 1.0 - eps)

        # ritorna un vettore (B,) di loss per-sample (Keras poi fa la media)
        return keras.losses.sparse_categorical_crossentropy(y_true, leaf_probs)

    def _hybrid_loss(self, y_true, leaf_probs):
        return self._leaf_loss(y_true, leaf_probs) + self._node_loss_weight * self._masked_local_node_loss(y_true, leaf_probs)

    def _compile_model(self, learning_rate=0.001):
        self._model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            #loss=self._masked_local_node_loss,
            #loss=self._leaf_loss,
            loss=self._hybrid_loss,
            metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
        )

    # #come carico i pesi base per nodo, prendo un file JSON che contiene i pesi per ognuno degli
    # #N split, per ogni nodo N1...N8 prendi il modello Keras dell’expert di quel nodo e carica il file
    # #.weights.h5 associato
    # def load_weights(self, weight_file: str):
    #     """
    #     weight_file deve essere un JSON con mapping nodo->file pesi (.weights.h5)
    #     """
    #     p = Path(weight_file) #cartella creata in tree_utils dove ci sono i 4 json con i pesi
    #     if p.suffix.lower() != ".json":
    #         raise ValueError(
    #             "TreeOfExpertsEEGNet.load_weights si aspetta un JSON nodo->pesi. "
    #             f"Ricevuto: {weight_file}"
    #         )
    #
    #     #Legge il JSON e lo trasforma in un dizionario Python dove mp è un dict tipo {"N1": "...h5", ..., "N8": "...h5"}
    #     # mp["N1"] è una stringa con il path ai pesi di N1
    #     with open(p, "r") as f:
    #         mp = json.load(f) #dizionario Python ottenuto leggendo il JSON
    #
    #     #n è il nome del nodo corrente nel loop
    #     for n in NODE_FOLDERS.keys():
    #         if n not in mp:
    #             raise KeyError(f"Nel JSON manca il nodo {n}")
    #         self._experts[n].load_weights(mp[n]) #Quindi self._experts[n] è il modello Keras dell’expert del nodo n
    #         #self._experts[n] = il modello Keras dell’expert del nodo n
    #         #mp[n] = il path al file .weights.h5 per quel nodo

    def load_weights(self, weight_file: str):
        """
        - Se .json: mapping nodo->pesi (come ora)
        - Altrimenti: carica i pesi dell'intero modello (tree end-to-end)
        """
        p = Path(weight_file)

        # Caso 1: JSON nodo -> pesi expert
        if p.suffix.lower() == ".json":
            with open(p, "r") as f:
                mp = json.load(f)

            for n in NODE_FOLDERS.keys():
                if n not in mp:
                    raise KeyError(f"Nel JSON manca il nodo {n}")
                self._experts[n].load_weights(mp[n])
            return

        #Caso 2: file pesi unico del tree (end-to-end)
        self._model.load_weights(str(p))

    #pensato per un futuro TL, congelo i layer con nome che inizia con "base_model"
    def train_base_model(self, trainable: bool, learning_rate=None, *, freeze_bn=True):
        """
        Congela/scongela i layer base_model* dentro OGNI expert.
        Questo serve perché in TL tu fai clf.train_base_model(False).
        """
        for expert in self._experts.values():
            for layer in expert.layers:
                if layer.name.startswith("base_model"):
                    layer.trainable = trainable

        if freeze_bn:
            for expert in self._experts.values():
                for layer in expert.layers:
                    if isinstance(layer, keras.layers.BatchNormalization):
                        layer.trainable = False

        if learning_rate is None:
            learning_rate = self._learning_rate
        self._compile_model(learning_rate)
