import sys
import time
import traceback
from enum import Enum
from json import dump as json_dump, load as json_load, JSONEncoder
from multiprocessing import Queue, Process
from pathlib import Path
from pickle import dump as pkl_dump, load as pkl_load
from sys import platform

import numpy as np
from joblib.externals.loky import get_reusable_executor
from mne import create_info, Annotations
from mne.io import read_raw, RawArray
from scipy import signal
from scipy.interpolate import interp1d
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import column_or_1d
from sklearn.utils._array_api import get_namespace
from sklearn.utils._encode import _encode
from sklearn.utils.validation import check_is_fitted, _num_samples

from .handlers import select_folder_in_explorer

# config options
CONFIG_FILE = 'bionic_apps.cfg'
BASE_DIR = 'base_dir'

END_COLOR = '\033[0m'

class LabelEncoderCustom(LabelEncoder):

    #nuovo parametro binary_grouping, se false usa labelencoder come in sklearn, se true lo usa come funzione definita sotto
    def __init__(
        self,
        *,
        binary_grouping: bool = False,
        first_set=None,
        second_set=None,
    ):
        super().__init__()
        self.binary_grouping = binary_grouping
        self.first_set = first_set if first_set is not None else None
        self.second_set = second_set if second_set is not None else None
        self.keep_mask_ = None         #spazio per salvare la maschera dei campioni “tenuti” dopo il fit
        self._first_set_resolved_ = None
        self._second_set_resolved_ = None
        self._original_labels_ = None

    def fit(self, y):
        """Fit label encoder."""

        #modalità macro-classi
        y = np.asarray(y)   #verifica che y sia un array numpy

        #Caso normale: uso il LabelEncoder standard
        if not self.binary_grouping:
            return super().fit(y)

        #Caso binario: uso la logica first_set / second_set
        if self.first_set is None or self.second_set is None:
            raise ValueError("Per binary_grouping=True devi passare first_set e second_set.")

        #se non mi hanno passato i gruppi, uso i default (se ci sono)
        first_set = set(self.first_set)
        second_set = set(self.second_set)

        #tengo solo le etichette che stanno in uno dei due gruppi
        keep_mask = np.isin(y, list(first_set) + list(second_set))
        y_keep = y[keep_mask]
        self.keep_mask_ = keep_mask

        #0 = first_set, 1 = second_set
        _ = np.where(np.isin(y_keep, list(second_set)), 1, 0)

        # # 🔴 DEBUG
        # print("\n[DEBUG] LabelEncoder.fit (binary_grouping=True)")
        # print("  n_totale campioni:", len(y))
        # print("  n_tenuti (keep_mask True):", keep_mask.sum())
        # print("  n_scartati:", len(y) - keep_mask.sum())
        # print("  classi presenti in y:", np.unique(y))
        # print("  first_set:", first_set)
        # print("  second_set:", second_set)

        #sklearn si aspetta che ci sia classes_
        self.classes_ = np.array([0, 1], dtype=int)
        #opzionale: salvare le label originali
        self._original_labels_ = y_keep
        #salvo anche i gruppi risolti, così li riuso in transform
        self._first_set_resolved_ = first_set
        self._second_set_resolved_ = second_set
        return self

    '''def fit_transform(self, y):
        """Fit label encoder and return encoded labels.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        y : array-like of shape (n_samples,)
            Encoded labels.
        """
        y = column_or_1d(y, warn=True)
        self.classes_, y = _unique(y, return_inverse=True)
        return y'''

    def transform(self, y, return_mask: bool = False):
        """Transform labels to normalized encoding."""

        #modalità macro classi
        y = np.asarray(y)


        #Caso normale: delego a LabelEncoder
        if not self.binary_grouping:
            y_enc = super().transform(y)
            if return_mask:
                keep_mask = np.ones_like(y, dtype=bool)
                return y_enc, keep_mask
            return y_enc

        first_set = self._first_set_resolved_
        second_set = self._second_set_resolved_

        #Caso binario
        if first_set is None or second_set is None:
            raise RuntimeError("LabelEncoderCustom binary non inizializzato correttamente (chiama fit prima).")

        #ricostruisco la mask e tengo solo label valide
        keep_mask = np.isin(y, list(first_set) + list(second_set))
        y_keep = y[keep_mask]
        y_bin = np.where(np.isin(y_keep, list(second_set)), 1, 0) #mappo le label tenute in 0 e 1

        # # 🔴 DEBUG
        # print("\n[DEBUG] LabelEncoder.transform (binary_grouping=True)")
        # print("  len(y) in ingresso:", len(y))
        # print("  n_tenuti (keep_mask True):", keep_mask.sum())
        # print("  len(y_bin) in uscita:", len(y_bin))

        if return_mask:
            return y_bin, keep_mask
        return y_bin

    def fit_transform(self, y):
        """Fit + transform rispettando la logica di binary_grouping."""
        y = np.asarray(y)

        if not self.binary_grouping:
            # Comportamento standard sklearn
            return super().fit_transform(y)

        # Comportamento binario: usa il nostro fit + transform
        self.fit(y)
        y_bin, _ = self.transform(y, return_mask=True)
        return y_bin