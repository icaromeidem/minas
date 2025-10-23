
import pandas as pd
read_csv = pd.read_csv

# Wrapper para train_test_split
try:
    from sklearn.model_selection import train_test_split as _sk_train_test_split
    def train_test_split(*args, **kwargs):
        return _sk_train_test_split(*args, **kwargs)
except ImportError:
    def train_test_split(*args, **kwargs):
        raise ImportError('scikit-learn não está instalado. train_test_split não disponível.')

import os
from . import preprocess
from . import tuning
from . import models
from . import evaluation
from .bolometric import magbol
from .tuning import hyperparameter_search

# Expose metrics as mg.metrics
from .evaluation import _metrics as metrics

PARAM_ALIASES = {
    'teff': ['teff', 'Teff', 'TEFF', 'TEFF_ADOP', 'TEFF_SPEC', 'T_EFF', 'teff_2'],
    'logg': ['logg', 'Logg', 'LOGG', 'log g', 'logg_2'],
    'feh': ['feh', 'Feh', 'FEH', '[Fe/H]', '[M/H]', 'feh_2', '[M/H]_2'],
}


# Filter definitions for minas
FILTERS = {
    "JPLUS": [
        "uJAVA", "J0378", "J0395", "J0410", "J0430", "gSDSS",
        "J0515", "rSDSS", "J0660", "iSDSS", "J0861", "zSDSS",
    ],
    "SPLUS": [
        "uJAVA", "J0378", "J0395", "J0410", "J0430", "gSDSS",
        "J0515", "rSDSS", "J0660", "iSDSS", "J0861", "zSDSS",
    ],
    "JPAS": [
        "uJAVA", "J0378", "J0390", "J0400", "J0410", "J0420", "J0430", "J0440", "J0450", "J0460",
        "J0470", "J0480", "J0490", "J0500", "J0510", "J0520", "J0530", "J0540", "J0550", "J0560",
        "J0570", "J0580", "J0590", "J0600", "J0610", "J0620", "J0630", "J0640", "J0650", "J0660",
        "J0670", "J0680", "J0690", "J0700", "J0710", "J0720", "J0730", "J0740", "J0750", "J0760",
        "J0770", "J0780", "J0790", "J0800", "J0810", "J0820", "J0830", "J0840", "J0850", "J0860",
        "J0870", "J0880", "J0890", "J0900", "J0910", "J1007", "iSDSS"
    ],
    "WISE": ["W1", "W2", "J", "H", "K"],
    "GALEX": ["NUVmag"],
    "GAIA": ["G", "BP", "RP"],
}

ERRORS = {
    "JPLUS": [f"{x}_err" for x in FILTERS["JPLUS"]],
    "SPLUS": [f"{x}_err" for x in FILTERS["SPLUS"]],
    "JPAS": [f"{x}_err" for x in FILTERS["JPAS"]],
}

CORRECTIONS = {
    "JPLUS": [f"Ax_{x}" for x in FILTERS["JPLUS"]],
    "SPLUS": [f"Ax_{x}" for x in FILTERS["SPLUS"]],
    "JPAS": [f"Ax_{x}" for x in FILTERS["JPAS"]],
}

ERRORS = {
    "JPLUS": [f"{x}_err" for x in FILTERS["JPLUS"]],
    "SPLUS": [f"{x}_err" for x in FILTERS["SPLUS"]],
    "JPAS": [f"{x}_err" for x in FILTERS["JPAS"]],
}

CORRECTIONS = {
    "JPLUS": [f"Ax_{x}" for x in FILTERS["JPLUS"]],
    "SPLUS": [f"Ax_{x}" for x in FILTERS["SPLUS"]],
    "JPAS": [f"Ax_{x}" for x in FILTERS["JPAS"]],
}
