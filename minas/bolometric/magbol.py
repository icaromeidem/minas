"""
minas/bolometric/magbol.py
===========================
Bolometric correction module for MINAS.

Applies pre-trained Machine Learning models (XGBoost or Random Forest)
to predict bolometric correction (BC) from stellar atmospheric parameters
(Teff, logg, [Fe/H]), based on the grid of Jordi et al. (2010).

Quick start
-----------
>>> import minas as mg
>>> df = mg.bolometric.apply_bc(
...     data='catalog.csv',
...     teff_col='Teff',
...     logg_col='logg',
...     feh_col='[M/H]',
...     model_type='XGB',
... )

Reference
---------
Jordi, C. et al. (2010), A&A 523, A48.
https://doi.org/10.1051/0004-6361/200913234
"""

import os
import glob
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────

# Pre-trained model files bundled with the package
_MODEL_DIR = Path(__file__).parent / 'models_bc'

# Validation tables generated during training (relative to project root)
_VALIDATION_DIR = (
    Path(__file__).parent.parent.parent.parent.parent
    / 'apply_models' / 'bolometric_correction'
)

# Default statistics used when no validation file is found
_DEFAULT_STATS = {
    'XGB': {'std': 0.0430, 'mad': 0.0062, 'r2': 0.9983, 'n_samples': None},
    'RF':  {'std': 0.0573, 'mad': 0.0065, 'r2': 0.9970, 'n_samples': None},
}


# ── Private helpers ───────────────────────────────────────────────────────────

def _predict_xgb(model, X):
    """
    Run prediction with an XGBoost model loaded via ``load_model``.

    Ensures correct feature alignment by reading feature names from the
    booster object.

    Parameters
    ----------
    model : xgboost.XGBRegressor
        Loaded XGBoost model.
    X : pd.DataFrame
        Input feature DataFrame.

    Returns
    -------
    np.ndarray
        Predicted values.
    """
    booster  = model.get_booster()
    features = booster.feature_names
    dmat     = xgb.DMatrix(X[features], feature_names=features)
    return booster.predict(dmat)


def _load_model_stats(model_type):
    """
    Load validation statistics for the specified model type.

    Searches for the most recent validation CSV in the validation directory.
    Falls back to hard-coded default values if no files are found.

    Parameters
    ----------
    model_type : str
        ``'XGB'`` or ``'RF'``.

    Returns
    -------
    dict
        Keys: ``'std'``, ``'mad'``, ``'r2'``, ``'n_samples'``,
        and optionally ``'validation_file'``.
    """
    pattern = str(
        _VALIDATION_DIR / ('table_XG' if model_type == 'XGB' else 'table_RF')
        / ('*_pred_XG.csv' if model_type == 'XGB' else '*_pred_RF.csv')
    )
    files = glob.glob(pattern)

    if not files:
        return _DEFAULT_STATS[model_type].copy()

    latest = max(files, key=lambda x: Path(x).stem)
    df     = pd.read_csv(latest)

    residuals = df['BC_pred'] - df['BC_real']
    try:
        r2 = float(Path(latest).stem.split('_')[2])
    except Exception:
        r2 = None

    return {
        'std'            : float(np.std(residuals)),
        'mad'            : float(np.median(np.abs(residuals))),
        'r2'             : r2,
        'n_samples'      : len(df),
        'validation_file': Path(latest).name,
    }


# ── Public classes ────────────────────────────────────────────────────────────

class BolometricCorrection:
    """
    Load and apply a pre-trained bolometric correction model.

    Parameters
    ----------
    model_type : str, optional
        ``'XGB'`` (XGBoost, default) or ``'RF'`` (Random Forest).

    Raises
    ------
    ValueError
        If an unsupported model_type is provided.
    FileNotFoundError
        If the model file is not found in the package directory.

    Examples
    --------
    >>> bc = BolometricCorrection(model_type='XGB')
    >>> bc_values = bc.model.predict(X)
    """

    def __init__(self, model_type='XGB'):
        self.model_type = model_type.upper()
        if self.model_type not in ('XGB', 'RF'):
            raise ValueError("model_type must be 'XGB' or 'RF'.")
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the pre-trained model from the package model directory."""
        if self.model_type == 'XGB':
            path = _MODEL_DIR / 'model_bc_XGB.json'
            if not path.exists():
                raise FileNotFoundError(f"XGBoost model not found: {path}")
            self.model = xgb.XGBRegressor()
            self.model.load_model(str(path))

        elif self.model_type == 'RF':
            path = _MODEL_DIR / 'model_bc_RF.sav'
            if not path.exists():
                raise FileNotFoundError(f"Random Forest model not found: {path}")
            with open(path, 'rb') as f:
                self.model = pickle.load(f)


# ── Public API ────────────────────────────────────────────────────────────────

def apply_bc(
    data,
    teff_col,
    logg_col,
    feh_col,
    model_type='XGB',
    output_file=None,
    output_col='BC_pred',
    sigma_multiplier=3.0,
):
    """
    Apply bolometric correction to a stellar catalog.

    Loads a pre-trained model, predicts BC for each object, adds the
    prediction and its uncertainty to the DataFrame, and optionally
    saves the result to a CSV file.

    The uncertainty is defined as ``sigma_multiplier × STD``, where STD
    is the standard deviation of residuals on the internal validation set.

    Parameters
    ----------
    data : str or pd.DataFrame
        Path to a CSV file or a pandas DataFrame containing the catalog.
    teff_col : str
        Name of the effective temperature column (K).
    logg_col : str
        Name of the surface gravity column (cgs).
    feh_col : str
        Name of the metallicity column ([M/H], dex).
    model_type : str, optional
        ``'XGB'`` (default) or ``'RF'``.
    output_file : str or None, optional
        If provided, the result DataFrame is saved to this CSV path.
    output_col : str, optional
        Name of the output BC column. Default: ``'BC_pred'``.
    sigma_multiplier : float, optional
        Multiplier applied to the model STD to compute uncertainty.
        Default: 3.0 (3σ).

    Returns
    -------
    pd.DataFrame
        Input DataFrame with two new columns:
        ``output_col`` (predicted BC) and ``'err_' + output_col``
        (uncertainty).

    Raises
    ------
    FileNotFoundError
        If ``data`` is a string path that does not exist.
    ValueError
        If ``model_type`` is not ``'XGB'`` or ``'RF'``.

    Examples
    --------
    >>> import minas as mg

    >>> # Apply XGBoost BC with 3-sigma uncertainty
    >>> df = mg.bolometric.apply_bc(
    ...     data='catalog.csv',
    ...     teff_col='Teff',
    ...     logg_col='logg',
    ...     feh_col='[M/H]',
    ...     model_type='XGB',
    ...     sigma_multiplier=3.0,
    ...     output_file='catalog_bc.csv',
    ... )

    >>> # Apply Random Forest BC with 2-sigma uncertainty
    >>> df = mg.bolometric.apply_bc(
    ...     data=df,
    ...     teff_col='Teff',
    ...     logg_col='logg',
    ...     feh_col='[M/H]',
    ...     model_type='RF',
    ...     sigma_multiplier=2.0,
    ... )
    """
    # Load data
    if isinstance(data, str):
        if not os.path.exists(data):
            raise FileNotFoundError(f"File not found: {data}")
        df = pd.read_csv(data)
        print(f"Table loaded: {data}")
    else:
        df = data.copy()

    model_type = model_type.upper()
    if model_type not in ('XGB', 'RF'):
        raise ValueError("model_type must be 'XGB' or 'RF'.")

    # Load validation statistics
    info      = _load_model_stats(model_type)
    std       = info['std']
    mad       = info['mad']
    r2        = info.get('r2', 'N/A')
    n_samples = info.get('n_samples', 'N/A')

    print(f"\n{'='*60}")
    print(f"MODEL: {model_type}")
    print(f"{'='*60}")
    if 'validation_file' in info:
        print(f"Validation file       : {info['validation_file']}")
    print(f"R² Score              : {r2}")
    print(f"Validation samples    : {n_samples}")
    print(f"Std deviation (STD)   : {std:.4f} mag")
    print(f"MAD                   : {mad:.4f} mag")
    print(f"Multiplier            : {sigma_multiplier:.1f}x")
    uncertainty = sigma_multiplier * std
    print(f"Uncertainty           : {uncertainty:.4f} mag ({sigma_multiplier:.1f} x {std:.4f})")
    print(f"{'='*60}\n")

    # Rename columns to model format
    orig_teff, orig_logg, orig_feh = teff_col, logg_col, feh_col
    df.rename(columns={teff_col: 'Teff', logg_col: 'logg', feh_col: '[M/H]'}, inplace=True)

    if model_type == 'XGB':
        df.rename(columns={'[M/H]': 'MH'}, inplace=True)
        feature_cols = ['Teff', 'logg', 'MH']
    else:
        feature_cols = ['Teff', 'logg', '[M/H]']

    # Predict
    bc_model = BolometricCorrection(model_type=model_type)
    bc_pred  = _predict_xgb(bc_model.model, df) if model_type == 'XGB' \
               else bc_model.model.predict(df[feature_cols])

    df[output_col]          = bc_pred
    df['err_' + output_col] = uncertainty

    # Validation error statistics
    try:
        pattern = str(
            _VALIDATION_DIR / ('table_XG' if model_type == 'XGB' else 'table_RF')
            / ('*_pred_XG.csv' if model_type == 'XGB' else '*_pred_RF.csv')
        )
        files = glob.glob(pattern)
        if files:
            latest   = max(files, key=lambda x: Path(x).stem)
            df_val   = pd.read_csv(latest)
            abs_err  = np.abs(df_val['BC_pred'] - df_val['BC_real'])
            pct_user = (abs_err < uncertainty).mean() * 100
            pct_5x   = (abs_err >= 5 * std).mean() * 100
            print(f"ERROR STATISTICS (validation set):")
            print(f"{'='*60}")
            print(f"  Errors below {sigma_multiplier:.1f}x STD : {pct_user:.2f}%")
            print(f"  Errors >= 5x STD      : {pct_5x:.2f}%")
            print(f"{'='*60}\n")
    except Exception as e:
        print(f"Warning: could not compute validation statistics: {e}\n")

    # Revert column names
    if model_type == 'XGB':
        df.rename(columns={'MH': '[M/H]'}, inplace=True)
    df.rename(columns={'Teff': orig_teff, 'logg': orig_logg, '[M/H]': orig_feh}, inplace=True)

    if output_file is not None:
        df.to_csv(output_file, index=False)
        print(f"Saved to: {output_file}\n")

    return df