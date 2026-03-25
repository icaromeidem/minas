"""
minas/evaluation/_metrics.py
=============================
Regression metrics for model evaluation.

Provides MAD, R² and a binned MAD summary table used throughout
the MINAS evaluation workflow.
"""

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import r2_score as _sk_r2_score
except ImportError:
    _sk_r2_score = None


def mad(y_true, y_pred):
    """
    Compute the Median Absolute Deviation (MAD) between predictions and true values.

    Parameters
    ----------
    y_true : array-like
        Ground-truth target values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    float
        Median absolute deviation: ``median(|y_pred - y_true|)``.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.median(np.abs(y_true - y_pred))


def r2_score(y_true, y_pred):
    """
    Compute the R² (coefficient of determination) score using scikit-learn.

    Parameters
    ----------
    y_true : array-like
        Ground-truth target values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    float
        R² score. Best value is 1.0.

    Raises
    ------
    ImportError
        If scikit-learn is not installed.
    """
    if _sk_r2_score is None:
        raise ImportError("scikit-learn is not installed. r2_score is unavailable.")
    return _sk_r2_score(y_true, y_pred)


def calculate_mad(predictions, true_values, bins, param_unit):
    """
    Compute MAD statistics per bin and for the full sample.

    Parameters
    ----------
    predictions : pd.Series
        Model predictions, indexed consistently with ``true_values``.
    true_values : pd.Series
        Ground-truth target values.
    bins : list of float
        Bin edges. If empty, a single bin covering the full range is used.
    param_unit : str
        Physical unit of the parameter (e.g. ``'K'``, ``'dex'``).
        Used for labelling the bin intervals.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``['bin', 'objects', 'mad']``, one row
        per bin plus a final row for the full sample.

    Examples
    --------
    >>> bins = [4000, 5000, 6000, 7000]
    >>> df_mad = calculate_mad(y_pred, y_true, bins, param_unit='K')
    """
    if not bins:
        bins = [min(true_values) - 1, max(true_values) + 1]

    df = pd.merge(
        left=true_values, left_index=True,
        right=predictions, right_index=True,
    )
    df.columns = ["TRUE_VALUE", "PREDICTION"]

    bins_intervals = []
    bins_sizes     = []
    bins_mads      = []

    for k in range(len(bins) - 1):
        bin_min, bin_max = bins[k], bins[k + 1]
        bins_intervals.append(f"[{bin_min} {param_unit}, {bin_max} {param_unit}]")
        df_bin = df[(df["TRUE_VALUE"] >= bin_min) & (df["TRUE_VALUE"] < bin_max)].copy()
        bins_sizes.append(len(df_bin))
        errors = df_bin["PREDICTION"] - df_bin["TRUE_VALUE"]
        bins_mads.append(np.median(np.abs(errors)))

    # Full sample row
    bins_intervals.append("Full Sample")
    bins_sizes.append(len(df))
    error = df["PREDICTION"] - df["TRUE_VALUE"]
    bins_mads.append(np.median(np.abs(error)))

    return pd.DataFrame({
        "bin"    : bins_intervals,
        "objects": bins_sizes,
        "mad"    : bins_mads,
    })