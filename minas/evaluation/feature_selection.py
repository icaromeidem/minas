"""
minas/evaluation/feature_selection.py
=======================================
Feature importance utilities for Random Forest and XGBoost regressors.

Provides two strategies:
  - **Impurity-based** (``get_important_features*``): fast, uses
    ``feature_importances_`` from the fitted estimator.
  - **Permutation-based** (``get_permutation_importance*``): slower but
    more reliable, especially for correlated features.

Both strategies return the top-N features and a summary DataFrame.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer, r2_score

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None


# ── Shared helpers ────────────────────────────────────────────────────────────

def _mad_score(y_true, y_pred):
    """MAD scorer for use with scikit-learn's make_scorer."""
    return float(np.median(np.abs(y_true - y_pred)))


def _resolve_scorer(scoring):
    """Convert a scoring string to a scikit-learn scorer object."""
    if scoring == 'mad':
        return make_scorer(_mad_score, greater_is_better=False)
    elif scoring == 'r2':
        return 'r2'
    return scoring


def _build_importance_df(feature_names, importances, n_features_to_save):
    """Build a sorted importance DataFrame and return the top-N features."""
    feature_names = np.array(feature_names)
    importances   = np.array(importances)
    sorted_idx    = np.argsort(importances)[::-1]

    sorted_features    = feature_names[sorted_idx]
    sorted_importances = importances[sorted_idx]
    cumulative         = np.cumsum(sorted_importances)

    df_feat = pd.DataFrame({
        'feature'              : sorted_features,
        'importance'           : sorted_importances,
        'cumulative_importance': cumulative,
    })
    return sorted_features[:n_features_to_save].tolist(), df_feat


def _build_xgb(params, random_state, n_estimators):
    """Instantiate an XGBRegressor with safe defaults."""
    if XGBRegressor is None:
        raise ImportError("xgboost is not installed. Run: pip install xgboost")
    if params is not None:
        p = params.copy()
        p.setdefault('random_state', random_state)
        p.setdefault('n_estimators', n_estimators)
        p.setdefault('n_jobs', -1)
        p.setdefault('verbosity', 0)
        return XGBRegressor(**p)
    return XGBRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        verbosity=0,
        n_jobs=-1,
    )


# ── Impurity-based importance ─────────────────────────────────────────────────

def get_important_features(X, y, n_features_to_save=10, params=None,
                           random_state=42, n_estimators=100):
    """
    Rank features by impurity-based importance using RandomForestRegressor.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : array-like
        Target variable.
    n_features_to_save : int, optional
        Number of top features to return. Default: 10.
    params : dict or None, optional
        Custom hyperparameters for RandomForestRegressor.
    random_state : int, optional
        Random seed. Default: 42.
    n_estimators : int, optional
        Number of trees (used only when ``params`` is None). Default: 100.

    Returns
    -------
    selected_features : list of str
        Names of the top-N most important features.
    df_feat : pd.DataFrame
        Columns: ``['feature', 'importance', 'cumulative_importance']``.

    Examples
    --------
    >>> features, df = get_important_features(work_df, catalog['Teff'], n_features_to_save=20)
    """
    rf = RandomForestRegressor(**params) if params else \
         RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    rf.fit(X, y)
    return _build_importance_df(X.columns, rf.feature_importances_, n_features_to_save)


def get_important_features_xgb(X, y, n_features_to_save=10, params=None,
                                random_state=42, n_estimators=100):
    """
    Rank features by impurity-based importance using XGBRegressor.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : array-like
        Target variable.
    n_features_to_save : int, optional
        Number of top features to return. Default: 10.
    params : dict or None, optional
        Custom hyperparameters for XGBRegressor.
    random_state : int, optional
        Random seed. Default: 42.
    n_estimators : int, optional
        Number of trees (used only when ``params`` is None). Default: 100.

    Returns
    -------
    selected_features : list of str
    df_feat : pd.DataFrame

    Raises
    ------
    ImportError
        If xgboost is not installed.
    """
    xgb = _build_xgb(params, random_state, n_estimators)
    xgb.fit(X, y)
    return _build_importance_df(X.columns, xgb.feature_importances_, n_features_to_save)


# ── Permutation importance ────────────────────────────────────────────────────

def get_permutation_importance_rf(X, y, n_features_to_save=10, params=None,
                                  random_state=42, n_estimators=100, scoring='r2'):
    """
    Rank features by permutation importance using RandomForestRegressor.

    Permutation importance is more reliable than impurity-based importance
    for correlated or high-cardinality features.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : array-like
        Target variable.
    n_features_to_save : int, optional
        Number of top features to return. Default: 10.
    params : dict or None, optional
        Custom hyperparameters for RandomForestRegressor.
    random_state : int, optional
        Random seed. Default: 42.
    n_estimators : int, optional
        Number of trees (used only when ``params`` is None). Default: 100.
    scoring : str, optional
        Scoring metric: ``'r2'`` (default) or ``'mad'``.

    Returns
    -------
    selected_features : list of str
    df_feat : pd.DataFrame

    Examples
    --------
    >>> features, df = get_permutation_importance_rf(
    ...     work_df, catalog['Teff'], n_features_to_save=20, scoring='r2')
    """
    rf = RandomForestRegressor(**params) if params else \
         RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    rf.fit(X, y)
    result = permutation_importance(
        rf, X, y, n_repeats=10,
        random_state=random_state,
        scoring=_resolve_scorer(scoring),
    )
    return _build_importance_df(X.columns, result.importances_mean, n_features_to_save)


def get_permutation_importance_xgb(X, y, n_features_to_save=10, params=None,
                                   random_state=42, n_estimators=100, scoring='r2'):
    """
    Rank features by permutation importance using XGBRegressor.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : array-like
        Target variable.
    n_features_to_save : int, optional
        Number of top features to return. Default: 10.
    params : dict or None, optional
        Custom hyperparameters for XGBRegressor.
    random_state : int, optional
        Random seed. Default: 42.
    n_estimators : int, optional
        Number of trees (used only when ``params`` is None). Default: 100.
    scoring : str, optional
        Scoring metric: ``'r2'`` (default) or ``'mad'``.

    Returns
    -------
    selected_features : list of str
    df_feat : pd.DataFrame

    Raises
    ------
    ImportError
        If xgboost is not installed.
    """
    xgb = _build_xgb(params, random_state, n_estimators)
    xgb.fit(X, y)
    result = permutation_importance(
        xgb, X, y, n_repeats=10,
        random_state=random_state,
        scoring=_resolve_scorer(scoring),
    )
    return _build_importance_df(X.columns, result.importances_mean, n_features_to_save)