from sklearn.metrics import make_scorer, r2_score
def mad_score(y_true, y_pred):
    return float(np.median(np.abs(y_true - y_pred)))
from sklearn.inspection import permutation_importance
def get_permutation_importance_rf(X, y, n_features_to_save=10, params=None, random_state=42, n_estimators=100, scoring='r2'):
    """
    Evaluates feature importance using permutation importance with RandomForestRegressor.
    Returns the n_features_to_save most important features.
    """
    if params is not None:
        rf = RandomForestRegressor(**params)
    else:
        rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    rf.fit(X, y)
    if scoring == 'mad':
        scorer = make_scorer(mad_score, greater_is_better=False)
    elif scoring == 'r2':
        scorer = 'r2'
    else:
        scorer = scoring
    result = permutation_importance(rf, X, y, n_repeats=10, random_state=random_state, scoring=scorer)
    importances = result.importances_mean
    feature_names = np.array(X.columns)
    sorted_idx = np.argsort(importances)[::-1]
    sorted_importances = importances[sorted_idx]
    sorted_features = feature_names[sorted_idx]
    selected_features = sorted_features[:n_features_to_save]
    cumulative_importance = np.cumsum(sorted_importances)
    df_feat = pd.DataFrame({
        'feature': sorted_features,
        'importance': sorted_importances,
        'cumulative_importance': cumulative_importance
    })
    return selected_features.tolist(), df_feat
def get_permutation_importance_xgb(X, y, n_features_to_save=10, params=None, random_state=42, n_estimators=100, scoring='r2'):
    """
    Evaluates feature importance using permutation importance with XGBRegressor.
    Returns the n_features_to_save most important features.
    """
    if params is not None:
        user_params = params.copy()
        if 'random_state' not in user_params:
            user_params['random_state'] = random_state
        if 'n_estimators' not in user_params:
            user_params['n_estimators'] = n_estimators
        if 'n_jobs' not in user_params:
            user_params['n_jobs'] = -1
        if 'verbosity' not in user_params:
            user_params['verbosity'] = 0
        xgb = XGBRegressor(**user_params)
    else:
        xgb = XGBRegressor(n_estimators=n_estimators, random_state=random_state, verbosity=0, n_jobs=-1)
    xgb.fit(X, y)
    if scoring == 'mad':
        scorer = make_scorer(mad_score, greater_is_better=False)
    elif scoring == 'r2':
        scorer = 'r2'
    else:
        scorer = scoring
    result = permutation_importance(xgb, X, y, n_repeats=10, random_state=random_state, scoring=scorer)
    importances = result.importances_mean
    feature_names = np.array(X.columns)
    sorted_idx = np.argsort(importances)[::-1]
    sorted_importances = importances[sorted_idx]
    sorted_features = feature_names[sorted_idx]
    selected_features = sorted_features[:n_features_to_save]
    cumulative_importance = np.cumsum(sorted_importances)
    df_feat = pd.DataFrame({
        'feature': sorted_features,
        'importance': sorted_importances,
        'cumulative_importance': cumulative_importance
    })
    return selected_features.tolist(), df_feat

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

def get_important_features(X, y, n_features_to_save=10, params=None, random_state=42, n_estimators=100):
    """
    Evaluates feature importance using RandomForestRegressor.
    Returns the n_features_to_save most important features.
    """
    if params is not None:
        rf = RandomForestRegressor(**params)
    else:
        rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    rf.fit(X, y)
    importances = rf.feature_importances_
    feature_names = np.array(X.columns)
    # Sort by descending importance
    sorted_idx = np.argsort(importances)[::-1]
    sorted_importances = importances[sorted_idx]
    sorted_features = feature_names[sorted_idx]
    # Select n_features_to_save most important features
    selected_features = sorted_features[:n_features_to_save]
    # DataFrame for visualization
    cumulative_importance = np.cumsum(sorted_importances)
    df_feat = pd.DataFrame({
        'feature': sorted_features,
        'importance': sorted_importances,
        'cumulative_importance': cumulative_importance
    })
    return selected_features.tolist(), df_feat


def get_important_features_xgb(X, y, n_features_to_save=10, params=None, random_state=42, n_estimators=100):
    """
    Evaluates feature importance using XGBRegressor.
    Returns the n_features_to_save most important features.
    """
    # If params is provided, use user settings, but ensure defaults for random_state, n_estimators, n_jobs if not present
    if params is not None:
        user_params = params.copy()
        if 'random_state' not in user_params:
            user_params['random_state'] = random_state
        if 'n_estimators' not in user_params:
            user_params['n_estimators'] = n_estimators
        if 'n_jobs' not in user_params:
            user_params['n_jobs'] = -1
        if 'verbosity' not in user_params:
            user_params['verbosity'] = 0
        xgb = XGBRegressor(**user_params)
    else:
        xgb = XGBRegressor(n_estimators=n_estimators, random_state=random_state, verbosity=0, n_jobs=-1)
    xgb.fit(X, y)
    importances = xgb.feature_importances_
    feature_names = np.array(X.columns)
    # Sort by descending importance
    sorted_idx = np.argsort(importances)[::-1]
    sorted_importances = importances[sorted_idx]
    sorted_features = feature_names[sorted_idx]
    # Select n_features_to_save most important features
    selected_features = sorted_features[:n_features_to_save]
    # DataFrame for visualization
    cumulative_importance = np.cumsum(sorted_importances)
    df_feat = pd.DataFrame({
        'feature': sorted_features,
        'importance': sorted_importances,
        'cumulative_importance': cumulative_importance
    })
    return selected_features.tolist(), df_feat
