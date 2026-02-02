
import joblib
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, make_scorer
import numpy as np
import os
from datetime import datetime



# Utility function for hyperparameter search for Random Forest or XGBoost
def hyperparameter_search(X, Y, model_type, param_dist, tuning_id, k_values=None, test_size=0.25, n_iter=20, cv=3, n_jobs=-1, random_state=42, save_dir='pipeline'):
    """
    Executes RandomizedSearchCV for Random Forest (RF) or XGBoost (XGB), using MAD as primary metric.
    User defines hyperparameter grid (param_dist) and k values (k_values).
    Best pipeline is saved as <tuning_id>.joblib in save_dir directory.

    Parameters:
        X, Y: input and output data
        model_type: 'RF' for RandomForest or 'XGB' for XGBoost
        param_dist: hyperparameter dictionary for RandomizedSearchCV
        tuning_id: saved file name
        k_values: list of k values for SelectKBest (optional)
        test_size, n_iter, cv, n_jobs, random_state, save_dir: control parameters

    Returns:
        best_pipeline, random_search
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Ajustar k_values se fornecido
    if k_values is not None:
        if 'selectkbest__k' in param_dist:
            param_dist['selectkbest__k'] = [k for k in k_values if k <= X.shape[1]]

    # Split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

    # Remove rows with missing values in X or Y
    train_mask = ~(X_train.isnull().any(axis=1) | Y_train.isnull())
    X_train = X_train[train_mask]
    Y_train = Y_train[train_mask]

    # Pipeline
    steps = []
    if model_type == 'RF':
        steps.append(('imputer', SimpleImputer(strategy='median')))
        steps.append(('selectkbest', SelectKBest(f_regression)))
        steps.append(('randomforestregressor', RandomForestRegressor(random_state=random_state)))
    elif model_type == 'XGB':
        steps.append(('selectkbest', SelectKBest(f_regression)))
        steps.append(('xgbregressor', XGBRegressor(objective='reg:squarederror', random_state=random_state)))
    else:
        raise ValueError('model_type deve ser "RF" ou "XGB"')
    pipeline = Pipeline(steps)

    # Function to calculate MAD
    def mad_score(y_true, y_pred):
        return np.median(np.abs(y_true - y_pred))

    scoring = {
        'R2': make_scorer(r2_score),
        'MAD': make_scorer(mad_score, greater_is_better=False)
    }

    # RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        scoring=scoring,
        refit='R2',
        cv=cv,
        verbose=0,
        n_iter=n_iter,
        n_jobs=n_jobs,
        random_state=random_state
    )
    random_search.fit(X_train, Y_train)

    best_pipeline = random_search.best_estimator_

    # Save pipeline with tuning_id name
    pipeline_path = os.path.join(save_dir, f'{tuning_id}.joblib')
    joblib.dump(best_pipeline, pipeline_path)
    # print(f'Model saved at: {pipeline_path}')

    return best_pipeline, random_search