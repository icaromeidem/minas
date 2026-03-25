"""
minas/tuning/_tuner.py
=======================
Hyperparameter search utilities for Random Forest and XGBoost pipelines.

Uses RandomizedSearchCV with MAD and R² scoring to find the best
hyperparameter combination for stellar parameter regression tasks.
"""

import os
import joblib
import numpy as np
from datetime import datetime

from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, make_scorer

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None


def hyperparameter_search(
    X,
    Y,
    model_type,
    param_dist,
    tuning_id,
    k_values=None,
    test_size=0.25,
    n_iter=20,
    cv=3,
    n_jobs=-1,
    random_state=42,
    save_dir='pipeline',
):
    """
    Run RandomizedSearchCV for a Random Forest or XGBoost regression pipeline.

    Optimises hyperparameters using R² as the refit metric and MAD as a
    secondary diagnostic. The best pipeline is saved to disk as a ``.joblib``
    file so it can be loaded directly for prediction.

    The pipeline structure is:
    - **RF**: ``SimpleImputer → SelectKBest → RandomForestRegressor``
    - **XGB**: ``SelectKBest → XGBRegressor``

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (magnitudes, colors, combinations).
    Y : pd.Series
        Target variable (e.g., Teff, logg, [Fe/H]).
    model_type : str
        ``'RF'`` for Random Forest or ``'XGB'`` for XGBoost.
    param_dist : dict
        Hyperparameter distributions for RandomizedSearchCV.
        Keys must be prefixed with the pipeline step name, e.g.:
        ``'randomforestregressor__n_estimators'`` or
        ``'selectkbest__k'``.
    tuning_id : str
        Base name for the saved ``.joblib`` file (without extension).
    k_values : list of int or None, optional
        Allowed values of k for SelectKBest. Values exceeding the number
        of input features are automatically removed. If None, the values
        in ``param_dist['selectkbest__k']`` are used as-is.
    test_size : float, optional
        Fraction of data held out for validation. Default: 0.25.
    n_iter : int, optional
        Number of random hyperparameter combinations to evaluate. Default: 20.
    cv : int, optional
        Number of cross-validation folds. Default: 3.
    n_jobs : int, optional
        Number of parallel jobs. ``-1`` uses all available CPUs. Default: -1.
    random_state : int, optional
        Random seed for reproducibility. Default: 42.
    save_dir : str, optional
        Directory where the best pipeline is saved. Created if it does
        not exist. Default: ``'pipeline'``.

    Returns
    -------
    best_pipeline : sklearn.pipeline.Pipeline
        The best pipeline found by RandomizedSearchCV.
    random_search : RandomizedSearchCV
        The fitted search object, providing access to ``cv_results_``,
        ``best_params_``, and ``best_score_``.

    Raises
    ------
    ImportError
        If XGBoost is requested but not installed.
    ValueError
        If an unsupported model_type is provided.

    Examples
    --------
    >>> import minas as mg
    >>> from minas.tuning import hyperparameter_search
    >>>
    >>> param_dist = {
    ...     'selectkbest__k'                        : [10, 20, 30],
    ...     'randomforestregressor__n_estimators'   : [100, 300, 500],
    ...     'randomforestregressor__min_samples_leaf': [1, 5, 10],
    ...     'randomforestregressor__max_features'   : ['sqrt', 'log2'],
    ...     'randomforestregressor__bootstrap'      : [True, False],
    ... }
    >>>
    >>> best_pipeline, search = hyperparameter_search(
    ...     X=work_df,
    ...     Y=catalog['Teff'],
    ...     model_type='RF',
    ...     param_dist=param_dist,
    ...     tuning_id='teff_rf',
    ...     k_values=list(range(5, work_df.shape[1] + 1, 5)),
    ...     n_iter=30,
    ...     save_dir='pipelines/',
    ... )
    >>> print(search.best_params_)
    """
    if model_type == 'XGB' and XGBRegressor is None:
        raise ImportError("xgboost is not installed. Run: pip install xgboost")

    os.makedirs(save_dir, exist_ok=True)

    # Clip k_values to the number of available features
    if k_values is not None and 'selectkbest__k' in param_dist:
        param_dist['selectkbest__k'] = [k for k in k_values if k <= X.shape[1]]

    # Train/test split
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state
    )

    # Drop rows with missing values in training set
    train_mask = ~(X_train.isnull().any(axis=1) | Y_train.isnull())
    X_train = X_train[train_mask]
    Y_train = Y_train[train_mask]

    # Build pipeline
    if model_type == 'RF':
        steps = [
            ('imputer',               SimpleImputer(strategy='median')),
            ('selectkbest',           SelectKBest(f_regression)),
            ('randomforestregressor', RandomForestRegressor(random_state=random_state)),
        ]
    elif model_type == 'XGB':
        steps = [
            ('selectkbest',  SelectKBest(f_regression)),
            ('xgbregressor', XGBRegressor(objective='reg:squarederror', random_state=random_state)),
        ]
    else:
        raise ValueError(
            f"model_type '{model_type}' is not supported. Use 'RF' or 'XGB'."
        )

    pipeline = Pipeline(steps)

    # Scoring — R² as primary metric, MAD as diagnostic
    def mad_score(y_true, y_pred):
        return np.median(np.abs(y_true - y_pred))

    scoring = {
        'R2' : make_scorer(r2_score),
        'MAD': make_scorer(mad_score, greater_is_better=False),
    }

    # Randomized search
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        scoring=scoring,
        refit='R2',
        cv=cv,
        verbose=0,
        n_iter=n_iter,
        n_jobs=n_jobs,
        random_state=random_state,
    )
    random_search.fit(X_train, Y_train)

    best_pipeline = random_search.best_estimator_

    # Save best pipeline
    pipeline_path = os.path.join(save_dir, f'{tuning_id}.joblib')
    joblib.dump(best_pipeline, pipeline_path)

    return best_pipeline, random_search