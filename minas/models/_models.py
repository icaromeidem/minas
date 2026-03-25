"""
minas/models/_models.py
========================
Factory function for creating scikit-learn compatible ML pipelines.

Supported model types:
  - 'RF-REG'  : Random Forest Regressor
  - 'RF-CLA'  : Random Forest Classifier
  - 'XGB-REG' : XGBoost Regressor
  - 'XGB-CLA' : XGBoost Classifier
"""

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

# XGBoost support — optional dependency
try:
    from xgboost import XGBRegressor, XGBClassifier
except ImportError:
    XGBRegressor = None
    XGBClassifier = None


def create_model(model_type, hp_combination=None):
    """
    Create a scikit-learn Pipeline for the specified model type.

    For Random Forest models, the pipeline includes an RFE feature selector
    followed by the estimator. For XGBoost models, only the estimator is
    included (feature selection is handled externally via SelectKBest in the
    tuning step).

    Parameters
    ----------
    model_type : str
        Type of model to create. One of:
        - ``'RF-REG'``  : Random Forest Regressor
        - ``'RF-CLA'``  : Random Forest Classifier
        - ``'XGB-REG'`` : XGBoost Regressor
        - ``'XGB-CLA'`` : XGBoost Classifier
    hp_combination : tuple or None, optional
        Hyperparameter combination returned by the tuning step.
        If None, default hyperparameters are used.

        For RF models: ``(n_features, n_trees, min_samples_leaf, bootstrap, max_features)``
        For RF-CLA: ``(n_features, n_trees, min_samples_leaf, bootstrap, max_features, class_weight)``
        For XGB models: ``(colsample_bytree, gamma, learning_rate, max_depth, n_estimators, subsample)``

    Returns
    -------
    sklearn.pipeline.Pipeline
        Configured pipeline ready for fitting.

    Raises
    ------
    ImportError
        If XGBoost is requested but not installed.
    ValueError
        If an unsupported model_type is provided.

    Examples
    --------
    >>> from minas.models import create_model

    >>> # Default Random Forest regressor
    >>> model = create_model('RF-REG')
    >>> model.fit(X_train, y_train)

    >>> # XGBoost regressor with tuned hyperparameters
    >>> hp = (0.8, 0.1, 0.05, 6, 500, 0.8)
    >>> model = create_model('XGB-REG', hp_combination=hp)
    """
    if model_type == "RF-REG":
        if hp_combination:
            n_features, n_trees, min_samples_leaf, bootstrap, max_features = hp_combination
            feature_selector = RFE(
                estimator=DecisionTreeRegressor(),
                n_features_to_select=n_features,
                verbose=0,
                step=100,
            )
            rf = RandomForestRegressor(
                n_estimators=n_trees,
                bootstrap=bootstrap,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
            )
        else:
            feature_selector = RFE(estimator=DecisionTreeRegressor())
            rf = RandomForestRegressor()
        pipeline = Pipeline(steps=[("feature_selector", feature_selector), ("model", rf)])

    elif model_type == "RF-CLA":
        if hp_combination:
            n_features, n_trees, min_samples_leaf, bootstrap, max_features, class_weight = hp_combination
            feature_selector = RFE(
                estimator=DecisionTreeRegressor(),
                n_features_to_select=n_features,
                verbose=0,
                step=100,
            )
            rf = RandomForestClassifier(
                n_estimators=n_trees,
                bootstrap=bootstrap,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                class_weight=class_weight,
            )
        else:
            feature_selector = RFE(estimator=DecisionTreeRegressor())
            rf = RandomForestClassifier()
        pipeline = Pipeline(steps=[("feature_selector", feature_selector), ("model", rf)])

    elif model_type == "XGB-REG":
        if XGBRegressor is None:
            raise ImportError("xgboost is not installed. Run: pip install xgboost")
        if hp_combination:
            colsample_bytree, gamma, learning_rate, max_depth, n_estimators, subsample = hp_combination
            xgb = XGBRegressor(
                colsample_bytree=colsample_bytree,  # fixed typo: was 'coolsample_bytree'
                gamma=gamma,
                learning_rate=learning_rate,
                max_depth=max_depth,
                n_estimators=n_estimators,
                subsample=subsample,
                tree_method='hist',
            )
        else:
            xgb = XGBRegressor(tree_method='hist')
        pipeline = Pipeline(steps=[("model", xgb)])

    elif model_type == "XGB-CLA":
        if XGBClassifier is None:
            raise ImportError("xgboost is not installed. Run: pip install xgboost")
        if hp_combination:
            colsample_bytree, gamma, learning_rate, max_depth, n_estimators, subsample = hp_combination
            xgb = XGBClassifier(
                colsample_bytree=colsample_bytree,
                gamma=gamma,
                learning_rate=learning_rate,
                max_depth=max_depth,
                n_estimators=n_estimators,
                subsample=subsample,
                tree_method='hist',
            )
        else:
            xgb = XGBClassifier(tree_method='hist')
        pipeline = Pipeline(steps=[("model", xgb)])

    else:
        raise ValueError(
            f"Model type '{model_type}' is not supported.\n"
            f"Available options: 'RF-REG', 'RF-CLA', 'XGB-REG', 'XGB-CLA'"
        )

    return pipeline