
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

# Suporte a XGBoost
try:
    from xgboost import XGBRegressor, XGBClassifier
except ImportError:
    XGBRegressor = None
    XGBClassifier = None


def create_model(model_type, hp_combination = None):
    if model_type == "RF-REG":
        if hp_combination:
            n_features, n_trees, min_samples_leaf, bootstrap, max_features = hp_combination
            FeatureSelector = RFE(
                estimator=DecisionTreeRegressor(),
                n_features_to_select=n_features,
                verbose=0,
                step=100,
            )
            RF = RandomForestRegressor(
                n_estimators=n_trees,
                bootstrap=bootstrap,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
            )
        else:
            FeatureSelector = RFE(estimator=DecisionTreeRegressor())
            RF = RandomForestRegressor()
        pipeline = Pipeline(steps=[("Feature Selector", FeatureSelector), ("Model", RF)])

    elif model_type == "RF-CLA":
        if hp_combination:
            n_features, n_trees, min_samples_leaf, bootstrap, max_features, class_weight = hp_combination
            FeatureSelector = RFE(
                estimator=DecisionTreeRegressor(),
                n_features_to_select=n_features,
                verbose=0,
                step=100,
            )
            RF = RandomForestClassifier(
                n_estimators=n_trees,
                bootstrap=bootstrap,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                class_weight=class_weight
            )
        else:
            FeatureSelector = RFE(estimator=DecisionTreeRegressor())
            RF = RandomForestClassifier()
        pipeline = Pipeline(steps=[("Feature Selector", FeatureSelector), ("Model", RF)])

    elif model_type == "XGB-REG":
        if XGBRegressor is None:
            raise ImportError("xgboost não está instalado.")
        if hp_combination:
            n_estimators, max_depth, learning_rate, subsample, colsample_bytree = hp_combination
            XGB = XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                tree_method='hist',
            )
        else:
            XGB = XGBRegressor(tree_method='hist')
        pipeline = Pipeline(steps=[("Model", XGB)])

    elif model_type == "XGB-CLA":
        if XGBClassifier is None:
            raise ImportError("xgboost não está instalado.")
        if hp_combination:
            n_estimators, max_depth, learning_rate, subsample, colsample_bytree = hp_combination
            XGB = XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                tree_method='hist',
            )
        else:
            XGB = XGBClassifier(tree_method='hist')
        pipeline = Pipeline(steps=[("Model", XGB)])

    else:
        raise ValueError("Modelo não suportado")

    return pipeline
