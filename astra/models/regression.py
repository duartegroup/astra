"""
This module contains variables for instantiating regressors and their hyperparameter search grids.

Attributes
----------
REGRESSORS : dict[str, BaseEstimator]
    A dictionary mapping model names to their corresponding scikit-learn regressor instances. [`source <https://github.com/duartegroup/astra/blob/main/astra/models/regression.py#L32-L44>`_]
REGRESSOR_PARAMS : dict[str, dict[str, list]]
    A dictionary mapping model names to dictionaries of hyperparameters to search over. [`source <https://github.com/duartegroup/astra/blob/main/astra/models/regression.py#L46-L115>`_]
REGRESSOR_PARAMS_OPTUNA : dict[str, dict[str, optuna.distributions]]
    A dictionary mapping model names to dictionaries of hyperparameters to search over using Optuna. [`source <https://github.com/duartegroup/astra/blob/main/astra/models/regression.py#L117-L234>`_]
"""

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from optuna.distributions import (
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

REGRESSORS = {
    "XGBRegressor": XGBRegressor(random_state=42),
    "RandomForestRegressor": RandomForestRegressor(random_state=42),
    "GradientBoostingRegressor": GradientBoostingRegressor(random_state=42),
    "HistGradientBoostingRegressor": HistGradientBoostingRegressor(random_state=42),
    "KNeighborsRegressor": KNeighborsRegressor(),
    "SVR": SVR(),
    "Ridge": Ridge(random_state=42),
    "BayesianRidge": BayesianRidge(),
    "KernelRidge": KernelRidge(),
    "LGBMRegressor": LGBMRegressor(random_state=42, force_row_wise=True, verbosity=-1),
    "CatBoostRegressor": CatBoostRegressor(random_state=42, verbose=False),
}

REGRESSOR_PARAMS = {
    "XGBRegressor": dict(
        n_estimators=[10, 100, 200, 500],
        max_leaves=[10, 30, 50, 0],
        learning_rate=[1e-3, 1e-2, 1e-1],
        max_depth=[3, 6, 10, 30],
    ),
    "RandomForestRegressor": dict(
        n_estimators=[10, 100, 200, 400],
        max_depth=[5, 10, 20, None],
        min_samples_split=[2, 5, 10],
        min_samples_leaf=[1, 2, 5],
        max_leaf_nodes=[100, 1000, None],
        bootstrap=[True, False],
        max_features=["log2", "sqrt", None],
    ),
    "GradientBoostingRegressor": dict(
        loss=["squared_error", "absolute_error"],
        learning_rate=[0.01, 0.1, 1.0],
        n_estimators=[10, 100, 200],
        min_samples_split=[2, 5, 10],
        min_samples_leaf=[1, 2, 5],
        max_depth=[1, 3, 5, 8],
        max_leaf_nodes=[100, 1000, None],
    ),
    "HistGradientBoostingRegressor": dict(
        loss=["squared_error", "absolute_error"],
        learning_rate=[0.01, 0.1, 1.0],
        max_iter=[10, 100, 200],
        max_leaf_nodes=[10, 30, 50],
        min_samples_leaf=[5, 10, 20],
    ),
    "KNeighborsRegressor": dict(
        n_neighbors=[1, 3, 5, 7, 9, 15],
        weights=["uniform", "distance"],
        p=[1, 2],
    ),
    "SVR": dict(
        kernel=["linear", "poly", "rbf", "sigmoid"],
        C=[1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0],
        gamma=["scale", "auto", 1e-3, 1e-2, 1e-1],
    ),
    "Ridge": dict(
        alpha=[1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0, 100.0],
        solver=["auto", "svd", "cholesky"],
    ),
    "BayesianRidge": dict(
        max_iter=[100, 300, 500],
        alpha_1=[1e-9, 1e-7, 1e-5, 1e-3, 1e-1],
        alpha_2=[1e-9, 1e-7, 1e-5, 1e-3, 1e-1],
        lambda_1=[1e-9, 1e-7, 1e-5, 1e-3, 1e-1],
        lambda_2=[1e-9, 1e-7, 1e-5, 1e-3, 1e-1],
    ),
    "KernelRidge": dict(
        alpha=[1e-6, 1e-4, 1e-2, 1e0, 1e1],
        kernel=["linear", "poly", "polynomial", "rbf", "laplacian", "sigmoid"],
    ),
    "LGBMRegressor": dict(
        boosting_type=["gbdt", "dart"],
        num_leaves=[10, 30, 50],
        max_depth=[10, 30, -1],
        learning_rate=[1e-3, 1e-2, 1e-1],
        n_estimators=[10, 100, 200, 500],
    ),
    "CatBoostRegressor": dict(
        iterations=[50, 100, 500],
        learning_rate=[1e-3, 1e-2, 1e-1],
        depth=[2, 4, 6, 8, 10],
    ),
}

REGRESSOR_PARAMS_OPTUNA = {
    "XGBRegressor": dict(
        n_estimators=IntDistribution(10, 1000),
        max_leaves=IntDistribution(0, 50),
        learning_rate=FloatDistribution(1e-4, 1e-0, log=True),
        max_depth=IntDistribution(3, 30),
        grow_policy=CategoricalDistribution(["depthwise", "lossguide"]),
        booster=CategoricalDistribution(["gbtree", "gblinear", "dart"]),
        tree_method=CategoricalDistribution(["exact", "approx", "hist"]),
        min_child_weight=IntDistribution(1, 10),
        gamma=FloatDistribution(0.0, 5.0),
        subsample=FloatDistribution(0.4, 1.0),
        sampling_method=CategoricalDistribution(["uniform", "gradient_based"]),
        colsample_bytree=FloatDistribution(0.3, 1.0),
        reg_alpha=FloatDistribution(1e-6, 10.0, log=True),
        reg_lambda=FloatDistribution(1e-3, 10.0, log=True),
    ),
    "RandomForestRegressor": dict(
        n_estimators=IntDistribution(10, 1000),
        max_depth=IntDistribution(1, 100),
        min_samples_split=IntDistribution(2, 20),
        min_samples_leaf=IntDistribution(1, 10),
        max_leaf_nodes=IntDistribution(2, 1000),
        bootstrap=CategoricalDistribution([True, False]),
        max_features=CategoricalDistribution(["log2", "sqrt", 0.2, 0.5, None]),
        min_weight_fraction_leaf=FloatDistribution(0.0, 0.5),
    ),
    "GradientBoostingRegressor": dict(
        loss=CategoricalDistribution(
            ["squared_error", "absolute_error", "huber", "quantile"]
        ),
        learning_rate=FloatDistribution(1e-4, 1e-0, log=True),
        n_estimators=IntDistribution(10, 500),
        min_samples_split=IntDistribution(2, 20),
        min_samples_leaf=IntDistribution(1, 20),
        max_depth=IntDistribution(1, 10),
        max_leaf_nodes=IntDistribution(2, 1000),
        subsample=FloatDistribution(0.4, 1.0),
        max_features=CategoricalDistribution([None, "sqrt", "log2", 0.5]),
        criterion=CategoricalDistribution(["friedman_mse", "squared_error"]),
        min_weight_fraction_leaf=FloatDistribution(0.0, 0.5),
        min_impurity_decrease=FloatDistribution(0.0, 1.0),
        n_iter_no_change=IntDistribution(1, 50),
    ),
    "HistGradientBoostingRegressor": dict(
        loss=CategoricalDistribution(
            ["squared_error", "absolute_error", "gamma", "poisson", "quantile"]
        ),
        learning_rate=FloatDistribution(1e-4, 1e-0, log=True),
        max_iter=IntDistribution(10, 500),
        max_leaf_nodes=IntDistribution(5, 125),
        min_samples_leaf=IntDistribution(1, 50),
        l2_regularization=FloatDistribution(0.0, 1.0),
        max_bins=IntDistribution(32, 256),
        early_stopping=CategoricalDistribution([True, False]),
        max_depth=IntDistribution(1, 30),
        max_features=FloatDistribution(0.3, 1.0),
    ),
    "KNeighborsRegressor": dict(
        n_neighbors=IntDistribution(3, 20),
        weights=CategoricalDistribution(["uniform", "distance"]),
        p=CategoricalDistribution([1, 2]),
    ),
    "SVR": dict(
        kernel=CategoricalDistribution(["linear", "poly", "rbf", "sigmoid"]),
        C=FloatDistribution(1e-3, 100, log=True),
        gamma=CategoricalDistribution(["scale", "auto", 1e-3, 1e-2, 1e-1]),
    ),
    "Ridge": dict(
        alpha=FloatDistribution(1e-6, 100, log=True),
        solver=CategoricalDistribution(["auto", "svd", "cholesky"]),
    ),
    "BayesianRidge": dict(
        max_iter=IntDistribution(100, 500),
        alpha_1=FloatDistribution(1e-9, 1e-1, log=True),
        alpha_2=FloatDistribution(1e-9, 1e-1, log=True),
        lambda_1=FloatDistribution(1e-9, 1e-1, log=True),
        lambda_2=FloatDistribution(1e-9, 1e-1, log=True),
    ),
    "KernelRidge": dict(
        kernel=CategoricalDistribution(
            ["linear", "poly", "polynomial", "rbf", "laplacian", "sigmoid"]
        ),
        alpha=FloatDistribution(1e-6, 1e1, log=True),
    ),
    "LGBMRegressor": dict(
        boosting_type=CategoricalDistribution(["gbdt", "dart"]),
        num_leaves=IntDistribution(5, 512),
        max_depth=IntDistribution(-1, 30),
        learning_rate=FloatDistribution(1e-4, 1e-0, log=True),
        n_estimators=IntDistribution(10, 1000),
        min_data_in_leaf=IntDistribution(1, 100),
        subsample=FloatDistribution(0.4, 1.0),
        subsample_freq=IntDistribution(0, 10),
        colsample_bytree=FloatDistribution(0.3, 1.0),
        reg_alpha=FloatDistribution(0.0, 1.0),
        reg_lambda=FloatDistribution(1e-6, 10.0, log=True),
        bagging_freq=IntDistribution(0, 10),
        min_split_gain=FloatDistribution(0.0, 1.0),
        min_child_weight=FloatDistribution(1e-3, 10.0, log=True),
        min_child_samples=IntDistribution(5, 100),
    ),
    "CatBoostRegressor": dict(
        iterations=IntDistribution(30, 1000),
        learning_rate=FloatDistribution(1e-4, 1e-0, log=True),
        depth=IntDistribution(2, 12),
        l2_leaf_reg=FloatDistribution(1e-3, 10.0, log=True),
        rsm=FloatDistribution(0.3, 1.0),
        loss_function=CategoricalDistribution(["RMSE", "MAE", "Quantile"]),
        border_count=IntDistribution(32, 255),
        feature_border_type=CategoricalDistribution(
            ["Median", "Uniform", "UniformAndQuantiles", "MaxLogSum", "MinEntropy"]
        ),
        random_strength=FloatDistribution(1e-6, 10.0, log=True),
        bootstrap_type=CategoricalDistribution(["Bayesian", "Bernoulli", "MVS"]),
    ),
}
