"""
This module contains variables for instantiating regressors and their hyperparameter search grids.

Attributes
----------
REGRESSORS : dict[str, BaseEstimator]
    A dictionary mapping model names to their corresponding scikit-learn regressor instances.
REGRESSOR_PARAMS : dict[str, dict[str, list]]
    A dictionary mapping model names to dictionaries of hyperparameters to search over.
"""

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
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
        learning_rate=[0.01, 0.1, 1],
        max_depth=[10, 30],
    ),
    "RandomForestRegressor": dict(
        n_estimators=[10, 100, 200],
        max_depth=[10, 50, 100, None],
        min_samples_split=[2, 5, 10],
        min_samples_leaf=[1, 2, 5],
        max_leaf_nodes=[100, 1000, None],
        bootstrap=[True, False],
    ),
    "GradientBoostingRegressor": dict(
        loss=["squared_error", "absolute_error"],
        learning_rate=[0.01, 0.1, 1.0],
        n_estimators=[10, 100, 200],
        min_samples_split=[2, 5, 10],
        min_samples_leaf=[1, 2, 5],
        max_depth=[1, 3, 5],
        max_leaf_nodes=[100, 1000, None],
    ),
    "HistGradientBoostingRegressor": dict(
        loss=["squared_error", "absolute_error"],
        learning_rate=[0.01, 0.1, 1.0],
        max_iter=[10, 100, 200],
        max_leaf_nodes=[10, 30, 50],
        min_samples_leaf=[10, 20, 30],
    ),
    "KNeighborsRegressor": dict(
        n_neighbors=[3, 5, 10, 20],
        weights=["uniform", "distance"],
    ),
    "SVR": dict(
        kernel=["linear", "poly", "rbf", "sigmoid"],
        C=[1, 10, 100],
    ),
    "Ridge": dict(alpha=[0.1, 0.5, 1, 2, 5]),
    "BayesianRidge": dict(
        max_iter=[100, 300, 500],
        alpha_1=[1e-5, 1e-6, 1e-7],
        alpha_2=[1e-5, 1e-6, 1e-7],
        lambda_1=[1e-5, 1e-6, 1e-7],
        lambda_2=[1e-5, 1e-6, 1e-7],
    ),
    "KernelRidge": dict(
        alpha=[0.5, 1, 1.5],
        kernel=["linear", "poly", "polynomial", "rbf", "laplacian", "sigmoid"],
    ),
    "LGBMRegressor": dict(
        boosting_type=["gbdt", "dart"],
        num_leaves=[10, 30, 50],
        max_depth=[10, 30, -1],
        learning_rate=[0.01, 0.1, 1],
        n_estimators=[10, 100, 200, 500],
    ),
    "CatBoostRegressor": dict(
        iterations=[30, 50, 100, 150, 200],
        learning_rate=[0.01, 0.1],
        depth=[2, 4, 6, 8, 10],
    ),
}
