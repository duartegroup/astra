"""
Description
-----------
This module contains variables for instantiating classifiers and their hyperparameter search grids.

Attributes
----------
CLASSIFIERS : dict[str, BaseEstimator]
    A dictionary mapping model names to their corresponding scikit-learn classifier instances.
CLASSIFIER_PARAMS : dict[str, dict[str, list]]
    A dictionary mapping model names to dictionaries of hyperparameters to search over.
NON_PROBABILISTIC_MODELS : list[str]
    A list of model names that do not have a `predict_proba` method.
"""

from xgboost import XGBClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier,
    BaggingClassifier,
    HistGradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import (
    LogisticRegression,
    RidgeClassifier,
    SGDClassifier,
    Perceptron,
    PassiveAggressiveClassifier,
)
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from sklearn.gaussian_process.kernels import (
    RBF,
    DotProduct,
    Matern,
    RationalQuadratic,
    WhiteKernel,
)

CLASSIFIERS = {
    "LogisticRegression": LogisticRegression(random_state=42, max_iter=100000),
    "GaussianProcessClassifier": GaussianProcessClassifier(random_state=42),
    "BernoulliNB": BernoulliNB(),
    "GaussianNB": GaussianNB(),
    "MultinomialNB": MultinomialNB(),
    "DecisionTreeClassifier": DecisionTreeClassifier(random_state=42),
    "ExtraTreeClassifier": ExtraTreeClassifier(random_state=42),
    "ExtraTreesClassifier": ExtraTreesClassifier(random_state=42),
    "RandomForestClassifier": RandomForestClassifier(random_state=42),
    "GradientBoostingClassifier": GradientBoostingClassifier(random_state=42),
    "BaggingClassifier": BaggingClassifier(random_state=42),
    "HistGradientBoostingClassifier": HistGradientBoostingClassifier(random_state=42),
    "AdaBoostClassifier": AdaBoostClassifier(random_state=42),
    "KNeighborsClassifier": KNeighborsClassifier(),
    "NearestCentroid": NearestCentroid(),
    "LinearDiscriminantAnalysis": LinearDiscriminantAnalysis(),
    "LinearSVC": LinearSVC(random_state=42, max_iter=1000000, dual="auto"),
    "SVC": SVC(random_state=42),
    "RidgeClassifier": RidgeClassifier(random_state=42),
    "SGDClassifier": SGDClassifier(loss="log_loss", random_state=42),
    "Perceptron": Perceptron(random_state=42),
    "PassiveAggressiveClassifier": PassiveAggressiveClassifier(random_state=42),
    "MLPClassifier": MLPClassifier(random_state=42, max_iter=2000),
    "LGBMClassifier": LGBMClassifier(
        random_state=42, force_row_wise=True, verbosity=-1
    ),
    "XGBClassifier": XGBClassifier(random_state=42),
}

NON_PROBABILISTIC_MODELS = [
    "NearestCentroid",
    "LinearSVC",
    "SVC",
    "RidgeClassifier",
    "Perceptron",
    "PassiveAggressiveClassifier",
]

CLASSIFIER_PARAMS = {
    "LogisticRegression": dict(
        solver=["newton-cg", "lbfgs", "liblinear"],
        penalty=["l2"],
        C=[100, 10, 1.0, 0.1, 0.01],
    ),
    "GaussianProcessClassifier": dict(
        kernel=[
            1 * RBF(),
            1 * DotProduct(),
            1 * Matern(),
            1 * RationalQuadratic(),
            1 * WhiteKernel(),
        ],
    ),
    "BernoulliNB": dict(
        alpha=[0.01, 0.1, 0.5, 1.0, 10.0],
        fit_prior=[True, False],
    ),
    "GaussianNB": dict(
        var_smoothing=[1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6],
    ),
    "MultinomialNB": dict(
        alpha=[0.01, 0.1, 0.5, 1.0, 10.0],
        fit_prior=[True, False],
    ),
    "DecisionTreeClassifier": dict(
        criterion=["gini", "entropy", "log_loss"],
        max_depth=[10, 50, 100, None],
        min_samples_split=[2, 5, 10],
        min_samples_leaf=[1, 2, 4],
        max_features=["log2", "sqrt", None],
        max_leaf_nodes=[100, 1000, None],
    ),
    "ExtraTreeClassifier": dict(
        criterion=["gini", "entropy", "log_loss"],
        max_depth=[10, 50, 100, None],
        min_samples_split=[2, 5, 10],
        min_samples_leaf=[1, 2, 4],
        max_features=["log2", "sqrt"],
        max_leaf_nodes=[100, 1000, None],
    ),
    "ExtraTreesClassifier": dict(
        n_estimators=[10, 100, 200, 500],
        criterion=["gini", "entropy", "log_loss"],
        bootstrap=[True, False],
        max_depth=[10, 50, 100, None],
        max_features=["log2", "sqrt"],
        min_samples_leaf=[1, 2, 4],
        min_samples_split=[2, 5, 10],
        max_leaf_nodes=[100, 1000, None],
    ),
    "RandomForestClassifier": dict(
        n_estimators=[10, 100, 500],
        bootstrap=[True, False],
        max_depth=[10, 50, 100, None],
        max_features=["log2", "sqrt"],
        min_samples_leaf=[1, 2, 4],
        min_samples_split=[2, 5, 10],
    ),
    "GradientBoostingClassifier": dict(
        learning_rate=[1, 0.1, 0.01],
        n_estimators=[10, 100, 500],
        max_depth=[1, 3, 5, None],
        min_samples_split=[2, 5, 10],
        min_samples_leaf=[1, 2, 4],
        max_features=["log2", "sqrt"],
    ),
    "BaggingClassifier": dict(
        n_estimators=[5, 10, 20],
        max_samples=[0.1, 0.5, 1.0],
        max_features=[0.1, 0.5, 1.0],
    ),
    "HistGradientBoostingClassifier": dict(
        learning_rate=[1, 0.1, 0.01],
        max_leaf_nodes=[3, 10, 30, 50],
        min_samples_leaf=[10, 20, 50],
    ),
    "AdaBoostClassifier": dict(
        n_estimators=[10, 50, 100],
        learning_rate=[0.1, 1.0, 10],
    ),
    "KNeighborsClassifier": dict(
        n_neighbors=[3, 5, 10, 20],
        weights=["uniform", "distance"],
    ),
    "NearestCentroid": dict(
        shrink_threshold=[0.1, 0.5, 1.0, None],
    ),
    "LinearDiscriminantAnalysis": dict(solver=["svd", "lsqr", "eigen"]),
    "LinearSVC": dict(
        penalty=["l1", "l2"],
        loss=["hinge", "squared_hinge"],
        C=[1, 10, 100],
    ),
    "SVC": dict(
        kernel=["linear", "rbf", "poly", "sigmoid"],
        gamma=["scale", 0.01, 0.1],
        C=[1, 10, 100],
    ),
    "RidgeClassifier": dict(
        alpha=[0.1, 0.5, 1.0, 2, 5],
    ),
    "SGDClassifier": dict(
        loss=["log_loss", "modified_huber"],
        penalty=["l2", "l1", "elasticnet", None],
        alpha=[0.00001, 0.0001, 0.001],
    ),
    "Perceptron": dict(
        penalty=["l2", "l1", "elasticnet", None], alpha=[0.00001, 0.0001, 0.001]
    ),
    "PassiveAggressiveClassifier": dict(C=[0.1, 1, 2, 5, 10]),
    "MLPClassifier": dict(
        hidden_layer_sizes=[
            [
                100,
            ],
            [100, 100],
            [
                200,
            ],
            [
                50,
            ],
            [200, 100],
        ],
        activation=["identity", "logistic", "relu"],
        solver=["lbfgs", "adam"],
        alpha=[0.00001, 0.0001, 0.001],
        learning_rate=["constant", "invscaling", "adaptive"],
        learning_rate_init=[0.0001, 0.001, 0.01],
        early_stopping=[False, True],
    ),
    "LGBMClassifier": dict(
        n_estimators=[10, 100, 200, 500],
        num_leaves=[10, 30, 50],
        learning_rate=[0.01, 0.1, 1],
        max_depth=[10, 30, -1],
    ),
    "XGBClassifier": dict(
        n_estimators=[10, 100, 200, 500],
        max_leaves=[10, 30, 50, 0],
        learning_rate=[0.01, 0.1, 1],
        max_depth=[
            10,
            30,
        ],
    ),
}
