"""
This module contains variables for instantiating classifiers and their hyperparameter search grids.

Attributes
----------
CLASSIFIERS : dict[str, BaseEstimator]
    A dictionary mapping model names to their corresponding scikit-learn classifier instances. [`source <https://github.com/duartegroup/astra/blob/main/astra/models/classification.py#L58-L86>`_]
CLASSIFIER_PARAMS : dict[str, dict[str, list]]
    A dictionary mapping model names to dictionaries of hyperparameters to search over. [`source <https://github.com/duartegroup/astra/blob/main/astra/models/classification.py#L96-L247>`_]
CLASSIFIER_PARAMS_OPTUNA : dict[str, dict[str, optuna.distributions]]
    A dictionary mapping model names to dictionaries of hyperparameters to search over using Optuna. [`source <https://github.com/duartegroup/astra/blob/main/astra/models/classification.py#L249-L452>`_]
NON_PROBABILISTIC_MODELS : list[str]
    A list of model names that do not have a `predict_proba` method. [`source <https://github.com/duartegroup/astra/blob/main/astra/models/classification.py#L88-L94>`_]
"""

import warnings

from lightgbm import LGBMClassifier
from optuna.distributions import (
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import (
    RBF,
    DotProduct,
    Matern,
    RationalQuadratic,
    WhiteKernel,
)
from sklearn.linear_model import (
    LogisticRegression,
    PassiveAggressiveClassifier,
    Perceptron,
    RidgeClassifier,
    SGDClassifier,
)
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from xgboost import XGBClassifier

# catch UserWarning from Optuna
warnings.filterwarnings("ignore", category=UserWarning)

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
    "LinearSVC",
    "SVC",
    "RidgeClassifier",
    "Perceptron",
    "PassiveAggressiveClassifier",
]

CLASSIFIER_PARAMS = {
    "LogisticRegression": dict(
        solver=["newton-cg", "lbfgs", "liblinear", "saga"],
        penalty=["l2", "l1", "elasticnet", "none"],
        C=[1e-5, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 1e2, 1e4],
        class_weight=[None, "balanced"],
        l1_ratio=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0],
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
        alpha=[1e-3, 1e-2, 1e-1, 1.0, 10.0],
        fit_prior=[True, False],
        binarize=[None, 0.0, 0.5],
    ),
    "GaussianNB": dict(
        var_smoothing=[1e-9, 1e-8, 1e-7, 1e-6],
    ),
    "MultinomialNB": dict(
        alpha=[1e-3, 1e-2, 1e-1, 1.0, 10.0],
        fit_prior=[True, False],
    ),
    "DecisionTreeClassifier": dict(
        criterion=["gini", "entropy", "log_loss"],
        max_depth=[3, 5, 10, 50, 100, None],
        min_samples_split=[2, 5, 10],
        min_samples_leaf=[1, 2, 4],
        max_features=["log2", "sqrt", None],
        max_leaf_nodes=[100, 1000, None],
        class_weight=[None, "balanced"],
    ),
    "ExtraTreeClassifier": dict(
        criterion=["gini", "entropy", "log_loss"],
        max_depth=[3, 5, 10, 50, 100, None],
        min_samples_split=[2, 5, 10],
        min_samples_leaf=[1, 2, 4],
        max_features=["log2", "sqrt", None],
        max_leaf_nodes=[100, 1000, None],
        class_weight=[None, "balanced"],
    ),
    "ExtraTreesClassifier": dict(
        n_estimators=[10, 100, 200, 500],
        criterion=["gini", "entropy", "log_loss"],
        bootstrap=[True, False],
        max_depth=[5, 10, 20, None],
        max_features=["log2", "sqrt", 0.3, 0.6],
        min_samples_leaf=[1, 2, 4],
        min_samples_split=[2, 5, 10],
        max_leaf_nodes=[100, 1000, None],
        class_weight=[None, "balanced"],
    ),
    "RandomForestClassifier": dict(
        n_estimators=[10, 100, 500],
        bootstrap=[True, False],
        max_depth=[5, 10, 20, 50, None],
        max_features=["log2", "sqrt", 0.2, 0.5, None],
        min_samples_leaf=[1, 2, 4, 8],
        min_samples_split=[2, 5, 10],
        class_weight=[None, "balanced"],
    ),
    "GradientBoostingClassifier": dict(
        learning_rate=[0.01, 0.05, 0.1],
        n_estimators=[50, 100, 200],
        max_depth=[3, 5, 8, None],
        min_samples_split=[2, 5, 10],
        min_samples_leaf=[1, 2, 4],
        max_features=[None, "sqrt", 0.5],
    ),
    "BaggingClassifier": dict(
        max_samples=[0.5, 0.7, 1.0],
        max_features=[0.5, 0.7, 1.0],
        n_estimators=[10, 50, 100],
        bootstrap=[True, False],
        bootstrap_features=[True, False],
    ),
    "HistGradientBoostingClassifier": dict(
        learning_rate=[0.01, 0.05, 0.1],
        max_leaf_nodes=[3, 10, 30, 50],
        min_samples_leaf=[5, 10, 20],
        l2_regularization=[0.0, 1e-3, 1e-2],
    ),
    "AdaBoostClassifier": dict(
        n_estimators=[50, 100, 200],
        learning_rate=[0.01, 0.1, 1.0],
    ),
    "KNeighborsClassifier": dict(
        n_neighbors=[1, 3, 5, 7, 9, 15],
        weights=["uniform", "distance"],
        p=[1, 2],
    ),
    "NearestCentroid": dict(
        shrink_threshold=[None, 0.1, 0.2, 0.5],
        metric=["euclidean", "manhattan"],
    ),
    "LinearDiscriminantAnalysis": dict(solver=["svd", "lsqr", "eigen"]),
    "LinearSVC": dict(
        penalty=["l1", "l2"],
        loss=["hinge", "squared_hinge"],
        C=[1e-3, 1e-2, 1e-1, 1.0, 10.0],
        dual=[True, False],
        class_weight=[None, "balanced"],
    ),
    "SVC": dict(
        kernel=["linear", "rbf", "poly", "sigmoid"],
        gamma=["scale", "auto", 1e-3, 1e-2, 1e-1, 1.0],
        C=[1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0],
    ),
    "RidgeClassifier": dict(
        alpha=[1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0],
        solver=["auto", "svd", "cholesky"],
    ),
    "SGDClassifier": dict(
        loss=["log_loss", "modified_huber"],
        penalty=["l2", "l1", "elasticnet", None],
        alpha=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
    ),
    "Perceptron": dict(
        penalty=["l2", "l1", "elasticnet", None], alpha=[0.0, 1e-6, 1e-4, 1e-3]
    ),
    "PassiveAggressiveClassifier": dict(
        C=[0.1, 0.5, 1.0], fit_intercept=[True, False], tol=[1e-3, 1e-4]
    ),
    "MLPClassifier": dict(
        hidden_layer_sizes=[(32,), (64,), (64, 32), (128,)],
        activation=["identity", "logistic", "relu"],
        solver=["sgd", "adam"],
        alpha=[1e-6, 1e-5, 1e-4, 1e-3],
        learning_rate=["constant", "invscaling", "adaptive"],
        learning_rate_init=[1e-4, 1e-3, 1e-2],
        early_stopping=[False, True],
    ),
    "LGBMClassifier": dict(
        n_estimators=[50, 100, 200],
        num_leaves=[7, 15, 31],
        learning_rate=[1e-3, 1e-2, 1e-1],
        max_depth=[5, 10, -1],
    ),
    "XGBClassifier": dict(
        n_estimators=[10, 100, 200, 500],
        max_leaves=[10, 30, 50, 0],
        learning_rate=[1e-3, 1e-2, 1e-1, 0.3],
        max_depth=[3, 6, 10],
        gamma=[0, 0.1, 1.0],
    ),
}

CLASSIFIER_PARAMS_OPTUNA = {
    "LogisticRegression": dict(
        solver=CategoricalDistribution(["newton-cg", "lbfgs", "liblinear", "saga"]),
        penalty=CategoricalDistribution(["l2", "l1", "elasticnet", "none"]),
        C=FloatDistribution(1e-5, 1e4, log=True),
        l1_ratio=FloatDistribution(0.0, 1.0),
        class_weight=CategoricalDistribution([None, "balanced"]),
    ),
    "GaussianProcessClassifier": dict(
        kernel=CategoricalDistribution(
            [
                1 * RBF(),
                1 * DotProduct(),
                1 * Matern(),
                1 * RationalQuadratic(),
                1 * WhiteKernel(),
            ]
        ),
    ),
    "BernoulliNB": dict(
        alpha=FloatDistribution(1e-4, 10.0, log=True),
        fit_prior=CategoricalDistribution([True, False]),
        binarize=FloatDistribution(0.0, 1.0),
    ),
    "GaussianNB": dict(
        var_smoothing=FloatDistribution(1e-12, 1e-6, log=True),
    ),
    "MultinomialNB": dict(
        alpha=FloatDistribution(1e-4, 10.0, log=True),
        fit_prior=CategoricalDistribution([True, False]),
    ),
    "DecisionTreeClassifier": dict(
        criterion=CategoricalDistribution(["gini", "entropy", "log_loss"]),
        max_depth=IntDistribution(0, 100),
        min_samples_split=IntDistribution(2, 20),
        min_samples_leaf=IntDistribution(1, 10),
        max_features=CategoricalDistribution(["log2", "sqrt", None, 0.2, 0.5]),
        max_leaf_nodes=IntDistribution(100, 1000),
        class_weight=CategoricalDistribution([None, "balanced"]),
    ),
    "ExtraTreeClassifier": dict(
        criterion=CategoricalDistribution(["gini", "entropy", "log_loss"]),
        max_depth=IntDistribution(0, 100),
        min_samples_split=IntDistribution(2, 20),
        min_samples_leaf=IntDistribution(1, 10),
        max_features=CategoricalDistribution(["log2", "sqrt", None, 0.2, 0.5]),
        max_leaf_nodes=IntDistribution(0, 1000),
        class_weight=CategoricalDistribution([None, "balanced"]),
    ),
    "ExtraTreesClassifier": dict(
        n_estimators=IntDistribution(10, 500),
        criterion=CategoricalDistribution(["gini", "entropy", "log_loss"]),
        bootstrap=CategoricalDistribution([True, False]),
        max_depth=IntDistribution(0, 100),
        max_features=CategoricalDistribution(["log2", "sqrt", 0.2, 0.5]),
        min_samples_leaf=IntDistribution(1, 10),
        min_weight_fraction_leaf=FloatDistribution(0.0, 0.5),
        min_impurity_decrease=FloatDistribution(0.0, 1.0),
        min_samples_split=IntDistribution(2, 20),
        max_leaf_nodes=IntDistribution(0, 1000),
        class_weight=CategoricalDistribution([None, "balanced"]),
    ),
    "RandomForestClassifier": dict(
        n_estimators=IntDistribution(10, 500),
        criterion=CategoricalDistribution(["gini", "entropy", "log_loss"]),
        bootstrap=CategoricalDistribution([True, False]),
        max_depth=IntDistribution(0, 100),
        max_features=CategoricalDistribution(["log2", "sqrt", 0.2, 0.5, None]),
        min_samples_split=IntDistribution(2, 20),
        min_samples_leaf=IntDistribution(1, 10),
        max_leaf_nodes=IntDistribution(0, 1000),
        min_weight_fraction_leaf=FloatDistribution(0.0, 0.5),
        min_impurity_decrease=FloatDistribution(0.0, 1.0),
        class_weight=CategoricalDistribution([None, "balanced"]),
    ),
    "GradientBoostingClassifier": dict(
        n_estimators=IntDistribution(10, 500),
        criterion=CategoricalDistribution(["friedman_mse", "squared_error"]),
        learning_rate=FloatDistribution(1e-3, 1e-0, log=True),
        max_depth=IntDistribution(0, 10),
        min_samples_split=IntDistribution(2, 20),
        min_samples_leaf=IntDistribution(1, 10),
        min_weight_fraction_leaf=FloatDistribution(0.0, 0.5),
        min_impurity_decrease=FloatDistribution(0.0, 1.0),
        max_features=CategoricalDistribution(["log2", "sqrt", None, 0.5]),
        max_leaf_nodes=IntDistribution(0, 1000),
        subsample=FloatDistribution(0.4, 1.0),
        n_iter_no_change=IntDistribution(0, 50),
    ),
    "BaggingClassifier": dict(
        n_estimators=IntDistribution(5, 200),
        max_samples=FloatDistribution(0.4, 1.0),
        max_features=FloatDistribution(0.4, 1.0),
        bootstrap=CategoricalDistribution([True, False]),
        bootstrap_features=CategoricalDistribution([False, True]),
    ),
    "HistGradientBoostingClassifier": dict(
        learning_rate=FloatDistribution(1e-3, 5e-1, log=True),
        max_leaf_nodes=IntDistribution(3, 125),
        max_depth=IntDistribution(0, 30),
        min_samples_leaf=IntDistribution(5, 50),
        max_iter=IntDistribution(30, 300),
        l2_regularization=FloatDistribution(0.0, 1.0),
        max_features=FloatDistribution(0.3, 1.0),
        max_bins=IntDistribution(32, 256),
    ),
    "AdaBoostClassifier": dict(
        n_estimators=IntDistribution(10, 300),
        learning_rate=FloatDistribution(1e-3, 1.0, log=True),
    ),
    "KNeighborsClassifier": dict(
        n_neighbors=IntDistribution(3, 20),
        weights=CategoricalDistribution(["uniform", "distance"]),
        p=CategoricalDistribution([1, 2]),
    ),
    "NearestCentroid": dict(
        shrink_threshold=FloatDistribution(0.1, 1.0),
        metric=CategoricalDistribution(["euclidean", "manhattan"]),
    ),
    "LinearDiscriminantAnalysis": dict(
        solver=CategoricalDistribution(["svd", "lsqr", "eigen"]),
    ),
    "LinearSVC": dict(
        penalty=CategoricalDistribution(["l1", "l2"]),
        loss=CategoricalDistribution(["hinge", "squared_hinge"]),
        C=FloatDistribution(1e-4, 1e3, log=True),
        dual=CategoricalDistribution([True, False]),
        class_weight=CategoricalDistribution([None, "balanced"]),
    ),
    "SVC": dict(
        kernel=CategoricalDistribution(["linear", "rbf", "poly", "sigmoid"]),
        gamma=FloatDistribution(1e-6, 1e1, log=True),
        C=FloatDistribution(1e-3, 1e3, log=True),
    ),
    "RidgeClassifier": dict(
        alpha=FloatDistribution(1e-6, 1e2, log=True),
        solver=CategoricalDistribution(["auto", "svd", "cholesky"]),
    ),
    "SGDClassifier": dict(
        loss=CategoricalDistribution(["hinge", "log", "squared_hinge", "perceptron"]),
        penalty=CategoricalDistribution(["l2", "l1", "elasticnet", None]),
        alpha=FloatDistribution(1e-8, 1e-1, log=True),
        max_iter=IntDistribution(1000, 5000),
        learning_rate=CategoricalDistribution(["optimal", "invscaling", "adaptive"]),
        eta0=FloatDistribution(1e-4, 1.0, log=True),
        class_weight=CategoricalDistribution([None, "balanced"]),
    ),
    "Perceptron": dict(
        penalty=CategoricalDistribution(["l2", "l1", "elasticnet", None]),
        alpha=FloatDistribution(1e-8, 1e-1, log=True),
        eta0=FloatDistribution(1e-4, 1.0, log=True),
    ),
    "PassiveAggressiveClassifier": dict(
        C=FloatDistribution(0.1, 5.0),
        fit_intercept=CategoricalDistribution([True, False]),
        tol=FloatDistribution(1e-5, 1e-2, log=True),
    ),
    "MLPClassifier": dict(
        hidden_layer_sizes=dict(
            hidden_layer_size_1=IntDistribution(16, 256),
            hidden_layer_size_2=IntDistribution(0, 256),
        ),
        activation=CategoricalDistribution(["identity", "logistic", "relu"]),
        solver=CategoricalDistribution(["lbfgs", "adam", "sgd"]),
        alpha=FloatDistribution(1e-6, 1e-2, log=True),
        learning_rate=CategoricalDistribution(["constant", "invscaling", "adaptive"]),
        learning_rate_init=FloatDistribution(1e-5, 1e-1, log=True),
        early_stopping=CategoricalDistribution([False, True]),
        batch_size=IntDistribution(16, 256),
    ),
    "LGBMClassifier": dict(
        boosting_type=CategoricalDistribution(["gbdt", "dart"]),
        n_estimators=IntDistribution(10, 500),
        num_leaves=IntDistribution(5, 125),
        learning_rate=FloatDistribution(1e-4, 1e-0, log=True),
        max_depth=IntDistribution(-1, 30),
        class_weight=CategoricalDistribution([None, "balanced"]),
        min_split_gain=FloatDistribution(0.0, 1.0),
        min_child_weight=FloatDistribution(1e-3, 10.0, log=True),
        min_child_samples=IntDistribution(5, 100),
        min_data_in_leaf=IntDistribution(1, 50),
        subsample=FloatDistribution(0.4, 1.0),
        subsample_freq=IntDistribution(0, 10),
        colsample_bytree=FloatDistribution(0.3, 1.0),
        reg_alpha=FloatDistribution(0.0, 1.0),
        reg_lambda=FloatDistribution(1e-6, 10.0, log=True),
    ),
    "XGBClassifier": dict(
        n_estimators=IntDistribution(10, 500),
        learning_rate=FloatDistribution(1e-4, 1e-0, log=True),
        max_leaves=IntDistribution(0, 50),
        max_depth=IntDistribution(1, 20),
        grow_policy=CategoricalDistribution(["depthwise", "lossguide"]),
        booster=CategoricalDistribution(["gbtree", "gblinear", "dart"]),
        tree_method=CategoricalDistribution(["exact", "approx", "hist"]),
        gamma=FloatDistribution(0.0, 5.0),
        min_child_weight=IntDistribution(1, 10),
        subsample=FloatDistribution(0.4, 1.0),
        sampling_method=CategoricalDistribution(["uniform", "gradient_based"]),
        colsample_bytree=FloatDistribution(0.3, 1.0),
        reg_alpha=FloatDistribution(0.0, 1.0),
        reg_lambda=FloatDistribution(1e-3, 10.0, log=True),
    ),
}
