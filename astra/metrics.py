"""
This module contains functions for model selection and evaluation.

Attributes
----------
CLASSIFICATION_METRICS : dict
    A dictionary mapping classification metric names to their corresponding functions.
REGRESSION_METRICS : dict
    A dictionary mapping regression metric names to their corresponding functions.
MULTICLASS_METRICS : dict
    A dictionary mapping multiclass classification metric names to their corresponding functions.
KNOWN_METRICS : dict
    A dictionary mapping all metric names to their corresponding functions.
SCORING : dict
    A dictionary mapping all metric names to their corresponding scoring functions.
HIGHER_BETTER : list
    A list of metrics for which higher scores are better.
LOWER_BETTER : list
    A list of metrics for which lower scores are better.
"""

from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    f1_score,
    make_scorer,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)


def get_kendalltau_score(y_true, y_pred):
    """
    Calculate the Kendall Tau correlation coefficient.

    Parameters
    ----------
    y_true : list
        True values.
    y_pred : list
        Predicted values.

    Returns
    -------
    float
        The Kendall Tau correlation coefficient.
    """
    return kendalltau(y_true, y_pred).statistic


def get_pearsonr_score(y_true, y_pred):
    """
    Calculate the Pearson correlation coefficient.

    Parameters
    ----------
    y_true : list
        True values.
    y_pred : list
        Predicted values.

    Returns
    -------
    float
        The Pearson correlation coefficient.
    """
    return pearsonr(y_true, y_pred).statistic


def get_spearmanr_score(y_true, y_pred):
    """
    Calculate the Spearman correlation coefficient.

    Parameters
    ----------
    y_true : list
        True values.
    y_pred : list
        Predicted values.

    Returns
    -------
    float
        The Spearman correlation
    """
    return spearmanr(y_true, y_pred).statistic


CLASSIFICATION_METRICS = {
    "accuracy": accuracy_score,
    "balanced_accuracy": balanced_accuracy_score,
    "brier_score": brier_score_loss,
    "f1": f1_score,
    "pr_auc": average_precision_score,
    "roc_auc": roc_auc_score,
    "mcc": matthews_corrcoef,
    "precision": precision_score,
    "recall": recall_score,
}

REGRESSION_METRICS = {
    "r2": r2_score,
    "rmse": root_mean_squared_error,
    "mse": mean_squared_error,
    "mae": mean_absolute_error,
    "kendalltau": get_kendalltau_score,
    "pearsonr": get_pearsonr_score,
    "spearmanr": get_spearmanr_score,
}

# Evaluation metrics for ordinal classification:
# Weighted Cohen Kappa Score, Ref.: https://aclanthology.org/2021.acl-long.214.pdf
# RMSE and MSE, Ref.: https://link.springer.com/chapter/10.1007/978-3-642-01818-3_25
MULTICLASS_METRICS = {
    "cohen_kappa": cohen_kappa_score,
    "rmse": root_mean_squared_error,
    "mse": mean_squared_error,
}

KNOWN_METRICS = {**CLASSIFICATION_METRICS, **REGRESSION_METRICS, **MULTICLASS_METRICS}

lin_kappa_score = make_scorer(cohen_kappa_score, weights="linear")
kendalltau_score = make_scorer(get_kendalltau_score)
pearsonr_score = make_scorer(get_pearsonr_score)
spearmanr_score = make_scorer(get_spearmanr_score)

SCORING = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "brier_score": "brier_score_loss",
    "f1": "f1",
    "pr_auc": "average_precision",
    "roc_auc": "roc_auc",
    "mcc": "matthews_corrcoef",
    "precision": "precision",
    "recall": "recall",
    "r2": "r2",
    "rmse": "neg_root_mean_squared_error",
    "mse": "neg_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "kendalltau": kendalltau_score,
    "pearsonr": pearsonr_score,
    "spearmanr": spearmanr_score,
    "cohen_kappa": lin_kappa_score,
}

HIGHER_BETTER = [
    "accuracy",
    "balanced_accuracy",
    "brier_score",
    "recall",
    "roc_auc",
    "f1",
    "pr_auc",
    "roc_auc",
    "mcc",
    "precision",
    "r2",
    "cohen_kappa",
]
LOWER_BETTER = ["rmse", "mse", "mae"]
