"""
Description
-----------
This module contains utility functions used in the package.

Functions
---------
get_estimator_name(estimator)
    Get the name of a scikit-learn estimator.
get_scores(cv_results_df, main_metric, sec_metrics, n_folds)
    Get means and standard deviations of the main and secondary metrics from the CV results.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from .model_selection import LOWER_BETTER


def get_estimator_name(model: BaseEstimator) -> str:
    """
    Get the name of the estimator from a scikit-learn model.

    Parameters
    ----------
    model : BaseEstimator
        A scikit-learn model. Can be a Pipeline or a direct estimator.

    Returns
    -------
    str
        The name of the estimator.
    """
    if hasattr(model, "steps"):  # Pipeline
        estimator = model.steps[-1][1]
    else:  # direct estimator
        estimator = model
    return estimator.__class__.__name__

def get_scores(cv_results_df: pd.DataFrame, main_metric: str, sec_metrics: list[str], n_folds: int) -> tuple[float, float, dict[str, tuple[float, float]]]:
    """
    Get means and standard deviations of the main and secondary metrics from the CV results.

    Parameters
    ----------
    cv_results_df : pd.DataFrame
        DataFrame containing the CV results.
    main_metric : str
        The main metric to extract results for.
    sec_metrics : list of str
        Secondary metrics to extract results for.
    n_folds : int
        Number of folds used in the CV.

    Returns
    -------
    tuple of floats and dict of str to tuple of floats
        Mean and standard deviation of the main metric, and means and standard deviations of the secondary metrics.
    """
    required_columns = [f"rank_test_{metric}" for metric in [main_metric] + sec_metrics] + [
        f"split{i}_test_{metric}" for metric in [main_metric] + sec_metrics for i in range(n_folds)
    ]
    assert all([col in cv_results_df.columns for col in required_columns]), (
        f"CV results do not contain all required columns: {required_columns}"
    )

    all_main_scores = [
        cv_results_df[cv_results_df[f"rank_test_{main_metric}"] == 1].iloc[0][
            f"split{i}_test_{main_metric}"
        ]
        for i in range(n_folds)
    ]
    mean_score_main = (
        -np.mean(all_main_scores) if (main_metric in LOWER_BETTER) else np.mean(all_main_scores)
    )
    std_score_main = np.std(all_main_scores)

    sec_metrics_scores = {}
    for metric in sec_metrics:
        all_scores = [
            cv_results_df[cv_results_df[f"rank_test_{metric}"] == 1].iloc[0][
                f"split{i}_test_{metric}"
            ]
            for i in range(n_folds)
        ]
        sec_metrics_scores[metric] = (
            -np.mean(all_scores) if (metric in LOWER_BETTER) else np.mean(all_scores),
            np.std(all_scores),
        )

    return mean_score_main, std_score_main, sec_metrics_scores