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
load_config(file_path)
    Load configuration from a YAML file.
print_performance(model_name, results_dict, file=None)
    Print the performance of a model based on the results dictionary.
print_file_console(file, message, mode='a', end='\n')
    Print a message to a file and the console.
print_final_results(final_model_name, final_hyperparameters, main_metric,
                    mean_score_main, std_score_main, median_score_main,
                    sec_metrics_scores, file=None)
    Print final results of the model training and evaluation.
"""

import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator
from .metrics import LOWER_BETTER


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


def get_scores(
    cv_results_df: pd.DataFrame, main_metric: str, sec_metrics: list[str], n_folds: int
) -> tuple[float, float, float, dict[str, tuple[float, float, float]]]:
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
        Mean, standard deviation and median of the main metric, and means, standard deviations,
        and medians of the secondary metrics.
    """
    required_columns = [
        f"rank_test_{metric}" for metric in [main_metric] + sec_metrics
    ] + [
        f"split{i}_test_{metric}"
        for metric in [main_metric] + sec_metrics
        for i in range(n_folds)
    ]
    assert all(
        [col in cv_results_df.columns for col in required_columns]
    ), f"CV results do not contain all required columns: {required_columns}"

    all_main_scores = [
        cv_results_df[cv_results_df[f"rank_test_{main_metric}"] == 1].iloc[0][
            f"split{i}_test_{main_metric}"
        ]
        for i in range(n_folds)
    ]
    mean_score_main = (
        -np.mean(all_main_scores)
        if (main_metric in LOWER_BETTER)
        else np.mean(all_main_scores)
    )
    std_score_main = np.std(all_main_scores)
    median_score_main = (
        -np.median(all_main_scores)
        if (main_metric in LOWER_BETTER)
        else np.median(all_main_scores)
    )

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
            (
                -np.median(all_scores)
                if (metric in LOWER_BETTER)
                else np.median(all_scores)
            ),
        )

    return mean_score_main, std_score_main, median_score_main, sec_metrics_scores


def load_config(file_path: str) -> tuple:
    """
    Load configuration from a YAML file.

    Parameters
    ----------
    file_path : str
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Configuration loaded from the YAML file.
    """
    with open(file_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def print_performance(
    model_name: str, results_dict: dict[str, list[float]], file: str | None = None
):
    """
    Print the performance of a model based on the results dictionary.

    Parameters
    ----------
    model_name : str
        Name of the model.
    results_dict : dict[str, list[float]]
        Dictionary containing performance metrics.
    file : str or None, default None
        If provided, the output will additionally be written to this file.

    Returns
    -------
    None
    """
    log_str = " " * 20 + f"Performance for {model_name}:\n"
    for metric, scores in results_dict.items():
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        median_score = np.median(scores)
        log_str += (
            " " * 20
            + f"{metric}: {mean_score:.3f} ± {std_score:.3f} (median: {median_score:.3f})\n"
        )

    if file:
        with open(file, "a") as f:
            f.write(log_str)

    print(log_str, end="")


def print_file_console(file: str, message: str, mode: str = "a", end: str = "\n"):
    """
    Print a message to a file and the console.

    Parameters
    ----------
    file : str
        Path to the file where the message will be written.
    message : str
        The message to print.
    mode : str, default 'a'
        File mode for writing. Default is append mode.
    end : str, default '\n'
        String appended after the message. Default is newline.

    Returns
    -------
    None
    """
    with open(file, mode) as f:
        f.write(message + end)
    print(message, end=end)


def print_final_results(
    final_model_name: str,
    final_hyperparameters: dict[str, int | float | str],
    main_metric: str,
    mean_score_main: float,
    std_score_main: float,
    median_score_main: float,
    sec_metrics_scores: dict[str, tuple[float, float, float]],
    file: str | None = None,
):
    """
    Print final results.

    Parameters
    ----------
    final_model_name : str
        Name of the final model.
    final_hyperparameters : dict[str, int | float | str]
        Hyperparameters of the final model.
    main_metric : str
        The main metric used for evaluation.
    mean_score_main : float
        Mean score of the main metric.
    std_score_main : float
        Standard deviation of the main metric score.
    median_score_main : float
        Median score of the main metric.
    sec_metrics_scores : dict[str, tuple[float, float, float]]
        Dictionary containing secondary metrics scores (mean, std, median).
    file : str or None, default None
        If provided, the output will additionally be written to this file.

    Returns
    -------
    None
    """
    print_file_console(message=" " * 20 + "-" * 13, file=file)
    print_file_console(message=" " * 20 + "Final results", file=file)
    print_file_console(message=" " * 20 + "-" * 13, file=file)
    print_file_console(
        message=" " * 20 + f"Final model: {final_model_name}",
        file=file,
    )
    print_file_console(message=" " * 20 + "Hyperparameters:", file=file)
    for f in final_hyperparameters:
        print_file_console(
            message=" " * 20 + f"{f}: {final_hyperparameters[f]}",
            file=file,
        )
    print_file_console(
        message=" " * 20
        + f"Mean {main_metric}: {mean_score_main:.3f} ± {std_score_main:.3f}.",
        file=file,
    )
    print_file_console(
        message=" " * 20 + f"Median {main_metric}: {median_score_main:.3f}.",
        file=file,
    )
    for metric in sec_metrics_scores:
        print_file_console(
            message=" " * 20
            + f"Mean {metric}: {sec_metrics_scores[metric][0]:.3f} ± {sec_metrics_scores[metric][1]:.3f}.",
            file=file,
        )
        print_file_console(
            message=" " * 20 + f"Median {metric}: {sec_metrics_scores[metric][2]:.3f}.",
            file=file,
        )
