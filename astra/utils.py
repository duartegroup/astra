"""
Description
-----------
This module contains utility functions used in the package.

Functions
---------
get_data(data, features)
    Load data from a file into a pandas DataFrame.
get_models(main_metric, sec_metrics, scaler=None, custom_models=None)
    Get models and their hyperparameters based on the main metric and secondary metrics.
build_model(model_class, impute=None, remove_constant=None, remove_correlated=None, scaler=None)
    Build a scikit-learn model with optional preprocessing steps.
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import VarianceThreshold
from .data.processing import CorrelationFilter
from sklearn.pipeline import make_pipeline
import logging
import ast
from .metrics import LOWER_BETTER, REGRESSION_METRICS, CLASSIFICATION_METRICS
from .models.regression import REGRESSORS, REGRESSOR_PARAMS
from .models.classification import (
    CLASSIFIERS,
    CLASSIFIER_PARAMS,
    NON_PROBABILISTIC_MODELS,
)


def get_data(data: str, features: str) -> pd.DataFrame:
    """
    Load data from a file into a pandas DataFrame.

    Parameters
    ----------
    data : str
        Path to the data file. Supported formats: CSV, pickle, or parquet.
    features : str
        Name of the column containing features.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the loaded data.

    Raises
    ------
    ValueError
        If the file format is unsupported.
    """
    if data.endswith(".csv"):
        data_df = pd.read_csv(data)
        # convert features column from string representation of list to list (if it exists)
        if features in data_df.columns:
            try:
                data_df[features] = data_df[features].apply(
                    lambda x: ast.literal_eval(x)
                )
            except (ValueError, SyntaxError):
                try:
                    data_df[features] = data_df[features].apply(
                        lambda x: np.fromstring(x.strip("[]"), sep=" ")
                    )
                except ValueError:
                    logging.warning(
                        f"Could not convert {features} column to list. "
                        "Using it as is, but this may cause issues."
                    )
    elif data.endswith(".pkl") or data.endswith(".pickle"):
        data_df = pd.read_pickle(data)
    elif data.endswith(".parquet"):
        data_df = pd.read_parquet(data)
    else:
        raise ValueError("Unsupported file format. Use CSV, pickle, or parquet.")

    return data_df


def get_models(
    main_metric: str,
    sec_metrics: list[str],
    scaler: str | None = None,
    custom_models: (
        dict[str, None | dict[str, dict] | tuple[dict[str, dict], dict[str, dict]]]
        | None
    ) = None,
) -> tuple[
    dict[str, BaseEstimator],
    dict[str, dict[str, list]],
    dict[str, dict[str, list]] | None,
]:
    """
    Get models and their hyperparameters based on the main metric and secondary metrics.

    Parameters
    ----------
    main_metric : str
        The main metric to determine the type of models.
    sec_metrics : list of str
        List of secondary metrics to validate against the main metric.
    scaler : str or None, default None
        The type of scaler used. If 'Standard', some models are excluded.
    custom_models : dict[str, None | dict[str, dict] | tuple[dict[str, dict], dict[str, dict]]] or None, default None
        Dictionary of models to use for benchmarking. If None, default models will be used.
        The keys should be the model names, and the values should be dictionaries of starting
        hyperparameters for the model, and/or a dictionary of hyperparameter search grids.

    Returns
    -------
    tuple of dicts
        A tuple containing:
        - A dictionary of models with their names as keys and scikit-learn estimators as values.
        - A dictionary of hyperparameters for each model.
        - A dictionary of custom hyperparameters if provided, otherwise None.

    Raises
    ------
    ValueError
        If the main metric or any secondary metric is not recognized.
    """
    if main_metric in REGRESSION_METRICS:
        for metric in sec_metrics:
            assert (
                metric in REGRESSION_METRICS
            ), f"Secondary metric '{metric}' is not a regression metric."

        models = REGRESSORS
        params = REGRESSOR_PARAMS
        logging.info("Benchmarking regression models.")

    elif main_metric in CLASSIFICATION_METRICS:
        for metric in sec_metrics:
            assert (
                metric in CLASSIFICATION_METRICS
            ), f"Secondary metric '{metric}' is not a classification metric."

        if (
            main_metric in ["roc_auc", "pr_auc"]
            or ("roc_auc" in sec_metrics)
            or ("pr_auc" in sec_metrics)
        ):
            models = {
                c: CLASSIFIERS[c]
                for c in CLASSIFIERS
                if c not in NON_PROBABILISTIC_MODELS
            }
        else:
            models = CLASSIFIERS
        params = CLASSIFIER_PARAMS
        logging.info("Benchmarking classification models.")

    else:
        raise ValueError(
            "Invalid metrics specified. Known metrics are:",
            REGRESSION_METRICS,
            "and",
            CLASSIFICATION_METRICS,
        )

    # drop MultinomialNB for standard scaler
    if scaler == "Standard" and "MultinomialNB" in models and not custom_models:
        models.pop("MultinomialNB")
        params.pop("MultinomialNB")

    if custom_models is not None:
        logging.info("Using provided models.")
        for model in custom_models:
            assert model in REGRESSORS or model in CLASSIFIERS, (
                f"Model '{model}' is not a valid model. "
                "Please provide a valid model from astra.models."
            )
        if (
            main_metric in ["roc_auc", "pr_auc"]
            or ("roc_auc" in sec_metrics)
            or ("pr_auc" in sec_metrics)
        ):
            custom_models = {
                model: custom_models[model]
                for model in custom_models
                if model not in NON_PROBABILISTIC_MODELS
            }
            logging.info(
                "Removing non-probabilistic models as one of the metrics is ROC AUC or PR AUC."
            )
        models = {model: models[model] for model in custom_models}
        custom_params = {
            model: custom_models[model]["params"]
            for model in custom_models
            if custom_models[model]["params"]
        }
        custom_hparams = {
            model: custom_models[model]["hparam_grid"]
            for model in custom_models
            if custom_models[model]["hparam_grid"]
        }
        params = {
            model: custom_hparams[model] if model in custom_hparams else params[model]
            for model in models
        }
        return models, params, custom_params
    else:
        return models, params, None


def build_model(
    model_class: BaseEstimator,
    impute: str | float | int | None = None,
    remove_constant: float | None = None,
    remove_correlated: float | None = None,
    scaler: str | None = None,
) -> BaseEstimator | Pipeline:
    """
    Build a scikit-learn model with optional preprocessing steps.

    Parameters
    ----------
    model_class : BaseEstimator
        A scikit-learn model class to be instantiated.
    impute : str | float | int or None, default None
        Imputation strategy to apply before the model. Valid options are 'mean', 'median',
        'knn', or a numeric value for constant imputation. If None, no imputation is applied.
    remove_constant : float or None, default None
        Threshold for variance to remove constant features. If None, no features are removed.
    remove_correlated : float or None, default None
        Threshold for correlation to remove correlated features. If None, no features are removed.
    scaler : str or None, default None
        Type of scaler to apply before the model. Valid options are 'MinMax' or 'Standard'.
        If None, no scaling is applied.

    Returns
    -------
    BaseEstimator or Pipeline
        A scikit-learn model or a Pipeline with the specified preprocessing steps.

    Raises
    ------
    ValueError
        If an unknown scaler or imputation strategy is provided, or if remove_constant or
        remove_correlated are not numeric values.
    """
    pipeline_steps = []

    if impute:
        if isinstance(impute, str):
            if impute == "mean":
                imputer = SimpleImputer(strategy="mean")
            elif impute == "median":
                imputer = SimpleImputer(strategy="median")
            elif impute == "knn":
                imputer = KNNImputer()
            else:
                raise ValueError(
                    "Unknown imputation strategy. Must be 'mean' or 'median'"
                )
        elif isinstance(impute, (int, float)):
            imputer = SimpleImputer(strategy="constant", fill_value=impute)
        else:
            raise ValueError("Imputation strategy must be a string or a numeric value")
        pipeline_steps.append(imputer)

    if remove_constant:
        if not isinstance(remove_constant, float):
            raise ValueError(
                "remove_constant must be a numeric value (a threshold for variance)."
            )
        selector = VarianceThreshold(threshold=remove_constant)
        pipeline_steps.append(selector)

    if remove_correlated:
        if not isinstance(remove_correlated, float):
            raise ValueError(
                "remove_correlated must be a numeric value (a threshold for correlation)."
            )
        filter = CorrelationFilter(threshold=remove_correlated)
        pipeline_steps.append(filter)

    if scaler:
        if scaler == "MinMax":
            s = MinMaxScaler()
        elif scaler == "Standard":
            s = StandardScaler()
        else:
            raise ValueError("Unknown scaler. Must be either MinMax or Standard")
        pipeline_steps.append(s)

    return (
        make_pipeline(*pipeline_steps, model_class) if pipeline_steps else model_class
    )


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
