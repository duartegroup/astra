"""
Description
-----------
This module contains functions for model selection and evaluation.

Functions
---------
find_n_best_models(results_dic, metric, bf_corr=True)
    Find the n best models that don't perform significantly differently with respect to a given metric as determined using the Friedman test.
perform_statistical_tests(results_dic, metric)
    Perform Conover post-hoc and rank-sum tests on the performance of models.
check_best_model(results_dic, test_statistics, metric)
    Check if there is a model that is significantly better than the others.
get_cv_performance(model_class, df, n_folds, fold_col, metric_list, scaler=None)
    Get the cross-validated performance of a model.
get_optimised_cv_performance(model_class, df, n_folds, fold_col, metric_list, main_metric, parameters, n_jobs, scaler=None)
    Get the cross-validated performance of a model with optimised hyperparameters using grid search with nested cross-validation.
get_best_hparams(model_class, df, n_folds, fold_col, metric, parameters, n_jobs, scaler=None)
    Get the best hyperparameters for a model using grid search with (non-nested) cross-validation.
get_best_model(results_dict, main_metric, secondary_metrics, bf_corr=True)
    Get the best model from a dictionary of model results.

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

import pandas as pd
import numpy as np
import pingouin as pg
import scikit_posthocs as sp
from scipy.stats import ranksums
from sklearn.base import clone, BaseEstimator
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    f1_score,
    cohen_kappa_score,
    matthews_corrcoef,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    root_mean_squared_error,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    make_scorer,
)
from .models.classification import non_probabilistic_models

# Evaluation metrics for ordinal classification:
# Weighted Cohen Kappa Score, Ref.: https://aclanthology.org/2021.acl-long.214.pdf
# RMSE and MSE, Ref.: https://link.springer.com/chapter/10.1007/978-3-642-01818-3_25
CLASSIFICATION_METRICS = {
    "F1": f1_score,
    "PR AUC": average_precision_score,
    "ROC AUC": roc_auc_score,
    "MCC": matthews_corrcoef,
}
REGRESSION_METRICS = {
    "R2": r2_score,
    "RMSE": root_mean_squared_error,
    "MSE": mean_squared_error,
    "MAE": mean_absolute_error,
}
MULTICLASS_METRICS = {
    "Cohen Kappa": cohen_kappa_score,
    "RMSE": root_mean_squared_error,
    "MSE": mean_squared_error,
}
KNOWN_METRICS = {**CLASSIFICATION_METRICS, **REGRESSION_METRICS, **MULTICLASS_METRICS}
lin_kappa_score = make_scorer(cohen_kappa_score, weights="linear")
SCORING = {
    "F1": "f1",
    "PR AUC": "average_precision",
    "ROC AUC": "roc_auc",
    "MCC": "matthews_corrcoef",
    "R2": "r2",
    "RMSE": "neg_root_mean_squared_error",
    "MSE": "neg_mean_squared_error",
    "MAE": "neg_mean_absolute_error",
    "Cohen Kappa": lin_kappa_score,
}
HIGHER_BETTER = ["F1", "PR AUC", "ROC AUC", "MCC", "R2", "Cohen Kappa"]
LOWER_BETTER = ["RMSE", "MSE", "MAE"]


def find_n_best_models(
    results_dic: dict[str, dict[str, list[float]]], metric: str, bf_corr: bool = True
) -> list[str]:
    """
    Find the n best models that don't perform significantly differently with respect to
    a given metric as determined using the Friedman test.

    Parameters
    ----------
    results_dic : dict[str, dict[str, list[float]]]
        A dictionary mapping model names to dictionaries of metric names and scores.
    metric : str
        The metric to use for model comparison.
    bf_corr : bool, default=True
        Whether to apply Bonferroni correction to the significance level.

    Returns
    -------
    list[str]
        A list of the n best models.
    """
    assert (
        metric in KNOWN_METRICS
    ), f"Unknown metric. Known metrics are: {', '.join(KNOWN_METRICS)}"
    maximise = True if metric in HIGHER_BETTER else False

    # Create a dataframe from the results dictionary
    results_df = pd.DataFrame.from_dict(results_dic)
    stat_df = pd.DataFrame(
        results_df.loc[metric].tolist(), index=results_df.loc[metric].index
    ).T
    stat_for_test = stat_df.dropna(axis=1)

    best_models = []
    for n_models in range(len(stat_for_test.columns), 1, -1):
        # Perform Friedman test
        friedman = pg.friedman(stat_for_test)["p-unc"].values[0]

        # Bonferroni correction of significance level
        if bf_corr:
            threshold = 0.05 / n_models
        else:
            threshold = 0.05

        # Check if there is a statistically significant difference
        if friedman < threshold:  # significant difference
            # Remove model with the worst median score
            model_labels = stat_for_test.columns
            median_scores = stat_for_test.median()
            combined = list(zip(median_scores, model_labels))
            sorted_scores = sorted(combined, key=lambda x: x[0], reverse=maximise)
            worst_model = sorted_scores[-1][1]
            stat_for_test = stat_for_test.drop(worst_model, axis=1)
        else:  # no significant difference
            best_models = list(stat_for_test.columns)
            break

    return best_models


def perform_statistical_tests(
    results_dic: dict[str, dict[str, list[float]]], metric: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform Conover post-hoc and rank-sum tests on the performance of models.

    Parameters
    ----------
    results_dic : dict[str, dict[str, list[float]]]
        A dictionary mapping model names to dictionaries of metric names and scores.
    metric : str
        The metric to use for model comparison.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        A tuple containing the post-hoc and rank-sum test results.
    """
    assert (
        metric in KNOWN_METRICS
    ), f"Unknown metric. Known metrics are: {', '.join(KNOWN_METRICS)}"

    # Create a dataframe from the results dictionary
    results_df = pd.DataFrame.from_dict(results_dic)
    stat_df = pd.DataFrame(
        results_df.loc[metric].tolist(), index=results_df.loc[metric].index
    ).T
    stat_for_test = stat_df.dropna(axis=1)

    # Perform post-hoc test
    post_hoc_stats = sp.posthoc_conover_friedman(stat_for_test, p_adjust="holm")

    # Perform rank-sum test
    rank_sum_p_values = np.empty(
        (len(stat_for_test.columns), len(stat_for_test.columns))
    )
    for n, col1 in enumerate(stat_for_test):
        for m, col2 in enumerate(stat_for_test):
            rank_sum_p_values[n, m] = ranksums(
                stat_for_test[col1],
                stat_for_test[col2],
            ).pvalue
    rank_sum_stats = pd.DataFrame(
        rank_sum_p_values, columns=stat_for_test.columns, index=stat_for_test.columns
    )

    return post_hoc_stats, rank_sum_stats


def check_best_model(
    results_dic: dict[str, dict[str, list[float]]],
    test_statistics: pd.DataFrame,
    metric: str,
) -> str | None:
    """
    Check if there is a model that is significantly better than the others.

    Parameters
    ----------
    results_dic : dict[str, dict[str, list[float]]]
        A dictionary mapping model names to dictionaries of metric names and scores.
    test_statistics : pd.DataFrame
        A dataframe containing the results of the statistical test.
    metric : str
        The metric to use for model comparison.

    Returns
    -------
    str or None
        The name of the best model, or None if no model is significantly better.
    """
    assert (
        metric in KNOWN_METRICS
    ), f"Unknown metric. Known metrics are: {', '.join(KNOWN_METRICS)}"

    # get model ranking according to median score
    scores = [np.median(results_dic[model][metric]) for model in results_dic]
    names = [model for model in results_dic]
    if metric in HIGHER_BETTER:
        best_models = [names[i] for i in np.argsort(scores)[::-1]]
    else:
        best_models = [names[i] for i in np.argsort(scores)]

    # get dictionary of models that are significantly different from the others,
    # sorted according to how many models perform significantly different
    sig_diff_models = (
        test_statistics.where(test_statistics < 0.05)
        .dropna(axis=0, how="all")
        .dropna(axis=1, how="all")
        .count()
        .sort_values(ascending=False)
        .to_dict()
    )

    # if no models are significantly different, return None
    if len(sig_diff_models) == 0:
        return None

    # loop over these models, and check if they are particularly
    # good (top half) or bad
    final_models = []
    models = list(sig_diff_models.keys())
    for model in models:
        rank = best_models.index(model)
        if rank < 0.5 * len(names):
            final_models.append(model)
            # handle case if more than one model is significantly
            # better than the same number of models
            done = True
            for other_model in models[models.index(model) + 1 :]:
                if sig_diff_models[other_model] == sig_diff_models[model]:
                    done = False
            if done:
                break

    # if more than one model is significantly better than the others,
    # choose the one with the lowest sum of p-values.
    # This is likely better than choosing the model with the best
    # median score, because it takes the distribution of scores into account.
    if len(final_models) > 1:
        model_pvalues = (
            test_statistics.where(test_statistics < 0.05)
            .dropna(axis=0, how="all")
            .dropna(axis=1, how="all")
            .to_dict(orient="list")
        )
        pvalue_scores = [np.nansum(model_pvalues[model]) for model in final_models]
        smallest_pvalue = min(pvalue_scores)
        if pvalue_scores.count(smallest_pvalue) == 1:
            final_model = final_models[np.argmin(pvalue_scores)]
        else:
            # If there is more than one model with the same sum of p-values,
            # choose the one with the best median score.
            best_model_idxs = np.where(np.array(pvalue_scores) == smallest_pvalue)[
                0
            ].tolist()
            best_model_scores = [
                np.median(results_dic[model][metric])
                for i, model in enumerate(final_models)
                if i in best_model_idxs
            ]
            if metric in HIGHER_BETTER:
                final_model_idx = best_model_idxs[np.argmax(best_model_scores)]
            else:
                final_model_idx = best_model_idxs[np.argmin(best_model_scores)]
            final_model = final_models[final_model_idx]

    # if none of the models are in top half, return None
    elif len(final_models) == 0:
        final_model = None

    # if only one model is significantly better than the others, return that model
    else:
        final_model = final_models[0]

    return final_model


def get_best_model(
    results_dict: dict[str, dict[str, list[float]]],
    main_metric: str,
    secondary_metrics: list[str],
    bf_corr: bool = True,
) -> tuple[str, str]:
    """
    Get the best model from a dictionary of model results.

    Parameters
    ----------
    results_dict : dict[str, dict[str, list[float]]]
        A dictionary mapping model names to dictionaries of metric names and scores.
    main_metric : str
        The main metric to use for model comparison.
    secondary_metrics : list[str]
        A list of secondary metrics to use for model comparison.
    bf_corr : bool, default=True
        Whether to apply Bonferroni correction to the significance level.

    Returns
    -------
    tuple[str, str]
        A tuple containing the name of the best model and the reason for its selection.
    """
    # Perform Friedman test to find the n best models
    n_best_models = find_n_best_models(results_dict, main_metric, bf_corr)
    results_dict_best = {model: results_dict[model] for model in n_best_models}

    # Perform statistical tests on the best models
    post_hoc_stats, rank_sum_stats = perform_statistical_tests(
        results_dict_best, main_metric
    )

    # Check if the post-hoc test yields a best model
    best_model = check_best_model(results_dict_best, post_hoc_stats, main_metric)
    reason = "post-hoc test"

    # Fall back to rank-sum test
    if not best_model:
        best_model = check_best_model(results_dict_best, rank_sum_stats, main_metric)
        reason = "rank-sum test"

    # Fall back to secondary metrics if the main one doesn't yield a best model
    if not best_model:
        for metric in secondary_metrics:
            post_hoc_stats, rank_sum_stats = perform_statistical_tests(
                results_dict_best, metric
            )
            best_model = check_best_model(results_dict_best, post_hoc_stats, metric)
            reason = f"post-hoc test using {metric}"
            if best_model:
                break
            best_model = check_best_model(results_dict_best, rank_sum_stats, metric)
            reason = f"rank-sum test using {metric}"
            if best_model:
                break

    # If there are no statistically significant differences between the models using any of the metrics,
    # select the model with the best median score.
    if not best_model:
        scores = [
            np.median(results_dict_best[model][main_metric])
            for model in results_dict_best
        ]
        names = [model for model in results_dict_best]
        if main_metric in HIGHER_BETTER:
            best_model = names[np.argmax(scores)]
        else:
            best_model = names[np.argmin(scores)]
        reason = "median score"

    return best_model, reason


def get_cv_performance(
    model_class: BaseEstimator,
    df: pd.DataFrame,
    n_folds: int,
    fold_col: str,
    metric_list: list[str],
    scaler: str | None = None,
) -> dict[str, list[float]]:
    """
    Get the cross-validated performance of a model.

    Parameters
    ----------
    model_class : BaseEstimator
        A scikit-learn model.
    df : pd.DataFrame
        A dataframe containing the features and target values. Must contain a column
        with the fold indices. The features must be in a column called 'Features' and
        the target values in a column called 'Target'.
    n_folds : int
        The number of folds in the cross-validation.
    fold_col : str
        The name of the column containing the fold indices.
    metric_list : list[str]
        A list of metrics to use for evaluation.
    scaler : str or None, default=None
        The type of scaler to use. Valid choices are 'MinMax' and 'Standard'.

    Returns
    -------
    dict[str, list[float]]
        A dictionary mapping metrics to lists of scores.
    """
    for metric in metric_list:
        assert (
            metric in KNOWN_METRICS.keys()
        ), f"Unknown metric. Known metrics are: {', '.join(KNOWN_METRICS.keys())}"
    metrics_dict = {metric: [] for metric in metric_list}

    classification = True if metric_list[0] in CLASSIFICATION_METRICS else False

    all_folds = [df[df[fold_col] == i] for i in range(n_folds)]

    if scaler:
        if scaler == "MinMax":
            s = MinMaxScaler()
        elif scaler == "Standard":
            s = StandardScaler()
        else:
            raise ValueError("Unknown scaler. Must be either MinMax or Standard")
        model = make_pipeline(s, model_class)
    else:
        model = model_class

    for test_fold in range(n_folds):
        # train model
        train_folds = [
            f
            for i, f in enumerate(all_folds)
            if i
            not in [
                test_fold,
            ]
        ]
        train_data = pd.concat([f for f in train_folds])
        X = np.vstack(train_data["Features"].to_numpy())
        y = np.vstack(train_data["Target"].to_numpy()).ravel()
        m = clone(model)
        m.fit(X, y)

        # evaluate model
        test_data = all_folds[test_fold]
        X_test = np.vstack(test_data["Features"].to_numpy())
        y_test = np.vstack(test_data["Target"].to_numpy()).ravel()
        y_pred = m.predict(X_test)
        if model_class not in non_probabilistic_models and classification:
            y_prob = m.predict_proba(X_test)[:, 1]

        for metric in metric_list:
            if (
                metric in ["PR AUC", "ROC AUC"]
                and model_class not in non_probabilistic_models
            ):
                metrics_dict[metric].append(KNOWN_METRICS[metric](y_test, y_prob))
            elif metric == "Cohen Kappa":
                metrics_dict[metric].append(
                    KNOWN_METRICS[metric](y_test, y_pred, weights="linear")
                )
            else:
                metrics_dict[metric].append(KNOWN_METRICS[metric](y_test, y_pred))

    return metrics_dict


def get_optimised_cv_performance(
    model_class: BaseEstimator,
    df: pd.DataFrame,
    n_folds: int,
    fold_col: str,
    metric_list: list[str],
    main_metric: str,
    parameters: dict[str, list],
    n_jobs: int,
    scaler: str | None = None,
) -> dict[str, list[float]]:
    """
    Get the cross-validated performance of a model with optimised hyperparameters. The
    hyperparameters are optimised using grid search with nested cross-validation.

    Parameters
    ----------
    model_class : BaseEstimator
        A scikit-learn model.
    df : pd.DataFrame
        A dataframe containing the features and target values. Must contain a column
        with the fold indices. The features must be in a column called 'Features' and
        the target values in a column called 'Target'.
    n_folds : int
        The number of folds in the cross-validation.
    fold_col : str
        The name of the column containing the fold indices.
    metric_list : list[str]
        A list of metrics to use for evaluation.
    main_metric : str
        The main metric to optimise hyperparameters for.
    parameters : dict[str, list]
        A dictionary of hyperparameters to search over.
    n_jobs : int
        The number of jobs to run in parallel during grid search.
    scaler : str or None, default=None
        The type of scaler to use. Valid choices are 'MinMax' and 'Standard'.

    Returns
    -------
    dict[str, list[float]]
        A dictionary mapping metrics to lists of scores.
    """
    for metric in metric_list:
        assert (
            metric in KNOWN_METRICS.keys()
        ), f"Unknown metric. Known metrics are: {', '.join(KNOWN_METRICS.keys())}"
    metrics_dict = {metric: [] for metric in metric_list}

    assert (
        main_metric in KNOWN_METRICS.keys()
    ), f"Unknown main metric. Known metrics are: {', '.join(KNOWN_METRICS.keys())}"
    scoring = SCORING[main_metric]

    classification = True if metric_list[0] in CLASSIFICATION_METRICS else False

    all_folds = [df[df[fold_col] == i] for i in range(n_folds)]

    if scaler:
        if scaler == "MinMax":
            s = MinMaxScaler()
        elif scaler == "Standard":
            s = StandardScaler()
        else:
            raise ValueError("Unknown scaler. Must be either MinMax or Standard")
        model = make_pipeline(s, model_class)
        model_step_name = list(model.named_steps.keys())[-1]
        parameters = {
            f"{model_step_name}__{key}": value for key, value in parameters.items()
        }
    else:
        model = model_class

    # outer CV loop
    for test_fold in range(n_folds):
        # get data for inner CV loop
        cv_folds = [
            f
            for i, f in enumerate(all_folds)
            if i
            not in [
                test_fold,
            ]
        ]
        cv_data = pd.concat([df for df in cv_folds]).reset_index()

        # for each cv iteration, get the indices of train and val data points
        train_val_idx = [
            f.index.to_list()
            for f in [
                cv_data[cv_data[fold_col] == i]
                for i in range(n_folds)
                if i != test_fold
            ]
        ]
        cv = []
        for val_fold in range(n_folds - 1):
            train_idx = [
                f
                for i, f in enumerate(train_val_idx)
                if i
                not in [
                    val_fold,
                ]
            ]
            train_idx = [idx for idxs in train_idx for idx in idxs]
            val_idx = train_val_idx[val_fold]
            curr_idx = train_idx, val_idx
            cv.append(curr_idx)

        # perform hyperparameter search
        clf = GridSearchCV(
            estimator=model,
            param_grid=parameters,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=True,
            cv=cv,
            pre_dispatch="n_jobs",
        )
        X = np.vstack(cv_data["Features"].to_numpy())
        y = np.vstack(cv_data["Target"].to_numpy()).ravel()
        clf.fit(X, y)

        # evaluate model
        test_data = all_folds[test_fold]
        X_test = np.vstack(test_data["Features"].to_numpy())
        y_test = np.vstack(test_data["Target"].to_numpy()).ravel()
        y_pred = clf.predict(test_data)
        if model_class not in non_probabilistic_models and classification:
            y_prob = clf.predict_proba(X_test)[:, 1]

        for metric in metric_list:
            if (
                metric in ["PR AUC", "ROC AUC"]
                and model_class not in non_probabilistic_models
            ):
                metrics_dict[metric].append(KNOWN_METRICS[metric](y_test, y_prob))
            elif metric == "Cohen Kappa":
                metrics_dict[metric].append(
                    KNOWN_METRICS[metric](y_test, y_pred, weights="linear")
                )
            else:
                metrics_dict[metric].append(KNOWN_METRICS[metric](y_test, y_pred))

    return metrics_dict


def get_best_hparams(
    model_class: BaseEstimator,
    df: pd.DataFrame,
    n_folds: int,
    fold_col: str,
    main_metric: str,
    sec_metrics: list[str],
    parameters: dict[str, list],
    n_jobs: int,
    scaler: str | None = None,
) -> GridSearchCV:
    """
    Get the best hyperparameters for a model using grid search with (non-nested) cross-validation.

    Parameters
    ----------
    model_class : BaseEstimator
        A scikit-learn model.
    df : pd.DataFrame
        A dataframe containing the features and target values. Must contain a column
        with the fold indices. The features must be in a column called 'Features' and
        the target values in a column called 'Target'.
    n_folds : int
        The number of folds in the cross-validation.
    fold_col : str
        The name of the column containing the fold indices.
    main_metric : str
        The metric to optimise hyperparameters for.
    sec_metrics : list[str]
        A list of secondary metrics to track during hyperparameter search.
    parameters : dict[str, list]
        A dictionary of hyperparameters to search over.
    n_jobs : int
        The number of jobs to run in parallel during grid search.
    scaler : str or None, default=None
        The type of scaler to use. Valid choices are 'MinMax' and 'Standard'.

    Returns
    -------
    GridSearchCV
        A GridSearchCV object containing the best hyperparameters.
    """
    assert (
        main_metric in KNOWN_METRICS.keys()
    ), f"Unknown metric. Known metrics are: {', '.join(KNOWN_METRICS.keys())}"
    for metric in sec_metrics:
        assert (
            metric in KNOWN_METRICS.keys()
        ), f"Unknown metric. Known metrics are: {', '.join(KNOWN_METRICS.keys())}"
    scoring = {metric: SCORING[metric] for metric in [main_metric] + sec_metrics}

    all_folds = [df[df[fold_col] == i] for i in range(n_folds)]

    if scaler:
        if scaler == "MinMax":
            s = MinMaxScaler()
        elif scaler == "Standard":
            s = StandardScaler()
        else:
            raise ValueError("Unknown scaler. Must be either MinMax or Standard")
        model = make_pipeline(s, model_class)
        model_step_name = list(model.named_steps.keys())[-1]
        parameters = {
            f"{model_step_name}__{key}": value for key, value in parameters.items()
        }
    else:
        model = model_class

    # for each cv iteration, get the indices of train and val data points
    train_val_idx = [f.index.to_list() for f in all_folds]
    cv = []
    for val_fold in range(n_folds):
        train_idx = [
            f
            for i, f in enumerate(train_val_idx)
            if i
            not in [
                val_fold,
            ]
        ]
        train_idx = [idx for idxs in train_idx for idx in idxs]
        val_idx = train_val_idx[val_fold]
        curr_idx = train_idx, val_idx
        cv.append(curr_idx)

    # perform hyperparameter search
    clf = GridSearchCV(
        estimator=model,
        param_grid=parameters,
        scoring=scoring,
        n_jobs=n_jobs,
        refit=main_metric,
        cv=cv,
        pre_dispatch="n_jobs",
    )
    X = np.vstack(df["Features"].to_numpy())
    y = np.vstack(df["Target"].to_numpy()).ravel()
    clf.fit(X, y)

    return clf
