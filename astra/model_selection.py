"""
This module contains functions for model selection and evaluation.
"""

import logging
import os
import pickle
import warnings

import numpy as np
import optuna
import pandas as pd
import pingouin as pg
import scikit_posthocs as sp
<<<<<<< HEAD
=======
from optuna.distributions import BaseDistribution
from optuna.exceptions import ExperimentalWarning
from optuna.integration import OptunaSearchCV
>>>>>>> 7ef386c88a406a87d0553e2d1a5c5f4139271492
from scipy.stats import levene, shapiro, ttest_rel, wilcoxon
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import GridSearchCV
from statsmodels.stats.libqsturng import psturng

from .metrics import (
    CLASSIFICATION_METRICS,
    HIGHER_BETTER,
    KNOWN_METRICS,
    SCORING,
)
from .models.classification import NON_PROBABILISTIC_MODELS
from .utils import build_model, print_performance

warnings.filterwarnings("ignore", category=ExperimentalWarning)


def check_assumptions(
    results_dict: dict[str, dict[str, list[float]]], verbose: bool = True
) -> bool:
    """
    Check homogeneity of variances and normality assumed by parametric statistical tests.

    Parameters
    ----------
    results_dict : dict[str, dict[str, list[float]]]
        A dictionary mapping model names to dictionaries of metric names and scores.
    verbose : bool, default=True
        Whether to print warnings if assumptions are violated.

    Returns
    -------
    bool
        True if all assumptions are met, False otherwise.
    """
    # Get metrics to check
    metrics = []
    for model in results_dict:
        metrics.append(list(results_dict[model].keys()))
    # assert that all models have the same metrics
    if not all(metrics[0] == m for m in metrics):
        raise ValueError("All models must have the same metrics.")
    metrics = metrics[0]

    # Run Levene's test for homogeneity of variances
    pvals_levene = []
    for metric in metrics:
        groups = [results_dict[model][metric] for model in results_dict]
        _, pvalue = levene(*groups)
        pvals_levene.append(pvalue)
        # Check if p-value is above 0.05
        if pvalue < 0.05 and verbose:
            print(
                "Warning: Homogeneity of variances assumption violated for metric "
                f"{metric}. Consider using non-parametric tests."
            )
    # Check if any p-values are above 0.05
    homogeneity_of_variances = all(pval > 0.05 for pval in pvals_levene)

    # If homogeneity of variances is met, we can assume that fold variances are also met
    if not homogeneity_of_variances:
        # Check max fold difference of variances
        max_fold_differences = []
        for metric in metrics:
            variances_by_method = pd.Series(
                [np.var(results_dict[model][metric]) for model in results_dict]
            )
            max_fold_diff = (
                variances_by_method.max() / variances_by_method.min()
                if variances_by_method.min() > 0
                else np.inf
            )
            if max_fold_diff > 9 and verbose:
                print(
                    "Warning: Variances of folds differ too much for metric "
                    f"{metric}. Consider using non-parametric tests."
                )
            max_fold_differences.append(max_fold_diff)
        # Check if any max fold differences are above 9
        fold_variances = all(
            max_fold_diff <= 9 for max_fold_diff in max_fold_differences
        )
        if fold_variances:
            homogeneity_of_variances = True
            if verbose:
                print("Info: All fold variances are within acceptable limits (< 9).")
    else:
        fold_variances = True

    # Run Shapiro-Wilk test for normality
    pvals_shapiro = []
    for metric in metrics:
        for model in results_dict:
            scores = results_dict[model][metric]
            _, pvalue = shapiro(scores)
            if pvalue < 0.05 and verbose:
                print(
                    "Warning: Normality assumption violated for model "
                    f"{model} and metric {metric}. Consider using non-parametric tests."
                )
            pvals_shapiro.append(pvalue)
    # Check if any p-values are above 0.05
    normality = all(pval > 0.05 for pval in pvals_shapiro)

    # If any of the assumptions are violated, return False
    if not (homogeneity_of_variances and fold_variances and normality):
        return False
    return True


def tukey_hsd(
    mse: float, residual_dof: int, score_means: pd.Series, n_folds: int
) -> pd.DataFrame:
    """
    Performs Tukey's HSD test using repeated measures ANOVA output.

    Parameters
    ----------
    mse : float
        Mean squared error from ANOVA.
    residual_dof : int
        Residual degrees of freedom.
    score_means : pd.Series
        Mean scores.
    n_folds: int
        Total number of folds per model.

    Returns
    -------
    pd.DataFrame
        p-values for pairwise comparisons between models.
    """
    # Get models to compare
    models = list(score_means.index)
    # Get number of models to compare
    n_models = len(models)
    # Calculate Tukey standard error
    tukey_se = np.sqrt(2 * mse / n_folds)

    p_values = np.ones((n_models, n_models))
    for i, model1 in enumerate(models):
        for j, model2 in enumerate(models):
            # Calculate the difference between the mean scores
            mean_diff = score_means.loc[model1] - score_means.loc[model2]
            # Calculate the studentised range
            studentised_range = np.abs(mean_diff) / tukey_se
            # Calculate the adjusted p-value
            adjusted_p = psturng(studentised_range * np.sqrt(2), n_models, residual_dof)
            # psturng sometimes returns an array containing a single float for unknown reasons
            if isinstance(adjusted_p, np.ndarray):
                adjusted_p = adjusted_p[0]
            # Store results
            p_values[i, j] = adjusted_p
    np.fill_diagonal(p_values, 1.0)

    return pd.DataFrame(p_values, columns=models, index=models)


def find_n_best_models(
    results_dic: dict[str, dict[str, list[float]]],
    metric: str,
    parametric: bool = False,
    bf_corr: bool = True,
) -> list[str]:
    """
    Find the n best models that don't perform significantly differently with respect to
    a given metric as determined using repeated measures ANOVA (if parametric=True) or
    the Friedman test (if parametric=False). The function iteratively removes the model
    with the worst median score until no statistically significant difference is found or
    only one model remains.

    Parameters
    ----------
    results_dic : dict[str, dict[str, list[float]]]
        A dictionary mapping model names to dictionaries of metric names and scores.
    metric : str
        The metric to use for model comparison.
    parametric : bool, default=False
        Whether to use parametric tests instead of non-parametric tests.
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
    for n_models in range(len(stat_for_test.columns), 0, -1):
        if n_models == 1:  # only one model left, no need to test
            best_models = list(stat_for_test.columns)
            break
        if parametric:
            # Perform repeated measures ANOVA
            pvalue = pg.rm_anova(stat_for_test)["p-unc"].values[0]
        else:
            # Perform Friedman test
            pvalue = pg.friedman(stat_for_test)["p-unc"].values[0]

        # Bonferroni correction of significance level
        if bf_corr:
            threshold = 0.05 / n_models
        else:
            threshold = 0.05

        # Check if there is a statistically significant difference
        if pvalue < threshold:  # significant difference
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
    results_dic: dict[str, dict[str, list[float]]],
    metric: str,
    parametric: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform Tukey's HSD and paired t-test (if parametric=True) or Conover post-hoc and
    Wilcoxon signed-rank tests (if parametric=False) on the performance of models.

    Parameters
    ----------
    results_dic : dict[str, dict[str, list[float]]]
        A dictionary mapping model names to dictionaries of metric names and scores.
    metric : str
        The metric to use for model comparison.
    parametric : bool, default=False
        Whether to use parametric tests instead of non-parametric tests.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        A tuple containing the test results for the two statistical tests.
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

    if parametric:
        # perform repeated measures ANOVA
        anova = pg.rm_anova(data=stat_for_test, detailed=True)
        # extract mean squared error and residual degrees of freedom
        mse = float(anova.loc[1, "MS"])
        residual_dof = int(anova.loc[1, "DF"])
        # calculate mean scores per model
        means = stat_for_test.mean(axis=0)
        n_folds = stat_for_test.shape[0]
        # perform Tukey's HSD test
        post_hoc_stats = tukey_hsd(mse, residual_dof, means, n_folds)
    else:
        # perform Conover post-hoc test with Holm-Bonferroni adjustment
        post_hoc_stats = sp.posthoc_conover_friedman(stat_for_test, p_adjust="holm")

    if parametric:
        # Perform paired t-test
        test = ttest_rel
    else:
        # Perform Wilcoxon signed-rank test
        test = wilcoxon
    naive_p_values = np.empty((len(stat_for_test.columns), len(stat_for_test.columns)))
    for n, col1 in enumerate(stat_for_test):
        for m, col2 in enumerate(stat_for_test):
            # Skip self-comparisons
            if col1 == col2:
                naive_p_values[n, m] = 1.0
                continue
            naive_p_values[n, m] = test(
                stat_for_test[col1],
                stat_for_test[col2],
            ).pvalue
    naive_stats = pd.DataFrame(
        naive_p_values, columns=stat_for_test.columns, index=stat_for_test.columns
    )

    return post_hoc_stats, naive_stats


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
    parametric: bool = False,
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
    parametric : bool, default=False
        Whether to use parametric tests instead of non-parametric tests.
    bf_corr : bool, default=True
        Whether to apply Bonferroni correction to the significance level.

    Returns
    -------
    tuple[str, str]
        A tuple containing the name of the best model and the reason for its selection.
    """
    # Perform tests to find the n best models
    n_best_models = find_n_best_models(results_dict, main_metric, parametric, bf_corr)
    results_dict_best = {model: results_dict[model] for model in n_best_models}

    if len(results_dict_best) == 1:
        # If only one model is left, return it as the best model
        return list(results_dict_best.keys())[0], (
            "Repeated measure ANOVA" if parametric else "Friedman test"
        )

    # Perform statistical tests on the best models
    post_hoc_stats, naive_stats = perform_statistical_tests(
        results_dict_best, main_metric, parametric
    )

    # Check if Tukey's HSD test/Conover post-hoc test yield a best model
    best_model = check_best_model(results_dict_best, post_hoc_stats, main_metric)
    reason = "Tukey's HSD test" if parametric else "Conover post-hoc test"

    # Fall back to naive test
    if not best_model:
        best_model = check_best_model(results_dict_best, naive_stats, main_metric)
        reason = "paired t-test" if parametric else "Wilcoxon signed-rank test"

    # Fall back to secondary metrics if the main one doesn't yield a best model
    if not best_model:
        for metric in secondary_metrics:
            post_hoc_stats, naive_stats = perform_statistical_tests(
                results_dict_best, metric, parametric
            )
            best_model = check_best_model(results_dict_best, post_hoc_stats, metric)
            test_used = "Tukey's HSD test" if parametric else "Conover post-hoc test"
            reason = f"{test_used} using {metric}"
            if best_model:
                break
            best_model = check_best_model(results_dict_best, naive_stats, metric)
            test_used = "paired t-test" if parametric else "Wilcoxon signed-rank test"
            reason = f"{test_used} using {metric}"
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
    features_col: str,
    target_col: str,
    fold_col: str,
    metric_list: list[str],
    impute: str | float | int | None = None,
    remove_constant: float | None = None,
    remove_correlated: float | None = None,
    scaler: str | None = None,
    custom_params: dict[str, list] | None = None,
) -> dict[str, list[float]]:
    """
    Get the cross-validated performance of a model.

    Parameters
    ----------
    model_class : BaseEstimator
        A scikit-learn model.
    df : pd.DataFrame
        A dataframe containing the features and target values.
    features_col : str
        The name of the column containing the features.
    target_col : str
        The name of the column containing the target values.
    fold_col : str
        The name of the column containing the fold indices.
    metric_list : list[str]
        A list of metrics to use for evaluation.
    impute : str or float or int or None, default=None
        The imputation strategy to use for missing values. If None, no imputation is performed.
        Valid choices are 'mean', 'median', 'knn', or a float or int value for constant imputation.
    remove_constant : float or None, default=None
        If specified, features with variance below this threshold will be removed.
        If None, no features are removed.
    remove_correlated : float or None, default=None
        If specified, features with correlation above this threshold will be removed.
        If None, no features are removed.
    scaler : str or None, default=None
        The type of scaler to use. Valid choices are 'MinMax' and 'Standard'.
    custom_params : dict[str, list] or None, default=None
        A dictionary of custom parameters for the model. If None, default parameters are used.

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

    n_folds = df[fold_col].nunique()
    all_folds = [df[df[fold_col] == i] for i in range(n_folds)]

    if custom_params:
        model_class.set_params(**custom_params)

    model = build_model(model_class, impute, remove_constant, remove_correlated, scaler)

    for test_fold in range(n_folds):
        # train model
        train_folds = [f for i, f in enumerate(all_folds) if i not in [test_fold]]
        train_data = pd.concat([f for f in train_folds])
        X = np.vstack(train_data[features_col].to_numpy())
        y = np.vstack(train_data[target_col].to_numpy()).ravel()
        m = clone(model)
        m.fit(X, y)

        # evaluate model
        test_data = all_folds[test_fold]
        X_test = np.vstack(test_data[features_col].to_numpy())
        y_test = np.vstack(test_data[target_col].to_numpy()).ravel()
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore", UserWarning
            )  # Suppress UserWarnings for LightGBM
            y_pred = m.predict(X_test)
            if (
                model_class.__class__.__name__ not in NON_PROBABILISTIC_MODELS
                and classification
            ):
                y_prob = m.predict_proba(X_test)[:, 1]

        for metric in metric_list:
            if (
                metric in ["pr_auc", "roc_auc"]
                and model_class.__class__.__name__ not in NON_PROBABILISTIC_MODELS
            ):
                metrics_dict[metric].append(KNOWN_METRICS[metric](y_test, y_prob))
            elif metric == "cohen_kappa":
                metrics_dict[metric].append(
                    KNOWN_METRICS[metric](y_test, y_pred, weights="linear")
                )
            else:
                metrics_dict[metric].append(KNOWN_METRICS[metric](y_test, y_pred))

    return metrics_dict


def run_CV(
    name: str,
    data_df: pd.DataFrame,
    features: str,
    target: str,
    fold_col: str,
    models: dict[str, BaseEstimator],
    metric_list: list[str],
    impute: str | float | int | None = None,
    remove_constant: float | None = None,
    remove_correlated: float | None = None,
    scaler: str | None = None,
    custom_params: dict[str, dict[str, list]] | None = None,
    repeated: bool = False,
):
    """
    Run cross-validation for multiple models and save the results.

    Parameters
    ----------
    name : str
        Name for the results directory and the experiment.
    data_df : pd.DataFrame
        DataFrame containing the data for cross-validation.
    features : str
        Name of the column containing features.
    target : str
        Name of the column containing the target variable.
    fold_col : str
        Name of the column containing fold assignments.
    models : dict[str, BaseEstimator]
        Dictionary of models to evaluate, with model names as keys and scikit-learn-like estimators as values.
    metric_list : list of str
        List of metrics to evaluate during cross-validation.
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
    custom_params : dict[str, dict[str, list]] or None, default None
        Dictionary of custom hyperparameter grids for each model. If None, default grids will be used.
    repeated : bool, default False
        Whether the cross-validation is repeated. If True, results are saved with the fold column name.

    Returns
    -------
    dict
        Dictionary containing cross-validation results for each model.
    """
    results_name = f"default_CV_{fold_col}" if repeated else "default_CV"
    ckpt_name = f"{name}_CV_{fold_col}_ckpt" if repeated else f"{name}_CV_ckpt"

    if os.path.exists(f"results/{name}/{results_name}.pkl"):
        logging.info("Loading existing results.")
        with open(f"results/{name}/{results_name}.pkl", "br") as f:
            results = pickle.load(f)

    else:
        try:
            with open(f"cache/{ckpt_name}.pkl", "br") as f:
                results = pickle.load(f)
            logging.info("Loaded checkpoint.")
        except FileNotFoundError:
            results = {}

        logging.info("Running CV.")
        for model in models.keys():
            logging.info(f"Running {model}.")

            if model in results:
                logging.info("Found existing results. Skipping.")
                continue

            results[model] = get_cv_performance(
                model_class=models[model],
                df=data_df,
                features_col=features,
                target_col=target,
                fold_col=fold_col,
                metric_list=metric_list,
                impute=impute,
                remove_constant=remove_constant,
                remove_correlated=remove_correlated,
                scaler=scaler,
                custom_params=custom_params.get(model, None) if custom_params else None,
            )
            file_handler = next(
                (
                    h
                    for h in logging.getLogger().handlers
                    if isinstance(h, logging.FileHandler)
                ),
                None,
            )
            print_performance(
                model_name=model,
                results_dict=results[model],
                file=file_handler.stream.name if file_handler else None,
            )
            with open(f"cache/{ckpt_name}.pkl", "wb") as f:
                pickle.dump(results, f)

        with open(f"results/{name}/{results_name}.pkl", "wb") as f:
            pickle.dump(results, f)
        os.remove(f"cache/{ckpt_name}.pkl")
        logging.info("Done!")

    return results


def get_optimised_cv_performance(
    model_class: BaseEstimator,
    df: pd.DataFrame,
    features_col: str,
    target_col: str,
    fold_col: str,
    metric_list: list[str],
    main_metric: str,
    parameters: dict[str, list] | dict[str, BaseDistribution],
    n_jobs: int,
    use_optuna: bool = False,
    n_trials: int = 100,
    timeout: int = 3600,
    impute: str | float | int | None = None,
    remove_constant: float | None = None,
    remove_correlated: float | None = None,
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
        A dataframe containing the features and target values.
    features_col : str
        The name of the column containing the features.
    target_col : str
        The name of the column containing the target values.
    fold_col : str
        The name of the column containing the fold indices.
    metric_list : list[str]
        A list of metrics to use for evaluation.
    main_metric : str
        The main metric to optimise hyperparameters for.
    parameters : dict[str, list] or dict[str, BaseDistribution]
        A dictionary of hyperparameters to search over.
    n_jobs : int
        The number of jobs to run in parallel during grid search.
    use_optuna : bool, default=False
        Whether to use Optuna for hyperparameter optimisation instead of grid search.
    n_trials : int, default=100
        The number of trials to run during Optuna hyperparameter optimisation.
    timeout : int, default=3600
        The maximum time in seconds to run Optuna hyperparameter optimisation.
    impute : str or float or int or None, default=None
        The imputation strategy to use for missing values. If None, no imputation is performed.
        Valid choices are 'mean', 'median', 'knn', or a float or int value for constant imputation.
    remove_constant : float or None, default=None
        If specified, features with variance below this threshold will be removed.
        If None, no features are removed.
    remove_correlated : float or None, default=None
        If specified, features with correlation above this threshold will be removed.
        If None, no features are removed.
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

    n_folds = df[fold_col].nunique()
    all_folds = [df[df[fold_col] == i] for i in range(n_folds)]

    model = build_model(model_class, impute, remove_constant, remove_correlated, scaler)
    if model.__class__.__name__ == "Pipeline":
        model_step_name = list(model.named_steps.keys())[-1]
        parameters = {
            f"{model_step_name}__{key}": value for key, value in parameters.items()
        }

    # outer CV loop
    for test_fold in range(n_folds):
        # get data for inner CV loop
        cv_folds = [f for i, f in enumerate(all_folds) if i not in [test_fold]]
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
            train_idx = [f for i, f in enumerate(train_val_idx) if i not in [val_fold]]
            train_idx = [idx for idxs in train_idx for idx in idxs]
            val_idx = train_val_idx[val_fold]
            curr_idx = train_idx, val_idx
            cv.append(curr_idx)

        # perform hyperparameter search
        if use_optuna:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            clf = OptunaSearchCV(
                estimator=model,
                param_distributions=parameters,
                cv=cv,
                n_jobs=n_jobs,
                scoring=scoring[main_metric],
                n_trials=n_trials,
                timeout=timeout,
                refit=True,
                verbose=0,
                random_state=42,
            )
        else:
            clf = GridSearchCV(
                estimator=model,
                param_grid=parameters,
                scoring=scoring,
                n_jobs=n_jobs,
                refit=True,
                cv=cv,
                pre_dispatch="n_jobs",
            )
        X = np.vstack(cv_data[features_col].to_numpy())
        y = np.vstack(cv_data[target_col].to_numpy()).ravel()
        # Suppress warnings for LGBM, needs to be done using os.environ, as GridSearchCV uses
        # parallel processing, which means that warnings.filterwarnings does not work
        if model_class.__class__.__name__[:4] == "LGBM":
            os.environ["PYTHONWARNINGS"] = "ignore"
        clf.fit(X, y)

        # evaluate model
        test_data = all_folds[test_fold]
        X_test = np.vstack(test_data[features_col].to_numpy())
        y_test = np.vstack(test_data[target_col].to_numpy()).ravel()
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore", UserWarning
            )  # Suppress UserWarnings for LightGBM
            y_pred = clf.predict(X_test)
            if (
                model_class.__class__.__name__ not in NON_PROBABILISTIC_MODELS
                and classification
            ):
                y_prob = clf.predict_proba(X_test)[:, 1]

        for metric in metric_list:
            if (
                metric in ["pr_auc", "roc_auc"]
                and model_class.__class__.__name__ not in NON_PROBABILISTIC_MODELS
            ):
                metrics_dict[metric].append(KNOWN_METRICS[metric](y_test, y_prob))
            elif metric == "cohen_kappa":
                metrics_dict[metric].append(
                    KNOWN_METRICS[metric](y_test, y_pred, weights="linear")
                )
            else:
                metrics_dict[metric].append(KNOWN_METRICS[metric](y_test, y_pred))

    return metrics_dict


def get_best_hparams(
    model_class: BaseEstimator,
    df: pd.DataFrame,
    features_col: str,
    target_col: str,
    fold_col: str,
    main_metric: str,
    sec_metrics: list[str],
    parameters: dict[str, list] | dict[str, BaseDistribution],
    n_jobs: int,
    use_optuna: bool = False,
    n_trials: int = 100,
    timeout: int = 3600,
    impute: str | float | int | None = None,
    remove_constant: float | None = None,
    remove_correlated: float | None = None,
    scaler: str | None = None,
) -> GridSearchCV | OptunaSearchCV:
    """
    Get the best hyperparameters for a model using grid search with (non-nested) cross-validation.

    Parameters
    ----------
    model_class : BaseEstimator
        A scikit-learn model.
    df : pd.DataFrame
        A dataframe containing the features and target values.
    features_col : str
        The name of the column containing the features.
    target_col : str
        The name of the column containing the target values.
    fold_col : str
        The name of the column containing the fold indices.
    main_metric : str
        The metric to optimise hyperparameters for.
    sec_metrics : list[str]
        A list of secondary metrics to track during hyperparameter search.
    parameters : dict[str, list] or dict[str, BaseDistribution]
        A dictionary of hyperparameters to search over.
    n_jobs : int
        The number of jobs to run in parallel during grid search.
    use_optuna : bool, default=False
        Whether to use Optuna for hyperparameter optimisation instead of grid search.
    n_trials : int, default=100
        The number of trials to run during Optuna hyperparameter optimisation.
    timeout : int, default=3600
        The maximum time in seconds to run Optuna hyperparameter optimisation.
    impute : str or float or int or None, default=None
        The imputation strategy to use for missing values. If None, no imputation is performed.
        Valid choices are 'mean', 'median', 'knn', or a float or int value for constant imputation.
    remove_constant : float or None, default=None
        If specified, features with variance below this threshold will be removed.
        If None, no features are removed.
    remove_correlated : float or None, default=None
        If specified, features with correlation above this threshold will be removed.
        If None, no features are removed.
    scaler : str or None, default=None
        The type of scaler to use. Valid choices are 'MinMax' and 'Standard'.

    Returns
    -------
    GridSearchCV or OptunaSearchCV
        A GridSearchCV or OptunaSearchCV object containing the best hyperparameters.
    """
    assert (
        main_metric in KNOWN_METRICS.keys()
    ), f"Unknown metric. Known metrics are: {', '.join(KNOWN_METRICS.keys())}"
    for metric in sec_metrics:
        assert (
            metric in KNOWN_METRICS.keys()
        ), f"Unknown metric. Known metrics are: {', '.join(KNOWN_METRICS.keys())}"
    scoring = {metric: SCORING[metric] for metric in [main_metric] + sec_metrics}

    df = df.copy().reset_index()

    n_folds = df[fold_col].nunique()
    all_folds = [df[df[fold_col] == i] for i in range(n_folds)]

    model = build_model(model_class, impute, remove_constant, remove_correlated, scaler)
    if model.__class__.__name__ == "Pipeline":
        model_step_name = list(model.named_steps.keys())[-1]
        parameters = {
            f"{model_step_name}__{key}": value for key, value in parameters.items()
        }

    # for each cv iteration, get the indices of train and val data points
    train_val_idx = [f.index.to_list() for f in all_folds]
    cv = []
    for val_fold in range(n_folds):
        train_idx = [f for i, f in enumerate(train_val_idx) if i not in [val_fold]]
        train_idx = [idx for idxs in train_idx for idx in idxs]
        val_idx = train_val_idx[val_fold]
        curr_idx = train_idx, val_idx
        cv.append(curr_idx)

    # perform hyperparameter search
    # TODO: Integrate Optuna
    if use_optuna:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        clf = OptunaSearchCV(
            estimator=model,
            param_distributions=parameters,
            cv=cv,
            n_jobs=n_jobs,
            scoring=scoring[main_metric],
            n_trials=n_trials,
            timeout=timeout,
            refit=True,
            verbose=0,
            random_state=42,
        )
    else:
        clf = GridSearchCV(
            estimator=model,
            param_grid=parameters,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=main_metric,
            cv=cv,
            pre_dispatch="n_jobs",
        )
    X = np.vstack(df[features_col].to_numpy())
    y = np.vstack(df[target_col].to_numpy()).ravel()
    # Suppress warnings for LGBM, needs to be done using os.environ, as GridSearchCV uses
    # parallel processing, which means that warnings.filterwarnings does not work
    if model_class.__class__.__name__[:4] == "LGBM":
        os.environ["PYTHONWARNINGS"] = "ignore"
    clf.fit(X, y)

    return clf
