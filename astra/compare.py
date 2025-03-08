import os
import pandas as pd
import logging
from .model_selection import (
    find_n_best_models,
    perform_statistical_tests,
    check_best_model,
    get_best_model,
    LOWER_BETTER,
)
from .utils import get_scores


def run(CV_results: list[str], main_metric: str, sec_metrics: list[str], n_folds: int) -> None:
    """
    Compare the results of different models using the CV results.

    Parameters
    ----------
    CV_results : list of str
        List of paths to directories containing the CV results.
    main_metric : str
        The main metric to use for comparison.
    sec_metrics : list of str
        Secondary metrics to use for comparison.
    n_folds : int
        Number of CV folds.

    Returns
    -------
    None
    """
    logging.basicConfig(
        level=logging.INFO,
        datefmt="%d-%m %H:%M",
        format="%(asctime)s - %(levelname)s: %(message)s",
    )

    all_results = {}

    CV_results = [
        CV_results_path + "/" if CV_results_path[-1] != "/" else CV_results_path
        for CV_results_path in CV_results
    ]

    for CV_results_path in CV_results:
        if not os.path.exists(CV_results_path):
            raise ValueError(f"{CV_results_path} does not exist")
        if not os.path.isdir(CV_results_path):
            raise ValueError(f"{CV_results_path} is not a directory")
        if not os.listdir(CV_results_path):
            raise ValueError(f"{CV_results_path} is empty")

    all_in_one_dir = True if len(CV_results) == 1 else False

    for CV_results_path in CV_results:

        for file in os.listdir(CV_results_path):
    
            if file.endswith("_CV_results.csv"):
                cv_results_df = pd.read_csv(CV_results_path + file)
    
                assert (
                    f"rank_test_{main_metric}" in cv_results_df.columns
                ), f"{file} does not contain results for {main_metric}"
                for metric in sec_metrics:
                    assert (
                        f"rank_test_{metric}" in cv_results_df.columns
                    ), f"{file} does not contain results for {metric}"
    
                if all_in_one_dir:
                    # remove the "_CV_results.csv" part of the filename
                    model_name = file.split("_CV_results.csv")[0]
                else:
                    # use the parent directory as the model name
                    model_name = CV_results_path.split("/")[-2]

                all_results[model_name] = {}
    
                for metric in [main_metric] + sec_metrics:
                    test_score_columns = [
                        col
                        for col in cv_results_df.columns
                        if ("split" in col) and (f"_test_{metric}" in col)
                    ]
                    scores = [
                        cv_results_df[cv_results_df[f"rank_test_{main_metric}"] == 1].iloc[
                            0
                        ][col]
                        for col in test_score_columns
                    ]
                    # for metrics where lower is better, we need to negate the scores
                    # as they will have been negated in the CV
                    if metric in LOWER_BETTER:
                        scores = [-score for score in scores]
                    all_results[model_name][metric] = scores
    
    if len(all_results) == 0:
        raise ValueError("No CV results found")

    elif len(all_results) == 1:
        raise ValueError("Only one CV result found, cannot compare")

    elif len(all_results) == 2:
        logging.info(
            "Two CV results found, comparing them using Wilcoxon rank-sum test"
        )
        _, rank_sum = perform_statistical_tests(all_results, main_metric)
        better_model = check_best_model(all_results, rank_sum, main_metric)
        if better_model:
            logging.info(
                f"{better_model} is significantly better according to {main_metric}"
            )
        else:
            for metric in sec_metrics:
                _, rank_sum = perform_statistical_tests(all_results, metric)
                better_model = check_best_model(all_results, rank_sum, metric)
                if better_model:
                    logging.info(
                        f"{better_model} is significantly better according to {metric}"
                    )
                    break
        if not better_model:
            logging.info("No significant difference found between the models")

    else:
        logging.info(
            f"{len(all_results)} CV results found, comparing them using Friedman test"
        )
        n_best_models = find_n_best_models(
            results_dic=all_results, metric=main_metric, bf_corr=True
        )
        logging.info(f"Best models based on {main_metric}:")
        for model in n_best_models:
            print(" " * 20 + f"{model}")

        best_results = {model: all_results[model] for model in n_best_models}
        best_model, reason = get_best_model(
            results_dict=best_results,
            main_metric=main_metric,
            secondary_metrics=sec_metrics,
            bf_corr=True,
        )
        logging.info(f"Best model overall: {best_model}. Reason: {reason}.")
        if all_in_one_dir:
            cv_results_df = pd.read_csv(CV_results[0] + best_model + "_CV_results.csv")
        else:
            for CV_results_path in CV_results:
                if best_model == CV_results_path.split("/")[-2]:
                    for file in os.listdir(CV_results_path):
                        if file.endswith("_CV_results.csv"):
                            cv_results_df = pd.read_csv(CV_results_path + file)
                            break
                    break
        mean_score_main, std_score_main, sec_metrics_scores = get_scores(
            cv_results_df, main_metric, sec_metrics, n_folds
        )
        print("-" * 50)
        print("Results:")
        print("-" * 50)
        print(f"Mean {main_metric}:", f"{mean_score_main:.3f} ± {std_score_main:.3f}.")
        for metric in sec_metrics:
            print(
                f"Mean {metric}:",
                f"{sec_metrics_scores[metric][0]:.3f} ± {sec_metrics_scores[metric][1]:.3f}.",
            )
        print("-" * 50)
