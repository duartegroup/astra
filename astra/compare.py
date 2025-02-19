import os
import pandas as pd
import argparse
from argparse import RawTextHelpFormatter
import logging
from model_selection import (
    find_n_best_models,
    perform_statistical_tests,
    check_best_model,
    get_best_model,
)


def get_argparser() -> argparse.ArgumentParser:
    """
    Get the argument parser. Arguments are as follows:
    - CV_results_path: Path to the directory containing the CV results
    - main_metric: The main metric to use for comparison
    - sec_metrics: Secondary metrics to use for comparison
    """
    parser = argparse.ArgumentParser(
        description="Compare the results of multiple models",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "--CV_results_path",
        type=str,
        required=True,
        help="Path to the directory containing the CV results",
    )
    parser.add_argument(
        "--main_metric",
        type=str,
        required=True,
        help="The main metric to use for comparison",
    )
    parser.add_argument(
        "--sec_metrics",
        type=str,
        nargs="+",
        help="Secondary metrics to use for comparison",
    )
    return parser


if __name__ == "__main__":
    args = get_argparser().parse_args()
    CV_results_path = args.CV_results_path
    main_metric = args.main_metric
    sec_metrics = args.sec_metrics

    logging.basicConfig(
        level=logging.INFO,
        datefmt="%d-%m %H:%M",
        format="%(asctime)s - %(levelname)s: %(message)s",
    )

    all_results = {}
    for file in os.listdir(CV_results_path):

        if file.endswith("_CV_results.csv"):
            cv_results_df = pd.read_csv(CV_results_path + file)

            assert (
                f"rank_test_{main_metric}" in cv_results_df.columns
            ), f"{file} does not contain results for {main_metric}"
            for metric in sec_metrics:
                assert (
                    f"mean_test_{metric}" in cv_results_df.columns
                ), f"{file} does not contain results for {metric}"

            # remove the "_CV_results.csv" part of the filename
            model_name = file.split("_CV_results.csv")[0]
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
                all_results[model_name][metric] = scores

    if len(all_results) == 0:
        raise ValueError("No CV results found in the specified directory")

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
        logging.info(f"Best models based on {main_metric}: {n_best_models}")

        best_results = {model: all_results[model] for model in n_best_models}
        best_model, reason = get_best_model(
            results_dict=best_results,
            main_metric=main_metric,
            secondary_metrics=sec_metrics,
            bf_corr=True,
        )
        logging.info(f"Best model: {best_model}. Reason: {reason}.")
