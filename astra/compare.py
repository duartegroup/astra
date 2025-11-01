import logging
import os
import pickle

import numpy as np

from .model_selection import (
    check_assumptions,
    check_best_model,
    find_n_best_models,
    get_best_model,
    perform_statistical_tests,
)


def run(
    CV_results: list[str],
    main_metric: str,
    sec_metrics: list[str],
    parametric: str | bool = "auto",
) -> None:
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
    parametric : str or bool, default="auto"
        Whether to use parametric tests. If 'auto', the assumptions of parametric tests
        will be checked, and parametric tests will be used if the assumptions are met.

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
        if not os.listdir(CV_results_path):
            raise ValueError(f"{CV_results_path} is empty")

    all_in_one_dir = True if len(CV_results) == 1 else False

    for CV_results_path in CV_results:
        for file in os.listdir(CV_results_path):
            if file.endswith("final_CV.pkl"):
                with open(CV_results_path + file, "rb") as f:
                    cv_results = pickle.load(f)

                assert (
                    main_metric in cv_results
                ), f"{file} does not contain results for {main_metric}"
                for metric in sec_metrics:
                    assert (
                        metric in cv_results
                    ), f"{file} does not contain results for {metric}"

                if all_in_one_dir:
                    model_name = file.split("final_CV.pkl")[0]
                else:
                    model_name = CV_results_path.split("/")[-2]

                all_results[model_name] = {
                    metric: cv_results[metric] for metric in [main_metric] + sec_metrics
                }

    logging.info("Starting comparison of CV results.")

    if parametric == "auto":
        logging.info("Will check assumptions for parametric tests and use them if met.")
    elif parametric is True:
        logging.info("Using parametric tests.")
    elif parametric is False:
        logging.info("Using non-parametric tests.")
    else:
        raise ValueError(
            "`parametric` must be one of [True, False, 'auto']. "
            f"Got {parametric} instead."
        )

    if len(all_results) == 0:
        raise ValueError("No CV results found")

    elif len(all_results) == 1:
        raise ValueError("Only one CV result found, cannot compare")

    elif len(all_results) == 2:
        logging.info("Two CV results found. Performing pairwise comparison.")

        if parametric == "auto":
            logging.info("Checking assumptions for parametric tests.")
            parametric = check_assumptions(results_dict=all_results, verbose=False)
            logging.info(f"Assumptions of parametric tests met: {parametric}.")
        elif parametric is True:
            logging.info("Checking assumptions for parametric tests.")
            _ = check_assumptions(results_dict=all_results, verbose=True)

        _, naive_stats = perform_statistical_tests(all_results, main_metric, parametric)
        better_model = check_best_model(all_results, naive_stats, main_metric)

        if better_model:
            logging.info(
                f"{better_model} is significantly better according to {main_metric}"
            )

        else:
            for metric in sec_metrics:
                _, naive_stats = perform_statistical_tests(all_results, metric)
                better_model = check_best_model(all_results, naive_stats, metric)
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

        if parametric == "auto":
            logging.info("Checking assumptions for parametric tests.")
            parametric = check_assumptions(results_dict=all_results, verbose=False)
            logging.info(f"Assumptions of parametric tests met: {parametric}.")
        elif parametric is True:
            logging.info("Checking assumptions for parametric tests.")
            _ = check_assumptions(results_dict=all_results, verbose=True)

        n_best_models = find_n_best_models(
            results_dic=all_results,
            metric=main_metric,
            parametric=parametric,
            bf_corr=True,
        )
        logging.info(f"Best models based on {main_metric}:")
        for model in n_best_models:
            print(" " * 20 + f"{model}")

        best_results = {model: all_results[model] for model in n_best_models}
        best_model, reason = get_best_model(
            results_dict=best_results,
            main_metric=main_metric,
            secondary_metrics=sec_metrics,
            parametric=parametric,
            bf_corr=True,
        )
        logging.info(f"Best model overall: {best_model}. Reason: {reason}.")

        if all_in_one_dir:
            with open(CV_results[0] + best_model + "final_CV.pkl", "rb") as f:
                cv_results = pickle.load(f)
        else:
            for CV_results_path in CV_results:
                if best_model == CV_results_path.split("/")[-2]:
                    for file in os.listdir(CV_results_path):
                        if file.endswith("final_CV.pkl"):
                            with open(CV_results_path + file, "rb") as f:
                                cv_results = pickle.load(f)
                            break
                    break

        print("-" * 50)
        print("Results:")
        print("-" * 50)
        print(
            f"Mean {main_metric}:",
            f"{np.mean(cv_results[main_metric]):.3f} ± {np.std(cv_results[main_metric]):.3f}.",
        )
        print(f"Median {main_metric}:", f"{np.median(cv_results[main_metric]):.3f}.")
        for metric in sec_metrics:
            print(
                f"Mean {metric}:",
                f"{np.mean(cv_results[metric]):.3f} ± {np.std(cv_results[metric]):.3f}.\n",
                f"Median {metric}:",
                f"{np.median(cv_results[metric]):.3f}.",
            )
        print("-" * 50)
