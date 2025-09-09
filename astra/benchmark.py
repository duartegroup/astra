import logging
import os
import pickle

import pandas as pd

from .model_selection import (
    check_assumptions,
    find_n_best_models,
    get_best_hparams,
    get_best_model,
    get_optimised_cv_performance,
    run_CV,
)
from .utils import (
    get_data,
    get_estimator_name,
    get_models,
    get_scores,
    print_file_console,
    print_final_results,
    print_performance,
)


def run(
    data: str,
    name: str | None = None,
    features: str = "Features",
    target: str = "Target",
    run_nested_CV: bool = False,
    fold_col: str | list[str] = "Fold",
    main_metric: str = "R2",
    sec_metrics: list[str] = ["MSE", "MAE"],
    parametric: str | bool = "auto",
    impute: str | float | int | None = None,
    remove_constant: float | None = None,
    remove_correlated: float | None = None,
    scaler: str | None = None,
    custom_models: (
        dict[str, None | dict[str, dict] | tuple[dict[str, dict], dict[str, dict]]]
        | None
    ) = None,
    n_jobs: int = 1,
) -> None:
    """
    Run the benchmark.

    Parameters
    ----------
    data : str
        Path to the dataset to train and evaluate models on. This should be a CSV, pickle,
        or parquet file.
    name : str or None, default=None
        Name of the experiment. Results will be saved in a folder with this name in the
        'results' directory. Will be used to load cached results if they exist. If None,
        the name will be the file name of the data file without extension.
    features : str, default='Features'
        Name of the column containing the features. Default: Features.
    target : str, default='Target'
        Name of the column containing the target. Default: Target.
    run_nested_CV : bool, default=False
        Whether to run nested CV with hyperparameter tuning for the best models.
    fold_col : str or list[str], default='Fold'
        Name(s) of the column(s) containing the CV fold number(s). If a list is provided,
        models will be benchmarked in an nxk-fold CV, where n is the number of repeats
        and k is the number of folds. If a single string is provided, it will be treated
        as a single fold column. nxk-fold CV does not currently support nested CV and
        final hyperparameter tuning.
    main_metric : str, default='R2'
        Main metric to use for model selection. This will be used to infer the
        prediction task (classification or regression).
    sec_metrics : list[str], default=['MSE', 'MAE']
        Secondary metrics to use for model selection.
    parametric : str or bool, default='auto'
        Whether to use parametric tests. If 'auto', the assumptions of parametric tests
        will be checked, and parametric tests will be used if the assumptions are met.
    impute : str or float or int or None, default=None
        Method to use for imputing missing values. If None, no imputation will be performed.
        Valid choices are 'mean', 'median', 'knn', or a float or int value for constant imputation.
    remove_constant : float or None, default=None
        If specified, features with variance below this threshold will be removed.
        If None, no features are removed.
    remove_correlated : float or None, default=None
        If specified, features with correlation above this threshold will be removed.
        If None, no features are removed.
    scaler : str or None, default=None
        Type of scaler to use, if the data is to be scaled first. Valid choices are
        'Standard' and 'MinMax'.
    custom_models : dict[str, None | dict[str, dict] | tuple[dict[str, dict], dict[str, dict]]] or None, default=None
        Dictionary of models to use for benchmarking. If None, default models will be used.
        The keys should be the model names, and the values should be dictionaries of starting
        hyperparameters for the model, and/or a dictionary of hyperparameter search grids.
        Default models are defined in astra.models.classification and astra.models.regression.
    n_jobs : int, default=1
        Number of jobs to run in parallel for hyperparameter tuning.

    Raises
    ------
    ValueError
        If any of the arguments are invalid.

    Returns
    -------
    None
    """
    if name is None:
        name = os.path.splitext(os.path.basename(data))[0]
    os.makedirs("cache", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(f"results/{name}", exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        datefmt="%d-%m %H:%M",
        format="%(asctime)s - %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler(f"results/{name}/benchmark.log"),
            logging.StreamHandler(),
        ],
    )

    logging.info(f"Starting benchmark for {name}.")

    logging.info("Loading data.")
    data_df = get_data(data, features=features)

    if isinstance(fold_col, str):
        n_folds = data_df[fold_col].nunique()
        assert (
            fold_col in data_df.columns
        ), f"Data does not contain a '{fold_col}' column."
        repeated_CV = False
    elif isinstance(fold_col, list):
        n_folds = [data_df[col].nunique() for col in fold_col]
        if len(set(n_folds)) != 1:
            raise ValueError(
                "All fold columns must have the same number of folds. "
                f"Got {n_folds} instead."
            )
        for col in fold_col:
            assert col in data_df.columns, f"Data does not contain a '{col}' column."
        repeated_CV = True
        n_folds = n_folds[0]
    else:
        raise ValueError(
            "`fold_col` must be a string or a list of strings. "
            f"Got {type(fold_col)} instead."
        )

    assert features in data_df.columns, f"Data does not contain a '{features}' column."
    assert target in data_df.columns, f"Data does not contain a '{target}' column."

    logging.info("Starting benchmarking.")
    logging.info(f"Features column: {features}")
    logging.info(f"Target column: {target}")
    if repeated_CV:
        logging.info(f"Running {len(fold_col)}x{n_folds} repeated CV.")
        logging.info(f"Fold columns: {fold_col}")
    else:
        logging.info(f"Running {n_folds}-fold CV.")
        logging.info(f"Fold column: {fold_col}")

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

    logging.info("Getting models and parameters.")
    models, params, custom_params = get_models(
        main_metric=main_metric,
        sec_metrics=sec_metrics,
        scaler=scaler,
        custom_models=custom_models,
    )
    logging.info(f"Main metric: {main_metric}")
    logging.info(f"Secondary metrics: {sec_metrics}")

    if impute is not None:
        if isinstance(impute, str):
            logging.info(f"Imputing missing values using {impute} strategy.")
        elif isinstance(impute, (int, float)):
            logging.info(f"Imputing missing values with constant value: {impute}.")
        else:
            raise ValueError(
                "`impute` must be a string or a number. " f"Got {impute} instead."
            )
    if remove_constant is not None:
        if not isinstance(remove_constant, float) and not remove_constant == 0:
            raise ValueError(
                "`remove_constant` must be a float. " f"Got {remove_constant} instead."
            )
        logging.info(f"Removing features with variance below {remove_constant}.")
    if remove_correlated is not None:
        if not isinstance(remove_correlated, float):
            raise ValueError(
                "`remove_correlated` must be a float. "
                f"Got {remove_correlated} instead."
            )
        logging.info(f"Removing features with correlation above {remove_correlated}.")
    if scaler is not None:
        if scaler not in ["Standard", "MinMax"]:
            raise ValueError(
                "`scaler` must be one of ['Standard', 'MinMax']. "
                f"Got {scaler} instead."
            )
        logging.info(f"Using {scaler.lower()} scaler.")

    logging.info("Starting CV for all models using default hyperparameters.")

    if repeated_CV:
        if not os.path.exists(f"results/{name}/default_CV_all_folds.pkl"):
            results = {}
            for col in fold_col:
                logging.info(f"Running CV for fold column: {col}.")
                results_col = run_CV(
                    name=name,
                    data_df=data_df,
                    features=features,
                    target=target,
                    fold_col=col,
                    models=models,
                    metric_list=[main_metric] + sec_metrics,
                    impute=impute,
                    remove_constant=remove_constant,
                    remove_correlated=remove_correlated,
                    scaler=scaler,
                    custom_params=custom_params,
                    repeated=True,
                )
                for model in results_col:
                    if model not in results:
                        results[model] = results_col[model]
                    else:
                        for metric in results_col[model]:
                            results[model][metric].extend(results_col[model][metric])
            for model in results:
                for metric in results[model]:
                    assert len(results[model][metric]) == n_folds * len(fold_col), (
                        f"Model {model} has {len(results[model][metric])} entries for "
                        f"metric {metric}, expected {n_folds * len(fold_col)}."
                    )
            with open(f"results/{name}/default_CV_all_folds.pkl", "wb") as f:
                pickle.dump(results, f)
        else:
            logging.info("Loading existing results.")
            with open(f"results/{name}/default_CV_all_folds.pkl", "br") as f:
                results = pickle.load(f)
    else:
        results = run_CV(
            name=name,
            data_df=data_df,
            features=features,
            target=target,
            fold_col=fold_col,
            models=models,
            metric_list=[main_metric] + sec_metrics,
            impute=impute,
            remove_constant=remove_constant,
            remove_correlated=remove_correlated,
            scaler=scaler,
            custom_params=custom_params,
            repeated=False,
        )

    logging.info("Finished CV for all models.")
    if parametric == "auto":
        logging.info("Checking assumptions for parametric tests.")
        parametric = check_assumptions(results_dict=results, verbose=False)
        logging.info(f"Assumptions of parametric tests met: {parametric}.")
    elif parametric is True:
        logging.info("Checking assumptions for parametric tests.")
        _ = check_assumptions(results_dict=results, verbose=True)

    if run_nested_CV and not repeated_CV:
        logging.info(
            "Starting nested CV with hyperparameter tuning for the best models."
        )
        if os.path.exists(f"results/{name}/nested_CV.pkl"):
            logging.info("Loading existing results.")
            with open(f"results/{name}/nested_CV.pkl", "br") as f:
                results = pickle.load(f)
        else:
            logging.info("Selecting best models.")
            n_best_models = find_n_best_models(
                results_dic=results,
                metric=main_metric,
                parametric=parametric,
                bf_corr=True,
            )
            logging.info(f"Best models based on {main_metric}:")
            for model in n_best_models:
                print_file_console(
                    file=logging.getLogger().handlers[0].stream.name,
                    message=" " * 20 + f"{model}",
                )

            try:
                with open(f"cache/{name}_nested_CV_ckpt.pkl", "br") as f:
                    results = pickle.load(f)
                logging.info("Loaded checkpoint.")
            except FileNotFoundError:
                results = {}

            logging.info("Running nested CV with hyperparameter tuning.")
            for model in n_best_models:
                logging.info(f"Running {model}.")
                if model in results:
                    logging.info("Found existing results. Skipping.")
                    continue
                results[model] = get_optimised_cv_performance(
                    model_class=models[model],
                    df=data_df,
                    features_col=features,
                    target_col=target,
                    fold_col=fold_col,
                    metric_list=[main_metric] + sec_metrics,
                    main_metric=main_metric,
                    parameters=params[model],
                    n_jobs=n_jobs,
                    impute=impute,
                    remove_constant=remove_constant,
                    remove_correlated=remove_correlated,
                    scaler=scaler,
                )
                print_performance(
                    model_name=model,
                    results_dict=results[model],
                    file=logging.getLogger().handlers[0].stream.name,
                )
                with open(f"cache/{name}_nested_CV_ckpt.pkl", "wb") as f:
                    pickle.dump(results, f)
            with open(f"results/{name}/nested_CV.pkl", "wb") as f:
                pickle.dump(results, f)
            os.remove(f"cache/{name}_nested_CV_ckpt.pkl")
            logging.info("Done!")

    logging.info("Finding best model.")
    best_model, reason = get_best_model(
        results_dict=results,
        main_metric=main_metric,
        secondary_metrics=sec_metrics,
        parametric=parametric,
        bf_corr=True,
    )
    logging.info(f"Best model: {best_model}. Reason: {reason}.")

    if repeated_CV:
        print_performance(
            model_name=best_model,
            results_dict=results[best_model],
            file=logging.getLogger().handlers[0].stream.name,
        )
    else:
        logging.info("Starting final hyperparameter tuning.")
        model = get_best_hparams(
            model_class=models[best_model],
            df=data_df,
            features_col=features,
            target_col=target,
            fold_col=fold_col,
            main_metric=main_metric,
            sec_metrics=sec_metrics,
            parameters=params[best_model],
            n_jobs=n_jobs,
            impute=impute,
            remove_constant=remove_constant,
            remove_correlated=remove_correlated,
            scaler=scaler,
        )
        logging.info("Done!")

        final_model = model.best_estimator_
        final_model_name = get_estimator_name(final_model)
        final_hyperparameters = model.best_params_
        final_hyperparameters = {
            key.replace(f"{final_model_name.lower()}__", ""): value
            for key, value in final_hyperparameters.items()
        }
        cv_results_df = pd.DataFrame(model.cv_results_)
        mean_score_main, std_score_main, median_score_main, sec_metrics_scores = (
            get_scores(cv_results_df, main_metric, sec_metrics, n_folds)
        )
        with open(f"results/{name}/final_model.pkl", "wb") as f:
            pickle.dump(final_model, f)
        with open(f"results/{name}/final_hyperparameters.pkl", "wb") as f:
            pickle.dump(final_hyperparameters, f)
        cv_results_df.to_csv(f"results/{name}/final_CV_results.csv")

        print_final_results(
            final_model_name=final_model_name,
            final_hyperparameters=final_hyperparameters,
            main_metric=main_metric,
            mean_score_main=mean_score_main,
            std_score_main=std_score_main,
            median_score_main=median_score_main,
            sec_metrics_scores=sec_metrics_scores,
            file=logging.getLogger().handlers[0].stream.name,
        )
