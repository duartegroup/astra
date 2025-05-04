import os
import pandas as pd
import numpy as np
import pickle
import logging

# from .data.splitting import get_splits  # TODO: Move away from deepchem
from .featurisation.features import get_fingerprints, RDKit_descriptors
from .model_selection import (
    CLASSIFICATION_METRICS,
    REGRESSION_METRICS,
    get_cv_performance,
    get_optimised_cv_performance,
    get_best_hparams,
    find_n_best_models,
    get_best_model,
)
from .utils import get_estimator_name, get_scores

# TODO: extract some of the code here and in compare.py into functions


def run(
    data: str,
    name: str,
    features: str = "Features",
    target: str = "Target",
    run_nested_CV: bool = False,
    fold_col: str = "Fold",
    main_metric: str = "R2",
    sec_metrics: list[str] = ["MSE", "MAE"],
    split: str | None = None,
    n_folds: int = 5,
    fingerprint: str | None = None,
    incl_RDKit_feats: bool = False,
    scaler: str | None = None,
    n_jobs: int = 1,
) -> None:
    """
    Run the benchmark.

    Parameters
    ----------
    data : str
        Path to the dataset to train and evaluate models on. This should be a pickled
        pd.DataFrame. If the data is not prefeaturised and presplit, it should contain
        a column called 'SMILES'.
    name : str
        Name of the experiment. Results will be saved in a folder with this name in the
        'results' directory. Will be used to load cached results if they exist.
    - features: Name of the column containing the features. Default: Features.
    - target: Name of the column containing the target. Default: Target.
    run_nested_CV : bool, default=False
        Whether or not to run nested CV with hyperparameter tuning for the best models.
    fold_col : str, default='Fold'
        Name of the column containing the CV fold number.
    main_metric : str, default='R2'
        Main metric to use for model selection. This will be used to infer the
        prediction task (classification or regression).
    sec_metrics : list[str], default=['MSE', 'MAE']
        Secondary metrics to use for model selection.
    split : str or None, default=None
        Type of split to use, if the data is to be resplit first. Valid choices are
        'Scaffold' and 'Fingerprint'. Results (fold number) will be saved in a column
        called 'Fold'.
    n_folds : int, default=5
        Number of CV folds.
    fingerprint : str or None, default=None
        Type of fingerprint to use, if the data is to be featurised first.
        Valid choices are 'Morgan', 'Avalon', 'RDKit', 'MACCS', 'AtomPair', 'TopTorsion'.
        Results will be saved in a column called 'Features'. For Morgan fingerprints,
        specify the radius and fingerprint size as 'Morgan_{radius}_{fpsize}'.
    incl_RDKit_feats : bool, default=False
        Whether or not to include RDKit features, if the data is to be featurised first.
        If 'fingerprint' isn't specified, this argument is ignored.
    scaler : str or None, default=None
        Type of scaler to use, if the data is to be scaled first. Valid choices are
        'Standard' and 'MinMax'.
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
    logging.basicConfig(
        level=logging.INFO,
        datefmt="%d-%m %H:%M",
        format="%(asctime)s - %(levelname)s: %(message)s",
    )

    logging.info("Loading data.")
    data = pd.read_pickle(data)

    if main_metric in REGRESSION_METRICS:
        from .models.regression import regressors, regressor_params

        for metric in sec_metrics:
            assert (
                metric in REGRESSION_METRICS
            ), "Secondary metrics must be regression metrics too."

        models = regressors
        params = regressor_params
        logging.info("Benchmarking regression models.")
    elif main_metric in CLASSIFICATION_METRICS:
        from .models.classification import (
            classifiers,
            classifier_params,
            non_probabilistic_models,
        )

        for metric in sec_metrics:
            assert (
                metric in CLASSIFICATION_METRICS
            ), "Secondary metrics must be classification metrics too."

        if main_metric in ["ROC_AUC", "PR_AUC"]:
            classifiers = {
                c: classifiers[c]
                for c in classifiers
                if c not in non_probabilistic_models
            }
        models = classifiers
        params = classifier_params
        logging.info("Benchmarking classification models.")
    else:
        raise ValueError(
            "Invalid metrics specified. Known metrics are:",
            REGRESSION_METRICS,
            "and",
            CLASSIFICATION_METRICS,
        )

    os.makedirs("cache", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(f"results/{name}", exist_ok=True)

    if split is not None:
        logging.info("Splitting data.")
        assert "SMILES" in data.columns, "Data does not contain a 'SMILES' column."
        assert split in ["Scaffold", "Fingerprint"], "Invalid split type."
        data = get_splits(data, split, n_folds)

    if fingerprint is not None:
        logging.info("Featurising data.")
        assert "SMILES" in data.columns, "Data does not contain a 'SMILES' column."
        assert fingerprint in [
            "Morgan",
            "Avalon",
            "RDKit",
            "MACCS",
            "AtomPair",
            "TopTorsion",
        ], "Invalid fingerprint type."

        if fingerprint.startswith("Morgan"):
            radius, fpsize = map(int, fingerprint.split("_")[1:])
            fingerprint = "Morgan"
        else:
            radius, fpsize = None, None

        smiles_list = data["SMILES"].tolist()
        fingerprints = get_fingerprints(smiles_list, fingerprint, radius, fpsize)
        if incl_RDKit_feats:
            rdkit_feats = RDKit_descriptors(smiles_list)
            fingerprints = np.concatenate([fingerprints, rdkit_feats], axis=1)

        data["Features"] = fingerprints
        logging.info("Featurisation complete. Saving data.")
        data.to_pickle("cache/featurised_data.pkl")

    assert features in data.columns, f"Data does not contain a '{features}' column."
    assert target in data.columns, f"Data does not contain a '{target}' column."
    assert fold_col in data.columns, f"Data does not contain a '{fold_col}' column."

    logging.info("Starting benchmarking.")

    logging.info(
        f"Starting {n_folds}-fold CV for all models using default hyperparameters."
    )
    if os.path.exists(f"results/{name}/default_CV.pkl"):
        logging.info("Loading existing results.")
        with open(f"results/{name}/default_CV.pkl", "br") as f:
            results = pickle.load(f)
    else:
        try:
            with open(f"cache/{name}_CV_ckpt.pkl", "br") as f:
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
                df=data,
                features_col=features,
                target_col=target,
                n_folds=n_folds,
                fold_col=fold_col,
                metric_list=[main_metric] + sec_metrics,
                scaler=scaler,
            )
            with open(f"cache/{name}_CV_ckpt.pkl", "wb") as f:
                pickle.dump(results, f)
        with open(f"results/{name}/default_CV.pkl", "wb") as f:
            pickle.dump(results, f)
        os.remove(f"cache/{name}_CV_ckpt.pkl")
        logging.info("Done!")

    if run_nested_CV:
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
                results_dic=results, metric=main_metric, bf_corr=True
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
                    df=data,
                    features_col=features,
                    target_col=target,
                    n_folds=n_folds,
                    fold_col=fold_col,
                    metric_list=[main_metric] + sec_metrics,
                    main_metric=main_metric,
                    parameters=params[model],
                    n_jobs=n_jobs,
                    scaler=scaler,
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
        bf_corr=True,
    )
    logging.info(f"Best model: {best_model}. Reason: {reason}")

    logging.info("Starting final hyperparameter tuning.")
    model = get_best_hparams(
        model_class=models[best_model],
        df=data,
        n_folds=n_folds,
        fold_col=fold_col,
        main_metric=main_metric,
        sec_metrics=sec_metrics,
        parameters=params[best_model],
        n_jobs=n_jobs,
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
    mean_score_main, std_score_main, sec_metrics_scores = get_scores(
        cv_results_df, main_metric, sec_metrics, n_folds
    )
    with open(f"results/{name}/final_model.pkl", "wb") as f:
        pickle.dump(final_model, f)
    with open(f"results/{name}/final_hyperparameters.pkl", "wb") as f:
        pickle.dump(final_hyperparameters, f)
    cv_results_df.to_csv(f"results/{name}/final_CV_results.csv")

    print("-" * 50)
    print("Final results")
    print("-" * 50)
    print(f"Final model: {final_model_name}")
    print("Hyperparameters:")
    for f in final_hyperparameters:
        print(f + ":", final_hyperparameters[f])
    print(f"Mean {main_metric}:", f"{mean_score_main:.3f} ± {std_score_main:.3f}.")
    for metric in sec_metrics_scores:
        print(
            f"Mean {metric}:",
            f"{sec_metrics_scores[metric][0]:.3f} ± {sec_metrics_scores[metric][1]:.3f}.",
        )
    print("-" * 50)
