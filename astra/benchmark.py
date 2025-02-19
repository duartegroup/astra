import os
import pandas as pd
import numpy as np
import pickle
import argparse
from argparse import RawTextHelpFormatter
import logging
from data.splitting import get_splits  # TODO: Move away from deepchem
from featurisation.features import get_fingerprints, RDKit_descriptors
from model_selection import (
    CLASSIFICATION_METRICS,
    REGRESSION_METRICS,
    get_cv_performance,
    get_optimised_cv_performance,
    get_best_hparams,
    find_n_best_models,
    get_best_model,
)

# TODO: extract some of the code here and in compare.py into functions


def get_argparser() -> argparse.ArgumentParser:
    """
    Get the argument parser. Arguments are as follows:
    - data: Path to the dataset to train and evaluate models on. This should be a
            pickled pd.DataFrame, at least with a column called 'Target', containing the
            target variable. Precomputed features should be in column called 'Features'.
            If the data is not prefeaturised, it should contain a column called 'SMILES'.
    - name: Name of the experiment to save the results to. Will be used to load cached
            results if they exist. Default: results.
    - run_nested_CV: Whether or not to run nested CV with hyperparameter tuning for the best
            models. Default: False.
    - fold_col: Name of the column containing the CV fold number. Default: fold.
    - main_metric: Main metric to use for model selection. This will be used to infer the
            prediction task (classification or regression). Default: R2.
    - sec_metrics: Secondary metrics to use for model selection. Default: MSE MAE.
    - split: Type of split to use, if the data is to be resplit first. Valid choices are
            'Scaffold' and 'Fingerprint'. If this is not specified, the data is assumed to
            already be split. If the data is not to be split, do not specify this argument,
            as it will override the original split. Default: None.
    - n_folds: Number of folds to split the data into, if the data is to be resplit first.
            Default: 5.
    - fingerprint: Type of fingerprint to use, if the data is to be featurised first.
            Valid choices are 'Morgan', 'Avalon', 'RDKit', 'MACCS', 'AtomPair', 'TopTorsion'.
            If this is not specified, the data is assumed to already be featurised. If the data
            is not to be featurised, do not specify this argument, as it will override the
            'Features' column. For Morgan fingerprints, specify the radius and fingerprint size
            as 'Morgan_{radius}_{fpsize}'. Default: None.
    - incl_RDKit_feats: Whether or not to include RDKit features, if the data is to be featurised first.
            If 'fingerprint' isn't specified, this argument is ignored.
    - scaler: Type of scaler to use, if the data is to be scaled first.
            Valid choices are 'Standard' and 'MinMax'. Default: None.
    - n_jobs: Number of jobs to run in parallel for hyperparameter tuning. Default: 1.

    Returns
    -------
    argparse.ArgumentParser
        The argument parser.
    """
    argparser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    argparser.add_argument(
        "data",
        type=str,
        help="Path to the dataset to train and evaluate models on. This should be a\n"
        "pickled pd.DataFrame, at least with a column called 'Target', containing the\n"
        "target variable. Precomputed features should be in column called 'Features'.\n"
        "If the data is not prefeaturised and presplit, it should contain a column\n"
        "called 'SMILES'.",
    )
    argparser.add_argument(
        "--name",
        type=str,
        default="results",
        help="Name of the experiment to save the results to. Will be used to load\n"
        "existing results if they exist. Default: results.",
    )
    argparser.add_argument(
        "--run_nested_CV",
        action="store_true",
        default=False,
        help="Whether or not to run nested CV with hyperparameter tuning for the best\n"
        "models. Default: False.",
    )
    argparser.add_argument(
        "--fold_col",
        type=str,
        default="Fold",
        help="Name of the column containing the CV fold number. Default: fold.",
    )
    argparser.add_argument(
        "--main_metric",
        type=str,
        default="R2",
        help="Main metric to use for model selection. This will be used to infer the\n"
        "prediction task (classification or regression). Default: R2.",
    )
    argparser.add_argument(
        "--sec_metrics",
        type=str,
        nargs="+",
        default=["MSE", "MAE"],
        help="Secondary metrics to use for model selection. Default: MSE MAE.",
    )
    argparser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Type of split to use, if the data is to be resplit first. Valid choices are\n"
        "'Scaffold' and 'Fingerprint'. If this is not specified, the data is assumed to\n"
        "already be split. If the data is not to be split, do not specify this argument,\n"
        "as it will override the original split. Default: None.",
    )
    argparser.add_argument(
        "--n_folds",
        type=int,
        default=5,
        help="Number of folds to split the data into, if the data is to be resplit first.\n"
        "Default: 5.",
    )
    argparser.add_argument(
        "--fingerprint",
        type=str,
        default=None,
        help="Type of fingerprint to use, if the data is to be featurised first.\n"
        "Valid choices are 'Morgan', 'Avalon', 'RDKit', 'MACCS', 'AtomPair', 'TopTorsion'.\n"
        "If this is not specified, the data is assumed to already be featurised. If the data\n"
        "is not to be featurised, do not specify this argument, as it will override the\n"
        "'Features' column. For Morgan fingerprints, specify the radius and fingerprint size\n"
        "as 'Morgan_{radius}_{fpsize}'. Default: None.",
    )
    argparser.add_argument(
        "--incl_RDKit_feats",
        action="store_true",
        default=False,
        help="Whether or not to include RDKit features, if the data is to be featurised first."
        "If 'fingerprint' isn't specified, this argument is ignored.",
    )
    argparser.add_argument(
        "--scaler",
        type=str,
        default=None,
        help="Type of scaler to use, if the data is to be scaled first.\n"
        "Valid choices are 'Standard' and 'MinMax'. Default: None.",
    )
    argparser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Number of jobs to run in parallel for hyperparameter tuning. Default: 1.",
    )
    return argparser


if __name__ == "__main__":
    argparser = get_argparser()
    args = argparser.parse_args()

    print(
        r"""
        ----------------------------------------------
                             _
                  __ _  ___ | |_  _ __   __ _
                 / _` |/ __|| __|| '__| / _` |
                | (_| |\__ \| |_ | |   | (_| |
                 \__,_||___/ \__||_|    \__,_|

        ----------------------------------------------
        """,
        flush=True,
    )

    logging.basicConfig(
        level=logging.INFO,
        datefmt="%d-%m %H:%M",
        format="%(asctime)s - %(levelname)s: %(message)s",
    )

    logging.info("Loading data.")
    data = pd.read_pickle(args.data)
    name = args.name
    run_nested_CV = args.run_nested_CV
    fold_col = args.fold_col
    main_metric = args.main_metric
    sec_metrics = args.sec_metrics
    split = args.split
    n_folds = args.n_folds
    fp_type = args.fingerprint
    incl_RDKit_feats = args.incl_RDKit_feats
    scaler = args.scaler
    n_jobs = args.n_jobs

    if main_metric in REGRESSION_METRICS:
        from models.regression import regressors, regressor_params

        for metric in sec_metrics:
            assert (
                metric in REGRESSION_METRICS
            ), "Secondary metrics must be regression metrics too."

        models = regressors
        params = regressor_params
        mode = "regression"
        logging.info("Benchmarking regression models.")
    elif main_metric in CLASSIFICATION_METRICS:
        from models.classification import (
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
        mode = "classification"
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

    if split is not None:
        logging.info("Splitting data.")
        assert "SMILES" in data.columns, "Data does not contain a 'SMILES' column."
        assert split in ["Scaffold", "Fingerprint"], "Invalid split type."
        data = get_splits(data, split, n_folds)

    if fp_type is not None:
        logging.info("Featurising data.")
        assert "SMILES" in data.columns, "Data does not contain a 'SMILES' column."
        assert fp_type in [
            "Morgan",
            "Avalon",
            "RDKit",
            "MACCS",
            "AtomPair",
            "TopTorsion",
        ], "Invalid fingerprint type."

        if fp_type.startswith("Morgan"):
            radius, fpsize = map(int, fp_type.split("_")[1:])
            fp_type = "Morgan"
        else:
            radius, fpsize = None, None

        smiles_list = data["SMILES"].tolist()
        fingerprints = get_fingerprints(smiles_list, fp_type, radius, fpsize)
        if incl_RDKit_feats:
            rdkit_feats = RDKit_descriptors(smiles_list)
            fingerprints = np.concatenate([fingerprints, rdkit_feats], axis=1)

        data["Features"] = fingerprints
        logging.info("Featurisation complete. Saving data.")
        data.to_pickle("cache/featurised_data.pkl")

    assert "Features" in data.columns, "Data does not contain a 'Features' column."
    assert "Target" in data.columns, "Data does not contain a 'Target' column."
    assert "Fold" in data.columns, "Data does not contain a 'Fold' column."

    logging.info("Starting benchmarking.")

    logging.info(
        f"Starting {n_folds}-fold CV for all models using default hyperparameters."
    )
    if os.path.exists(f"results/{name}_CV.pkl"):
        logging.info("Loading existing results.")
        with open(f"results/{name}_CV.pkl", "br") as f:
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
                n_folds=n_folds,
                fold_col=fold_col,
                metric_list=[main_metric] + sec_metrics,
                scaler=scaler,
            )
            with open(f"cache/{name}_CV_ckpt.pkl", "wb") as f:
                pickle.dump(results, f)
        with open(f"results/{name}_CV.pkl", "wb") as f:
            pickle.dump(results, f)
        os.remove(f"cache/{name}_CV_ckpt.pkl")
        logging.info("Done!")

    if run_nested_CV:
        logging.info(
            "Starting nested CV with hyperparameter tuning for the best models."
        )
        if os.path.exists(f"results/{name}_nested_CV.pkl"):
            logging.info("Loading existing results.")
            with open(f"results/{name}_nested_CV.pkl", "br") as f:
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
            with open(f"results/{name}_nested_CV.pkl", "wb") as f:
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
    final_hyperparameters = model.best_params_
    final_hyperparameters = {
        key.replace("model__", ""): value
        for key, value in final_hyperparameters.items()
    }
    cv_results_df = pd.DataFrame(model.cv_results_)
    all_scores = [
        cv_results_df[cv_results_df[f"rank_test_{main_metric}"] == 1].iloc[0][
            f"split{i}_test_{main_metric}"
        ]
        for i in range(n_folds)
    ]
    mean_score = np.mean(all_scores)
    std_score = np.std(all_scores)
    with open(f"results/{name}_final_model.pkl", "wb") as f:
        pickle.dump(final_model, f)
    with open(f"results/{name}_final_hyperparameters.pkl", "wb") as f:
        pickle.dump(final_hyperparameters, f)
    cv_results_df.to_csv(f"results/{name}_cv_results.csv")

    print("-" * 50)
    print("Final results")
    print("-" * 50)
    print("Final model:", final_model)
    print("Hyperparameters:")
    for f in final_hyperparameters:
        print(f + ":", final_hyperparameters[f])
    print(f"Mean {main_metric}:", f"{mean_score:.3f} ± {std_score:.3f}.")
    print("-" * 50)
