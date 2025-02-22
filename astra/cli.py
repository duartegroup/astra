import argparse
from argparse import RawTextHelpFormatter
from . import benchmark
from . import compare


def get_CLI_parser() -> argparse.ArgumentParser:
    """
    Get the argument parser for the CLI. There are two commands available:
    - benchmark: Benchmark model performance
    - compare: Compare model performance

    Benchmark command arguments:
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

    Compare command arguments:
    - CV_results_path: Path to the directory containing the CV results
    - main_metric: The main metric to use for comparison
    - sec_metrics: Secondary metrics to use for comparison

    Returns
    -------
    argparse.ArgumentParser
        Argument parser for the CLI.
    """
    parser = argparse.ArgumentParser(
        description="ASTRA - Automated model selection using statistical testing"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Benchmark model performance",
        formatter_class=RawTextHelpFormatter,
    )
    benchmark_parser.add_argument(
        "data",
        type=str,
        help="Path to the dataset to train and evaluate models on. This should be a\n"
        "pickled pd.DataFrame, at least with a column called 'Target', containing the\n"
        "target variable. Precomputed features should be in column called 'Features'.\n"
        "If the data is not prefeaturised and presplit, it should contain a column\n"
        "called 'SMILES'.",
    )
    benchmark_parser.add_argument(
        "--name",
        type=str,
        help="Name of the experiment. Results will be saved in a folder with this name\n"
        "in the 'results' directory. Will be used to load cached results if they exist.",
        required=True,
    )
    benchmark_parser.add_argument(
        "--run_nested_CV",
        action="store_true",
        default=False,
        help="Whether or not to run nested CV with hyperparameter tuning for the best\n"
        "models. Default: False.",
    )
    benchmark_parser.add_argument(
        "--fold_col",
        type=str,
        default="Fold",
        help="Name of the column containing the CV fold number. Default: fold.",
    )
    benchmark_parser.add_argument(
        "--main_metric",
        type=str,
        default="R2",
        help="Main metric to use for model selection. This will be used to infer the\n"
        "prediction task (classification or regression). Default: R2.",
    )
    benchmark_parser.add_argument(
        "--sec_metrics",
        type=str,
        nargs="+",
        default=["MSE", "MAE"],
        help="Secondary metrics to use for model selection. Default: MSE MAE.",
    )
    benchmark_parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Type of split to use, if the data is to be resplit first. Valid choices are\n"
        "'Scaffold' and 'Fingerprint'. If this is not specified, the data is assumed to\n"
        "already be split. If the data is not to be split, do not specify this argument,\n"
        "as it will override the original split. Default: None.",
    )
    benchmark_parser.add_argument(
        "--n_folds",
        type=int,
        default=5,
        help="Number of folds to split the data into, if the data is to be resplit first.\n"
        "Default: 5.",
    )
    benchmark_parser.add_argument(
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
    benchmark_parser.add_argument(
        "--incl_RDKit_feats",
        action="store_true",
        default=False,
        help="Whether or not to include RDKit features, if the data is to be featurised first."
        "If 'fingerprint' isn't specified, this argument is ignored.",
    )
    benchmark_parser.add_argument(
        "--scaler",
        type=str,
        default=None,
        help="Type of scaler to use, if the data is to be scaled first.\n"
        "Valid choices are 'Standard' and 'MinMax'. Default: None.",
    )
    benchmark_parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Number of jobs to run in parallel for hyperparameter tuning. Default: 1.",
    )

    compare_parser = subparsers.add_parser("compare", help="Compare model performance")
    compare_parser.add_argument(
        "CV_results_path",
        type=str,
        help="Path to the directory containing the CV results",
    )
    compare_parser.add_argument(
        "--main_metric",
        type=str,
        required=True,
        help="The main metric to use for comparison",
    )
    compare_parser.add_argument(
        "--sec_metrics",
        type=str,
        nargs="+",
        help="Secondary metrics to use for comparison",
    )

    return parser


def main() -> int:
    """
    Main function for the CLI. Parses the arguments and runs the appropriate command.

    Returns
    -------
    int
        Exit code. 0 if successful, 1 if an error occurred.
    """
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
    parser = get_CLI_parser()
    args = parser.parse_args()

    if args.command == "benchmark":
        benchmark.run(
            data=args.data,
            name=args.name,
            run_nested_CV=args.run_nested_CV,
            fold_col=args.fold_col,
            main_metric=args.main_metric,
            sec_metrics=args.sec_metrics,
            split=args.split,
            n_folds=args.n_folds,
            fingerprint=args.fingerprint,
            incl_RDKit_feats=args.incl_RDKit_feats,
            scaler=args.scaler,
            n_jobs=args.n_jobs,
        )
    elif args.command == "compare":
        compare.run(
            results_dir=args.CV_results_path,
            output_file=args.main_metric,
            main_metric=args.main_metric,
            sec_metrics=args.sec_metrics,
        )
    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
