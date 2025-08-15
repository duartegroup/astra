import argparse
from argparse import RawTextHelpFormatter
from rich.console import Console
from . import benchmark
from . import compare
from .utils import load_config


def get_CLI_parser() -> argparse.ArgumentParser:
    """
    Get the argument parser for the CLI. There are two commands available:
    - benchmark: Benchmark model performance
    - compare: Compare model performance

    Benchmark command arguments:
    - data: Path to the dataset to train and evaluate models on. This should be a CSV, pickle,
            or parquet file.
    - name: Name of the experiment to save the results to. Will be used to load cached
            results if they exist. Default: `data` file name without extension.
    - features: Name of the column containing the features. Default: Features.
    - target: Name of the column containing the target. Default: Target.
    - run_nested_CV: Whether to run nested CV with hyperparameter tuning for the best
            models. Default: False.
    - fold_col: Name of the column containing the CV fold number. Default: fold.
    - main_metric: Main metric to use for model selection. This will be used to infer the
            prediction task (classification or regression). Default: R2.
    - sec_metrics: Secondary metrics to use for model selection. Default: MSE MAE.
    - parametric: Whether to use parametric statistical tests for model comparison.
    - fingerprint: Type of fingerprint to use, if the data is to be featurised first.
            Valid choices are 'Morgan', 'Avalon', 'RDKit', 'MACCS', 'AtomPair', 'TopTorsion'.
            Results will be saved in a column called 'Features'. For Morgan fingerprints,
            specify the radius and fingerprint size as 'Morgan_{radius}_{fpsize}'.
            Default: None.
    - incl_RDKit_feats: Whether to include RDKit features, if the data is to be featurised first.
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
    group = benchmark_parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "data",
        nargs="?",
        type=str,
        help="Path to the dataset to train and evaluate models on.",
    )
    group.add_argument(
        "--config",
        type=str,
        help="Path to a YAML config file. Must at least contain a 'data' key.",
    )
    benchmark_parser.add_argument(
        "--name",
        type=str,
        help="Name of the experiment. Results will be saved in a folder with this name\n"
        "in the 'results' directory. Will be used to load cached results if they exist. \n"
        "Default: `data` file name without extension.",
        default=None,
    )
    benchmark_parser.add_argument(
        "--features",
        type=str,
        default="Features",
        help="Name of the column containing the features. Default: Features.",
    )
    benchmark_parser.add_argument(
        "--target",
        type=str,
        default="Target",
        help="Name of the column containing the target. Default: Target.",
    )
    benchmark_parser.add_argument(
        "--run_nested_CV",
        action="store_true",
        default=False,
        help="Whether to run nested CV with hyperparameter tuning for the best\n"
        "models. Default: False.",
    )
    benchmark_parser.add_argument(
        "--fold_col",
        type=str,
        default="Fold",
        help="Name of the column containing the CV fold number. Default: Fold.",
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
        "--parametric",
        type=str,
        choices=["True", "False", "auto"],
        default="auto",
        help="Whether to use parametric statistical tests for model comparison.\n"
        "If 'auto' (default), the assumptions of parametric tests will be checked,\n"
        "and parametric tests will be used if the assumptions are met.",
    )
    benchmark_parser.add_argument(
        "--fingerprint",
        type=str,
        default=None,
        help="Type of fingerprint to use, if the data is to be featurised first.\n"
        "Valid choices are 'Morgan', 'Avalon', 'RDKit', 'MACCS', 'AtomPair', 'TopTorsion'.\n"
        "Results will be saved in a column called 'Features'. For Morgan fingerprints,\n"
        "specify the radius and fingerprint size as 'Morgan_{radius}_{fpsize}'. Default: None.",
    )
    benchmark_parser.add_argument(
        "--incl_RDKit_feats",
        action="store_true",
        default=False,
        help="Whether to include RDKit features, if the data is to be featurised first."
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
        "CV_results",
        type=str,
        nargs="+",
        help="Path to a single directory containing CV results, or a list of directories\n"
        "containing CV results, as returned by scikit-learn's BaseSearchCV.cv_results_.\n"
        "If a single directory is provided, CV results for the different models should be\n"
        "saved in a file ending with '_CV_results.csv', and this ending will be removed to\n"
        "get the model name.\n"
        "If multiple directories are provided, each should contain the CV results for a\n"
        "different model saved in a file ending with '_CV_results.csv', and the parent\n"
        "directory will be used as the model name.",
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
    compare_parser.add_argument(
        "--n_folds",
        type=int,
        default=5,
        help="Number of CV folds. Default: 5.",
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
    console = Console()
    console.print(
        "\n[bold blue]:wave: Welcome to ASTRA - Automated model selection using statistical testing[/bold blue]",
        justify="center",
    )
    astra_string = """
    ----------------------------------------------
    _
      __ _  ___ | |_  _ __   __ _
     / _` |/ __|| __|| '__| / _` |
    | (_| |\__ \| |_ | |   | (_| |
     \__,_||___/ \__||_|    \__,_|

    ----------------------------------------------
        """
    console.print(
        astra_string,
        style="bold blue",
        justify="center",
    )
    console.print(
        "[bold cyan]:thinking_face: For help, run: astra --help[/bold cyan]",
        justify="center",
    )
    console.print(
        "[bold cyan]:test_tube: To benchmark models, run: astra benchmark --help[/bold cyan]",
        justify="center",
    )
    console.print(
        "[bold cyan]:trophy: To compare models, run: astra compare --help[/bold cyan]",
        justify="center",
    )

    parser = get_CLI_parser()
    args = parser.parse_args()
    if args.config:
        config = load_config(args.config)

        # Override CLI arguments with config values
        for key, value in config.items():
            setattr(args, key, value)

        if args.data is None:
            raise ValueError("The config file must include a 'data' field.")

        # Custom model settings
        if "models" in config:
            args.models = {}
            for model in config["models"]:
                model_name = model["name"]
                model_params = model.get("params", None)
                hparam_grid = model.get("hparam_grid", None)
                args.models[model_name] = {
                    "params": model_params,
                    "hparam_grid": hparam_grid,
                }

    # convert all args.main_metric and args.sec_metrics to lowercase
    args.main_metric = args.main_metric.lower()
    args.sec_metrics = [metric.lower() for metric in args.sec_metrics]

    if args.command == "benchmark":
        benchmark.run(
            data=args.data,
            name=args.name,
            features=args.features,
            target=args.target,
            run_nested_CV=args.run_nested_CV,
            fold_col=args.fold_col,
            main_metric=args.main_metric,
            sec_metrics=args.sec_metrics,
            parametric=args.parametric,
            fingerprint=args.fingerprint,
            incl_RDKit_feats=args.incl_RDKit_feats,
            scaler=args.scaler,
            custom_models=args.models if hasattr(args, "models") else None,
            n_jobs=args.n_jobs,
        )
    elif args.command == "compare":
        compare.run(
            CV_results=args.CV_results,
            main_metric=args.main_metric,
            sec_metrics=args.sec_metrics,
            n_folds=args.n_folds,
        )
    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
