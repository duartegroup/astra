import os
import subprocess

import pytest


@pytest.fixture
def expected_output_files():
    return [
        "benchmark.log",
        "default_CV.pkl",
        "final_CV_results.csv",
        "final_hyperparameters.pkl",
        "final_model.pkl",
    ]


def check_output_files(experiment_name, expected_files):
    for filename in expected_files:
        assert os.path.exists(f"results/{experiment_name}/{filename}")


def run_benchmark(command):
    subprocess.run(command, check=True)


def test_basic_benchmark_config(expected_output_files):
    command = [
        "astra",
        "benchmark",
        "--config",
        "configs/example.yml",
    ]
    run_benchmark(command)
    check_output_files("example_experiment", expected_output_files)


def test_basic_benchmark(expected_output_files):
    command = [
        "astra",
        "benchmark",
        "astra/data/example_df.csv",
        "--name",
        "example_experiment_basic",
        "--main_metric",
        "MSE",
        "--sec_metrics",
        "R2",
        "--scaler",
        "Standard",
        "--fold_col",
        "Fold_0",
    ]
    run_benchmark(command)
    check_output_files("example_experiment_basic", expected_output_files)


def test_benchmark_repeated_CV(expected_output_files):
    command = [
        "astra",
        "benchmark",
        "astra/data/example_df.csv",
        "--name",
        "example_experiment_repeated",
        "--main_metric",
        "MSE",
        "--sec_metrics",
        "R2",
        "--scaler",
        "Standard",
        "--fold_col",
        "Fold_0 Fold_1 Fold_2 Fold_3 Fold_4",
    ]
    run_benchmark(command)
    check_output_files("example_experiment_repeated", expected_output_files)


def test_benchmark_nested_CV(expected_output_files):
    command = [
        "astra",
        "benchmark",
        "astra/data/example_df.csv",
        "--name",
        "example_experiment_nested",
        "--run_nested_CV",
        "--main_metric",
        "MSE",
        "--sec_metrics",
        "R2",
        "--scaler",
        "Standard",
        "--fold_col",
        "Fold_0",
    ]
    run_benchmark(command)
    check_output_files("example_experiment_nested", expected_output_files)
