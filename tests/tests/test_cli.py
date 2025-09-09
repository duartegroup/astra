import os
import subprocess


def test_benchmark_cli():
    # Run the benchmark command
    command = [
        "astra",
        "benchmark",
        "astra/data/example_df.csv",
        "--name",
        "example_experiment",
        "--main_metric",
        "MSE",
        "--sec_metrics",
        "R2",
        "--scaler",
        "Standard",
        "--fold_col",
        "Fold_0",
    ]
    subprocess.run(command, check=True)

    # Check if the output files are created
    assert os.path.exists("results/example_experiment/benchmark.log")
    assert os.path.exists("results/example_experiment/default_CV.pkl")
    assert os.path.exists("results/example_experiment/final_CV_results.csv")
    assert os.path.exists("results/example_experiment/final_hyperparameters.pkl")
    assert os.path.exists("results/example_experiment/final_model.pkl")
