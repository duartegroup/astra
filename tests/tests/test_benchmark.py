import logging
import os
import pickle
import subprocess
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from astra.benchmark import run

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def expected_output_files():
    return [
        "benchmark.log",
        "default_CV.pkl",
        "final_CV.pkl",
        "final_CV_hparam_search.csv",
        "final_hyperparameters.pkl",
        "final_model.pkl",
    ]


@pytest.fixture
def expected_output_files_repeated():
    return [
        "benchmark.log",
        "default_CV_all_folds.pkl",
        "default_CV_Fold_0.pkl",
        "default_CV_Fold_1.pkl",
        "default_CV_Fold_2.pkl",
        "default_CV_Fold_3.pkl",
        "default_CV_Fold_4.pkl",
    ]


@pytest.fixture
def single_fold_df():
    return pd.DataFrame(
        {
            "Features": [np.random.rand(3) for _ in range(10)],
            "Target": np.random.rand(10),
            "Fold_0": np.repeat([0, 1, 2, 3, 4], 2),
        }
    )


@pytest.fixture
def repeated_fold_df():
    return pd.DataFrame(
        {
            "Features": [np.random.rand(3) for _ in range(10)],
            "Target": np.random.rand(10),
            "Fold_0": np.repeat([0, 1, 2, 3, 4], 2),
            "Fold_1": np.repeat([0, 1, 2, 3, 4], 2),
        }
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def check_output_files(experiment_name, expected_files):
    for filename in expected_files:
        assert os.path.exists(f"results/{experiment_name}/{filename}")


def run_benchmark(command):
    subprocess.run(command, check=True)


def _setup_full_run_mocks(monkeypatch, data_df):
    """Wire up mocks for a complete non-repeated, non-nested benchmark run."""
    mock_search = MagicMock()
    mock_search.best_estimator_ = {"model": "fake"}  # picklable
    mock_search.best_params_ = {"fm__param": 1}
    mock_search.cv_results_ = {
        "mean_test_score": [0.1],
        "std_test_score": [0.01],
        "params": [{"fm__param": 1}],
        "rank_test_score": [1],
    }
    monkeypatch.setattr("astra.benchmark.get_data", lambda *_, **__: data_df)
    monkeypatch.setattr(
        "astra.benchmark.get_models",
        lambda **kw: ({"FM": MagicMock()}, {"FM": {}}, None),
    )
    monkeypatch.setattr(
        "astra.benchmark.run_CV",
        lambda **kw: {"FM": {"mse": [0.1] * 5, "r2": [0.9] * 5}},
    )
    monkeypatch.setattr("astra.benchmark.check_assumptions", lambda **kw: True)
    monkeypatch.setattr(
        "astra.benchmark.get_best_model", lambda **kw: ("FM", "mean CV score")
    )
    monkeypatch.setattr("astra.benchmark.get_best_hparams", lambda **kw: mock_search)
    monkeypatch.setattr(
        "astra.benchmark.get_scores",
        lambda *a, **kw: (
            {"mse": [0.1] * 5, "r2": [0.9] * 5},
            0.1,
            0.01,
            0.1,
            {"r2": [0.9, 0.01, 0.9]},
        ),
    )
    monkeypatch.setattr("astra.benchmark.print_final_results", lambda **kw: None)
    monkeypatch.setattr("astra.benchmark.get_estimator_name", lambda x: "FM")


def _setup_repeated_cv_mocks(monkeypatch, data_df):
    """Wire up mocks for a repeated-CV benchmark run (no hparam tuning)."""
    monkeypatch.setattr("astra.benchmark.get_data", lambda *_, **__: data_df)
    monkeypatch.setattr(
        "astra.benchmark.get_models",
        lambda **kw: ({"FM": MagicMock()}, {"FM": {}}, None),
    )
    monkeypatch.setattr("astra.benchmark.check_assumptions", lambda **kw: True)
    monkeypatch.setattr(
        "astra.benchmark.get_best_model", lambda **kw: ("FM", "mean CV score")
    )


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


def test_basic_benchmark_config(expected_output_files_repeated):
    command = [
        "astra",
        "benchmark",
        "--config",
        "configs/example.yml",
    ]
    run_benchmark(command)
    check_output_files("example_experiment", expected_output_files_repeated)


def test_basic_benchmark(expected_output_files):
    os.makedirs("results/example_experiment_basic", exist_ok=True)
    with open("results/example_experiment_basic/unit_test.log", "w") as f:
        f.write("Dummy log file for test mode.\n")
    run(
        data="astra/data/example_df.csv",
        name="example_experiment_basic",
        main_metric="mse",
        sec_metrics=["r2"],
        scaler="Standard",
        fold_col="Fold_0",
        test_mode=True,
    )
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


def test_benchmark_optuna(expected_output_files):
    os.makedirs("results/example_experiment_optuna", exist_ok=True)
    with open("results/example_experiment_optuna/unit_test.log", "w") as f:
        f.write("Dummy log file for test mode.\n")
    run(
        data="astra/data/example_df.csv",
        name="example_experiment_optuna",
        use_optuna=True,
        main_metric="mse",
        sec_metrics=["r2"],
        scaler="Standard",
        fold_col="Fold_0",
        test_mode=True,
    )
    command = [
        "astra",
        "benchmark",
        "astra/data/example_df.csv",
        "--name",
        "example_experiment_optuna",
        "--use_optuna",
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
    check_output_files("example_experiment_optuna", expected_output_files)


def test_benchmark_repeated_CV(expected_output_files_repeated):
    os.makedirs("results/example_experiment_repeated", exist_ok=True)
    with open("results/example_experiment_repeated/unit_test.log", "w") as f:
        f.write("Dummy log file for test mode.\n")
    run(
        data="astra/data/example_df.csv",
        name="example_experiment_repeated",
        main_metric="mse",
        sec_metrics=["r2"],
        scaler="Standard",
        fold_col=["Fold_0", "Fold_1", "Fold_2", "Fold_3", "Fold_4"],
        test_mode=True,
    )
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
        "Fold_0",
        "Fold_1",
        "Fold_2",
        "Fold_3",
        "Fold_4",
    ]
    run_benchmark(command)
    check_output_files("example_experiment_repeated", expected_output_files_repeated)


def test_benchmark_nested_CV(expected_output_files):
    os.makedirs("results/example_experiment_nested", exist_ok=True)
    with open("results/example_experiment_nested/unit_test.log", "w") as f:
        f.write("Dummy log file for test mode.\n")
    run(
        data="astra/data/example_df.csv",
        name="example_experiment_nested",
        run_nested_CV=True,
        main_metric="mse",
        sec_metrics=["r2"],
        scaler="Standard",
        fold_col="Fold_0",
        test_mode=True,
        n_jobs=4,
    )
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


# ---------------------------------------------------------------------------
# Validation-error tests
# ---------------------------------------------------------------------------


def test_benchmark_invalid_parametric(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="`parametric` must be one of"):
        run(data="dummy.csv", name="test_invalid_parametric", parametric="invalid")


def test_benchmark_invalid_fold_col_type(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data_df = pd.DataFrame(
        {
            "Features": [np.random.rand(3) for _ in range(10)],
            "Target": np.random.rand(10),
            "Fold": np.repeat([0, 1, 2, 3, 4], 2),
        }
    )
    monkeypatch.setattr("astra.benchmark.get_data", lambda *_, **__: data_df)
    with pytest.raises(ValueError, match="`fold_col` must be a string"):
        run(
            data="dummy.csv",
            name="test_invalid_fold_col",
            fold_col=123,  # type: ignore[arg-type]
            main_metric="mse",
            sec_metrics=["r2"],
        )


def test_benchmark_invalid_impute_type(tmp_path, monkeypatch, single_fold_df):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("astra.benchmark.get_data", lambda *_, **__: single_fold_df)
    with pytest.raises(ValueError, match="`impute` must be a string or a number"):
        run(
            data="dummy.csv",
            name="test_invalid_impute",
            fold_col="Fold_0",
            main_metric="mse",
            sec_metrics=["r2"],
            impute=[],  # type: ignore[arg-type]
        )


def test_benchmark_invalid_remove_constant(tmp_path, monkeypatch, single_fold_df):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("astra.benchmark.get_data", lambda *_, **__: single_fold_df)
    with pytest.raises(ValueError, match="`remove_constant` must be a float"):
        run(
            data="dummy.csv",
            name="test_invalid_remove_constant",
            fold_col="Fold_0",
            main_metric="mse",
            sec_metrics=["r2"],
            remove_constant=1,  # type: ignore[arg-type]
        )


def test_benchmark_invalid_remove_correlated(tmp_path, monkeypatch, single_fold_df):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("astra.benchmark.get_data", lambda *_, **__: single_fold_df)
    with pytest.raises(ValueError, match="`remove_correlated` must be a float"):
        run(
            data="dummy.csv",
            name="test_invalid_remove_correlated",
            fold_col="Fold_0",
            main_metric="mse",
            sec_metrics=["r2"],
            remove_correlated=2,  # type: ignore[arg-type]
        )


def test_benchmark_invalid_scaler(tmp_path, monkeypatch, single_fold_df):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("astra.benchmark.get_data", lambda *_, **__: single_fold_df)
    with pytest.raises(ValueError, match="`scaler` must be one of"):
        run(
            data="dummy.csv",
            name="test_invalid_scaler",
            fold_col="Fold_0",
            main_metric="mse",
            sec_metrics=["r2"],
            scaler="bad_scaler",
        )


# ---------------------------------------------------------------------------
# Full-run tests (mocked internals) — cover logging and post-CV branches
# ---------------------------------------------------------------------------


def test_benchmark_parametric_true(tmp_path, monkeypatch, single_fold_df):
    monkeypatch.chdir(tmp_path)
    _setup_full_run_mocks(monkeypatch, single_fold_df)
    run(
        data="dummy.csv",
        name="test_parametric_true",
        parametric=True,
        main_metric="mse",
        sec_metrics=["r2"],
        fold_col="Fold_0",
        test_mode=True,
    )


def test_benchmark_parametric_false(tmp_path, monkeypatch, single_fold_df):
    monkeypatch.chdir(tmp_path)
    _setup_full_run_mocks(monkeypatch, single_fold_df)
    run(
        data="dummy.csv",
        name="test_parametric_false",
        parametric=False,
        main_metric="mse",
        sec_metrics=["r2"],
        fold_col="Fold_0",
        test_mode=True,
    )


def test_benchmark_preprocessing_logging(tmp_path, monkeypatch, single_fold_df):
    monkeypatch.chdir(tmp_path)
    _setup_full_run_mocks(monkeypatch, single_fold_df)
    run(
        data="dummy.csv",
        name="test_preprocessing_logging",
        main_metric="mse",
        sec_metrics=["r2"],
        fold_col="Fold_0",
        impute="mean",
        remove_constant=0.0,
        remove_correlated=0.9,
        scaler="Standard",
        test_mode=True,
    )


def test_benchmark_impute_numeric_logging(tmp_path, monkeypatch, single_fold_df):
    monkeypatch.chdir(tmp_path)
    _setup_full_run_mocks(monkeypatch, single_fold_df)
    run(
        data="dummy.csv",
        name="test_impute_numeric",
        main_metric="mse",
        sec_metrics=["r2"],
        fold_col="Fold_0",
        impute=0.5,
        test_mode=True,
    )


# ---------------------------------------------------------------------------
# Repeated-CV tests
# ---------------------------------------------------------------------------


def test_benchmark_repeated_cv_cached(tmp_path, monkeypatch, repeated_fold_df):
    monkeypatch.chdir(tmp_path)

    fake_results = {"FM": {"mse": [0.1] * 10, "r2": [0.9] * 10}}
    os.makedirs("results/test_rc_cached", exist_ok=True)
    with open("results/test_rc_cached/default_CV_all_folds.pkl", "wb") as fh:
        pickle.dump(fake_results, fh)

    _setup_repeated_cv_mocks(monkeypatch, repeated_fold_df)

    run(
        data="dummy.csv",
        name="test_rc_cached",
        main_metric="mse",
        sec_metrics=["r2"],
        fold_col=["Fold_0", "Fold_1"],
        test_mode=True,
    )

    assert os.path.exists("results/test_rc_cached/best_model.txt")


def test_benchmark_nested_cv_repeated_cv_warning(
    tmp_path, monkeypatch, repeated_fold_df, caplog
):
    monkeypatch.chdir(tmp_path)

    fake_results = {"FM": {"mse": [0.1] * 10, "r2": [0.9] * 10}}
    os.makedirs("results/test_ncv_rc", exist_ok=True)
    with open("results/test_ncv_rc/default_CV_all_folds.pkl", "wb") as fh:
        pickle.dump(fake_results, fh)

    _setup_repeated_cv_mocks(monkeypatch, repeated_fold_df)

    with caplog.at_level(logging.WARNING):
        run(
            data="dummy.csv",
            name="test_ncv_rc",
            main_metric="mse",
            sec_metrics=["r2"],
            fold_col=["Fold_0", "Fold_1"],
            run_nested_CV=True,
            test_mode=True,
        )

    assert "run_nested_CV=True is not supported with repeated CV" in caplog.text
