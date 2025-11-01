import os
import pickle
import shutil
import subprocess

import pytest


@pytest.fixture
def temp_dir():
    dir_path = "temp_test_dir"
    os.makedirs(dir_path, exist_ok=True)
    yield dir_path
    shutil.rmtree(dir_path)


def create_model_data(dir_path, model_name, roc_auc, pr_auc):
    data = {"roc_auc": roc_auc, "pr_auc": pr_auc}
    with open(os.path.join(dir_path, f"{model_name}_final_CV.pkl"), "wb") as f:
        pickle.dump(data, f)


@pytest.fixture
def cv_results_single_dir_no_models(temp_dir):
    dir_path = os.path.join(temp_dir, "cv_results_no_models")
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


@pytest.fixture
def cv_results_single_dir_one_model(temp_dir):
    dir_path = os.path.join(temp_dir, "cv_results_one_model")
    os.makedirs(dir_path, exist_ok=True)
    create_model_data(
        dir_path,
        "model1",
        [0.9, 0.91, 0.92, 0.93, 0.94],
        [0.8, 0.81, 0.82, 0.83, 0.84],
    )
    return dir_path


@pytest.fixture
def cv_results_single_dir_two_models(temp_dir):
    dir_path = os.path.join(temp_dir, "cv_results_two_models")
    os.makedirs(dir_path, exist_ok=True)
    create_model_data(
        dir_path,
        "model1",
        [0.9, 0.91, 0.92, 0.93, 0.94],
        [0.8, 0.81, 0.82, 0.83, 0.84],
    )
    create_model_data(
        dir_path,
        "model2",
        [0.95, 0.96, 0.97, 0.98, 0.99],
        [0.85, 0.86, 0.87, 0.88, 0.89],
    )

    return dir_path


@pytest.fixture
def cv_results_single_dir(temp_dir):
    dir_path = os.path.join(temp_dir, "cv_results")
    os.makedirs(dir_path, exist_ok=True)
    create_model_data(
        dir_path,
        "model1",
        [0.9, 0.91, 0.92, 0.93, 0.94],
        [0.8, 0.81, 0.82, 0.83, 0.84],
    )
    create_model_data(
        dir_path,
        "model2",
        [0.95, 0.96, 0.97, 0.98, 0.99],
        [0.85, 0.86, 0.87, 0.88, 0.89],
    )
    create_model_data(
        dir_path,
        "model3",
        [0.88, 0.89, 0.90, 0.91, 0.92],
        [0.75, 0.76, 0.77, 0.78, 0.79],
    )
    create_model_data(
        dir_path,
        "model4",
        [0.93, 0.94, 0.95, 0.96, 0.97],
        [0.82, 0.83, 0.84, 0.85, 0.86],
    )
    create_model_data(
        dir_path,
        "model5",
        [0.91, 0.92, 0.93, 0.94, 0.95],
        [0.79, 0.80, 0.81, 0.82, 0.83],
    )

    return dir_path


@pytest.fixture
def cv_results_multiple_dirs(temp_dir):
    dir_paths = []
    for i, (roc_auc, pr_auc) in enumerate(
        [
            ([0.9, 0.91, 0.92, 0.93, 0.94], [0.8, 0.81, 0.82, 0.83, 0.84]),
            ([0.95, 0.96, 0.97, 0.98, 0.99], [0.85, 0.86, 0.87, 0.88, 0.89]),
            ([0.88, 0.89, 0.90, 0.91, 0.92], [0.75, 0.76, 0.77, 0.78, 0.79]),
            ([0.93, 0.94, 0.95, 0.96, 0.97], [0.82, 0.83, 0.84, 0.85, 0.86]),
            ([0.91, 0.92, 0.93, 0.94, 0.95], [0.79, 0.80, 0.81, 0.82, 0.83]),
        ]
    ):
        dir_path = os.path.join(temp_dir, f"model{i + 1}")
        os.makedirs(dir_path, exist_ok=True)
        create_model_data(dir_path, "CV_results", roc_auc, pr_auc)
        dir_paths.append(dir_path)
    return dir_paths


def run_command(command):
    return subprocess.run(command, capture_output=True, text=True)


def test_path_does_not_exist(temp_dir):
    command = [
        "astra",
        "compare",
        "non_existent_dir",
        "--main_metric",
        "roc_auc",
        "--sec_metrics",
        "pr_auc",
    ]
    result = run_command(command)
    assert "does not exist" in result.stderr


def test_directory_is_empty(temp_dir):
    dir_path = os.path.join(temp_dir, "empty_dir")
    os.makedirs(dir_path, exist_ok=True)
    command = [
        "astra",
        "compare",
        dir_path,
        "--main_metric",
        "roc_auc",
        "--sec_metrics",
        "pr_auc",
    ]
    result = run_command(command)
    assert "is empty" in result.stderr


def test_no_cv_results_found(temp_dir):
    dir_path = os.path.join(temp_dir, "no_results_dir")
    os.makedirs(dir_path, exist_ok=True)
    with open(os.path.join(dir_path, "some_file.txt"), "w") as f:
        f.write("dummy content")
    command = [
        "astra",
        "compare",
        dir_path,
        "--main_metric",
        "roc_auc",
        "--sec_metrics",
        "pr_auc",
    ]
    result = run_command(command)
    assert "No CV results found" in result.stderr


def test_only_one_cv_result_found(temp_dir):
    dir_path = os.path.join(temp_dir, "one_result_dir")
    os.makedirs(dir_path, exist_ok=True)
    data = {
        "roc_auc": [0.9, 0.91, 0.92, 0.93, 0.94],
        "pr_auc": [0.8, 0.81, 0.82, 0.83, 0.84],
    }
    with open(os.path.join(dir_path, "model1_final_CV.pkl"), "wb") as f:
        pickle.dump(data, f)
    command = [
        "astra",
        "compare",
        dir_path,
        "--main_metric",
        "roc_auc",
        "--sec_metrics",
        "pr_auc",
    ]
    result = run_command(command)
    assert "Only one CV result found" in result.stderr


def test_main_metric_missing(cv_results_single_dir):
    command = [
        "astra",
        "compare",
        cv_results_single_dir,
        "--main_metric",
        "accuracy",
        "--sec_metrics",
        "pr_auc",
    ]
    result = run_command(command)
    assert "does not contain results for accuracy" in result.stderr


def test_run_sec_metric_missing(cv_results_single_dir):
    command = [
        "astra",
        "compare",
        cv_results_single_dir,
        "--main_metric",
        "roc_auc",
        "--sec_metrics",
        "accuracy",
    ]
    result = run_command(command)
    assert "does not contain results for accuracy" in result.stderr


def test_invalid_parametric(cv_results_single_dir):
    command = [
        "astra",
        "compare",
        cv_results_single_dir,
        "--main_metric",
        "roc_auc",
        "--sec_metrics",
        "pr_auc",
        "--parametric",
        "invalid",
    ]
    result = run_command(command)
    assert "invalid choice: 'invalid'" in result.stderr


def test_two_models_single_dir(cv_results_single_dir_two_models):
    command = [
        "astra",
        "compare",
        cv_results_single_dir_two_models,
        "--main_metric",
        "roc_auc",
        "--sec_metrics",
        "pr_auc",
    ]
    result = run_command(command)
    assert result.returncode == 0


def test_multiple_models_single_dir(cv_results_single_dir):
    command = [
        "astra",
        "compare",
        cv_results_single_dir,
        "--main_metric",
        "roc_auc",
        "--sec_metrics",
        "pr_auc",
    ]
    result = run_command(command)
    assert result.returncode == 0


def test_two_models_two_dirs(cv_results_multiple_dirs):
    model_list = [
        cv_results_multiple_dirs[0],
        cv_results_multiple_dirs[1],
    ]
    command = [
        "astra",
        "compare",
        *model_list,
        "--main_metric",
        "roc_auc",
        "--sec_metrics",
        "pr_auc",
    ]
    result = run_command(command)
    assert result.returncode == 0


def test_multiple_models_multiple_dirs(cv_results_multiple_dirs):
    command = [
        "astra",
        "compare",
        *cv_results_multiple_dirs,
        "--main_metric",
        "roc_auc",
        "--sec_metrics",
        "pr_auc",
    ]
    result = run_command(command)
    assert result.returncode == 0


def test_two_models_single_dir_parametric_false(cv_results_single_dir_two_models):
    command = [
        "astra",
        "compare",
        cv_results_single_dir_two_models,
        "--main_metric",
        "roc_auc",
        "--sec_metrics",
        "pr_auc",
        "--parametric",
        "False",
    ]
    result = run_command(command)
    assert result.returncode == 0


def test_two_models_single_dir_parametric_true(cv_results_single_dir_two_models):
    command = [
        "astra",
        "compare",
        cv_results_single_dir_two_models,
        "--main_metric",
        "roc_auc",
        "--sec_metrics",
        "pr_auc",
        "--parametric",
        "True",
    ]
    result = run_command(command)
    assert result.returncode == 0


def test_multiple_models_single_dir_parametric_true(cv_results_single_dir):
    command = [
        "astra",
        "compare",
        cv_results_single_dir,
        "--main_metric",
        "roc_auc",
        "--sec_metrics",
        "pr_auc",
        "--parametric",
        "True",
    ]
    result = run_command(command)
    assert result.returncode == 0


def test_multiple_models_single_dir_parametric_false(cv_results_single_dir):
    command = [
        "astra",
        "compare",
        cv_results_single_dir,
        "--main_metric",
        "roc_auc",
        "--sec_metrics",
        "pr_auc",
        "--parametric",
        "False",
    ]
    result = run_command(command)
    assert result.returncode == 0


def test_two_models_two_dirs_parametric_false(cv_results_multiple_dirs):
    model_list = [
        cv_results_multiple_dirs[0],
        cv_results_multiple_dirs[1],
    ]
    command = [
        "astra",
        "compare",
        *model_list,
        "--main_metric",
        "roc_auc",
        "--sec_metrics",
        "pr_auc",
        "--parametric",
        "False",
    ]
    result = run_command(command)
    assert result.returncode == 0


def test_two_models_two_dirs_parametric_true(cv_results_multiple_dirs):
    model_list = [
        cv_results_multiple_dirs[0],
        cv_results_multiple_dirs[1],
    ]
    command = [
        "astra",
        "compare",
        *model_list,
        "--main_metric",
        "roc_auc",
        "--sec_metrics",
        "pr_auc",
        "--parametric",
        "True",
    ]
    result = run_command(command)
    assert result.returncode == 0


def test_multiple_models_multiple_dirs_parametric_true(cv_results_multiple_dirs):
    command = [
        "astra",
        "compare",
        *cv_results_multiple_dirs,
        "--main_metric",
        "roc_auc",
        "--sec_metrics",
        "pr_auc",
        "--parametric",
        "True",
    ]
    result = run_command(command)
    assert result.returncode == 0


def test_multiple_models_multiple_dirs_parametric_false(cv_results_multiple_dirs):
    command = [
        "astra",
        "compare",
        *cv_results_multiple_dirs,
        "--main_metric",
        "roc_auc",
        "--sec_metrics",
        "pr_auc",
        "--parametric",
        "False",
    ]
    result = run_command(command)
    assert result.returncode == 0
