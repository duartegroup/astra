import os
import pickle
import subprocess

import pytest

from astra.compare import run

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def create_model_data(dir_path, model_name, roc_auc, pr_auc):
    data = {"roc_auc": roc_auc, "pr_auc": pr_auc}
    with open(os.path.join(dir_path, f"{model_name}_final_CV.pkl"), "wb") as f:
        pickle.dump(data, f)


def run_command(command):
    return subprocess.run(command, capture_output=True, text=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cv_results_single_dir_two_models(tmp_path):
    dir_path = str(tmp_path / "cv_results_two_models")
    os.makedirs(dir_path)
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
def cv_results_single_dir(tmp_path):
    dir_path = str(tmp_path / "cv_results")
    os.makedirs(dir_path)
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
def cv_results_multiple_dirs(tmp_path):
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
        dir_path = str(tmp_path / f"model{i + 1}")
        os.makedirs(dir_path)
        create_model_data(dir_path, "CV_results", roc_auc, pr_auc)
        dir_paths.append(dir_path)
    return dir_paths


# ---------------------------------------------------------------------------
# Validation-error tests
# ---------------------------------------------------------------------------


def test_path_does_not_exist():
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
    with pytest.raises(ValueError, match="does not exist"):
        run(["non_existent_dir"], "roc_auc", ["pr_auc"])


def test_directory_is_empty(tmp_path):
    dir_path = str(tmp_path / "empty_dir")
    os.makedirs(dir_path)
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
    with pytest.raises(ValueError, match="is empty"):
        run([dir_path], "roc_auc", ["pr_auc"])


def test_no_cv_results_found(tmp_path):
    dir_path = str(tmp_path / "no_results_dir")
    os.makedirs(dir_path)
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
    with pytest.raises(ValueError, match="No CV results found"):
        run([dir_path], "roc_auc", ["pr_auc"])


def test_only_one_cv_result_found(tmp_path):
    dir_path = str(tmp_path / "one_result_dir")
    os.makedirs(dir_path)
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
    with pytest.raises(ValueError, match="Only one CV result found"):
        run([dir_path], "roc_auc", ["pr_auc"])


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
    with pytest.raises(AssertionError, match="does not contain results for accuracy"):
        run([cv_results_single_dir], "accuracy", ["pr_auc"])


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
    with pytest.raises(AssertionError, match="does not contain results for accuracy"):
        run([cv_results_single_dir], "roc_auc", ["accuracy"])


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
    with pytest.raises(ValueError, match="Got invalid instead."):
        run([cv_results_single_dir], "roc_auc", ["pr_auc"], parametric="invalid")


# ---------------------------------------------------------------------------
# Integration tests — single directory
# ---------------------------------------------------------------------------


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
    run([cv_results_single_dir_two_models], "roc_auc", ["pr_auc"])


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
    run([cv_results_single_dir_two_models], "roc_auc", ["pr_auc"], parametric=False)


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
    run([cv_results_single_dir_two_models], "roc_auc", ["pr_auc"], parametric=True)


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
    run([cv_results_single_dir], "roc_auc", ["pr_auc"])


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
    run([cv_results_single_dir], "roc_auc", ["pr_auc"], parametric=True)


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
    run([cv_results_single_dir], "roc_auc", ["pr_auc"], parametric=False)


# ---------------------------------------------------------------------------
# Integration tests — multiple directories
# ---------------------------------------------------------------------------


def test_two_models_two_dirs(cv_results_multiple_dirs):
    model_list = cv_results_multiple_dirs[:2]
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
    run(cv_results_multiple_dirs, "roc_auc", ["pr_auc"])


def test_two_models_two_dirs_parametric_false(cv_results_multiple_dirs):
    model_list = cv_results_multiple_dirs[:2]
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
    run(model_list, "roc_auc", ["pr_auc"], parametric=False)


def test_two_models_two_dirs_parametric_true(cv_results_multiple_dirs):
    model_list = cv_results_multiple_dirs[:2]
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
    run(model_list, "roc_auc", ["pr_auc"], parametric=True)


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
    run(cv_results_multiple_dirs, "roc_auc", ["pr_auc"])


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
    run(cv_results_multiple_dirs, "roc_auc", ["pr_auc"], parametric=True)


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
    run(cv_results_multiple_dirs, "roc_auc", ["pr_auc"], parametric=False)
