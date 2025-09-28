import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from astra.model_selection import (
    check_assumptions,
    check_best_model,
    find_n_best_models,
    get_best_hparams,
    get_best_model,
    get_cv_performance,
    get_optimised_cv_performance,
    perform_statistical_tests,
    run_CV,
    tukey_hsd,
)


@pytest.fixture
def results_dict():
    return {
        "model1": {
            "accuracy": [0.8, 0.82, 0.81, 0.83, 0.805],
            "f1": [0.7, 0.72, 0.71, 0.73, 0.705],
        },
        "model2": {
            "accuracy": [0.9, 0.91, 0.92, 0.905, 0.915],
            "f1": [0.8, 0.81, 0.82, 0.805, 0.815],
        },
        "model3": {
            "accuracy": [0.85, 0.86, 0.87, 0.855, 0.865],
            "f1": [0.75, 0.76, 0.77, 0.755, 0.765],
        },
    }


@pytest.fixture
def classification_df():
    # Ensure each fold has both classes
    features = [np.random.rand(5) for _ in range(20)]
    targets = [0] * 10 + [1] * 10
    folds = [0, 1, 2, 3, 4] * 4
    return pd.DataFrame({"Features": features, "Target": targets, "Fold": folds})


@pytest.fixture
def regression_df():
    return pd.DataFrame(
        {
            "Features": [np.random.rand(5) for _ in range(10)],
            "Target": np.random.rand(10),
            "Fold": np.repeat([0, 1, 2, 3, 4], 2),
        }
    )


def test_check_assumptions(results_dict):
    # Test with normal data
    assert check_assumptions(results_dict, verbose=False)

    # Test with non-normal data
    non_normal_results = results_dict.copy()
    # Uniform distribution is non-normal
    non_normal_results["model1"]["accuracy"] = np.random.uniform(0, 1, 30).tolist()
    assert not check_assumptions(non_normal_results, verbose=False)

    # Test with different metrics
    different_metrics_results = results_dict.copy()
    different_metrics_results["model1"] = {"precision": [0.8, 0.82, 0.81, 0.83, 0.805]}
    with pytest.raises(ValueError):
        check_assumptions(different_metrics_results)


def test_tukey_hsd():
    mse = 0.01
    residual_dof = 10
    score_means = pd.Series([0.8, 0.9, 0.85], index=["model1", "model2", "model3"])
    n_folds = 5
    p_values = tukey_hsd(mse, residual_dof, score_means, n_folds)
    assert isinstance(p_values, pd.DataFrame)
    assert p_values.shape == (3, 3)
    assert np.all(np.diag(p_values) == 1.0)


def test_find_n_best_models(results_dict):
    best_models = find_n_best_models(results_dict, "accuracy")
    assert isinstance(best_models, list)
    assert len(best_models) > 0
    assert "model2" in best_models
    best_models_parametric = find_n_best_models(
        results_dict, "accuracy", parametric=True
    )
    assert best_models == best_models_parametric


def test_perform_statistical_tests(results_dict):
    post_hoc, naive = perform_statistical_tests(results_dict, "accuracy")
    assert isinstance(post_hoc, pd.DataFrame)
    assert isinstance(naive, pd.DataFrame)
    assert post_hoc.shape == (3, 3)
    assert naive.shape == (3, 3)


def test_check_best_model(results_dict):
    post_hoc, naive_stats = perform_statistical_tests(results_dict, "accuracy")
    best_model = check_best_model(results_dict, post_hoc, "accuracy")
    assert best_model == "model2"
    best_model_naive = check_best_model(results_dict, naive_stats, "accuracy")
    assert best_model_naive == "model2"


def test_get_best_model(results_dict):
    best_model, reason = get_best_model(results_dict, "accuracy", ["f1"])
    assert best_model == "model2"
    assert isinstance(reason, str)


def test_get_cv_performance(classification_df, regression_df):
    # Classification
    metrics = get_cv_performance(
        LogisticRegression(),
        classification_df,
        "Features",
        "Target",
        "Fold",
        ["accuracy", "f1"],
    )
    assert "accuracy" in metrics
    assert "f1" in metrics
    assert len(metrics["accuracy"]) == 5

    # Regression
    metrics = get_cv_performance(
        LinearRegression(),
        regression_df,
        "Features",
        "Target",
        "Fold",
        ["mse", "r2"],
    )
    assert "mse" in metrics
    assert "r2" in metrics
    assert len(metrics["mse"]) == 5


def test_run_CV(classification_df, tmp_path, monkeypatch):
    # Mock get_cv_performance to avoid long run times
    def mock_get_cv_performance(*args, **kwargs):
        return {"accuracy": [0.8, 0.9, 0.85, 0.88, 0.91]}

    monkeypatch.setattr(
        "astra.model_selection.get_cv_performance", mock_get_cv_performance
    )

    # Create dummy directories
    (tmp_path / "results" / "test_exp").mkdir(parents=True)
    (tmp_path / "cache").mkdir(parents=True)

    # Change to tmp_path directory
    monkeypatch.chdir(tmp_path)

    models = {"LogisticRegression": LogisticRegression()}
    results = run_CV(
        "test_exp",
        classification_df,
        "Features",
        "Target",
        "Fold",
        models,
        ["accuracy"],
    )
    assert "LogisticRegression" in results
    assert "accuracy" in results["LogisticRegression"]


def test_get_optimised_cv_performance(classification_df, monkeypatch):
    # Mock GridSearchCV to avoid long run times
    class MockGridSearchCV:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, *args, **kwargs):
            pass

        def predict(self, X):
            return np.zeros(X.shape[0])

        def predict_proba(self, X):
            return np.zeros((X.shape[0], 2))

    monkeypatch.setattr("astra.model_selection.GridSearchCV", MockGridSearchCV)

    metrics = get_optimised_cv_performance(
        LogisticRegression(),
        classification_df,
        "Features",
        "Target",
        "Fold",
        ["accuracy", "roc_auc"],
        "accuracy",
        {"C": [0.1, 1]},
        1,
    )
    assert "accuracy" in metrics
    assert "roc_auc" in metrics
    assert len(metrics["accuracy"]) == 5


def test_get_best_hparams(classification_df, monkeypatch):
    # Mock GridSearchCV to avoid long run times
    class MockGridSearchCV:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, *args, **kwargs):
            pass

    monkeypatch.setattr("astra.model_selection.GridSearchCV", MockGridSearchCV)

    clf = get_best_hparams(
        LogisticRegression(),
        classification_df,
        "Features",
        "Target",
        "Fold",
        "accuracy",
        ["f1"],
        {"C": [0.1, 1]},
        1,
    )
    assert isinstance(clf, MockGridSearchCV)


def test_tukey_hsd_correctness():
    mse = 2.66
    residual_dof = 12
    score_means = pd.Series([87.75, 89, 94], index=["A", "B", "C"])
    n_folds = 5
    p_values = tukey_hsd(mse, residual_dof, score_means, n_folds)
    assert p_values.loc["A", "C"] < 0.05
    assert p_values.loc["A", "B"] > 0.05


def test_perform_statistical_tests_correctness():
    results = {
        "modelA": {"score": [1, 2, 3, 4, 5]},
        "modelB": {"score": [2, 3, 4, 5, 6]},
    }
    post_hoc, naive = perform_statistical_tests(results, "score", parametric=True)
    assert post_hoc.iloc[0, 1] < 0.05
    assert naive.iloc[0, 1] < 0.05
