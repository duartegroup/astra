import logging
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC

from astra.model_selection import (
    build_equivalent_ensemble,
    check_assumptions,
    check_best_model,
    check_pareto_dominant,
    corrected_ttest,
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


@pytest.fixture
def pareto_results():
    return {
        "strong": {
            "accuracy": [0.95, 0.93, 0.97, 0.91, 0.96, 0.94, 0.98, 0.92, 0.95, 0.97],
            "f1": [0.92, 0.90, 0.94, 0.88, 0.93, 0.91, 0.95, 0.89, 0.92, 0.94],
        },
        "weak": {
            "accuracy": [0.65, 0.67, 0.63, 0.70, 0.64, 0.66, 0.62, 0.68, 0.65, 0.67],
            "f1": [0.60, 0.62, 0.58, 0.65, 0.59, 0.61, 0.57, 0.63, 0.60, 0.62],
        },
        "mid": {
            "accuracy": [0.78, 0.80, 0.76, 0.82, 0.77, 0.79, 0.75, 0.81, 0.78, 0.80],
            "f1": [0.73, 0.75, 0.71, 0.77, 0.72, 0.74, 0.70, 0.76, 0.73, 0.75],
        },
    }


# ---------------------------------------------------------------------------
# check_assumptions
# ---------------------------------------------------------------------------


def test_check_assumptions(results_dict):
    # Test with normal data
    assert check_assumptions(results_dict, verbose=False)

    # Test with non-normal data
    non_normal_results = results_dict.copy()
    non_normal_results["model1"]["accuracy"] = np.random.uniform(0, 1, 30).tolist()
    assert not check_assumptions(non_normal_results, verbose=False)

    # Test with different metrics
    different_metrics_results = results_dict.copy()
    different_metrics_results["model1"] = {"precision": [0.8, 0.82, 0.81, 0.83, 0.805]}
    with pytest.raises(ValueError):
        check_assumptions(different_metrics_results)


def test_check_assumptions_verbose(capsys):
    # Fewer than 8 folds triggers the low-power print
    small = {
        "m1": {"accuracy": [0.8, 0.82, 0.81, 0.83, 0.80]},
        "m2": {"accuracy": [0.9, 0.91, 0.92, 0.905, 0.91]},
    }
    check_assumptions(small, verbose=True)
    assert "Warning" in capsys.readouterr().out

    # Bimodal two-value distribution is non-normal, triggers Shapiro-Wilk warning
    bimodal = [0.1] * 15 + [0.9] * 15
    non_normal = {
        "m1": {"accuracy": bimodal},
        "m2": {"accuracy": [0.2] * 15 + [0.8] * 15},
    }
    check_assumptions(non_normal, verbose=True)
    assert "Warning" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# tukey_hsd
# ---------------------------------------------------------------------------


def test_tukey_hsd():
    mse = 0.01
    residual_dof = 10
    score_means = pd.Series([0.8, 0.9, 0.85], index=["model1", "model2", "model3"])
    n_folds = 5
    p_values = tukey_hsd(mse, residual_dof, score_means, n_folds)
    assert isinstance(p_values, pd.DataFrame)
    assert p_values.shape == (3, 3)
    assert np.all(np.diag(p_values) == 1.0)


def test_tukey_hsd_correctness():
    mse = 2.66
    residual_dof = 12
    score_means = pd.Series([87.75, 89, 94], index=["A", "B", "C"])
    n_folds = 5
    p_values = tukey_hsd(mse, residual_dof, score_means, n_folds)
    assert p_values.loc["A", "C"] < 0.05
    assert p_values.loc["A", "B"] > 0.05


# ---------------------------------------------------------------------------
# corrected_ttest
# ---------------------------------------------------------------------------


def test_corrected_ttest_identical_arrays():
    a = np.array([0.8, 0.82, 0.81, 0.83, 0.805])
    assert corrected_ttest(a, a) == 1.0


def test_corrected_ttest_clearly_different():
    a = np.array([0.90, 0.88, 0.93, 0.86, 0.94])
    b = np.array([0.30, 0.25, 0.35, 0.20, 0.40])
    p = corrected_ttest(a, b)
    assert isinstance(p, float)
    assert 0.0 <= p <= 1.0
    assert p < 0.05


def test_corrected_ttest_n_folds_parameter():
    # Non-constant differences so both calls give a proper t-statistic.
    a = np.array([0.90, 0.88, 0.93, 0.86, 0.94, 0.87, 0.91, 0.85, 0.92, 0.89])
    b = np.array([0.30, 0.25, 0.35, 0.20, 0.40, 0.28, 0.33, 0.22, 0.38, 0.26])
    p_default = corrected_ttest(a, b)
    p_with_folds = corrected_ttest(a, b, n_folds=5)
    assert isinstance(p_default, float)
    assert isinstance(p_with_folds, float)
    # n_folds changes the rho correction factor, so p-values must differ.
    assert p_default != p_with_folds


def test_corrected_ttest_returns_valid_pvalue():
    rng = np.random.default_rng(42)
    a = rng.normal(0.8, 0.01, 10)
    b = rng.normal(0.75, 0.01, 10)
    p = corrected_ttest(a, b)
    assert 0.0 <= p <= 1.0


# ---------------------------------------------------------------------------
# _min_detectable_effect
# ---------------------------------------------------------------------------


def test_min_detectable_effect():
    from astra.model_selection import _min_detectable_effect

    mde = _min_detectable_effect(n_total=10, n_folds=5, alpha=0.05)
    assert isinstance(mde, float)
    assert mde > 0
    # More data leads to a smaller minimum detectable effect
    assert _min_detectable_effect(n_total=50, n_folds=5, alpha=0.05) < mde


# ---------------------------------------------------------------------------
# find_n_best_models
# ---------------------------------------------------------------------------


def test_find_n_best_models(results_dict):
    best_models = find_n_best_models(results_dict, "accuracy")
    assert isinstance(best_models, list)
    assert len(best_models) > 0
    assert "model2" in best_models
    best_models_parametric = find_n_best_models(
        results_dict, "accuracy", parametric=True
    )
    assert "model2" in best_models_parametric


def test_find_n_best_models_no_bonferroni(results_dict):
    # Exercises the bf_corr=False branch
    best = find_n_best_models(results_dict, "accuracy", bf_corr=False)
    assert isinstance(best, list)
    assert len(best) > 0


def test_find_n_best_models_no_significant_difference():
    # Perfectly alternating scores -> Friedman p = 1 -> both models retained
    close = {
        "m1": {
            "accuracy": [
                0.801,
                0.799,
                0.801,
                0.799,
                0.801,
                0.799,
                0.801,
                0.799,
                0.801,
                0.799,
            ]
        },
        "m2": {
            "accuracy": [
                0.799,
                0.801,
                0.799,
                0.801,
                0.799,
                0.801,
                0.799,
                0.801,
                0.799,
                0.801,
            ]
        },
    }
    best = find_n_best_models(close, "accuracy")
    assert "m1" in best
    assert "m2" in best


# ---------------------------------------------------------------------------
# perform_statistical_tests
# ---------------------------------------------------------------------------


def test_perform_statistical_tests(results_dict):
    post_hoc, naive = perform_statistical_tests(results_dict, "accuracy")
    assert isinstance(post_hoc, pd.DataFrame)
    assert isinstance(naive, pd.DataFrame)
    assert post_hoc.shape == (3, 3)
    assert naive.shape == (3, 3)


def test_perform_statistical_tests_correctness():
    # Non-constant differences so corrected t-test variance > 0 and p < 0.05
    results = {
        "modelA": {"mae": [1.0, 1.5, 2.0, 1.8, 1.2]},
        "modelB": {"mae": [3.5, 4.5, 3.0, 5.0, 4.0]},
    }
    post_hoc, naive = perform_statistical_tests(results, "mae", parametric=True)
    assert post_hoc.iloc[0, 1] < 0.05
    assert naive.iloc[0, 1] < 0.05


def test_perform_statistical_tests_all_equal():
    # Identical scores -> Wilcoxon guard fires and returns p = 1.0
    results = {
        "m1": {"accuracy": [0.8, 0.8, 0.8, 0.8, 0.8]},
        "m2": {"accuracy": [0.8, 0.8, 0.8, 0.8, 0.8]},
    }
    _, naive = perform_statistical_tests(results, "accuracy")
    assert naive.iloc[0, 1] == 1.0


# ---------------------------------------------------------------------------
# check_best_model
# ---------------------------------------------------------------------------


def test_check_best_model(results_dict):
    post_hoc, naive_stats = perform_statistical_tests(results_dict, "accuracy")
    best_model = check_best_model(results_dict, post_hoc, "accuracy")
    assert best_model == "model2"
    best_model_naive = check_best_model(results_dict, naive_stats, "accuracy")
    assert best_model_naive in [None, "model2"]
    post_hoc_parametric, naive_parametric = perform_statistical_tests(
        results_dict, "accuracy", parametric=True
    )
    best_model_parametric = check_best_model(
        results_dict, post_hoc_parametric, "accuracy"
    )
    assert best_model_parametric == "model2"
    best_model_naive_parametric = check_best_model(
        results_dict, naive_parametric, "accuracy"
    )
    assert best_model_naive_parametric == "model2"


def test_check_best_model_minimization_metric():
    # Exercises the `else` branch for lower-is-better metrics
    results = {
        "low_error": {
            "mae": [0.10, 0.12, 0.09, 0.11, 0.10, 0.12, 0.09, 0.11, 0.10, 0.12]
        },
        "high_error": {
            "mae": [0.50, 0.52, 0.48, 0.53, 0.49, 0.51, 0.47, 0.54, 0.50, 0.51]
        },
    }
    _, naive = perform_statistical_tests(results, "mae")
    best = check_best_model(results, naive, "mae", use_mean=False)
    assert best == "low_error"


# ---------------------------------------------------------------------------
# check_pareto_dominant
# ---------------------------------------------------------------------------


def test_check_pareto_dominant_finds_dominant(pareto_results):
    result = check_pareto_dominant(pareto_results, "accuracy", ["f1"], parametric=False)
    assert result == "strong"


def test_check_pareto_dominant_none_when_equivalent():
    # Nearly identical scores -> no significant differences -> no Pareto dominant model
    equiv = {
        "m1": {
            "accuracy": [0.800, 0.800, 0.800, 0.800, 0.800],
            "f1": [0.700, 0.700, 0.700, 0.700, 0.700],
        },
        "m2": {
            "accuracy": [0.801, 0.800, 0.800, 0.800, 0.800],
            "f1": [0.701, 0.700, 0.700, 0.700, 0.700],
        },
    }
    result = check_pareto_dominant(equiv, "accuracy", ["f1"], parametric=False)
    assert result is None


def test_check_pareto_dominant_no_secondary_metrics(pareto_results):
    result = check_pareto_dominant(pareto_results, "accuracy", [], parametric=False)
    assert result == "strong"


def test_check_pareto_dominant_returns_str_or_none(pareto_results):
    result = check_pareto_dominant(pareto_results, "accuracy", ["f1"], parametric=False)
    assert result is None or isinstance(result, str)


def test_check_pareto_dominant_ignores_missing_secondary():
    # Secondary metric absent from results dict is silently skipped
    results = {
        "m1": {"accuracy": [0.9, 0.91, 0.92, 0.905, 0.915]},
        "m2": {"accuracy": [0.7, 0.71, 0.72, 0.705, 0.715]},
    }
    result = check_pareto_dominant(results, "accuracy", ["f1"], parametric=False)
    assert result is None or isinstance(result, str)


def test_check_pareto_dominant_significantly_worse():
    # "bad" is significantly worse -> is_significantly_worse=True, not Pareto dominant
    results = {
        "good": {
            "accuracy": [0.95, 0.93, 0.97, 0.91, 0.96, 0.94, 0.98, 0.92, 0.95, 0.97]
        },
        "bad": {
            "accuracy": [0.50, 0.52, 0.48, 0.53, 0.49, 0.51, 0.47, 0.54, 0.50, 0.51]
        },
    }
    result = check_pareto_dominant(results, "accuracy", [], parametric=False)
    assert result == "good"


def test_check_pareto_dominant_lower_is_better():
    # Exercises the LOWER_BETTER branch
    results = {
        "good": {"mae": [0.10, 0.12, 0.09, 0.11, 0.10, 0.12, 0.09, 0.11, 0.10, 0.12]},
        "bad": {"mae": [0.50, 0.52, 0.48, 0.53, 0.49, 0.51, 0.47, 0.54, 0.50, 0.51]},
    }
    result = check_pareto_dominant(results, "mae", [], parametric=False)
    assert result == "good"


def test_check_pareto_dominant_multiple_dominant():
    # A and B both dominate C but are indistinguishable from each other
    # tiebreaker picks by mean main metric.
    acc_top = [0.88, 0.90, 0.89, 0.91, 0.88, 0.90, 0.89, 0.91, 0.88, 0.90]
    f1_top = [0.86, 0.88, 0.87, 0.89, 0.86, 0.88, 0.87, 0.89, 0.86, 0.88]
    results = {
        "A": {"accuracy": acc_top, "f1": f1_top},
        "B": {"accuracy": acc_top, "f1": f1_top},
        "C": {
            "accuracy": [0.60, 0.62, 0.58, 0.63, 0.59, 0.61, 0.60, 0.62, 0.58, 0.61],
            "f1": [0.55, 0.57, 0.53, 0.58, 0.54, 0.56, 0.55, 0.57, 0.53, 0.56],
        },
    }
    result = check_pareto_dominant(results, "accuracy", ["f1"], parametric=False)
    assert result in ["A", "B"]


# ---------------------------------------------------------------------------
# get_best_model
# ---------------------------------------------------------------------------


def test_get_best_model(results_dict):
    best_model, reason = get_best_model(results_dict, "accuracy", ["f1"])
    assert best_model == "model2"
    assert isinstance(reason, str)


def test_get_best_model_low_power_warning(results_dict, caplog):
    # 5 folds and 3 models triggers low-power warning
    with caplog.at_level(logging.WARNING):
        get_best_model(results_dict, "accuracy", [], parametric=True)
    assert any("Low statistical power" in m for m in caplog.messages)


def test_get_best_model_mean_fallback():
    # Perfectly alternating scores -> selection falls back to mean CV score.
    close = {
        "m1": {
            "accuracy": [
                0.801,
                0.799,
                0.801,
                0.799,
                0.801,
                0.799,
                0.801,
                0.799,
                0.801,
                0.799,
            ]
        },
        "m2": {
            "accuracy": [
                0.799,
                0.801,
                0.799,
                0.801,
                0.799,
                0.801,
                0.799,
                0.801,
                0.799,
                0.801,
            ]
        },
    }
    best, reason = get_best_model(close, "accuracy", [])
    assert best in ["m1", "m2"]
    assert reason == "mean CV score"


# ---------------------------------------------------------------------------
# get_cv_performance
# ---------------------------------------------------------------------------


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


def test_get_cv_performance_roc_auc(classification_df):
    # Exercises the predict_proba branch for roc_auc
    metrics = get_cv_performance(
        LogisticRegression(),
        classification_df,
        "Features",
        "Target",
        "Fold",
        ["roc_auc"],
    )
    assert "roc_auc" in metrics
    assert len(metrics["roc_auc"]) == 5


def test_get_cv_performance_cohen_kappa(classification_df):
    # Exercises the cohen_kappa branch with weights="linear"
    metrics = get_cv_performance(
        LogisticRegression(),
        classification_df,
        "Features",
        "Target",
        "Fold",
        ["cohen_kappa"],
    )
    assert "cohen_kappa" in metrics
    assert len(metrics["cohen_kappa"]) == 5


def test_get_cv_performance_custom_params(classification_df):
    # Exercises the custom_params branch
    metrics = get_cv_performance(
        LogisticRegression(),
        classification_df,
        "Features",
        "Target",
        "Fold",
        ["accuracy"],
        custom_params={"C": 0.1},
    )
    assert "accuracy" in metrics
    assert len(metrics["accuracy"]) == 5


# ---------------------------------------------------------------------------
# run_CV
# ---------------------------------------------------------------------------


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


def test_run_CV_loads_existing_results(classification_df, tmp_path, monkeypatch):
    # When a final results file already exists, run_CV loads it without recomputing
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results" / "existing_exp").mkdir(parents=True)
    (tmp_path / "cache").mkdir(parents=True)

    expected = {"LogisticRegression": {"accuracy": [0.9, 0.88, 0.92, 0.91, 0.89]}}
    with open(tmp_path / "results" / "existing_exp" / "default_CV.pkl", "wb") as f:
        pickle.dump(expected, f)

    results = run_CV(
        "existing_exp",
        classification_df,
        "Features",
        "Target",
        "Fold",
        {"LogisticRegression": LogisticRegression()},
        ["accuracy"],
    )
    assert results == expected


def test_run_CV_checkpoint(classification_df, tmp_path, monkeypatch):
    # When a checkpoint exists, run_CV loads it and skips already-computed models
    def mock_get_cv_performance(*args, **kwargs):
        return {"accuracy": [0.8, 0.9, 0.85, 0.88, 0.91]}

    monkeypatch.setattr(
        "astra.model_selection.get_cv_performance", mock_get_cv_performance
    )
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results" / "ckpt_exp").mkdir(parents=True)
    (tmp_path / "cache").mkdir(parents=True)

    checkpoint = {"ModelA": {"accuracy": [0.85, 0.87, 0.83, 0.86, 0.84]}}
    with open(tmp_path / "cache" / "ckpt_exp_CV_ckpt.pkl", "wb") as f:
        pickle.dump(checkpoint, f)

    models = {"ModelA": LogisticRegression(), "ModelB": LogisticRegression()}
    results = run_CV(
        "ckpt_exp",
        classification_df,
        "Features",
        "Target",
        "Fold",
        models,
        ["accuracy"],
    )
    assert "ModelA" in results
    assert "ModelB" in results
    # ModelA came from checkpoint and must not have been overwritten by the mock
    assert results["ModelA"] == checkpoint["ModelA"]


# ---------------------------------------------------------------------------
# get_optimised_cv_performance
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# get_best_hparams
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Additional coverage tests
# ---------------------------------------------------------------------------


def test_check_assumptions_verbose_high_fold_diff(capsys):
    # m1 has tiny variance, m2 has large variance -> ratio >> 9 -> triggers warning
    results = {
        "m1": {
            "accuracy": [
                0.800,
                0.801,
                0.800,
                0.801,
                0.800,
                0.801,
                0.800,
                0.801,
                0.800,
                0.801,
            ]
        },
        "m2": {
            "accuracy": [0.30, 0.50, 0.70, 0.40, 0.60, 0.50, 0.30, 0.70, 0.40, 0.60]
        },
    }
    check_assumptions(results, verbose=True)
    out = capsys.readouterr().out
    assert "Warning" in out


def test_get_optimised_cv_performance_cohen_kappa(classification_df, monkeypatch):
    # cohen_kappa branch in get_optimised_cv_performance
    class MockGridSearchCV:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, *args, **kwargs):
            pass

        def predict(self, X):
            return np.zeros(X.shape[0], dtype=int)

        def predict_proba(self, X):
            return np.column_stack(
                [np.ones(X.shape[0]) * 0.5, np.ones(X.shape[0]) * 0.5]
            )

    monkeypatch.setattr("astra.model_selection.GridSearchCV", MockGridSearchCV)

    metrics = get_optimised_cv_performance(
        LogisticRegression(),
        classification_df,
        "Features",
        "Target",
        "Fold",
        ["accuracy", "cohen_kappa"],
        "accuracy",
        {"C": [0.1, 1]},
        1,
    )
    assert "cohen_kappa" in metrics
    assert len(metrics["cohen_kappa"]) == 5


def test_get_optimised_cv_performance_optuna_n_jobs_warning(
    classification_df, monkeypatch
):
    class MockOptunaSearchCV:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, *args, **kwargs):
            pass

        def predict(self, X):
            return np.zeros(X.shape[0], dtype=int)

        def predict_proba(self, X):
            return np.column_stack(
                [np.ones(X.shape[0]) * 0.5, np.ones(X.shape[0]) * 0.5]
            )

    monkeypatch.setattr("astra.model_selection.OptunaSearchCV", MockOptunaSearchCV)

    from optuna.distributions import FloatDistribution

    with pytest.warns(UserWarning, match="n_jobs=2 runs Optuna trials in parallel"):
        get_optimised_cv_performance(
            LogisticRegression(),
            classification_df,
            "Features",
            "Target",
            "Fold",
            ["accuracy"],
            "accuracy",
            {"C": FloatDistribution(0.01, 10.0)},
            n_jobs=2,
            use_optuna=True,
            timeout=60,
        )


def test_get_best_hparams_optuna_n_jobs_warning(classification_df, monkeypatch):
    class MockOptunaSearchCV:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, *args, **kwargs):
            pass

    monkeypatch.setattr("astra.model_selection.OptunaSearchCV", MockOptunaSearchCV)

    from optuna.distributions import FloatDistribution

    with pytest.warns(UserWarning, match="n_jobs=2 runs Optuna trials in parallel"):
        get_best_hparams(
            LogisticRegression(),
            classification_df,
            "Features",
            "Target",
            "Fold",
            "accuracy",
            ["f1"],
            {"C": FloatDistribution(0.01, 10.0)},
            n_jobs=2,
            use_optuna=True,
            timeout=60,
        )


def test_get_best_model_high_power_no_warning(caplog):
    rng = np.random.default_rng(42)
    results = {
        "m1": {"accuracy": (0.80 + rng.normal(0, 0.01, 50)).tolist()},
        "m2": {"accuracy": (0.70 + rng.normal(0, 0.01, 50)).tolist()},
    }
    with caplog.at_level(logging.WARNING):
        get_best_model(results, "accuracy", [], parametric=True)
    assert not any("Low statistical power" in m for m in caplog.messages)


def test_get_best_model_naive_fallback(monkeypatch):
    results = {
        "m1": {
            "accuracy": [0.80, 0.81, 0.82, 0.80, 0.81, 0.82, 0.80, 0.81, 0.82, 0.80]
        },
        "m2": {
            "accuracy": [0.70, 0.71, 0.72, 0.70, 0.71, 0.72, 0.70, 0.71, 0.72, 0.70]
        },
    }
    models = list(results.keys())

    # post-hoc: no significant differences (p >= 0.05)
    post_hoc = pd.DataFrame([[1.0, 0.3], [0.3, 1.0]], index=models, columns=models)
    # naive: m1 significantly beats m2 (p < 0.05, large effect size guaranteed by the data)
    naive = pd.DataFrame([[1.0, 0.01], [0.01, 1.0]], index=models, columns=models)

    monkeypatch.setattr(
        "astra.model_selection.find_n_best_models", lambda *a, **kw: models
    )
    monkeypatch.setattr(
        "astra.model_selection.perform_statistical_tests",
        lambda *a, **kw: (post_hoc, naive),
    )

    best, reason = get_best_model(results, "accuracy", [])
    assert reason == "Wilcoxon signed-rank test"
    assert best == "m1"


def test_get_best_model_pareto_fallback(monkeypatch):
    results = {
        "m1": {"accuracy": [0.80] * 10, "f1": [0.75] * 10},
        "m2": {"accuracy": [0.79] * 10, "f1": [0.74] * 10},
    }
    models = list(results.keys())

    equal_stats = pd.DataFrame([[1.0, 1.0], [1.0, 1.0]], index=models, columns=models)

    monkeypatch.setattr(
        "astra.model_selection.find_n_best_models", lambda *a, **kw: models
    )
    monkeypatch.setattr(
        "astra.model_selection.perform_statistical_tests",
        lambda *a, **kw: (equal_stats, equal_stats),
    )
    monkeypatch.setattr(
        "astra.model_selection.check_pareto_dominant", lambda *a, **kw: "m1"
    )

    best, reason = get_best_model(results, "accuracy", ["f1"])
    assert best == "m1"
    assert reason == "Pareto dominance across metrics"


# ---------------------------------------------------------------------------
# build_equivalent_ensemble
# ---------------------------------------------------------------------------


def test_build_equivalent_ensemble_regression(regression_df):
    X = np.vstack(regression_df["Features"].to_numpy())
    y = regression_df["Target"].to_numpy()

    est_a = LinearRegression().fit(X, y)
    est_b = LinearRegression().fit(X, y)

    ensemble = build_equivalent_ensemble(
        top_n_models=["A", "B"],
        estimators={"A": est_a, "B": est_b},
        X=X,
        y=y,
        classification=False,
    )

    preds = ensemble.predict(X)
    assert preds.shape == (len(y),)


def test_build_equivalent_ensemble_classification_soft(classification_df):
    X = np.vstack(classification_df["Features"].to_numpy())
    y = classification_df["Target"].to_numpy()

    est_a = LogisticRegression(max_iter=1000).fit(X, y)
    est_b = LogisticRegression(max_iter=1000, C=0.1).fit(X, y)

    ensemble = build_equivalent_ensemble(
        top_n_models=["A", "B"],
        estimators={"A": est_a, "B": est_b},
        X=X,
        y=y,
        classification=True,
    )

    assert ensemble.voting == "soft"
    assert hasattr(ensemble, "predict_proba")
    preds = ensemble.predict(X)
    assert preds.shape == (len(y),)


def test_build_equivalent_ensemble_classification_hard_fallback(
    classification_df, caplog
):
    X = np.vstack(classification_df["Features"].to_numpy())
    y = classification_df["Target"].to_numpy()

    # SVC without probability=True does not have predict_proba
    est_a = LogisticRegression(max_iter=1000).fit(X, y)
    est_b = SVC(kernel="linear").fit(X, y)

    with caplog.at_level(logging.INFO):
        ensemble = build_equivalent_ensemble(
            top_n_models=["lr", "svc"],
            estimators={"lr": est_a, "svc": est_b},
            X=X,
            y=y,
            classification=True,
        )

    assert ensemble.voting == "hard"
    assert "hard voting" in caplog.text
    preds = ensemble.predict(X)
    assert preds.shape == (len(y),)
