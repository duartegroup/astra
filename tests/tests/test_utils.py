import numpy as np
import pandas as pd
import pytest
import yaml
from numpy import ndarray
from optuna.distributions import (
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from astra.models.classification import (
    CLASSIFIER_PARAMS,
    CLASSIFIER_PARAMS_OPTUNA,
    CLASSIFIERS,
    NON_PROBABILISTIC_MODELS,
)
from astra.models.regression import (
    REGRESSOR_PARAMS,
    REGRESSOR_PARAMS_OPTUNA,
    REGRESSORS,
)
from astra.utils import (
    build_model,
    get_data,
    get_estimator_name,
    get_models,
    get_optuna_grid,
    get_scores,
    load_config,
    print_file_console,
    print_final_results,
    print_performance,
)


def test_get_data():
    for file in [
        "tests/test_data/example_df.csv",
        "tests/test_data/example_df.pkl",
        "tests/test_data/example_df.parquet",
    ]:
        df = get_data(data=file, features="Features")
        assert "SMILES" in df.columns
        assert "Target" in df.columns
        assert "Features" in df.columns
        assert isinstance(df["Features"].iloc[0], ndarray)
        assert not df.empty
    # Test for unsupported file format
    with pytest.raises(ValueError):
        get_data(data="astra/data/example_df.txt", features="Features")


def test_get_models_regression():
    models, hyperparams, custom_hyperparams = get_models(
        main_metric="mse", sec_metrics=["r2"], scaler="Standard", custom_models=None
    )
    assert isinstance(models, dict)
    assert isinstance(hyperparams, dict)
    assert custom_hyperparams is None
    assert models == REGRESSORS
    assert hyperparams == REGRESSOR_PARAMS


def test_get_models_classification():
    models, hyperparams, custom_hyperparams = get_models(
        main_metric="accuracy",
        sec_metrics=["f1"],
        scaler=None,
        custom_models=None,
    )
    assert isinstance(models, dict)
    assert isinstance(hyperparams, dict)
    assert custom_hyperparams is None
    assert models == CLASSIFIERS
    assert hyperparams == CLASSIFIER_PARAMS


@pytest.mark.parametrize("metric", ["roc_auc", "pr_auc"])
def test_get_models_auc_metrics(metric):
    models, _, _ = get_models(
        main_metric=metric,
        sec_metrics=["accuracy"],
        scaler=None,
        custom_models=None,
    )
    assert models == {
        c: CLASSIFIERS[c] for c in CLASSIFIERS if c not in NON_PROBABILISTIC_MODELS
    }

    models, _, _ = get_models(
        main_metric="accuracy",
        sec_metrics=[metric],
        scaler=None,
        custom_models=None,
    )
    assert models == {
        c: CLASSIFIERS[c] for c in CLASSIFIERS if c not in NON_PROBABILISTIC_MODELS
    }


def test_get_models_scaler():
    models, hyperparams, _ = get_models(
        main_metric="accuracy",
        sec_metrics=["f1"],
        scaler="Standard",
        custom_models=None,
    )
    assert "MultinomialNB" not in models
    assert "MultinomialNB" not in hyperparams


@pytest.mark.parametrize(
    "main_metric, sec_metrics",
    [("mse", ["accuracy"]), ("accuracy", ["mse"])],
)
def test_get_models_incompatible_metrics(main_metric, sec_metrics):
    with pytest.raises(AssertionError):
        get_models(
            main_metric=main_metric,
            sec_metrics=sec_metrics,
            scaler=None,
            custom_models=None,
        )


@pytest.mark.parametrize(
    "custom_models, expected_hyperparams, expected_custom_hyperparams",
    [
        (
            {"LogisticRegression": None, "RandomForestClassifier": None},
            {
                "LogisticRegression": CLASSIFIER_PARAMS["LogisticRegression"],
                "RandomForestClassifier": CLASSIFIER_PARAMS["RandomForestClassifier"],
            },
            {},
        ),
        (
            {
                "LogisticRegression": {"params": {"C": 0.1}},
                "RandomForestClassifier": {"params": {"n_estimators": 50}},
            },
            {
                "LogisticRegression": CLASSIFIER_PARAMS["LogisticRegression"],
                "RandomForestClassifier": CLASSIFIER_PARAMS["RandomForestClassifier"],
            },
            {
                "LogisticRegression": {"C": 0.1},
                "RandomForestClassifier": {"n_estimators": 50},
            },
        ),
        (
            {
                "LogisticRegression": {"hparam_grid": {"C": [0.01, 0.1, 1]}},
                "RandomForestClassifier": {
                    "hparam_grid": {"n_estimators": [10, 50, 100]}
                },
            },
            {
                "LogisticRegression": {"C": [0.01, 0.1, 1]},
                "RandomForestClassifier": {"n_estimators": [10, 50, 100]},
            },
            {},
        ),
        (
            {
                "LogisticRegression": {
                    "params": {"C": 0.1},
                    "hparam_grid": {"C": [0.01, 0.1, 1]},
                },
                "RandomForestClassifier": {
                    "params": {"n_estimators": 50},
                    "hparam_grid": {"n_estimators": [10, 50, 100]},
                },
            },
            {
                "LogisticRegression": {"C": [0.01, 0.1, 1]},
                "RandomForestClassifier": {"n_estimators": [10, 50, 100]},
            },
            {
                "LogisticRegression": {"C": 0.1},
                "RandomForestClassifier": {"n_estimators": 50},
            },
        ),
        (
            {
                "LogisticRegression": None,
                "RandomForestClassifier": {
                    "params": {"n_estimators": 50},
                    "hparam_grid": {"n_estimators": [10, 50, 100]},
                },
            },
            {
                "LogisticRegression": CLASSIFIER_PARAMS["LogisticRegression"],
                "RandomForestClassifier": {"n_estimators": [10, 50, 100]},
            },
            {"RandomForestClassifier": {"n_estimators": 50}},
        ),
        (
            {
                "LogisticRegression": {"params": {"C": 0.1}},
                "RandomForestClassifier": {
                    "hparam_grid": {"n_estimators": [10, 50, 100]}
                },
            },
            {
                "LogisticRegression": CLASSIFIER_PARAMS["LogisticRegression"],
                "RandomForestClassifier": {"n_estimators": [10, 50, 100]},
            },
            {"LogisticRegression": {"C": 0.1}},
        ),
    ],
)
def test_get_models_custom(
    custom_models, expected_hyperparams, expected_custom_hyperparams
):
    models, hyperparams, custom_hyperparams = get_models(
        main_metric="accuracy",
        sec_metrics=["f1"],
        scaler=None,
        custom_models=custom_models,
    )
    expected_models = {
        "LogisticRegression": LogisticRegression(random_state=42, max_iter=100000),
        "RandomForestClassifier": RandomForestClassifier(random_state=42),
    }
    assert models.keys() == expected_models.keys()
    for model_name in models:
        assert isinstance(models[model_name], type(expected_models[model_name]))
        assert (
            models[model_name].get_params() == expected_models[model_name].get_params()
        )
    assert hyperparams == expected_hyperparams
    assert custom_hyperparams == expected_custom_hyperparams


def test_get_models_regression_optuna():
    models, hyperparams, custom_hyperparams = get_models(
        main_metric="mse",
        sec_metrics=["r2"],
        scaler="Standard",
        custom_models=None,
        use_optuna=True,
    )
    assert isinstance(models, dict)
    assert isinstance(hyperparams, dict)
    assert custom_hyperparams is None
    assert models == REGRESSORS
    assert hyperparams == REGRESSOR_PARAMS_OPTUNA


def test_get_models_classification_optuna():
    models, hyperparams, custom_hyperparams = get_models(
        main_metric="accuracy",
        sec_metrics=["f1"],
        scaler=None,
        custom_models=None,
        use_optuna=True,
    )
    assert isinstance(models, dict)
    assert isinstance(hyperparams, dict)
    assert custom_hyperparams is None
    assert models == CLASSIFIERS
    assert hyperparams == CLASSIFIER_PARAMS_OPTUNA


def test_get_models_custom_optuna():
    custom_models = {
        "LogisticRegression": {
            "hparam_grid": {"C": [0.01, 0.1, 1.0]},
        },
        "RandomForestClassifier": {"hparam_grid": {"n_estimators": [10, 50, 100]}},
    }
    _, hyperparams, _ = get_models(
        main_metric="accuracy",
        sec_metrics=["f1"],
        scaler=None,
        custom_models=custom_models,
        use_optuna=True,
    )
    assert isinstance(hyperparams["LogisticRegression"]["C"], FloatDistribution)
    assert hyperparams["LogisticRegression"]["C"].low == 0.01
    assert hyperparams["LogisticRegression"]["C"].high == 1.0
    assert isinstance(
        hyperparams["RandomForestClassifier"]["n_estimators"], IntDistribution
    )
    assert hyperparams["RandomForestClassifier"]["n_estimators"].low == 10
    assert hyperparams["RandomForestClassifier"]["n_estimators"].high == 100


def test_get_optuna_grid():
    hparam_grid = {
        "int_param": [1, 2, 3, 4, 5],
        "float_param": [0.1, 0.5, 1.0],
        "cat_param": ["a", "b", "c"],
        "mixed_param": [1, 0.5, "a"],
    }
    optuna_grid = get_optuna_grid(hparam_grid)
    assert isinstance(optuna_grid["int_param"], IntDistribution)
    assert optuna_grid["int_param"].low == 1
    assert optuna_grid["int_param"].high == 5
    assert isinstance(optuna_grid["float_param"], FloatDistribution)
    assert optuna_grid["float_param"].low == 0.1
    assert optuna_grid["float_param"].high == 1.0
    assert isinstance(optuna_grid["cat_param"], CategoricalDistribution)
    assert optuna_grid["cat_param"].choices == ("a", "b", "c")
    assert isinstance(optuna_grid["mixed_param"], CategoricalDistribution)
    assert optuna_grid["mixed_param"].choices == (1, 0.5, "a")


def test_get_models_unsupported():
    custom_models_invalid = {
        "LogisticRegression": None,
        "RandomForestClassifier": None,
        "UnsupportedModel": None,
    }
    with pytest.raises(AssertionError):
        get_models(
            main_metric="accuracy",
            sec_metrics=["f1"],
            scaler=None,
            custom_models=custom_models_invalid,
        )


def test_build_model():
    # Test with no preprocessing
    model = build_model(LogisticRegression())
    assert isinstance(model, LogisticRegression)

    # Test with imputation
    model = build_model(LogisticRegression(), impute="mean")
    assert isinstance(model, Pipeline)
    assert "simpleimputer" in model.named_steps

    # Test with variance threshold
    model = build_model(LogisticRegression(), remove_constant=0.1)
    assert isinstance(model, Pipeline)
    assert "variancethreshold" in model.named_steps

    # Test with correlation filter
    model = build_model(LogisticRegression(), remove_correlated=0.9)
    assert isinstance(model, Pipeline)
    assert "correlationfilter" in model.named_steps

    # Test with scaler
    model = build_model(LogisticRegression(), scaler="Standard")
    assert isinstance(model, Pipeline)
    assert "standardscaler" in model.named_steps

    # Test with a combination of steps
    model = build_model(
        LogisticRegression(),
        impute="mean",
        remove_constant=0.1,
        remove_correlated=0.9,
        scaler="Standard",
    )
    assert isinstance(model, Pipeline)
    assert "simpleimputer" in model.named_steps
    assert "variancethreshold" in model.named_steps
    assert "correlationfilter" in model.named_steps
    assert "standardscaler" in model.named_steps

    # Test for invalid impute value
    with pytest.raises(ValueError):
        build_model(LogisticRegression(), impute="invalid_impute")

    # Test for invalid scaler value
    with pytest.raises(ValueError):
        build_model(LogisticRegression(), scaler="invalid_scaler")


def test_get_estimator_name():
    # Test with a direct estimator
    model = LogisticRegression()
    assert get_estimator_name(model) == "LogisticRegression"

    # Test with a pipeline
    pipeline = Pipeline([("scaler", "passthrough"), ("model", model)])
    assert get_estimator_name(pipeline) == "LogisticRegression"


def test_get_scores():
    # Create a sample cv_results_df
    cv_results_df = pd.DataFrame(
        {
            "rank_test_mse": [1, 2, 3],
            "rank_test_r2": [3, 2, 1],
            "split0_test_mse": [-0.1, -0.2, -0.3],
            "split1_test_mse": [-0.11, -0.21, -0.31],
            "split0_test_r2": [0.9, 0.8, 0.7],
            "split1_test_r2": [0.89, 0.79, 0.69],
        }
    )

    # Test with a metric where lower is better
    results_dict, mean_score, std_score, median_score, sec_scores = get_scores(
        cv_results_df, "mse", ["r2"], 2
    )
    assert np.isclose(mean_score, 0.105)
    assert np.isclose(std_score, 0.005)
    assert np.isclose(median_score, 0.105)
    assert np.isclose(sec_scores["r2"][0], 0.895)
    assert np.isclose(sec_scores["r2"][1], 0.005)
    assert np.isclose(sec_scores["r2"][2], 0.895)
    assert "mse" in results_dict
    assert "r2" in results_dict
    assert results_dict["mse"] == [0.1, 0.11]
    assert results_dict["r2"] == [0.9, 0.89]

    # Test with a metric where higher is better
    cv_results_df = pd.DataFrame(
        {
            "rank_test_accuracy": [1, 2, 3],
            "rank_test_f1": [3, 2, 1],
            "split0_test_accuracy": [0.9, 0.8, 0.7],
            "split1_test_accuracy": [0.89, 0.79, 0.69],
            "split0_test_f1": [0.8, 0.7, 0.6],
            "split1_test_f1": [0.79, 0.69, 0.59],
        }
    )
    results_dict, mean_score, std_score, median_score, sec_scores = get_scores(
        cv_results_df, "accuracy", ["f1"], 2
    )
    assert np.isclose(mean_score, 0.895)
    assert np.isclose(std_score, 0.005)
    assert np.isclose(median_score, 0.895)
    assert np.isclose(sec_scores["f1"][0], 0.795)
    assert np.isclose(sec_scores["f1"][1], 0.005)
    assert np.isclose(sec_scores["f1"][2], 0.795)
    assert "accuracy" in results_dict
    assert "f1" in results_dict
    assert results_dict["accuracy"] == [0.9, 0.89]
    assert results_dict["f1"] == [0.8, 0.79]

    # Test for missing columns
    with pytest.raises(AssertionError):
        get_scores(cv_results_df, "accuracy", ["precision"], 2)


def test_load_config(tmp_path):
    config_data = {"key": "value", "number": 123}
    config_file = tmp_path / "config.yml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    config = load_config(config_file)
    assert config == config_data


def test_print_performance(capsys, tmp_path):
    results_dict = {"metric1": [0.8, 0.9, 1.0], "metric2": [0.6, 0.7, 0.8]}
    file = tmp_path / "test_output.txt"

    print_performance("TestModel", results_dict, file=file)

    captured = capsys.readouterr()
    assert "Performance for TestModel:" in captured.out
    assert "metric1: 0.900" in captured.out
    assert "metric2: 0.700" in captured.out

    with open(file, "r") as f:
        content = f.read()
        assert "Performance for TestModel:" in content
        assert "metric1: 0.900" in content
        assert "metric2: 0.700" in content


def test_print_file_console(capsys, tmp_path):
    file = tmp_path / "test_output.txt"
    message = "Test message"

    print_file_console(file=file, message=message)

    captured = capsys.readouterr()
    assert message in captured.out

    with open(file, "r") as f:
        content = f.read()
        assert message in content


def test_print_final_results(capsys, tmp_path):
    file = tmp_path / "test_output.txt"
    final_model_name = "FinalModel"
    final_hyperparameters = {"param1": 10, "param2": "abc"}
    main_metric = "accuracy"
    mean_score_main = 0.95
    std_score_main = 0.02
    median_score_main = 0.96
    sec_metrics_scores = {"f1": (0.92, 0.03, 0.93)}

    print_final_results(
        final_model_name,
        final_hyperparameters,
        main_metric,
        mean_score_main,
        std_score_main,
        median_score_main,
        sec_metrics_scores,
        file=file,
    )

    captured = capsys.readouterr()
    assert "Final results" in captured.out
    assert f"Final model: {final_model_name}" in captured.out
    assert "Hyperparameters:" in captured.out
    assert "param1: 10" in captured.out
    assert "param2: abc" in captured.out
    assert f"Mean {main_metric}: {mean_score_main:.3f}" in captured.out
    assert f"Median {main_metric}: {median_score_main:.3f}" in captured.out
    assert "Mean f1: 0.920" in captured.out
    assert "Median f1: 0.930" in captured.out

    with open(file, "r") as f:
        content = f.read()
        assert "Final results" in content
        assert f"Final model: {final_model_name}" in content
        assert "Hyperparameters:" in content
        assert "param1: 10" in content
        assert "param2: abc" in content
        assert f"Mean {main_metric}: {mean_score_main:.3f}" in content
        assert f"Median {main_metric}: {median_score_main:.3f}" in content
        assert "Mean f1: 0.920" in content
        assert "Median f1: 0.930" in content
