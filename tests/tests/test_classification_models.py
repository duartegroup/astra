import pytest

from astra.models.classification import (
    CLASSIFIER_PARAMS,
    CLASSIFIER_PARAMS_OPTUNA,
    CLASSIFIERS,
    NON_PROBABILISTIC_MODELS,
)


def test_classifiers_and_params_keys_match():
    assert CLASSIFIERS.keys() == CLASSIFIER_PARAMS.keys()
    assert CLASSIFIER_PARAMS_OPTUNA.keys() == CLASSIFIER_PARAMS.keys()


@pytest.mark.parametrize("model_name", CLASSIFIERS.keys())
def test_classifier_instantiation_and_params(model_name):
    model = CLASSIFIERS[model_name]
    params = CLASSIFIER_PARAMS[model_name]

    # Check instantiation
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")

    # Check hyperparameter validity
    model_params = model.get_params().keys()
    for param in params:
        assert param in model_params


def test_non_probabilistic_models():
    for model_name in NON_PROBABILISTIC_MODELS:
        assert model_name in CLASSIFIERS
        model = CLASSIFIERS[model_name]
        assert not hasattr(model, "predict_proba")
