import pytest

from astra.models.regression import REGRESSOR_PARAMS, REGRESSORS


def test_regressors_and_params_keys_match():
    assert REGRESSORS.keys() == REGRESSOR_PARAMS.keys()


@pytest.mark.parametrize("model_name", REGRESSORS.keys())
def test_regressor_instantiation_and_params(model_name):
    model = REGRESSORS[model_name]
    params = REGRESSOR_PARAMS[model_name]

    # Check instantiation
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")

    # Check hyperparameter validity
    if model_name == "CatBoostRegressor":
        # get_params() for CatBoostRegressor only returns parameters
        # that were explicitly specified
        return

    model_params = model.get_params().keys()
    for param in params:
        assert param in model_params
