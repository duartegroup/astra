import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from astra.utils import get_estimator_name


def test_get_estimator_name():
    # Test with a direct estimator
    model = LogisticRegression()
    assert get_estimator_name(model) == "LogisticRegression"

    # Test with a pipeline
    pipeline = Pipeline([("scaler", "passthrough"), ("model", model)])
    assert get_estimator_name(pipeline) == "LogisticRegression"
