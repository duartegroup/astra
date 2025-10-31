import numpy as np
import pytest
from sklearn.utils.validation import NotFittedError

from astra.data.processing import CorrelationFilter


@pytest.fixture
def correlated_data():
    X = np.zeros((10, 4))
    X[:, 0] = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    X[:, 1] = X[:, 0] * 2  # correlated with 0
    X[:, 2] = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
    X[:, 3] = X[:, 2] * 3  # correlated with 2
    return X


def test_correlation_filter_init():
    cf = CorrelationFilter(threshold=0.9)
    assert cf.threshold == 0.9


def test_correlation_filter_fit(correlated_data):
    cf = CorrelationFilter(threshold=0.9)
    cf.fit(correlated_data)
    assert hasattr(cf, "to_drop")
    assert cf.to_drop == {1, 3}


def test_correlation_filter_transform(correlated_data):
    cf = CorrelationFilter(threshold=0.9)
    cf.fit(correlated_data)
    transformed_X = cf.transform(correlated_data)
    expected_X = correlated_data[:, [0, 2]]
    np.testing.assert_array_equal(transformed_X, expected_X)


def test_correlation_filter_fit_transform(correlated_data):
    cf = CorrelationFilter(threshold=0.9)
    transformed_X = cf.fit_transform(correlated_data)
    expected_X = correlated_data[:, [0, 2]]
    np.testing.assert_array_equal(transformed_X, expected_X)


def test_correlation_filter_get_feature_names_out(correlated_data):
    cf = CorrelationFilter(threshold=0.9)
    cf.fit(correlated_data)
    input_features = ["A", "B", "C", "D"]
    output_features = cf.get_feature_names_out(input_features)
    assert output_features == ["A", "C"]


def test_correlation_filter_no_correlated_features():
    X = np.random.rand(10, 5)
    cf = CorrelationFilter(threshold=0.9)
    cf.fit(X)
    assert len(cf.to_drop) == 0
    transformed_X = cf.transform(X)
    np.testing.assert_array_equal(transformed_X, X)


def test_correlation_filter_transform_not_fitted():
    cf = CorrelationFilter()
    with pytest.raises(NotFittedError):
        cf.transform(np.array([[1, 2, 3]]))


def test_correlation_filter_get_feature_names_out_not_fitted():
    cf = CorrelationFilter()
    with pytest.raises(NotFittedError):
        cf.get_feature_names_out(["A", "B", "C"])
