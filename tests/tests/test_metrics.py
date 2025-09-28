import numpy as np
import pytest

from astra.metrics import (
    CLASSIFICATION_METRICS,
    HIGHER_BETTER,
    KNOWN_METRICS,
    LOWER_BETTER,
    MULTICLASS_METRICS,
    REGRESSION_METRICS,
    SCORING,
    get_kendalltau_score,
    get_pearsonr_score,
    get_spearmanr_score,
)


@pytest.fixture
def sample_data():
    return [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]


@pytest.fixture
def random_data():
    return [1, 2, 3, 4, 5], [5, 4, 3, 2, 1]


def test_get_kendalltau_score(sample_data, random_data):
    y_true, y_pred = sample_data
    assert np.isclose(get_kendalltau_score(y_true, y_pred), 1.0)

    y_true, y_pred = random_data
    assert np.isclose(get_kendalltau_score(y_true, y_pred), -1.0)


def test_get_pearsonr_score(sample_data, random_data):
    y_true, y_pred = sample_data
    assert np.isclose(get_pearsonr_score(y_true, y_pred), 1.0)

    y_true, y_pred = random_data
    assert np.isclose(get_pearsonr_score(y_true, y_pred), -1.0)


def test_get_spearmanr_score(sample_data, random_data):
    y_true, y_pred = sample_data
    assert np.isclose(get_spearmanr_score(y_true, y_pred), 1.0)

    y_true, y_pred = random_data
    assert np.isclose(get_spearmanr_score(y_true, y_pred), -1.0)


def test_metric_dictionaries():
    # Check that all metrics are in KNOWN_METRICS
    for metric in CLASSIFICATION_METRICS:
        assert metric in KNOWN_METRICS
    for metric in REGRESSION_METRICS:
        assert metric in KNOWN_METRICS
    for metric in MULTICLASS_METRICS:
        assert metric in KNOWN_METRICS

    # Check that all known metrics are in SCORING
    for metric in KNOWN_METRICS:
        assert metric in SCORING

    # Check that all metrics in HIGHER_BETTER and LOWER_BETTER are known
    for metric in HIGHER_BETTER:
        assert metric in KNOWN_METRICS
    for metric in LOWER_BETTER:
        assert metric in KNOWN_METRICS

    # Check for no overlap between HIGHER_BETTER and LOWER_BETTER
    assert len(set(HIGHER_BETTER) & set(LOWER_BETTER)) == 0
