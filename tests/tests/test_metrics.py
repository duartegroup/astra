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

# ---------------------------------------------------------------------------
# Correlation metrics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "y_true,y_pred,expected",
    [
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], 1.0),
        ([1, 2, 3, 4, 5], [5, 4, 3, 2, 1], -1.0),
    ],
)
def test_get_kendalltau_score(y_true, y_pred, expected):
    assert np.isclose(get_kendalltau_score(y_true, y_pred), expected)


@pytest.mark.parametrize(
    "y_true,y_pred,expected",
    [
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], 1.0),
        ([1, 2, 3, 4, 5], [5, 4, 3, 2, 1], -1.0),
    ],
)
def test_get_pearsonr_score(y_true, y_pred, expected):
    assert np.isclose(get_pearsonr_score(y_true, y_pred), expected)


@pytest.mark.parametrize(
    "y_true,y_pred,expected",
    [
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], 1.0),
        ([1, 2, 3, 4, 5], [5, 4, 3, 2, 1], -1.0),
    ],
)
def test_get_spearmanr_score(y_true, y_pred, expected):
    assert np.isclose(get_spearmanr_score(y_true, y_pred), expected)


# ---------------------------------------------------------------------------
# Metric dictionaries
# ---------------------------------------------------------------------------


def test_metric_dictionaries():
    for metric in CLASSIFICATION_METRICS:
        assert metric in KNOWN_METRICS
    for metric in REGRESSION_METRICS:
        assert metric in KNOWN_METRICS
    for metric in MULTICLASS_METRICS:
        assert metric in KNOWN_METRICS

    for metric in KNOWN_METRICS:
        assert metric in SCORING

    for metric in HIGHER_BETTER:
        assert metric in KNOWN_METRICS
    for metric in LOWER_BETTER:
        assert metric in KNOWN_METRICS

    assert len(set(HIGHER_BETTER) & set(LOWER_BETTER)) == 0
    assert set(KNOWN_METRICS) == set(HIGHER_BETTER) | set(LOWER_BETTER)
