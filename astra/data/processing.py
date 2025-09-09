"""
This module contains the `CorrelationFilter` class, which is used to filter out features that are highly correlated
with each other based on a specified threshold.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """
    A transformer that removes features from a dataset that are highly correlated with each other.

    Parameters
    ----------
    threshold : float, default=0.95
        The correlation threshold above which features will be considered highly correlated and removed.

    Attributes
    ----------
    to_drop : set
        A set of indices of features to be dropped from the dataset after fitting.

    Methods
    -------
    fit(X: np.ndarray, y: None = None) -> CorrelationFilter
        Fit the transformer to the data by calculating the correlation matrix and identifying features to drop.
    transform(X: np.ndarray) -> np.ndarray
        Transform the input data by removing the features identified during fitting.
    fit_transform(X: np.ndarray, y: None = None) -> np.ndarray
        Fit the transformer and transform the input data in one step.
    get_feature_names_out(input_features: list[str]) -> list[str]
        Get the names of the features that are retained after transformation.

    Notes
    -----
    This transformer is compatible with the scikit-learn API, allowing it to be used
    seamlessly with other transformers and estimators.

    Examples
    --------
    >>> from astra.data.processing import CorrelationFilter
    >>> import numpy as np
    >>> X = np.array([[1, 2, 3],
    ...               [4, 5, 6],
    ...               [7, 8, 9],
    ...               [1, 2, 3]])
    >>> cf = CorrelationFilter(threshold=0.9)
    >>> cf.fit_transform(X)
    array([[1],
           [4],
           [7],
           [1]])
    """

    def __init__(self, threshold: float = 0.95) -> None:
        """
        Initialize the CorrelationFilter with a correlation threshold.

        Parameters
        ----------
        threshold : float, default=0.95
            The correlation threshold above which features will be considered highly correlated and removed.
        """
        self.threshold = threshold

    def fit(self, X: np.ndarray, y: None = None) -> "CorrelationFilter":
        """
        Fit the transformer to the data by calculating the correlation matrix and identifying features to drop.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data to fit the transformer on.
        y : None, default=None
            Ignored, exists for compatibility with the scikit-learn API.

        Returns
        -------
        self : CorrelationFilter
            Returns the instance itself.
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        X = check_array(X)
        # Get the correlation matrix
        with np.errstate(invalid="ignore"):
            corr_matrix = np.corrcoef(X, rowvar=False)
        # Get the indices of the upper triangle of the correlation matrix
        upper_triangle = np.triu_indices_from(corr_matrix, k=1)
        # Identify features to drop based on the correlation threshold
        self.to_drop = set()
        for i, j in zip(*upper_triangle):
            if abs(corr_matrix[i, j]) > self.threshold:
                self.to_drop.add(j)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the input data by removing the features identified during fitting.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data to transform.

        Returns
        -------
        np.ndarray
            The transformed data with highly correlated features removed.
        """
        check_is_fitted(self, attributes=["to_drop"])
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        X = check_array(X)
        # Drop the features identified during fitting
        return np.delete(X, list(self.to_drop), axis=1)

    def fit_transform(self, X: np.ndarray, y: None = None) -> np.ndarray:
        """
        Fit the transformer and transform the input data in one step.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data to fit and transform.
        y : None, default=None
            Ignored, exists for compatibility with the scikit-learn API.

        Returns
        -------
        np.ndarray
            The transformed data with highly correlated features removed.
        """
        self.fit(X, y)
        return self.transform(X)

    def get_feature_names_out(self, input_features: list[str]) -> list[str]:
        """
        Get the names of the features that are retained after transformation.

        Parameters
        ----------
        input_features : list of str
            The names of the input features.

        Returns
        -------
        list of str
            The names of the features that are retained after transformation.
        """
        check_is_fitted(self, attributes=["to_drop"])
        return [f for i, f in enumerate(input_features) if i not in self.to_drop]
