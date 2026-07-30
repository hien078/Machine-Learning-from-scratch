from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ml_first_principles.metrics import accuracy


class GaussianNB:
    """Gaussian Naive Bayes evaluated entirely in log space."""

    def __init__(self, var_smoothing: float = 1e-9) -> None:
        if var_smoothing <= 0.0:
            raise ValueError("var_smoothing must be positive")
        self.var_smoothing = var_smoothing
        self.classes_: NDArray | None = None
        self.mean_: NDArray[np.float64] | None = None
        self.var_: NDArray[np.float64] | None = None
        self.class_log_prior_: NDArray[np.float64] | None = None

    def fit(self, X: ArrayLike, y: ArrayLike) -> GaussianNB:
        """Estimate class priors and feature-wise Gaussian parameters."""
        features = np.asarray(X, dtype=float)
        target = np.asarray(y)
        if features.ndim != 2 or target.ndim != 1 or features.shape[0] != target.shape[0]:
            raise ValueError("X and y have incompatible shapes")
        if features.shape[0] == 0:
            raise ValueError("training data cannot be empty")
        self.classes_, encoded = np.unique(target, return_inverse=True)
        n_classes = len(self.classes_)
        n_features = features.shape[1]
        self.mean_ = np.empty((n_classes, n_features))
        self.var_ = np.empty((n_classes, n_features))
        counts = np.bincount(encoded, minlength=n_classes)
        epsilon = self.var_smoothing * max(float(np.var(features, axis=0).max()), 1.0)
        for index in range(n_classes):
            class_data = features[encoded == index]
            self.mean_[index] = class_data.mean(axis=0)
            self.var_[index] = class_data.var(axis=0) + epsilon
        self.class_log_prior_ = np.log(counts / features.shape[0])
        return self

    def _joint_log_likelihood(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.mean_ is None or self.var_ is None or self.class_log_prior_ is None:
            raise RuntimeError("fit must be called before prediction")
        difference = X[:, None, :] - self.mean_[None, :, :]
        log_density = -0.5 * (
            np.log(2.0 * np.pi * self.var_)[None, :, :] + difference**2 / self.var_[None, :, :]
        )
        return self.class_log_prior_[None, :] + log_density.sum(axis=2)

    def predict(self, X: ArrayLike) -> NDArray:
        """Predict the class with maximum joint log likelihood."""
        if self.classes_ is None or self.mean_ is None:
            raise RuntimeError("fit must be called before predict")
        features = np.asarray(X, dtype=float)
        if features.ndim != 2 or features.shape[1] != self.mean_.shape[1]:
            raise ValueError("X has an incompatible feature shape")
        return self.classes_[np.argmax(self._joint_log_likelihood(features), axis=1)]

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return classification accuracy."""
        return accuracy(y, self.predict(X))
