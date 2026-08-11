"""Support vector machine models."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ml_first_principles.metrics import accuracy
from ml_first_principles.optimizers import sgd


class LinearSVC:
    r"""Linear soft-margin SVM minimizing $\frac12\|w\|^2+C\,\mathrm{mean}(\text{hinge})$."""

    def __init__(
        self,
        C: float = 1.0,
        lr: float = 0.01,
        max_iter: int = 1000,
        batch_size: int = 32,
        random_state: int | None = 42,
    ) -> None:
        if C <= 0.0 or lr <= 0.0 or max_iter < 1 or batch_size < 1:
            raise ValueError("C, lr, max_iter, and batch_size must be positive")
        self.C = C
        self.lr = lr
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.random_state = random_state
        self.coef_: NDArray[np.float64] | None = None
        self.intercept_: float | None = None
        self.classes_: NDArray | None = None

    def fit(self, X: ArrayLike, y: ArrayLike) -> LinearSVC:
        """Fit a binary linear classifier with mini-batch subgradient descent."""
        features = np.asarray(X, dtype=float)
        target = np.asarray(y)
        if features.ndim != 2 or target.ndim != 1 or features.shape[0] != target.shape[0]:
            raise ValueError("X and y have incompatible shapes")
        self.classes_ = np.unique(target)
        if self.classes_.size != 2:
            raise ValueError("LinearSVC supports exactly two classes")
        encoded = np.where(target == self.classes_[0], -1.0, 1.0)
        design = np.column_stack((np.ones(features.shape[0]), features))

        def batch_gradient(
            theta: NDArray[np.float64], indices: NDArray[np.int64]
        ) -> NDArray[np.float64]:
            batch_X = design[indices]
            batch_y = encoded[indices]
            margins = batch_y * (batch_X @ theta)
            gradient = np.zeros_like(theta)
            gradient[1:] = theta[1:]
            violations = margins < 1.0
            if np.any(violations):
                gradient -= (
                    self.C
                    * np.sum(batch_y[violations, None] * batch_X[violations], axis=0)
                    / len(indices)
                )
            return gradient

        theta, _ = sgd(
            batch_gradient,
            np.zeros(design.shape[1]),
            n_samples=features.shape[0],
            lr=self.lr,
            max_iter=self.max_iter,
            batch_size=min(self.batch_size, features.shape[0]),
            random_state=self.random_state,
        )
        self.intercept_ = float(theta[0])
        self.coef_ = theta[1:]
        return self

    def decision_function(self, X: ArrayLike) -> NDArray[np.float64]:
        """Return signed distances up to the scale of the fitted weights."""
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("fit must be called before decision_function")
        features = np.asarray(X, dtype=float)
        if features.ndim != 2 or features.shape[1] != self.coef_.shape[0]:
            raise ValueError("X has an incompatible feature shape")
        return features @ self.coef_ + self.intercept_

    def predict(self, X: ArrayLike) -> NDArray:
        """Predict original class labels."""
        if self.classes_ is None:
            raise RuntimeError("fit must be called before predict")
        indices = (self.decision_function(X) >= 0.0).astype(int)
        return self.classes_[indices]

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return classification accuracy."""
        return accuracy(y, self.predict(X))
