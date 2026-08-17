"""Linear models: linear, ridge, lasso, logistic regression, polynomial features."""

from __future__ import annotations

from itertools import combinations_with_replacement

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ml_first_principles.metrics import accuracy, r2_score
from ml_first_principles.optimizers import coordinate_descent_lasso, gradient_descent


def _regression_arrays(
    X: ArrayLike, y: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    features = np.asarray(X, dtype=float)
    target = np.asarray(y, dtype=float)
    if features.ndim != 2 or target.ndim != 1 or features.shape[0] != target.shape[0]:
        raise ValueError("X must be two-dimensional and y must be a matching one-dimensional array")
    if features.shape[0] == 0:
        raise ValueError("training data cannot be empty")
    return features, target


class LinearRegression:
    """Ordinary least squares solved by least squares or gradient descent."""

    def __init__(self, solver: str = "normal", lr: float = 0.01, max_iter: int = 1000) -> None:
        if solver not in {"normal", "gd"}:
            raise ValueError("solver must be 'normal' or 'gd'")
        if lr <= 0.0 or max_iter < 1:
            raise ValueError("lr and max_iter must be positive")
        self.solver = solver
        self.lr = lr
        self.max_iter = max_iter
        self.coef_: NDArray[np.float64] | None = None
        self.intercept_: float | None = None

    def fit(self, X: ArrayLike, y: ArrayLike) -> LinearRegression:
        """Fit the model and return ``self``."""
        features, target = _regression_arrays(X, y)
        design = np.column_stack((np.ones(features.shape[0]), features))
        if self.solver == "normal":
            theta = np.linalg.lstsq(design, target, rcond=None)[0]
        else:
            n_samples = features.shape[0]

            def gradient(theta: NDArray[np.float64]) -> NDArray[np.float64]:
                return (2.0 / n_samples) * design.T @ (design @ theta - target)

            theta, _ = gradient_descent(
                gradient, np.zeros(design.shape[1]), lr=self.lr, max_iter=self.max_iter
            )
        self.intercept_ = float(theta[0])
        self.coef_ = np.asarray(theta[1:], dtype=float)
        return self

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        """Predict continuous targets."""
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("fit must be called before predict")
        features = np.asarray(X, dtype=float)
        if features.ndim != 2 or features.shape[1] != self.coef_.shape[0]:
            raise ValueError("X has an incompatible feature shape")
        return features @ self.coef_ + self.intercept_

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return $R^2$ on supplied data."""
        return r2_score(y, self.predict(X))


class PolynomialFeatures:
    """Generate monomials up to a specified degree without a bias column."""

    def __init__(self, degree: int = 2) -> None:
        if not isinstance(degree, int) or degree < 1:
            raise ValueError("degree must be a positive integer")
        self.degree = degree

    def fit_transform(self, X: ArrayLike) -> NDArray[np.float64]:
        """Return polynomial and interaction features."""
        features = np.asarray(X, dtype=float)
        if features.ndim != 2 or features.shape[1] == 0:
            raise ValueError("X must be a non-empty two-dimensional feature matrix")
        columns = []
        for degree in range(1, self.degree + 1):
            for combination in combinations_with_replacement(range(features.shape[1]), degree):
                columns.append(np.prod(features[:, combination], axis=1))
        return np.column_stack(columns)

    def transform(self, X: ArrayLike) -> NDArray[np.float64]:
        """Transform data using the configured polynomial degree."""
        return self.fit_transform(X)


class RidgeRegression:
    r"""Ridge regression using $\frac{1}{2n}\|r\|^2+\frac{\alpha}{2n}\|w\|^2$."""

    def __init__(
        self, alpha: float = 1.0, solver: str = "normal", lr: float = 0.01, max_iter: int = 1000
    ) -> None:
        if alpha < 0.0 or solver not in {"normal", "gd"}:
            raise ValueError("alpha must be non-negative and solver must be 'normal' or 'gd'")
        if lr <= 0.0 or max_iter < 1:
            raise ValueError("lr and max_iter must be positive")
        self.alpha = alpha
        self.solver = solver
        self.lr = lr
        self.max_iter = max_iter
        self.coef_: NDArray[np.float64] | None = None
        self.intercept_: float | None = None

    def fit(self, X: ArrayLike, y: ArrayLike) -> RidgeRegression:
        """Fit slopes on centered data so the intercept remains unpenalized."""
        features, target = _regression_arrays(X, y)
        feature_mean = features.mean(axis=0)
        target_mean = float(target.mean())
        centered_X = features - feature_mean
        centered_y = target - target_mean
        n_samples, n_features = centered_X.shape
        if self.solver == "normal":
            system = centered_X.T @ centered_X + self.alpha * np.eye(n_features)
            self.coef_ = np.linalg.solve(system, centered_X.T @ centered_y)
        else:

            def gradient(weights: NDArray[np.float64]) -> NDArray[np.float64]:
                return (
                    centered_X.T @ (centered_X @ weights - centered_y) / n_samples
                    + self.alpha * weights / n_samples
                )

            self.coef_, _ = gradient_descent(
                gradient, np.zeros(n_features), lr=self.lr, max_iter=self.max_iter
            )
        self.intercept_ = target_mean - float(feature_mean @ self.coef_)
        return self

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        """Predict continuous targets."""
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("fit must be called before predict")
        features = np.asarray(X, dtype=float)
        if features.ndim != 2 or features.shape[1] != self.coef_.shape[0]:
            raise ValueError("X has an incompatible feature shape")
        return features @ self.coef_ + self.intercept_

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return $R^2$ on supplied data."""
        return r2_score(y, self.predict(X))


class LassoRegression:
    r"""Lasso using $\frac{1}{2n}\|r\|^2+\alpha\|w\|_1$ and coordinate descent."""

    def __init__(self, alpha: float = 1.0, max_iter: int = 1000, tol: float = 1e-4) -> None:
        if alpha < 0.0 or max_iter < 1 or tol <= 0.0:
            raise ValueError("alpha must be non-negative; max_iter and tol must be positive")
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.coef_: NDArray[np.float64] | None = None
        self.intercept_: float | None = None

    def fit(self, X: ArrayLike, y: ArrayLike) -> LassoRegression:
        """Fit slopes on centered data so the intercept remains unpenalized."""
        features, target = _regression_arrays(X, y)
        feature_mean = features.mean(axis=0)
        target_mean = float(target.mean())
        centered_X = features - feature_mean
        centered_y = target - target_mean
        self.coef_ = coordinate_descent_lasso(
            centered_X, centered_y, self.alpha, self.max_iter, self.tol
        )
        self.intercept_ = target_mean - float(feature_mean @ self.coef_)
        return self

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        """Predict continuous targets."""
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("fit must be called before predict")
        features = np.asarray(X, dtype=float)
        if features.ndim != 2 or features.shape[1] != self.coef_.shape[0]:
            raise ValueError("X has an incompatible feature shape")
        return features @ self.coef_ + self.intercept_

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return $R^2$ on supplied data."""
        return r2_score(y, self.predict(X))


class LogisticRegression:
    r"""Binary logistic regression trained with full-batch gradient descent.

    Minimizes the mean negative log-likelihood plus an optional L2 penalty on
    the slopes, $\frac{1}{n}\sum_i \ell_i + \frac{l2}{2n}\|w\|^2$, where the
    intercept is never penalized. The default ``l2=0.0`` reproduces the
    unregularized behavior exactly.
    """

    def __init__(
        self, lr: float = 0.1, max_iter: int = 1000, tol: float = 1e-6, l2: float = 0.0
    ) -> None:
        if lr <= 0.0 or max_iter < 1 or tol <= 0.0:
            raise ValueError("lr, max_iter, and tol must be positive")
        if l2 < 0.0:
            raise ValueError("l2 must be non-negative")
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.l2 = l2
        self.coef_: NDArray[np.float64] | None = None
        self.intercept_: float | None = None
        self.classes_: NDArray | None = None

    @staticmethod
    def _sigmoid(z: NDArray[np.float64]) -> NDArray[np.float64]:
        clipped = np.clip(z, -250.0, 250.0)
        return 1.0 / (1.0 + np.exp(-clipped))

    def fit(self, X: ArrayLike, y: ArrayLike) -> LogisticRegression:
        """Fit a binary classifier and preserve the original class labels."""
        features = np.asarray(X, dtype=float)
        target = np.asarray(y)
        if features.ndim != 2 or target.ndim != 1 or features.shape[0] != target.shape[0]:
            raise ValueError("X and y have incompatible shapes")
        self.classes_ = np.unique(target)
        if self.classes_.size != 2:
            raise ValueError("LogisticRegression supports exactly two classes")
        encoded = (target == self.classes_[1]).astype(float)
        design = np.column_stack((np.ones(features.shape[0]), features))

        def gradient(theta: NDArray[np.float64]) -> NDArray[np.float64]:
            value = design.T @ (self._sigmoid(design @ theta) - encoded) / features.shape[0]
            if self.l2 > 0.0:
                value[1:] += self.l2 * theta[1:] / features.shape[0]
            return value

        theta, _ = gradient_descent(
            gradient, np.zeros(design.shape[1]), lr=self.lr, max_iter=self.max_iter, tol=self.tol
        )
        self.intercept_ = float(theta[0])
        self.coef_ = theta[1:]
        return self

    def predict_proba(self, X: ArrayLike) -> NDArray[np.float64]:
        """Return probabilities for the second class in ``classes_``."""
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("fit must be called before predict_proba")
        features = np.asarray(X, dtype=float)
        if features.ndim != 2 or features.shape[1] != self.coef_.shape[0]:
            raise ValueError("X has an incompatible feature shape")
        return self._sigmoid(features @ self.coef_ + self.intercept_)

    def predict(self, X: ArrayLike, threshold: float = 0.5) -> NDArray:
        """Predict original class labels at a probability threshold."""
        if self.classes_ is None:
            raise RuntimeError("fit must be called before predict")
        if not (0.0 <= threshold <= 1.0):
            raise ValueError("threshold must lie in [0, 1]")
        indices = (self.predict_proba(X) >= threshold).astype(int)
        return self.classes_[indices]

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return classification accuracy."""
        return accuracy(y, self.predict(X))
