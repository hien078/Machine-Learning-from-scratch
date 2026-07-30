from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

Gradient = Callable[[NDArray[np.float64]], NDArray[np.float64]]
BatchGradient = Callable[[NDArray[np.float64], NDArray[np.int64]], NDArray[np.float64]]


def _initial_point(x0: ArrayLike) -> NDArray[np.float64]:
    point = np.asarray(x0, dtype=float).copy()
    if point.ndim != 1 or point.size == 0 or np.any(~np.isfinite(point)):
        raise ValueError("x0 must be a non-empty finite one-dimensional array")
    return point


def _checked_gradient(gradient: ArrayLike, shape: tuple[int, ...]) -> NDArray[np.float64]:
    value = np.asarray(gradient, dtype=float)
    if value.shape != shape or np.any(~np.isfinite(value)):
        raise ValueError("gradient must be finite and have the same shape as the parameter")
    return value


def gradient_descent(
    gradient_fn: Gradient,
    x0: ArrayLike,
    lr: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> tuple[NDArray[np.float64], list[NDArray[np.float64]]]:
    """Minimize an objective using full-batch gradient descent."""
    if lr <= 0.0 or max_iter < 1 or tol <= 0.0:
        raise ValueError("lr, max_iter, and tol must be positive")
    x = _initial_point(x0)
    history = [x.copy()]
    for _ in range(max_iter):
        gradient = _checked_gradient(gradient_fn(x), x.shape)
        x_new = x - lr * gradient
        if np.any(~np.isfinite(x_new)):
            raise FloatingPointError("gradient descent produced a non-finite iterate")
        history.append(x_new.copy())
        if np.linalg.norm(x_new - x) <= tol:
            return x_new, history
        x = x_new
    return x, history


def sgd(
    gradient_batch_fn: BatchGradient,
    x0: ArrayLike,
    n_samples: int,
    lr: float = 0.01,
    max_iter: int = 100,
    batch_size: int = 32,
    random_state: int | None = None,
) -> tuple[NDArray[np.float64], list[NDArray[np.float64]]]:
    """Minimize an empirical objective using shuffled mini-batches."""
    if n_samples < 1 or lr <= 0.0 or max_iter < 1 or not 1 <= batch_size <= n_samples:
        raise ValueError("invalid sample, learning-rate, iteration, or batch-size value")
    x = _initial_point(x0)
    rng = np.random.default_rng(random_state)
    history = [x.copy()]
    for _ in range(max_iter):
        indices = rng.permutation(n_samples)
        for start in range(0, n_samples, batch_size):
            batch = indices[start : start + batch_size]
            gradient = _checked_gradient(gradient_batch_fn(x, batch), x.shape)
            x = x - lr * gradient
            if np.any(~np.isfinite(x)):
                raise FloatingPointError("SGD produced a non-finite iterate")
        history.append(x.copy())
    return x, history


def adam(
    gradient_fn: Gradient,
    x0: ArrayLike,
    lr: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> tuple[NDArray[np.float64], list[NDArray[np.float64]]]:
    """Minimize an objective using Adam with bias correction."""
    if lr <= 0.0 or eps <= 0.0 or max_iter < 1 or tol <= 0.0:
        raise ValueError("lr, eps, max_iter, and tol must be positive")
    if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
        raise ValueError("beta1 and beta2 must lie in [0, 1)")
    x = _initial_point(x0)
    first_moment = np.zeros_like(x)
    second_moment = np.zeros_like(x)
    history = [x.copy()]
    for iteration in range(1, max_iter + 1):
        gradient = _checked_gradient(gradient_fn(x), x.shape)
        first_moment = beta1 * first_moment + (1.0 - beta1) * gradient
        second_moment = beta2 * second_moment + (1.0 - beta2) * gradient**2
        corrected_first = first_moment / (1.0 - beta1**iteration)
        corrected_second = second_moment / (1.0 - beta2**iteration)
        x_new = x - lr * corrected_first / (np.sqrt(corrected_second) + eps)
        if np.any(~np.isfinite(x_new)):
            raise FloatingPointError("Adam produced a non-finite iterate")
        history.append(x_new.copy())
        if np.linalg.norm(x_new - x) <= tol:
            return x_new, history
        x = x_new
    return x, history


def coordinate_descent_lasso(
    X: ArrayLike,
    y: ArrayLike,
    lam: float,
    max_iter: int = 1000,
    tol: float = 1e-4,
) -> NDArray[np.float64]:
    r"""Minimize $\frac{1}{2n}\|y-Xw\|_2^2+\lambda\|w\|_1$ by coordinate descent."""
    features = np.asarray(X, dtype=float)
    target = np.asarray(y, dtype=float)
    if features.ndim != 2 or target.ndim != 1 or features.shape[0] != target.shape[0]:
        raise ValueError("X and y have incompatible shapes")
    if features.shape[0] == 0 or lam < 0.0 or max_iter < 1 or tol <= 0.0:
        raise ValueError("data must be non-empty and parameters must be valid")
    n_samples, n_features = features.shape
    weights = np.zeros(n_features)
    squared_norms = np.sum(features**2, axis=0) / n_samples
    active = squared_norms > 0.0

    for _ in range(max_iter):
        previous = weights.copy()
        for feature in range(n_features):
            if not active[feature]:
                weights[feature] = 0.0
                continue
            residual_without_feature = (
                target - features @ weights + features[:, feature] * weights[feature]
            )
            correlation = features[:, feature] @ residual_without_feature / n_samples
            thresholded = np.sign(correlation) * max(abs(correlation) - lam, 0.0)
            weights[feature] = thresholded / squared_norms[feature]
        if np.max(np.abs(weights - previous), initial=0.0) <= tol:
            break
    return weights
