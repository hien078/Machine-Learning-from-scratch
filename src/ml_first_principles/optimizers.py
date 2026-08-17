"""First-order optimizers: gradient descent, SGD, Adam, and coordinate descent."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

Gradient = Callable[[NDArray[np.float64]], NDArray[np.float64]]
BatchGradient = Callable[[NDArray[np.float64], NDArray[np.int64]], NDArray[np.float64]]
ParamDict = dict[str, NDArray[np.float64]]


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
    keep_history: bool = True,
) -> tuple[NDArray[np.float64], list[NDArray[np.float64]]]:
    """Minimize an objective using Adam with bias correction.

    When ``keep_history`` is ``False`` the returned history is a single-element
    list containing only the final iterate, which avoids storing every
    intermediate point on long runs. The default keeps the full iterate path.
    """
    if lr <= 0.0 or eps <= 0.0 or max_iter < 1 or tol <= 0.0:
        raise ValueError("lr, eps, max_iter, and tol must be positive")
    if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
        raise ValueError("beta1 and beta2 must lie in [0, 1)")
    x = _initial_point(x0)
    first_moment = np.zeros_like(x)
    second_moment = np.zeros_like(x)
    history = [x.copy()] if keep_history else []
    for iteration in range(1, max_iter + 1):
        gradient = _checked_gradient(gradient_fn(x), x.shape)
        first_moment = beta1 * first_moment + (1.0 - beta1) * gradient
        second_moment = beta2 * second_moment + (1.0 - beta2) * gradient**2
        corrected_first = first_moment / (1.0 - beta1**iteration)
        corrected_second = second_moment / (1.0 - beta2**iteration)
        x_new = x - lr * corrected_first / (np.sqrt(corrected_second) + eps)
        if np.any(~np.isfinite(x_new)):
            raise FloatingPointError("Adam produced a non-finite iterate")
        if keep_history:
            history.append(x_new.copy())
        if np.linalg.norm(x_new - x) <= tol:
            return x_new, history if keep_history else [x_new.copy()]
        x = x_new
    return x, history if keep_history else [x.copy()]


class Adam:
    """Step-based Adam optimizer over dictionaries of named parameter arrays.

    Unlike :func:`adam`, which drives its own full optimization loop over a
    single flat vector, this class applies one bias-corrected Adam update per
    ``step`` call, so it can be embedded in external (e.g. minibatch) training
    loops. First and second moments are kept per tensor and all tensors share
    a single timestep.

    ``step`` updates the parameter arrays **in place** and returns ``None``,
    so callers keep using the same ``params`` dictionary across steps.

    Example:
        >>> optimizer = Adam(learning_rate=0.1)
        >>> params = {"w": np.array([5.0])}
        >>> for _ in range(100):
        ...     optimizer.step(params, {"w": 2.0 * params["w"]})
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        """Configure the optimizer hyperparameters.

        Args:
            learning_rate: Positive step size applied to each update.
            beta1: Exponential decay rate of the first moment, in ``[0, 1)``.
            beta2: Exponential decay rate of the second moment, in ``[0, 1)``.
            epsilon: Positive constant added to the denominator for stability.

        Raises:
            ValueError: If any hyperparameter lies outside its valid range.
        """
        if learning_rate <= 0.0 or epsilon <= 0.0:
            raise ValueError("learning_rate and epsilon must be positive")
        if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
            raise ValueError("beta1 and beta2 must lie in [0, 1)")
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.timestep = 0
        self._first_moment: ParamDict = {}
        self._second_moment: ParamDict = {}

    def step(self, params: ParamDict, grads: ParamDict) -> None:
        """Apply one bias-corrected Adam update to every parameter in place.

        Args:
            params: Mapping from parameter name to a float array. The arrays
                are mutated in place; the key set must stay identical across
                calls. Extra keys in ``grads`` are ignored.
            grads: Mapping from parameter name to a gradient array with the
                same shape as the matching parameter.

        Raises:
            ValueError: If a gradient is missing, non-finite, or has a shape
                different from its parameter, or if the parameter key set
                changed between calls.
            FloatingPointError: If an update produces a non-finite parameter.
        """
        missing = sorted(set(params) - set(grads))
        if missing:
            raise ValueError(f"grads is missing entries for parameters: {missing}")
        checked = {name: _checked_gradient(grads[name], p.shape) for name, p in params.items()}
        if not self._first_moment:
            self._first_moment = {name: np.zeros_like(p, dtype=float) for name, p in params.items()}
            self._second_moment = {
                name: np.zeros_like(p, dtype=float) for name, p in params.items()
            }
        elif set(params) != set(self._first_moment):
            raise ValueError("params must keep the same keys across step calls")
        self.timestep += 1
        for name, parameter in params.items():
            gradient = checked[name]
            self._first_moment[name] = (
                self.beta1 * self._first_moment[name] + (1.0 - self.beta1) * gradient
            )
            self._second_moment[name] = (
                self.beta2 * self._second_moment[name] + (1.0 - self.beta2) * gradient**2
            )
            corrected_first = self._first_moment[name] / (1.0 - self.beta1**self.timestep)
            corrected_second = self._second_moment[name] / (1.0 - self.beta2**self.timestep)
            parameter -= (
                self.learning_rate * corrected_first / (np.sqrt(corrected_second) + self.epsilon)
            )
            if np.any(~np.isfinite(parameter)):
                raise FloatingPointError("Adam produced a non-finite parameter")


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
