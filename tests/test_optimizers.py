"""
Unit tests for optimizers (GD, SGD, Adam).
"""

import numpy as np
import pytest
from sklearn.linear_model import Lasso

from ml_first_principles.optimizers import (
    Adam,
    adam,
    coordinate_descent_lasso,
    gradient_descent,
    sgd,
)


def test_gradient_descent():
    # Minimize f(x) = x^2 + 2x + 1 = (x+1)^2
    # Minimum at x = -1
    def grad_fn(x):
        return 2 * x + 2

    x0 = np.array([5.0])
    x_opt, history = gradient_descent(grad_fn, x0, lr=0.1, max_iter=100)

    assert np.allclose(x_opt, [-1.0], atol=1e-3)
    assert len(history) > 1


def test_adam():
    # Minimize f(x, y) = x^2 + y^2
    # Minimum at (0, 0)
    def grad_fn(x):
        return 2 * x

    x0 = np.array([5.0, -3.0])
    x_opt, history = adam(grad_fn, x0, lr=0.1, max_iter=500)

    assert np.allclose(x_opt, [0.0, 0.0], atol=1e-2)


def test_sgd():
    # Linear regression with SGD
    np.random.seed(42)
    X = np.random.randn(100, 2)
    true_w = np.array([3.0, -1.5])
    y = X @ true_w

    def grad_batch_fn(w, batch_indices):
        X_b = X[batch_indices]
        y_b = y[batch_indices]
        return (2 / len(batch_indices)) * X_b.T @ (X_b @ w - y_b)

    w0 = np.zeros(2)
    w_opt, history = sgd(grad_batch_fn, w0, n_samples=100, lr=0.1, max_iter=50, batch_size=10)

    np.testing.assert_allclose(w_opt, true_w, rtol=1e-2)


def test_coordinate_descent_lasso_matches_sklearn():
    # Both objectives are (1/2n)||y - Xw||^2 + lam * ||w||_1 without intercept.
    rng = np.random.default_rng(0)
    X = rng.standard_normal((60, 4))
    true_w = np.array([2.0, 0.0, -3.0, 0.0])
    y = X @ true_w + 0.01 * rng.standard_normal(60)
    lam = 0.1
    weights = coordinate_descent_lasso(X, y, lam=lam, max_iter=2000, tol=1e-8)
    reference = Lasso(alpha=lam, fit_intercept=False, max_iter=50_000, tol=1e-10).fit(X, y)
    np.testing.assert_allclose(weights, reference.coef_, atol=1e-4)


def test_adam_keep_history_false_returns_only_final_iterate():
    def grad_fn(x):
        return 2 * x

    x0 = np.array([5.0, -3.0])
    x_full, history_full = adam(grad_fn, x0, lr=0.1, max_iter=500)
    x_final, history = adam(grad_fn, x0, lr=0.1, max_iter=500, keep_history=False)

    np.testing.assert_array_equal(x_final, x_full)
    assert len(history_full) > 1
    assert len(history) == 1
    np.testing.assert_array_equal(history[0], x_final)


def test_adam_class_step_matches_char_transformer_reference():
    # Reference math mirrors the local Adam in projects/char_transformer_tiny/src/ct_model.py:
    # per-tensor first/second moments, one shared timestep, bias correction, in-place update.
    def grads_of(params):
        return {name: 2.0 * (p - 3.0) for name, p in params.items()}

    rng = np.random.default_rng(0)
    init = {"w": rng.standard_normal(4), "b": rng.standard_normal(2)}

    params = {name: value.copy() for name, value in init.items()}
    optimizer = Adam(learning_rate=0.05)
    for _ in range(200):
        optimizer.step(params, grads_of(params))

    reference = {name: value.copy() for name, value in init.items()}
    m = {name: np.zeros_like(value) for name, value in init.items()}
    v = {name: np.zeros_like(value) for name, value in init.items()}
    for t in range(1, 201):
        grads = grads_of(reference)
        for name, p in reference.items():
            m[name] = 0.9 * m[name] + 0.1 * grads[name]
            v[name] = 0.999 * v[name] + 0.001 * grads[name] ** 2
            m_hat = m[name] / (1.0 - 0.9**t)
            v_hat = v[name] / (1.0 - 0.999**t)
            p -= 0.05 * m_hat / (np.sqrt(v_hat) + 1e-8)

    for name in init:
        np.testing.assert_allclose(params[name], reference[name])


def test_adam_class_converges_and_updates_in_place():
    params = {"w": np.array([5.0, -3.0])}
    original = params["w"]
    optimizer = Adam(learning_rate=0.1)
    for _ in range(500):
        optimizer.step(params, {"w": 2.0 * params["w"]})

    assert params["w"] is original
    assert optimizer.timestep == 500
    np.testing.assert_allclose(params["w"], [0.0, 0.0], atol=1e-2)


@pytest.mark.parametrize(
    "kwargs",
    [{"learning_rate": 0.0}, {"epsilon": 0.0}, {"beta1": 1.0}, {"beta2": -0.1}],
)
def test_adam_class_rejects_invalid_hyperparameters(kwargs):
    with pytest.raises(ValueError):
        Adam(**kwargs)


def test_adam_class_step_rejects_invalid_inputs():
    optimizer = Adam()
    params = {"w": np.zeros(3)}
    with pytest.raises(ValueError, match="missing"):
        optimizer.step(params, {})
    with pytest.raises(ValueError, match="shape"):
        optimizer.step(params, {"w": np.zeros(4)})
    optimizer.step(params, {"w": np.ones(3)})
    with pytest.raises(ValueError, match="keys"):
        optimizer.step({"v": np.zeros(3)}, {"v": np.ones(3)})
