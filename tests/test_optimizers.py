"""
Unit tests for optimizers (GD, SGD, Adam).
"""

import numpy as np
from sklearn.linear_model import Lasso

from ml_first_principles.optimizers import adam, coordinate_descent_lasso, gradient_descent, sgd


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
