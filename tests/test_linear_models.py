"""
Unit tests for linear models (Linear, Ridge, Lasso, Logistic).
"""

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.linear_model import Ridge as SklearnRidge

from ml_first_principles.data_utils import (
    generate_classification_data,
    generate_regression_data,
    standardize,
)
from ml_first_principles.linear_models import (
    LassoRegression,
    LinearRegression,
    LogisticRegression,
    PolynomialFeatures,
    RidgeRegression,
)


def test_linear_regression():
    X, y = generate_regression_data(n_samples=100, n_features=3, noise=0.1, random_state=42)

    # Test normal equation
    model = LinearRegression(solver="normal")
    model.fit(X, y)
    score = model.score(X, y)
    assert score > 0.9  # Should be very high on simple linear data

    # Test GD
    model_gd = LinearRegression(solver="gd", lr=0.1, max_iter=1000)
    model_gd.fit(X, y)
    score_gd = model_gd.score(X, y)
    assert score_gd > 0.9

    # Check that coefficients are similar
    np.testing.assert_allclose(model.coef_, model_gd.coef_, rtol=1e-2)

    reference = SklearnLinearRegression().fit(X, y)
    np.testing.assert_allclose(model.coef_, reference.coef_, atol=1e-10)
    assert np.isclose(model.intercept_, reference.intercept_, atol=1e-10)


def test_polynomial_features():
    X = np.array([[1], [2], [3]])
    poly = PolynomialFeatures(degree=3)
    X_poly = poly.fit_transform(X)

    assert X_poly.shape == (3, 3)
    np.testing.assert_array_equal(X_poly[0], [1, 1, 1])
    np.testing.assert_array_equal(X_poly[1], [2, 4, 8])
    np.testing.assert_array_equal(X_poly[2], [3, 9, 27])


def test_ridge_regression():
    X, y = generate_regression_data(n_samples=50, n_features=5, noise=0.5, random_state=42)
    X, _, _ = standardize(X)

    model = RidgeRegression(alpha=1.0, solver="normal")
    model.fit(X, y)

    model_gd = RidgeRegression(alpha=1.0, solver="gd", lr=0.1, max_iter=2000)
    model_gd.fit(X, y)

    np.testing.assert_allclose(model.coef_, model_gd.coef_, rtol=1e-2)

    reference = SklearnRidge(alpha=1.0).fit(X, y)
    np.testing.assert_allclose(model.coef_, reference.coef_, atol=1e-10)
    assert np.isclose(model.intercept_, reference.intercept_, atol=1e-10)


def test_lasso_regression():
    # Create data with sparse true weights
    np.random.seed(42)
    X = np.random.randn(100, 10)
    true_w = np.zeros(10)
    true_w[0] = 5.0
    true_w[3] = -3.0
    true_b = 2.0
    y = X @ true_w + true_b + np.random.randn(100) * 0.1

    X, _, _ = standardize(X)

    model = LassoRegression(alpha=0.5, max_iter=1000)
    model.fit(X, y)

    # Check sparsity (many coefficients should be exactly zero)
    zeros = np.sum(np.abs(model.coef_) < 1e-10)
    assert zeros >= 5  # At least half should be zeroed out

    # Check non-zero coefficients
    assert abs(model.coef_[0]) > 1.0
    assert abs(model.coef_[3]) > 1.0


def test_logistic_regression():
    X, y = generate_classification_data(n_samples=200, n_features=2, n_classes=2, random_state=42)
    X, _, _ = standardize(X)

    model = LogisticRegression(lr=0.1, max_iter=1000)
    model.fit(X, y)

    score = model.score(X, y)
    assert score > 0.8  # Should perform well on simple Gaussian clusters

    # Check probabilities
    probs = model.predict_proba(X)
    assert np.all((probs >= 0) & (probs <= 1))


def test_logistic_regression_preserves_original_labels():
    X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array(["negative", "negative", "positive", "positive"])
    model = LogisticRegression(lr=0.2, max_iter=2000).fit(X, y)
    np.testing.assert_array_equal(model.predict(X), y)


def test_logistic_regression_l2_zero_matches_unregularized():
    X, y = generate_classification_data(n_samples=100, n_features=3, n_classes=2, random_state=0)
    X, _, _ = standardize(X)
    base = LogisticRegression(lr=0.1, max_iter=500).fit(X, y)
    explicit = LogisticRegression(lr=0.1, max_iter=500, l2=0.0).fit(X, y)
    np.testing.assert_array_equal(base.coef_, explicit.coef_)
    assert base.intercept_ == explicit.intercept_


def test_logistic_regression_l2_shrinks_weights_and_still_fits():
    X, y = generate_classification_data(n_samples=200, n_features=2, n_classes=2, random_state=42)
    X, _, _ = standardize(X)
    plain = LogisticRegression(lr=0.1, max_iter=1000).fit(X, y)
    penalized = LogisticRegression(lr=0.1, max_iter=1000, l2=100.0).fit(X, y)
    assert np.linalg.norm(penalized.coef_) < np.linalg.norm(plain.coef_)
    assert penalized.score(X, y) > 0.8


def test_logistic_regression_rejects_negative_l2():
    with pytest.raises(ValueError, match="l2"):
        LogisticRegression(l2=-0.1)


def test_models_validate_solver_and_fitted_state():
    with pytest.raises(ValueError, match="solver"):
        RidgeRegression(solver="bad")
    with pytest.raises(RuntimeError, match="fit"):
        LinearRegression().predict(np.ones((2, 1)))
