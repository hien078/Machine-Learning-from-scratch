# Linear Regression (+ Polynomial)

> **Phase:** 1 | **Status:** ✅ Complete | **Prerequisites:** Linear Algebra, Calculus

## Overview

The entry point of the curriculum: fitting a linear model by least squares. The topic derives the MSE objective, proves its convexity via the Hessian, solves it three ways (normal equations, QR/SVD, gradient descent), interprets the solution geometrically as an orthogonal projection, and extends it to polynomial features — where overfitting appears for the first time.

## Scope

- **In scope:** the MSE objective and its gradient/Hessian, normal equations and their closed form, the projection view of least squares, gradient-descent fitting, polynomial feature maps, and train/test error as degree grows — all in pure NumPy.
- **Out of scope:** statistical inference on coefficients (confidence intervals, hypothesis tests), generalized least squares, and regularized variants — Ridge/Lasso are [Topic 03](../03_regularization/README.md).

## Contents

| File | Type | Description |
|------|------|-------------|
| [theory.md](theory.md) | Theory | MSE formulation, gradient/Hessian and convexity, normal equations, projection geometry, polynomial extension. |
| [first_principles.ipynb](first_principles.ipynb) | Code | NumPy least squares (closed form + gradient descent), sklearn comparison, polynomial fits, overfitting experiments. |
| [exercises.ipynb](exercises.ipynb) | Practice | Hand derivation of the normal equations, coding tasks with deterministic checks, conceptual questions. |

The reusable, unit-tested reference implementations live in [`src/ml_first_principles/linear_models.py`](https://github.com/hien078/Machine-Learning-from-scratch/blob/master/src/ml_first_principles/linear_models.py) (`LinearRegression`, `PolynomialFeatures`), covered by `tests/test_linear_models.py`.

## Connections

- **Prereqs:** [Linear Algebra](https://github.com/hien078/applied-mathematics-foundation), [Calculus](https://github.com/hien078/applied-mathematics-foundation)
- **Next:** [02 Gradient Descent](../02_gradient_descent/README.md), [03 Regularization](../03_regularization/README.md), [04 Logistic Regression](../04_logistic_regression/README.md)
