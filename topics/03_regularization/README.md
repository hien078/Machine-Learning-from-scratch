# Regularization (Ridge + Lasso)

> **Phase:** 1 | **Status:** ✅ Complete | **Prerequisites:** 01 Linear Regression, 02 GD

## Overview

Controlling overfitting by penalizing coefficient magnitude. The topic derives Ridge (L2) in closed form and interprets it as SVD shrinkage, derives Lasso (L1) via subgradients and soft-thresholding, explains geometrically why L1 produces sparsity, and reinterprets both penalties as Bayesian MAP estimation with Gaussian and Laplace priors.

## Scope

- **In scope:** the Ridge objective and its closed form, SVD shrinkage view, Lasso via subgradients and coordinate descent with soft-thresholding, elastic net, the constraint-region geometry of sparsity, and the MAP interpretation — all in pure NumPy.
- **Out of scope:** regularization inside neural networks (dropout, weight decay coupling with Adam) and cross-validation machinery beyond what the experiments need.

## Contents

| # | File | Type | Description |
|--:|---|---|---|
| 1 | [theory.md](theory.md) | Theory | WHY, objectives, closed forms, SVD shrinkage, subgradients, geometry, Bayesian views |
| 2 | [first_principles.ipynb](first_principles.ipynb) | Computation | WHY→WHAT→HOW→BUILD→VERIFY — Ridge & Lasso from scratch, sklearn comparison, failure cases |
| 3 | [exercises.ipynb](exercises.ipynb) | Practice | Hand derivation, soft-thresholding, sparse recovery, conceptual & failure-analysis questions |

The reusable, unit-tested reference implementations live in [`src/ml_first_principles/linear_models.py`](https://github.com/hien078/Machine-Learning-from-scratch/blob/master/src/ml_first_principles/linear_models.py) (`RidgeRegression`, `LassoRegression`), covered by `tests/test_linear_models.py`.

## Connections

- **Prereqs:** [01 Linear Regression](../01_linear_regression/README.md), [02 Gradient Descent](../02_gradient_descent/README.md)
- **Synthesis:** [Bias–Variance](../synthesis/bias_variance_tradeoff.md), [Geometry of ML](../synthesis/geometry_of_ml.md), [Regularization Across Models](../synthesis/regularization_across_models.md)
- **Next:** [04 Logistic Regression](../04_logistic_regression/README.md), [09 SVM](../09_svm/README.md) (regularized margin)
