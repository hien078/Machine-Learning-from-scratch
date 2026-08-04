# Regularization (Ridge + Lasso)

> **Phase:** 1 | **Status:** ✅ Complete | **Prerequisites:** 01 Linear Regression, 02 GD

## Overview

L2 penalty (Ridge), L1 penalty (Lasso), elastic net, constraint geometry,
sparsity, SVD shrinkage, soft-thresholding, Bayesian interpretation (MAP).

## Contents

| # | File | Type | Description |
|--:|---|---|---|
| 1 | `theory.md` | Theory | WHY, objectives, closed forms, SVD shrinkage, subgradients, geometry, Bayesian views |
| 2 | `first_principles.ipynb` | Computation | WHY→WHAT→HOW→BUILD→VERIFY — Ridge & Lasso from scratch, sklearn comparison, failure cases |
| 3 | `exercises.ipynb` | Practice | Hand derivation, soft-thresholding, sparse recovery, conceptual & failure-analysis questions |

## Connections

- **Prereqs:** [01 Linear Regression](../01_linear_regression/README.md), [02 Gradient Descent](../02_gradient_descent/README.md)
- **Synthesis:** [Bias–Variance](../../synthesis/bias_variance_tradeoff.md), [Geometry of ML](../../synthesis/geometry_of_ml.md), [Regularization Across Models](../../synthesis/regularization_across_models.md)
- **Next:** [04 Logistic Regression](../04_logistic_regression/README.md), [09 SVM](../09_svm/README.md) (regularized margin)
