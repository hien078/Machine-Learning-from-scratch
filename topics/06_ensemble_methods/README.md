# Ensemble Methods

> **Phase:** 2 | **Status:** ✅ Complete | **Prerequisites:** 05 Decision Tree, 02 Gradient Descent

## Overview

Combining multiple weak learners to build a stronger predictor. Covers bagging (bootstrap
aggregating for variance reduction), random forests (bagging + random feature subsets),
boosting (sequential models that correct predecessors' errors: AdaBoost reweights
examples, gradient boosting fits residuals), bias-variance decomposition of ensembles,
out-of-bag error estimation, and feature importance. Builds `RandomForestClassifier` and
`GradientBoostingRegressor` from scratch, compares with the `src/` library and sklearn,
and demonstrates failure modes (boosting overfitting on noisy data).

## Contents

| # | File | Type | Description |
|--:|---|---|---|
| 1 | `theory.md` | Theory | Bagging variance reduction derivation, random forest correlation argument, AdaBoost weight update, gradient boosting as functional gradient descent, OOB error, feature importance, failure cases |
| 2 | `first_principles.ipynb` | Computation | WHY→WHAT→HOW→BUILD→VERIFY — from-scratch RF and GBT, single tree vs ensemble comparison, OOB error, sklearn match, boosting-on-noise failure |
| 3 | `exercises.ipynb` | Practice | Hand calculation of AdaBoost weights, bagging implementation with deterministic check, conceptual question on random feature selection |

## Connections

- **Prereqs:** [05 Decision Tree](../05_decision_tree/README.md), [02 Gradient Descent](../02_gradient_descent/README.md)
- **Builds on:** [05 Decision Tree](../05_decision_tree/README.md) (ensembles of trees reduce single-tree variance)
- **Synthesis:** [Bias–Variance Tradeoff](../../synthesis/bias_variance_tradeoff.md)
- **Next:** [07 KNN](../07_knn/README.md)
