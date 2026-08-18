# Decision Tree

> **Phase:** 2 | **Status:** ✅ Complete | **Prerequisites:** Information Theory

## Overview

Recursive axis-aligned partitioning for classification and regression. Covers impurity
measures (Gini index, entropy, MSE), greedy split selection, information gain, stopping
criteria (max depth, min samples), cost-complexity pruning, and CART unification.
Builds a `DecisionTreeClassifier` from scratch, compares with the `src/` library
implementation and sklearn, and demonstrates instability (high variance) as motivation
for ensemble methods.

## Contents

| # | File | Type | Description |
|--:|---|---|---|
| 1 | `theory.md` | Theory | Impurity measures, information gain derivation, concavity proof (Jensen), greedy splitting, pruning, complexity $O(d \cdot n \log n)$, failure cases |
| 2 | `first_principles.ipynb` | Computation | WHY→WHAT→HOW→BUILD→VERIFY — from-scratch tree, Gini/entropy cells, depth experiment, sklearn comparison, instability demo |
| 3 | `exercises.ipynb` | Practice | Hand entropy/Gini calculation, information gain coding task, conceptual overfitting/pruning questions |

## Connections

- **Prereqs:** [Information Theory](https://github.com/hien078/applied-mathematics-foundation)
- **Builds on:** [04 Logistic Regression](../04_logistic_regression/README.md) (non-linear alternative)
- **Synthesis:** [Bias–Variance Tradeoff](../synthesis/bias_variance_tradeoff.md)
- **Next:** [06 Ensemble Methods](../06_ensemble_methods/README.md) (bagging/boosting reduce tree variance)
