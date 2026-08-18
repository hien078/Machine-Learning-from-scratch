# Support Vector Machines

> **Phase:** 2 | **Status:** ✅ Complete | **Prerequisites:** Linear Algebra, Calculus & Optimization

## Overview

Maximum margin classification, hard-margin and soft-margin SVM formulations,
Lagrangian duality and KKT conditions, support vectors, hinge loss view,
sub-gradient descent optimization, kernel trick (linear, polynomial, RBF),
effect of the regularization parameter $C$, feature scaling sensitivity.

## Contents

| # | File | Type | Description |
|--:|---|---|---|
| 1 | `theory.md` | Theory | Maximum margin principle, hard/soft-margin primal, dual derivation, KKT conditions, support vectors, hinge loss, kernel trick, common kernels, failure cases |
| 2 | `first_principles.ipynb` | Computation | WHY→BUILD→VERIFY — from-scratch LinearSVC via sub-gradient descent, margin geometry, support vector identification, C parameter effect, sklearn comparison, kernel demo, failure cases |
| 3 | `exercises.ipynb` | Practice | Hand calculation of margin and support vectors, hinge loss + gradient coding task, conceptual questions on support vector sparsity |

## Connections

- **Prereqs:** [Linear Algebra](https://github.com/hien078/applied-mathematics-foundation), [Calculus & Optimization](https://github.com/hien078/applied-mathematics-foundation)
- **Builds on:** [04 Logistic Regression](../04_logistic_regression/README.md) (margin view comparison), [03 Regularization](../03_regularization/README.md) (L2 penalty as margin maximization)
- **Synthesis:** [Geometry of ML](../synthesis/geometry_of_ml.md), [Loss Functions](../synthesis/loss_functions_map.md)
- **Next:** [13 Neural Networks](../13_neural_networks/README.md) (single-layer NN with hinge loss = linear SVM)
