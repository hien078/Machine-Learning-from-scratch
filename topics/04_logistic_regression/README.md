# Logistic Regression

> **Phase:** 1 | **Status:** ✅ Complete | **Prerequisites:** Probability, 02 GD

## Overview

Binary classification via sigmoid, cross-entropy / NLL loss, gradient and Hessian,
Newton / IRLS, decision boundary geometry, log-odds interpretation, regularization,
softmax multi-class extension, ROC / AUC, calibration.

## Contents

| # | File | Type | Description |
|--:|---|---|---|
| 1 | `theory.md` | Theory | Sigmoid identities, likelihood derivation, gradient & Hessian proofs, convexity, IRLS, regularization, softmax, statistical properties |
| 2 | `first_principles.ipynb` | Computation | WHY→BUILD→VERIFY — binary & softmax from scratch, GD vs Newton convergence, sklearn comparison, decision boundary, ROC/AUC, failure cases |
| 3 | `exercises.ipynb` | Practice | Sigmoid identities, cross-entropy calculation, gradient coding task, decision boundary visualization, perfect separation, log-odds interpretation |

## Connections

- **Prereqs:** [Probability & Statistics](https://github.com/hien078/applied-mathematics-foundation), [02 Gradient Descent](../02_gradient_descent/README.md)
- **Builds on:** [01 Linear Regression](../01_linear_regression/README.md), [03 Regularization](../03_regularization/README.md)
- **Synthesis:** [Loss Functions](../synthesis/loss_functions_map.md), [Probabilistic View](../synthesis/probabilistic_view_of_ml.md)
- **Next:** [09 SVM](../09_svm/README.md) (margin view), [13 Neural Networks](../13_neural_networks/README.md) (logistic = 1-layer NN)
