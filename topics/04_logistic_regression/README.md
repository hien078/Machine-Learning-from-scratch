# Logistic Regression

> **Phase:** 1 | **Status:** ✅ Complete | **Prerequisites:** Probability, 02 GD

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hien078/Machine-Learning-from-scratch/blob/master/topics/04_logistic_regression/first_principles.ipynb)

## Overview

Binary classification via sigmoid, cross-entropy / NLL loss, gradient and Hessian,
Newton / IRLS, decision boundary geometry, log-odds interpretation, regularization,
softmax multi-class extension, ROC / AUC, calibration.

## Contents

| # | File | Type | Description | Colab |
|--:|---|---|---|---|
| 1 | `theory.md` | Theory | Sigmoid identities, likelihood derivation, gradient & Hessian proofs, convexity, IRLS, regularization, softmax, statistical properties | — |
| 2 | `first_principles.ipynb` | Computation | WHY→BUILD→VERIFY — binary & softmax from scratch, GD vs Newton convergence, sklearn comparison, decision boundary, ROC/AUC, failure cases | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hien078/Machine-Learning-from-scratch/blob/master/topics/04_logistic_regression/first_principles.ipynb) |
| 3 | `exercises.ipynb` | Practice | Sigmoid identities, cross-entropy calculation, gradient coding task, decision boundary visualization, perfect separation, log-odds interpretation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hien078/Machine-Learning-from-scratch/blob/master/topics/04_logistic_regression/exercises.ipynb) |

## Connections

- **Prereqs:** [Probability & Statistics](../../foundations/probability_statistics/README.md), [02 Gradient Descent](../02_gradient_descent/README.md)
- **Builds on:** [01 Linear Regression](../01_linear_regression/README.md), [03 Regularization](../03_regularization/README.md)
- **Synthesis:** [Loss Functions](../../synthesis/loss_functions_map.md), [Probabilistic View](../../synthesis/probabilistic_view_of_ml.md)
- **Next:** [09 SVM](../09_svm/README.md) (margin view), [13 Neural Networks](../13_neural_networks/README.md) (logistic = 1-layer NN)
