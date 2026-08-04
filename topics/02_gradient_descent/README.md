# Gradient Descent

> **Phase:** 1 | **Status:** ✅ Complete | **Prerequisites:** [01 Linear Regression](../01_linear_regression/README.md), [Calculus & Optimization](../../foundations/calculus_optimization/README.md)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hien078/Machine-Learning-from-scratch/blob/master/topics/02_gradient_descent/first_principles.ipynb)

## Overview

The foundational iterative optimization algorithm for machine learning. Covers full-batch GD, SGD, mini-batch SGD, momentum (Polyak heavy-ball, Nesterov), adaptive methods (AdaGrad, RMSProp, Adam), learning rate schedules, convergence theory, and failure modes.

## Contents

| # | File | Type | Description | Colab |
|--:|---|---|---|---|
| 1 | `theory.md` | Theory | GD update rule, convergence rates (convex $O(1/T)$, strongly convex linear rate), smoothness condition $\alpha < 2/M$, momentum, adaptive methods, schedules | — |
| 2 | `first_principles.ipynb` | Computation | From-scratch GD, momentum, Adam; learning rate effects; convergence race; library comparison | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hien078/Machine-Learning-from-scratch/blob/master/topics/02_gradient_descent/first_principles.ipynb) |
| 3 | `exercises.ipynb` | Practice | Hand-calc GD steps, implement momentum GD, conceptual questions on Adam vs SGD | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hien078/Machine-Learning-from-scratch/blob/master/topics/02_gradient_descent/exercises.ipynb) |

## Connections

- **Prereqs:** [01 Linear Regression](../01_linear_regression/README.md), [Calculus & Optimization](../../foundations/calculus_optimization/README.md)
- **Synthesis:** [Optimization Methods Compared](../../synthesis/optimization_methods_compared.md)
- **Next:** [03 Regularization](../03_regularization/README.md), [04 Logistic Regression](../04_logistic_regression/README.md)
- **Used by:** Every iterative ML algorithm — logistic regression, SVM, neural networks, ensemble boosting
