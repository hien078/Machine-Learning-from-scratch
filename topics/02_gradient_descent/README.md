# Gradient Descent

> **Phase:** 1 | **Status:** ✅ Complete | **Prerequisites:** [01 Linear Regression](../01_linear_regression/README.md), [Calculus & Optimization](https://github.com/hien078/applied-mathematics-foundation)

## Overview

The foundational iterative optimization algorithm for machine learning. Covers full-batch GD, SGD, mini-batch SGD, momentum (Polyak heavy-ball, Nesterov), adaptive methods (AdaGrad, RMSProp, Adam), learning rate schedules, convergence theory, and failure modes.

## Scope

- **In scope:** the GD update rule and its convergence theory (convex $O(1/T)$, strongly convex linear rate, step-size condition $\alpha < 2/M$), SGD and mini-batching, momentum (heavy-ball, Nesterov), AdaGrad/RMSProp/Adam, learning-rate schedules, and divergence/oscillation failure modes — all in pure NumPy.
- **Out of scope:** second-order methods (Newton, L-BFGS), distributed/parallel SGD, and the backpropagation machinery that produces the gradients — that is [Topic 13](../13_neural_networks/README.md).

## Contents

| # | File | Type | Description |
|--:|---|---|---|
| 1 | [theory.md](theory.md) | Theory | GD update rule, convergence rates (convex $O(1/T)$, strongly convex linear rate), smoothness condition $\alpha < 2/M$, momentum, adaptive methods, schedules |
| 2 | [first_principles.ipynb](first_principles.ipynb) | Computation | From-scratch GD, momentum, Adam; learning rate effects; convergence race; library comparison |
| 3 | [exercises.ipynb](exercises.ipynb) | Practice | Hand-calc GD steps, implement momentum GD, conceptual questions on Adam vs SGD |

The reusable, unit-tested reference implementations live in [`src/ml_first_principles/optimizers.py`](https://github.com/hien078/Machine-Learning-from-scratch/blob/master/src/ml_first_principles/optimizers.py) (`SGD`, `Adam`), covered by `tests/test_optimizers.py`.

## Connections

- **Prereqs:** [01 Linear Regression](../01_linear_regression/README.md), [Calculus & Optimization](https://github.com/hien078/applied-mathematics-foundation)
- **Synthesis:** [Optimization Methods Compared](../synthesis/optimization_methods_compared.md)
- **Next:** [03 Regularization](../03_regularization/README.md), [04 Logistic Regression](../04_logistic_regression/README.md)
- **Used by:** Every iterative ML algorithm — logistic regression, SVM, neural networks, ensemble boosting
