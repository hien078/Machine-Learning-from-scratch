# Neural Networks

> **Phase:** 3 | **Status:** ✅ Complete | **Prerequisites:** 02 GD, 04 LogReg

## Overview

The step beyond linear models: multi-layer perceptrons. The topic derives the forward pass in matrix form, derives backpropagation layer by layer from the chain rule, frames it as reverse-mode differentiation on a computational graph, and covers initialization schemes, activation trade-offs, and the universal approximation theorem — verified against finite-difference gradient checks and PyTorch.

## Scope

- **In scope:** MLP forward pass, full backpropagation derivation, computational-graph view, Xavier/He initialization, activation functions and their gradients, the XOR problem, gradient checking, and vanishing-gradient failure cases — all in pure NumPy.
- **Out of scope:** convolutional and recurrent architectures ([Topic 14](../14_cnn/README.md), [Topic 15](../15_rnn_lstm/README.md)), attention ([Topic 16](../16_transformer/README.md)), and modern training tricks (batch norm, dropout).

## Contents

| # | File | Type | Description |
|--:|---|---|---|
| 1 | [theory.md](theory.md) | Theory | Motivation, notation, forward pass, backprop derivation, initialization, failure cases |
| 2 | [first_principles.ipynb](first_principles.ipynb) | Computation | MLP from scratch, XOR problem, gradient checking, PyTorch comparison, experiments |
| 3 | [exercises.ipynb](exercises.ipynb) | Practice | Hand calculation (forward + backprop), Dense layer gradient check, XOR capacity, vanishing gradients |

The reusable, unit-tested reference implementations live in [`src/ml_first_principles/nn_core.py`](https://github.com/hien078/Machine-Learning-from-scratch/blob/master/src/ml_first_principles/nn_core.py) (`Dense`, `ReLU`, `Sigmoid`, `Sequential`), covered by `tests/test_nn.py`.

## Connections

- **Prereqs:** [02 Gradient Descent](../02_gradient_descent/README.md), [04 Logistic Regression](../04_logistic_regression/README.md)
- **Synthesis:** Loss Functions, Optimization Methods
- **Next:** [14 CNN](../14_cnn/README.md), [15 RNN/LSTM](../15_rnn_lstm/README.md), [16 Transformer](../16_transformer/README.md), [17 Autoencoder](../17_autoencoder/README.md)
