# Convolutional Neural Networks

> **Phase:** 3 | **Status:** ✅ Complete | **Prerequisites:** 13 Neural Networks

## Overview

Convolution (cross-correlation), pooling, feature maps, translation equivariance,
parameter sharing, and building a small CNN from scratch. Covers the forward and
backward pass of convolutional and pooling layers, receptive fields, and common
architecture patterns (LeNet onward).

## Contents

| # | File | Type | Description |
|--:|---|---|---|
| 1 | `theory.md` | Theory | Convolution operation, padding, stride, pooling, backprop through conv, parameter counting, receptive field, architectures, failure cases |
| 2 | `first_principles.ipynb` | Computation | Conv2D and MaxPool2D from scratch, edge detection demo, small CNN on pattern task, PyTorch comparison, filter visualization, shuffled-pixel failure |
| 3 | `exercises.ipynb` | Practice | Hand convolution calculation, cross-correlation coding task, conceptual questions on parameter sharing and receptive field |

## Connections

- **Prereqs:** [13 Neural Networks](../13_neural_networks/README.md)
- **Synthesis:** Loss Functions, Optimization Methods
- **Next:** [15 RNN/LSTM](../15_rnn_lstm/README.md), [16 Transformer](../16_transformer/README.md)
