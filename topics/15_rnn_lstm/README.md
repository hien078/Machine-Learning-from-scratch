# RNN / LSTM

> **Phase:** 3 | **Status:** ✅ Complete | **Prerequisites:** 13 Neural Networks

## Overview

Recurrent neural networks for sequential data: vanilla RNN with parameter sharing
across time, backpropagation through time (BPTT), vanishing/exploding gradient problem,
LSTM gating mechanism (forget/input/output gates, cell state as gradient highway),
and GRU as a simplified alternative.

## Contents

| # | File | Type | Description |
|--:|---|---|---|
| 1 | `theory.md` | Theory | RNN equations, BPTT derivation, vanishing gradients, LSTM/GRU gates, failure cases |
| 2 | `first_principles.ipynb` | Computation | From-scratch RNN and LSTM cells, vanishing gradient demo, long-dependency comparison, PyTorch verification |
| 3 | `exercises.ipynb` | Practice | Hand calculation of RNN forward step, RNN forward pass implementation, LSTM gradient highway analysis |

## Connections

- **Prereqs:** [13 Neural Networks](../13_neural_networks/README.md)
- **Synthesis:** Sequence Modeling, Gradient Flow Analysis
- **Next:** [16 Transformer](../16_transformer/README.md) (attention replaces recurrence), [14 CNN](../14_cnn/README.md) (spatial vs temporal parameter sharing)
