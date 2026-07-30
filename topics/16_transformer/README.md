# Transformer

> **Phase:** 4 | **Status:** ✅ Complete | **Prerequisites:** 13 Neural Networks, 15 RNN/LSTM

## Overview

Self-attention mechanism, scaled dot-product attention, multi-head attention,
sinusoidal positional encoding, encoder block (self-attention + FFN + LayerNorm +
residual connections), decoder block (masked self-attention + cross-attention),
from-scratch NumPy implementation, comparison with PyTorch `nn.MultiheadAttention`.

## Contents

| # | File | Type | Description |
|--:|---|---|---|
| 1 | `theory.md` | Theory | Attention formula, scaling derivation, multi-head, positional encoding, encoder/decoder blocks, failure cases |
| 2 | `first_principles.ipynb` | Computation | WHY→BUILD→VERIFY — from-scratch attention, multi-head attention, positional encoding, encoder block, PyTorch comparison, attention pattern visualization, failure cases |
| 3 | `exercises.ipynb` | Practice | Hand calculation of attention weights, scaled dot-product attention coding task, conceptual questions on multi-head attention and causal masking |

## Connections

- **Prereqs:** [13 Neural Networks](../13_neural_networks/README.md), [15 RNN/LSTM](../15_rnn_lstm/README.md)
- **Builds on:** Feed-forward networks (MLP), softmax, cross-entropy loss
- **Next:** Large language models, vision transformers, attention variants (FlashAttention, sparse attention)
