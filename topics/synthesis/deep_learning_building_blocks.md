# Deep Learning Building Blocks — Cross-Topic Synthesis

> How the deep architectures (topics 13–17, 20) fit together as composable differentiable layers.
> See [INDEX.md](../../INDEX.md) for the full curriculum index.

---

## Overview

Every deep architecture is the same object: a composition of differentiable layers
trained end-to-end with backpropagation and an optimizer from
[Optimization Methods Compared](optimization_methods_compared.md). What distinguishes
MLP, CNN, RNN, attention, autoencoder, and GNN is **what each assumes about the
structure of the data** — and how that assumption is baked into the layer as an
inductive bias. Choosing an architecture is choosing a prior.

---

## Inductive Bias Table

| Architecture | Data assumption | Structural prior | Weight sharing over | Breaks when |
|---|---|---|---|---|
| [MLP (13)](../13_neural_networks/theory.md) | No structure — features are exchangeable | None (fully connected) | Nothing | High-dim structured inputs (too many parameters) |
| [CNN (14)](../14_cnn/theory.md) | Spatial locality + translation equivariance | Local receptive fields | Spatial positions | Permuted pixels, non-grid data |
| [RNN/LSTM (15)](../15_rnn_lstm/theory.md) | Sequential order, time-invariant dynamics | Recurrence $h_t = f(h_{t-1}, x_t)$ | Time steps | Very long dependencies, parallel hardware |
| [Attention (16)](../16_transformer/theory.md) | Pairwise relevance between elements (a set) | Content-based mixing, $O(1)$ path length | Positions (same $W^Q, W^K, W^V$ per token) | Long sequences ($O(n^2)$), missing positional signal |
| [Autoencoder (17)](../17_autoencoder/theory.md) | Data lies near a low-dim manifold (compressible) | Bottleneck $k < d$ | Optionally tied encoder/decoder | Bottleneck $\approx d$ (identity map), no manifold |
| [GNN (20)](../20_graph_neural_networks/theory.md) | Relations given by an explicit graph | Permutation-equivariant message passing | Nodes/edges (same $W^{(l)}$ everywhere) | Heterophily, deep stacks (over-smoothing) |

Each row *restricts* the MLP's hypothesis space: fewer free parameters, better sample
efficiency — **when the assumption holds**. CNN's failure on shuffled pixels
(topic 14, failure case 2) shows an inductive bias is a bet, not a free lunch.

---

## Parameter Sharing — The Common Trick

The single most reused idea: apply the **same** weights at every "location" the data's
symmetry group moves over.

```math
\text{CNN: } Y_{i,j} = \sum_{m,n} K_{m,n}\, X_{i+m,\, j+n}
\qquad
\text{RNN: } h_t = \tanh(W_{xh}\, x_t + W_{hh}\, h_{t-1} + b_h)
```

```math
\text{GNN: } H^{(l+1)} = \sigma\!\left(\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(l)} W^{(l)}\right)
```

- **CNN** ties $K$ across spatial positions → translation equivariance, 9 parameters
  for a $3 \times 3$ kernel regardless of image size.
- **RNN** ties $(W_{xh}, W_{hh})$ across time → variable-length sequences, fixed
  parameter count.
- **Transformer** ties $W^Q, W^K, W^V$ and the FFN across positions → permutation
  equivariance (order re-injected via positional encoding).
- **GNN** ties $W^{(l)}$ across nodes → permutation equivariance, generalization to
  unseen graph sizes.

Same mechanism, different symmetry: translation (space), time-shift (sequences),
permutation (sets and graphs).

---

## Gradient-Flow Pathologies and Their Fixes

Backpropagation multiplies Jacobians along the computational path. Every architecture
innovation in this phase is a response to some product of Jacobians shrinking or
exploding.

| Pathology | Where it appears | Mechanism | Fix | Topic |
|---|---|---|---|---|
| Vanishing through time | RNN, $\prod_s \text{diag}(\tanh'(z_s)) W_{hh}$ | $(\gamma \lambda_{\max})^{T-t} \to 0$ | LSTM gates: additive cell path $\partial c_T / \partial c_t = \prod_s f_s$ | [15](../15_rnn_lstm/theory.md) |
| Exploding through time | RNN, same product | $(\gamma \lambda_{\max})^{T-t} \to \infty$ | Gradient clipping | [15](../15_rnn_lstm/theory.md) |
| Vanishing through depth | Deep MLP/CNN stacks | Repeated multiplication + saturating activations | Residual connections $X + \text{SubLayer}(X)$, ReLU, He/Xavier init, LayerNorm | [13](../13_neural_networks/theory.md), [16](../16_transformer/theory.md) |
| Long gradient path over sequence length | LSTM at $T \gtrsim 500$ | Even $\prod f_s$ decays slowly | Attention: $O(1)$ path between any two tokens | [16](../16_transformer/theory.md) |
| Softmax saturation | Attention logits | $\text{Var}(q^\top k) = d_k$ pushes softmax to 0/1 | Scale by $\sqrt{d_k}$ | [16](../16_transformer/theory.md) |
| Over-smoothing | Deep GCN | Repeated low-pass filtering by $\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$ | Shallow stacks (2–4 layers), residuals | [20](../20_graph_neural_networks/theory.md) |

The unifying pattern: **replace a long multiplicative path with a short or additive
one**. The LSTM cell state, the residual connection, and attention's direct
token-to-token edge are three instances of the same idea at different granularities.

---

## GNNs Generalize Convolution

A CNN is a GNN on a grid graph where each pixel connects to its neighbors — with one
extra luxury: the grid gives neighbors a *canonical order* (up/down/left/right), so the
kernel can assign a distinct weight per relative position. General graphs have no such
order, so the aggregation must be permutation invariant:

```math
m_i^{(l+1)} = \sum_{j \in \mathcal{N}(i)} M_l\!\left(h_i^{(l)}, h_j^{(l)}, e_{ij}\right),
\qquad
h_i^{(l+1)} = U_l\!\left(h_i^{(l)}, m_i^{(l+1)}\right)
```

| Concept | CNN version | GNN version |
|---|---|---|
| Neighborhood | Fixed $k \times k$ window | $\mathcal{N}(i)$, variable size |
| Weights per neighbor | Distinct (position-indexed) | Shared or attention-weighted (GAT) |
| Receptive field growth | $r = L(k-1) + 1$ | $L$-hop neighborhood |
| Downsampling | Pooling | Graph pooling / readout |
| Depth limit | Vanishing gradients (fixed by residuals) | Over-smoothing |

GAT closes the circle: it reuses the [Transformer's](../16_transformer/theory.md)
attention on sparse neighborhoods. A Transformer is, in this view, a GNN on a fully
connected graph with positional features.

---

## Composability

Because every block maps tensors to tensors differentiably, they stack freely: CNN
encoder + LSTM decoder, CNN inside an autoencoder, attention inside a GNN (GAT), MLP
inside every Transformer block (the FFN sub-layer). The autoencoder shows most clearly
that the *objective* (reconstruction) is orthogonal to the *architecture* (any
encoder/decoder pair) — developed further in
[Generative and Self-Supervised Learning](generative_and_self_supervised.md).

---

## Decision Guide

```
What structure does the data have?
├── Tabular / no structure → MLP (13)
├── Grid (image, spectrogram) → CNN (14)
├── Sequence
│   ├── short, streaming, low compute → RNN/LSTM (15)
│   └── long-range dependencies, parallel training → Transformer (16)
├── Explicit graph → GNN (20)
└── Unlabeled, want compressed representation → Autoencoder (17)
```

---

## Connections

- **Topics:** [13 Neural Networks](../13_neural_networks/theory.md), [14 CNN](../14_cnn/theory.md), [15 RNN/LSTM](../15_rnn_lstm/theory.md), [16 Transformer](../16_transformer/theory.md), [17 Autoencoder](../17_autoencoder/theory.md), [20 Graph Neural Networks](../20_graph_neural_networks/theory.md)
- **Related synthesis:** [Optimization Methods Compared](optimization_methods_compared.md) (how these are trained), [Sequence Models and Attention](sequence_models_and_attention.md) (the RNN → Transformer arc in detail), [Bias-Variance Tradeoff](bias_variance_tradeoff.md) (inductive bias as variance reduction)
- **Maps:** [INDEX.md](../../INDEX.md)
