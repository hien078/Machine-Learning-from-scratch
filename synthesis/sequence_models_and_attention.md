# Sequence Models and Attention — Cross-Topic Synthesis

> The arc from RNN to LSTM to attention to Transformer to LLM engineering (topics 15, 16, 21).
> See [INDEX.md](../INDEX.md) for the full curriculum index.

---

## Overview

Sequence modeling from 1997 to today is one continuous story: each step exists because
the previous one hit a concrete wall. This document traces the chain and compares the
designs on the axes that actually drove the transitions — gradient path length,
parallelism, memory, and how order is represented.

---

## The Arc

| Step | Wall it hit | What the next step solved |
|---|---|---|
| [Vanilla RNN (15)](../topics/15_rnn_lstm/theory.md) | Gradients vanish/explode as $(\gamma \lambda_{\max})^{T-t}$; unusable for $T \gtrsim 20$ | LSTM: additive cell state, gated gradient highway |
| [LSTM/GRU (15)](../topics/15_rnn_lstm/theory.md) | Still sequential ($h_t$ needs $h_{t-1}$); fixed-size $h_t$ bottleneck; decays past $T \approx 500$ | Attention: direct access to all past states |
| [Attention + Transformer (16)](../topics/16_transformer/theory.md) | Needs order injected externally; $O(n^2)$ cost; parameter-hungry | Scale: pre-train once on raw text, adapt cheaply |
| [LLM engineering (21)](../topics/21_llm_engineering/theory.md) | Full fine-tuning and RL alignment too expensive/unstable | BPE, LoRA, DPO — the engineering layer |

---

## Complexity and Memory

| Property | RNN / LSTM | Self-attention |
|---|---|---|
| Compute per layer | $O(n \cdot m^2)$ | $O(n^2 \cdot d)$ |
| Memory in sequence length | $O(n)$ | $O(n^2)$ (attention matrix $A \in \mathbb{R}^{n \times n}$) |
| Sequential steps (training) | $O(n)$ — inherently serial | $O(1)$ — all positions in one matmul |
| Max gradient path between tokens $1$ and $n$ | $n - 1$ multiplicative steps | $1$ |
| State at inference | Fixed $h_t \in \mathbb{R}^m$ | Growing KV cache, $O(n \cdot d)$ |

This is the fundamental trade: recurrence is cheap in memory but serial and
gradient-hostile; attention is quadratic but parallel and gives every token an $O(1)$
path to every other. On GPU hardware, parallelism won.

---

## How Each Model Represents Order

- **Recurrence:** order is *implicit* — $h_t$ is computed after $h_{t-1}$ by
  construction. No extra machinery, but this is exactly what forces serial computation.
- **Attention:** permutation equivariant — it sees a *set*. Order must be added back:

```math
X_{\text{input}} = X_{\text{embed}} + \text{PE},
\qquad
\text{PE}(\text{pos}, 2i) = \sin\!\left(\frac{\text{pos}}{10000^{2i/d}}\right)
```

  Sinusoidal PE makes relative offsets a linear map; learned embeddings fit better but
  cannot extrapolate; RoPE rotates $q, k$ by position and dominates modern LLMs
  ([16 §4](../topics/16_transformer/theory.md)).
- **Causality:** the RNN is causal for free; the Transformer decoder must *enforce* it
  with the mask $M_{ij} = -\infty$ for $j > i$. Forgetting the mask is the classic
  silent bug — training looks fine, inference fails.

---

## Training Through Time: BPTT vs Teacher Forcing

Two orthogonal questions get conflated:

1. **How do gradients flow?** RNNs unroll the graph and run backprop through all $T$
   steps (BPTT) — depth equals sequence length, hence the vanishing-gradient product
   $\prod_s \text{diag}(\tanh'(z_s))\, W_{hh}$. Transformers have no recurrence to
   unroll; gradient depth is the *layer count*, handled by residuals and LayerNorm.
2. **What inputs does the model see during training?** Both RNN decoders and
   Transformer decoders use **teacher forcing**: feed the ground-truth previous tokens,
   predict the next one. The Transformer's causal mask makes all $n$ next-token
   predictions in parallel — teacher forcing is what makes the parallelism usable.
   Shared cost: **exposure bias** — at inference the model conditions on its own
   (possibly wrong) outputs, a mismatch never seen in training.

---

## Why Attention Enables Transfer Learning at Scale

The pre-train/fine-tune paradigm needed all of these at once — recurrence lacked the
first two:

1. **Parallel training** makes web-scale corpora feasible ($O(1)$ sequential steps
   vs $O(n)$).
2. **Stable depth scaling**: residuals + LayerNorm let blocks stack to dozens of
   layers, so capacity grows with compute (see
   [Deep Learning Building Blocks](deep_learning_building_blocks.md)).
3. **A self-supervised objective**: next-token prediction turns raw text into
   unlimited labels — cross-entropy from the
   [Loss Functions Map](loss_functions_map.md), no annotation required
   (see [Generative and Self-Supervised Learning](generative_and_self_supervised.md)).
4. **A task-agnostic interface**: one token-in/token-out model transfers to any text
   task by prompting or light fine-tuning.

Result: the modern pipeline — pre-train → SFT → alignment
([21 §1](../topics/21_llm_engineering/theory.md)).

---

## The Engineering Layer (Topic 21)

Each technique patches a specific cost of scale:

| Technique | Problem at scale | Core idea |
|---|---|---|
| **BPE** | Character tokens make $n$ huge ($O(n^2)$ attention); word vocab explodes with OOV | Greedily merge frequent pairs → subword vocabulary of chosen size $C$ |
| **LoRA** | Full fine-tuning stores params + grads + Adam states for billions of weights | Freeze $W_0$, train $\Delta W = \frac{\alpha}{r} BA$ with rank $r \ll \min(d,k)$; params drop from $dk$ to $r(d+k)$ |
| **DPO** | RLHF needs a reward model + unstable PPO loop | Closed-form optimal policy lets $Z(x)$ cancel; preferences train the policy directly by NLL |

Note the continuity: BPE trades sequence length against vocabulary size *because* of
attention's $O(n^2)$; LoRA targets exactly the attention projections $W^Q, W^V$; DPO
replaces the RL machinery of [topic 18](../topics/18_reinforcement_learning/theory.md)
with the supervised losses this curriculum started from.

---

## Decision Guide

```
Sequence task?
├── Short sequences, tiny compute, streaming state → LSTM/GRU (15)
├── Training from scratch on moderate data → Transformer (16)
└── Language task with a pre-trained LLM available
    ├── Behavior change, limited GPU → LoRA fine-tuning (21)
    └── Preference alignment → DPO (21); RLHF if you need an explicit reward model (18)
```

---

## Connections

- **Topics:** [15 RNN/LSTM](../topics/15_rnn_lstm/theory.md), [16 Transformer](../topics/16_transformer/theory.md), [21 LLM Engineering](../topics/21_llm_engineering/theory.md), [18 Reinforcement Learning](../topics/18_reinforcement_learning/theory.md) (RLHF), [20 GNN](../topics/20_graph_neural_networks/theory.md) (GAT reuses attention on graphs)
- **Related synthesis:** [Deep Learning Building Blocks](deep_learning_building_blocks.md) (inductive-bias view of the same architectures), [Generative and Self-Supervised Learning](generative_and_self_supervised.md) (next-token prediction as self-supervision), [Loss Functions Map](loss_functions_map.md)
- **Maps:** [INDEX.md](../INDEX.md)
