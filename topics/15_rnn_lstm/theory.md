# Recurrent Neural Networks and LSTM — Theory

## 1. WHY — Sequences Need Memory

Feed-forward networks (MLP, CNN) assume inputs are independent or have fixed spatial
structure. Many real-world problems involve **sequential data** of variable length where
the order matters:

- Language: sentences have 5–100+ words; meaning depends on word order.
- Time series: stock prices, sensor readings — past values predict future.
- Speech: audio frames arrive sequentially; phoneme identity depends on context.

A feed-forward network would need a separate input dimension for every possible sequence
length, and would not share knowledge across time positions. We need an architecture
that: (1) handles variable-length input, (2) maintains a running summary of past
information, and (3) shares parameters across time steps.

## 2. WHAT — Notation and Setup

### 2.1 Data

| Symbol | Type | Meaning |
|---|---:|---|
| $x_t \in \mathbb{R}^d$ | vector | input at time step $t$ |
| $h_t \in \mathbb{R}^m$ | vector | hidden state at time $t$ |
| $y_t \in \mathbb{R}^k$ | vector | output/target at time $t$ |
| $T$ | scalar | sequence length |
| $W_{xh} \in \mathbb{R}^{m \times d}$ | matrix | input-to-hidden weights |
| $W_{hh} \in \mathbb{R}^{m \times m}$ | matrix | hidden-to-hidden (recurrent) weights |
| $W_{hy} \in \mathbb{R}^{k \times m}$ | matrix | hidden-to-output weights |
| $b_h \in \mathbb{R}^m$ | vector | hidden bias |
| $b_y \in \mathbb{R}^k$ | vector | output bias |

### 2.2 Vanilla RNN equations

**Hidden state update:**

```math
h_t = \tanh(W_{xh}\, x_t + W_{hh}\, h_{t-1} + b_h)
```

**Output:**

```math
\hat{y}_t = W_{hy}\, h_t + b_y
```

The initial hidden state $h_0$ is typically set to zero. The same parameters
$(W_{xh}, W_{hh}, W_{hy}, b_h, b_y)$ are reused at every time step — this is
**parameter sharing across time**.

### 2.3 Assumptions

1. The hidden state $h_t$ is a sufficient summary of all past inputs $x_1, \ldots, x_t$.
2. The dynamics are time-invariant: the same transition function is used at every step.
3. The $\tanh$ nonlinearity keeps hidden activations in $[-1, 1]$.

### 2.4 Loss

For a sequence-level loss summed over time steps:

$$\mathcal{L} = \sum_{t=1}^{T} \ell(\hat{y}_t, y_t)$$

where $\ell$ is a per-step loss (e.g., cross-entropy for classification, MSE for
regression).

## 3. HOW — Backpropagation Through Time (BPTT)

### 3.1 Unrolling the computation graph

To compute gradients, we **unroll** the RNN for $T$ steps, treating each time step
as a layer in a deep feed-forward network. The "depth" equals the sequence length.

```
x_1 → [RNN] → h_1 → [RNN] → h_2 → ... → [RNN] → h_T
        ↑               ↑                          ↑
       h_0             h_1                        h_{T-1}
        ↓               ↓                          ↓
      ŷ_1             ŷ_2                        ŷ_T
```

### 3.2 BPTT gradient derivation

Define $z_t = W_{xh}\thinspace x_t + W_{hh}\thinspace h_{t-1} + b_h$ so that $h_t = \tanh(z_t)$.

The gradient of the total loss w.r.t. $h_t$ has two sources:
(1) the direct loss at step $t$, and (2) the influence on future steps through $h_{t+1}$:

```math
\frac{\partial \mathcal{L}}{\partial h_t}
= \frac{\partial \ell_t}{\partial h_t}
+ \frac{\partial h_{t+1}}{\partial h_t}^\top \frac{\partial \mathcal{L}}{\partial h_{t+1}}
```

The Jacobian of $h_{t+1}$ w.r.t. $h_t$ is:

```math
\frac{\partial h_{t+1}}{\partial h_t}
= \text{diag}\!\bigl(\tanh'(z_{t+1})\bigr)\, W_{hh}
```

where $\tanh'(z) = 1 - \tanh^2(z)$ is applied element-wise.

### 3.3 Long-range gradient: product of Jacobians

The gradient contribution from step $T$ to an earlier step $t$ involves a product
of Jacobians:

```math
\frac{\partial h_T}{\partial h_t}
= \prod_{s=t+1}^{T} \frac{\partial h_s}{\partial h_{s-1}}
= \prod_{s=t+1}^{T} \text{diag}\!\bigl(\tanh'(z_s)\bigr)\, W_{hh}
```

The weight gradient w.r.t. $W_{hh}$ accumulates contributions from all time steps:

```math
\frac{\partial \mathcal{L}}{\partial W_{hh}}
= \sum_{t=1}^{T} \frac{\partial \mathcal{L}}{\partial h_t}
\odot \tanh'(z_t) \cdot h_{t-1}^\top
```

**Result:** BPTT computes gradients by unrolling the recurrence and applying the chain
rule backward through all $T$ steps. The computational cost is $O(T)$ in both time
and memory.

## 4. Vanishing and Exploding Gradients

### 4.1 The problem

The product of Jacobians from §3.3 determines how gradients flow across time:

```math
\prod_{s=t+1}^{T} \text{diag}\!\bigl(\tanh'(z_s)\bigr)\, W_{hh}
```

Each factor has two components:
- $\text{diag}(\tanh'(z_s))$: diagonal with entries in $(0, 1]$ (since $\max \tanh'(z) = 1$).
- $W_{hh}$: the recurrent weight matrix.

### 4.2 Eigenvalue analysis

Let $\lambda_{\max}$ be the largest singular value of $W_{hh}$. The norm of the
Jacobian product satisfies approximately:

```math
\left\|\prod_{s=t+1}^{T} \text{diag}\!\bigl(\tanh'(z_s)\bigr)\, W_{hh}\right\|
\lesssim \left(\gamma \cdot \lambda_{\max}\right)^{T-t}
```

where $\gamma \leq 1$ accounts for the $\tanh'$ saturation.

- If $\gamma \cdot \lambda_{\max} < 1$: the product **shrinks exponentially** →
  **vanishing gradients**. Early time steps receive negligible gradient signal.
- If $\gamma \cdot \lambda_{\max} > 1$: the product **grows exponentially** →
  **exploding gradients**. Training becomes unstable.

### 4.3 Consequences

| Problem | Symptom | Mitigation |
|---|---|---|
| Vanishing | Model cannot learn long-range dependencies; early inputs are forgotten | LSTM/GRU gates, skip connections |
| Exploding | Loss spikes to NaN; weights diverge | Gradient clipping: $g \leftarrow g \cdot \min(1, \theta / \Vert g\Vert)$ |

**Result:** Vanilla RNNs can in principle capture long-range dependencies, but in
practice the gradient signal decays (or explodes) exponentially with sequence length,
making optimization difficult for $T \gtrsim 20$.

## 5. LSTM — Long Short-Term Memory

### 5.1 Key idea

LSTM (Hochreiter & Schmidhuber, 1997) introduces a **cell state** $c_t$ that flows
through time via an additive update, protected by learned **gates**. The additive
path acts as a "gradient highway" — gradients can flow through it without repeated
multiplication by the recurrent weight matrix.

### 5.2 Notation

| Symbol | Meaning | Range |
|---|---|---|
| $f_t$ | forget gate | $(0, 1)^m$ |
| $i_t$ | input gate | $(0, 1)^m$ |
| $o_t$ | output gate | $(0, 1)^m$ |
| $\tilde{c}_t$ | candidate cell | $(-1, 1)^m$ |
| $c_t$ | cell state | $\mathbb{R}^m$ |
| $h_t$ | hidden state (output) | $(-1, 1)^m$ |

### 5.3 LSTM equations

```math
f_t = \sigma(W_f\, [h_{t-1},\, x_t] + b_f) \quad \text{(forget gate)}
```

```math
i_t = \sigma(W_i\, [h_{t-1},\, x_t] + b_i) \quad \text{(input gate)}
```

```math
\tilde{c}_t = \tanh(W_c\, [h_{t-1},\, x_t] + b_c) \quad \text{(candidate)}
```

$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t \quad \text{(cell update)}$$

```math
o_t = \sigma(W_o\, [h_{t-1},\, x_t] + b_o) \quad \text{(output gate)}
```

$$h_t = o_t \odot \tanh(c_t) \quad \text{(hidden state)}$$

Here $[h_{t-1}, x_t]$ denotes concatenation, $\sigma$ is the sigmoid function,
and $\odot$ is element-wise (Hadamard) product. Each weight matrix
$W_f, W_i, W_c, W_o \in \mathbb{R}^{m \times (m+d)}$.

### 5.4 Why LSTM solves vanishing gradients

The cell state update $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$ is **additive**.

The gradient of $c_T$ w.r.t. an earlier cell state $c_t$ is:

$$\frac{\partial c_T}{\partial c_t} = \prod_{s=t+1}^{T} f_s$$

Each forget gate $f_s \in (0, 1)^m$. If the network learns $f_s \approx 1$
(i.e., "don't forget"), then $\prod f_s \approx 1$ and gradients flow unimpeded.
This is fundamentally different from the vanilla RNN where gradients must pass through
both nonlinear activations **and** weight matrices at every step.

**Result:** The LSTM cell state provides an additive gradient path controlled by the
forget gate. When $f_t \approx 1$, gradients propagate across long distances without
vanishing. The network **learns** when to remember and when to forget.

### 5.5 Intuition for each gate

| Gate | Role | Analogy |
|---|---|---|
| Forget $f_t$ | How much of the old cell state to keep | Erasing parts of a whiteboard |
| Input $i_t$ | How much of the new candidate to write | Writing new notes |
| Output $o_t$ | How much of the cell state to expose | Choosing what to say out loud |

### 5.6 Initialization note

The forget gate bias $b_f$ is often initialized to a positive value (e.g., 1 or 2)
so that the forget gate starts near 1, encouraging information flow at the beginning
of training (Jozefowicz et al., 2015).

## 6. GRU — Gated Recurrent Unit

GRU (Cho et al., 2014) simplifies the LSTM by merging the cell and hidden state
and using only two gates:

```math
z_t = \sigma(W_z\, [h_{t-1},\, x_t] + b_z) \quad \text{(update gate)}
```

```math
r_t = \sigma(W_r\, [h_{t-1},\, x_t] + b_r) \quad \text{(reset gate)}
```

```math
\tilde{h}_t = \tanh(W_h\, [r_t \odot h_{t-1},\, x_t] + b_h) \quad \text{(candidate)}
```

$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t \quad \text{(interpolation)}$$

| GRU component | LSTM equivalent |
|---|---|
| Update gate $z_t$ | Coupled forget + input ($f_t = 1 - z_t$, $i_t = z_t$) |
| Reset gate $r_t$ | Controls how much past state enters the candidate |
| No separate cell state | Hidden state directly stores long-term memory |

GRU has fewer parameters than LSTM (~75%) and often performs comparably on
moderate-length sequences.

## 7. Failure Cases

1. **Very long-range dependencies ($T > 500$).** Even LSTM struggles when relevant
   information is hundreds of steps away. The forget gate product $\prod f_s$ still
   decays, just more slowly. Attention mechanisms (Transformer) address this directly.

2. **Sequential computation prevents parallelism.** Computing $h_t$ requires $h_{t-1}$,
   so the forward pass cannot be parallelized across time. This makes RNNs slower to
   train than Transformers on modern GPU hardware.

3. **Exposure bias with teacher forcing.** During training, the model sees ground-truth
   previous tokens; at inference, it sees its own predictions. Mismatches accumulate.

4. **Gradient clipping is a band-aid.** Clipping addresses exploding gradients but
   does not recover the information lost to vanishing gradients.

5. **Hidden state bottleneck.** The fixed-size $h_t \in \mathbb{R}^m$ must compress
   the entire history. As $T$ grows, this becomes increasingly difficult. Attention
   mechanisms let the model look back at all past states.

6. **Evaluation and data leakage.** Sequence splits must respect temporal order
   (no future data in the training set). Padding masks and hidden-state reset policies
   at sequence boundaries must be explicit.

## 8. Connections

- **[Neural Networks](../13_neural_networks/README.md)** — RNN is an MLP with weight
  sharing across time; BPTT is backpropagation applied to the unrolled graph.
- **[Gradient Descent](../02_gradient_descent/README.md)** — optimization of recurrent
  networks; gradient clipping as a practical technique.
- **[CNN](../14_cnn/README.md)** — spatial (CNN) vs temporal (RNN) parameter sharing;
  1D convolutions can also process sequences.
- **[Transformer](../16_transformer/README.md)** — attention replaces recurrence,
  enabling parallelism and direct access to all positions; largely supersedes RNNs
  for NLP tasks.

---

## 9. References

- **Hochreiter, S., & Schmidhuber, J. (1997).** Long short-term memory. *Neural Computation*, 9(8), 1735–1780.
- **Cho, K., van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014).** Learning phrase representations using RNN encoder-decoder for statistical machine translation. *EMNLP*, 1724–1734.
- **Pascanu, R., Mikolov, T., & Bengio, Y. (2013).** On the difficulty of training recurrent neural networks. *International Conference on Machine Learning (ICML)*, 1310–1318.
- **Goodfellow, I., Bengio, Y., & Courville, A. (2016).** *Deep Learning*. MIT Press. Chapter 10: *Sequence Modeling: Recurrent and Recursive Nets*.

