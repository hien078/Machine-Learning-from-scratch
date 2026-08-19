# Transformer — Theory

## 0. Notation

| Symbol | Type | Meaning |
|---|---|---|
| $n$ | scalar | sequence length (number of tokens) |
| $d$ | scalar | model dimension (embedding width) |
| $d_k$ | scalar | key/query dimension per head |
| $d_v$ | scalar | value dimension per head |
| $h$ | scalar | number of attention heads |
| $X \in \mathbb{R}^{n \times d}$ | matrix | input embeddings (one row per token) |
| $W^Q \in \mathbb{R}^{d \times d_k}$ | matrix | query projection for one head |
| $W^K \in \mathbb{R}^{d \times d_k}$ | matrix | key projection for one head |
| $W^V \in \mathbb{R}^{d \times d_v}$ | matrix | value projection for one head |
| $W^O \in \mathbb{R}^{h d_v \times d}$ | matrix | output projection (after concatenating heads) |
| $Q \in \mathbb{R}^{n \times d_k}$ | matrix | queries: $Q = X W^Q$ |
| $K \in \mathbb{R}^{n \times d_k}$ | matrix | keys: $K = X W^K$ |
| $V \in \mathbb{R}^{n \times d_v}$ | matrix | values: $V = X W^V$ |
| $A \in \mathbb{R}^{n \times n}$ | matrix | attention weight matrix (row-stochastic) |
| $M \in \mathbb{R}^{n \times n}$ | matrix | mask (0 for allowed, $-\infty$ for blocked positions) |
| $\text{PE} \in \mathbb{R}^{n \times d}$ | matrix | positional encoding |
| $\text{FFN}(x)$ | function | position-wise feed-forward network |

**Convention.** Softmax is always applied row-wise. Typical configuration:
$d_k = d_v = d / h$, so concatenating $h$ heads yields dimension $h \cdot d_v = d$.

---

## 1. WHY — Replacing Recurrence with Attention

### 1.1 The problem with recurrence

RNNs and LSTMs process sequences token-by-token:

$$h_t = f(h_{t-1}, x_t).$$

This creates two bottlenecks:

1. **Sequential computation.** Token $t$ cannot be computed until token $t-1$ is finished.
   Training cannot parallelise across the time dimension, wasting modern GPU bandwidth.
2. **Long-range gradient path.** The gradient from token $t$ to token $1$ traverses $t-1$
   multiplicative steps. Even with gating (LSTM), information decays over hundreds of
   tokens.

### 1.2 The attention alternative

Self-attention computes the output for every position *simultaneously*, with each position
directly attending to every other position. This gives:

- **Full parallelism** during training — all positions computed in one matrix multiply.
- **$O(1)$ path length** between any two tokens — token 1 attends directly to token $n$.

**Trade-off:** self-attention has $O(n^2)$ memory and compute in sequence length,
vs $O(n)$ for recurrence. This is acceptable for moderate $n$ (up to several thousand)
and is the fundamental trade-off of the Transformer architecture.

---

## 2. WHAT — Scaled Dot-Product Attention

### 2.1 Key / Query / Value intuition

The attention mechanism uses three roles per token:

| Role | Intuition | Analogy |
|---|---|---|
| **Query** $q_i$ | "What am I looking for?" | A search query |
| **Key** $k_j$ | "What do I contain?" | An index entry / tag |
| **Value** $v_j$ | "What do I return if selected?" | The content retrieved |

Token $i$ computes a compatibility score with every token $j$ by comparing its query
$q_i$ against $j$'s key $k_j$. High score → token $j$'s value $v_j$ contributes more
to the output of token $i$.

### 2.2 The attention formula

Given queries $Q$, keys $K$, and values $V$:

```math
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V. \qquad (2.1)
```

Step by step:

1. **Compute raw scores.** $S = Q K^\top \in \mathbb{R}^{n \times n}$, where $S_{ij} = q_i^\top k_j$.
2. **Scale.** Divide by $\sqrt{d_k}$: $S' = S / \sqrt{d_k}$.
3. **Mask (optional).** Add mask $M$: $S' \leftarrow S' + M$ (with $M_{ij} = -\infty$ for
   blocked positions).
4. **Softmax.** Apply row-wise softmax to get attention weights $A = \text{softmax}(S')$.
5. **Weighted sum.** Output $= A V$. Row $i$ of the output is a weighted combination of
   all value vectors, with weights given by $A_i$.

### 2.3 Why scale by $\sqrt{d_k}$ — preventing softmax saturation

**Claim.** Without scaling, the dot products $q_i^\top k_j$ grow in magnitude with $d_k$,
pushing softmax into saturated regions where gradients vanish.

**Derivation.** Assume query and key entries are independent with zero mean and unit
variance: $q_{il}, k_{jl} \sim (0, 1)$ i.i.d. for $l = 1, \dots, d_k$.

The dot product is a sum of $d_k$ independent terms:

$$q_i^\top k_j = \sum_{l=1}^{d_k} q_{il} \cdot k_{jl}.$$

Each term has:

$$\mathbb{E}[q_{il} \cdot k_{jl}] = \mathbb{E}[q_{il}] \cdot \mathbb{E}[k_{jl}] = 0,$$

$$\text{Var}(q_{il} \cdot k_{jl}) = \mathbb{E}[q_{il}^2] \cdot \mathbb{E}[k_{jl}^2] = 1 \cdot 1 = 1.$$

By independence of the $d_k$ terms:

$$\mathbb{E}[q_i^\top k_j] = 0, \qquad \text{Var}(q_i^\top k_j) = d_k.$$

So $q_i^\top k_j$ has standard deviation $\sqrt{d_k}$. For $d_k = 64$, this means
typical dot products have magnitude $\sim 8$, producing softmax outputs very close to 0
or 1 — the saturated regime where $\partial \text{softmax} / \partial z \approx 0$.

Dividing by $\sqrt{d_k}$ restores unit variance:

```math
\text{Var}\!\left(\frac{q_i^\top k_j}{\sqrt{d_k}}\right) = \frac{d_k}{d_k} = 1.
```

**Result:** Scaling by $\sqrt{d_k}$ keeps the softmax input in a moderate range where
gradients are informative, regardless of the dimension $d_k$.

---

## 3. HOW — Multi-Head Attention

### 3.1 Motivation

A single attention head computes one set of attention weights — one "mixing pattern."
Different linguistic relationships (syntactic, semantic, positional) may need different
patterns. Multi-head attention runs $h$ independent attention operations in parallel,
each with its own learned projections.

### 3.2 Formulation

For head $i \in \lbrace1, \dots, h\rbrace$:

$$Q_i = X W_i^Q, \quad K_i = X W_i^K, \quad V_i = X W_i^V,$$

$$\text{head}_i = \text{Attention}(Q_i, K_i, V_i).$$

Concatenate and project:

```math
\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) \, W^O. \qquad (3.1)
```

where $W_i^Q, W_i^K \in \mathbb{R}^{d \times d_k}$, $W_i^V \in \mathbb{R}^{d \times d_v}$,
and $W^O \in \mathbb{R}^{h d_v \times d}$.

### 3.3 Parameter count

With the standard choice $d_k = d_v = d/h$:

- Per head: $3 \cdot d \cdot (d/h) = 3 d^2 / h$ parameters.
- All heads: $h \cdot 3 d^2 / h = 3 d^2$ parameters.
- Output projection: $d^2$.
- **Total for multi-head attention: $4 d^2$.**

This is the *same* total as if we had used a single head with $d_k = d$. Multi-head
attention does not increase parameter count — it only restructures the computation.

---

## 4. Positional Encoding

### 4.1 The problem

Self-attention is **permutation equivariant**: if the input tokens are reordered,
the output is reordered in exactly the same way. The operation treats the input as a
*set*, not a *sequence*. Position information must be injected externally.

### 4.2 Sinusoidal encoding

The original Transformer uses fixed sinusoidal functions:

```math
\text{PE}(\text{pos}, 2i) = \sin\!\left(\frac{\text{pos}}{10000^{2i/d}}\right), \qquad
\text{PE}(\text{pos}, 2i+1) = \cos\!\left(\frac{\text{pos}}{10000^{2i/d}}\right), \qquad (4.1)
```

for position $\text{pos} = 0, 1, \dots, n-1$ and dimension index $i = 0, 1, \dots, d/2 - 1$.

**Properties:**

1. **Unique encoding per position.** Each position gets a distinct $d$-dimensional vector.
2. **Bounded magnitude.** All entries are in $[-1, 1]$.
3. **Relative position via linear transformation.** For any fixed offset $k$,
   $\text{PE}(\text{pos} + k)$ is a linear function of $\text{PE}(\text{pos})$ — a
   rotation in the $(\sin, \cos)$ plane. This allows the model to learn relative
   position through attention weights.
4. **Smooth interpolation.** Nearby positions have similar encodings; distant positions
   differ more.

### 4.3 Usage

The positional encoding is **added** to the token embeddings:

$$X_{\text{input}} = X_{\text{embed}} + \text{PE}. \qquad (4.2)$$

This sum is the input to the first encoder (or decoder) layer.

### 4.4 Alternatives

- **Learned positional embeddings:** a trainable $\mathbb{R}^{n_{\max} \times d}$ matrix.
  Used in BERT, GPT. Slightly better empirically, but cannot extrapolate to lengths beyond
  training.
- **Relative positional encodings (RPE):** encode the *distance* between positions rather
  than absolute position (Shaw et al. 2018). More principled for translation-invariant
  tasks.
- **Rotary Position Embedding (RoPE):** applies rotation matrices to query/key vectors
  based on position (Su et al. 2021). Enables efficient relative position without
  extra parameters.

---

## 5. Encoder Block

A single Transformer encoder block applies two sub-layers with residual connections and
layer normalisation:

### 5.1 Architecture

```
Input X
  │
  ├──→ MultiHeadAttention(X, X, X) ──→ + ──→ LayerNorm ──→ Z₁
  │                                    ↑
  └────────────────────────────────────┘  (residual)

  Z₁
  │
  ├──→ FFN(Z₁) ──→ + ──→ LayerNorm ──→ Z₂
  │                 ↑
  └────────────────┘  (residual)
```

In equations (Post-LN, original Transformer):

```math
Z_1 = \text{LayerNorm}\!\big(X + \text{MultiHead}(X, X, X)\big), \qquad (5.1)
```

```math
Z_2 = \text{LayerNorm}\!\big(Z_1 + \text{FFN}(Z_1)\big). \qquad (5.2)
```

Each sub-layer is wrapped in a residual connection followed by a LayerNorm, so the block
adds an increment to its input instead of replacing it.

### 5.2 Feed-forward network (FFN)

A two-layer MLP applied independently to each position:

```math
\text{FFN}(x) = W_2 \, \text{ReLU}(W_1 x + b_1) + b_2, \qquad (5.3)
```

where $W_1 \in \mathbb{R}^{d \times d_{\text{ff}}}$, $W_2 \in \mathbb{R}^{d_{\text{ff}} \times d}$,
and typically $d_{\text{ff}} = 4d$.

### 5.3 Layer normalisation

Normalises across the feature dimension (not the batch dimension like BatchNorm):

$$\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta, \qquad (5.4)$$

where $\mu, \sigma^2$ are the mean and variance of $x$ across the $d$ features, and
$\gamma, \beta \in \mathbb{R}^d$ are learned scale and shift parameters.

### 5.4 Residual connections

The $X + \text{SubLayer}(X)$ structure ensures:

- **Gradient flow.** The gradient has a direct additive path through the residual,
  preventing vanishing gradients in deep stacks.
- **Easy-to-learn identity.** If a sub-layer is not useful, the network can learn
  $\text{SubLayer}(X) \approx 0$ and pass the input through unchanged.

---

## 6. Decoder Block

The decoder has three sub-layers:

### 6.1 Masked self-attention

At training time, the decoder processes the entire target sequence in parallel. To prevent
token $t$ from attending to future tokens $t+1, t+2, \dots$, a **causal mask** is applied:

```math
M_{ij} = \begin{cases} 0 & \text{if } j \le i \\ -\infty & \text{if } j > i \end{cases}
```

This upper-triangular mask sets future attention weights to zero after softmax, preserving
the autoregressive property: the prediction for position $t$ depends only on positions
$\le t$.

### 6.2 Cross-attention

The second sub-layer attends to the encoder output $E$:

- Queries come from the decoder: $Q = X_{\text{dec}} W^Q$.
- Keys and values come from the encoder: $K = E W^K$, $V = E W^V$.

This allows each decoder position to "read" the entire source sequence.

### 6.3 Full decoder block

```math
Z_1 = \text{LayerNorm}\!\big(X + \text{MaskedMultiHead}(X, X, X)\big),
```

```math
Z_2 = \text{LayerNorm}\!\big(Z_1 + \text{MultiHead}(Z_1, E, E)\big),
```

```math
Z_3 = \text{LayerNorm}\!\big(Z_2 + \text{FFN}(Z_2)\big).
```

The three sub-layers run in sequence, each wrapped in a residual connection and a
LayerNorm: masked self-attention, then cross-attention into the encoder output $E$, then
the position-wise FFN.

---

## 7. The Full Transformer

The original Transformer (Vaswani et al. 2017, "Attention Is All You Need"):

1. **Encoder:** $N$ stacked encoder blocks (typically $N = 6$). Input: source tokens +
   positional encoding. Output: contextualised representations $E$.
2. **Decoder:** $N$ stacked decoder blocks. Input: target tokens (shifted right) +
   positional encoding, plus encoder output $E$ via cross-attention. Output: predicted
   next-token logits.
3. **Output head:** Linear projection + softmax over vocabulary to produce token
   probabilities.

Training uses **teacher forcing**: the decoder receives the ground-truth target sequence
(shifted by one) and predicts each next token in parallel.

---

## 8. Failure Cases

1. **Quadratic memory in sequence length.** The attention matrix $A \in \mathbb{R}^{n \times n}$
   requires $O(n^2)$ memory and compute. For $n = 10{,}000$, a single attention head
   stores a $10{,}000 \times 10{,}000$ matrix.

2. **Practical limit.** This limits standard Transformers to sequences of a few thousand
   tokens without specialised techniques (FlashAttention, sparse attention, linear
   attention).

3. **No inherent notion of order.** Without positional encoding, the Transformer is
   permutation equivariant — it treats the input as a set. If the positional encoding
   fails to convey order (e.g., extrapolating beyond training lengths), the model loses
   sequence structure.

4. **Positional encoding extrapolation.** Sinusoidal encodings are defined for any
   position, but the model has never seen positions beyond its training range. Learned
   embeddings fail entirely beyond the maximum trained length. This is a key motivation
   for RoPE and ALiBi.

5. **Missing or wrong masks.** Forgetting the causal mask in the decoder leaks future
   information during training, producing a model that appears to learn well but fails
   at inference (where future tokens are unavailable).

6. **Padding masks.** Forgetting padding masks causes attention to leak into padding
   tokens.

7. **Attention is not explanation.** Attention weights are data-dependent mixing
   coefficients, not faithfully attributing importance.

8. **Reading the weights.** High attention weight on token $j$ does not mean the model
   "understands" token $j$ is important — it means the *current* query happened to align
   with that key. Attention patterns can be misleading for interpretability.

9. **Large parameter count.** A single encoder block with $d = 512$ has $\sim 2.4\text{M}$
   parameters. Stacking $N = 6$ gives $\sim 14\text{M}$ for the encoder alone. Transformers are
   parameter-hungry and require large datasets to generalise.

---

## 9. Connections

- [Neural Networks](../13_neural_networks/README.md) — the FFN sub-layer is a two-layer
  MLP; attention + FFN is the Transformer's alternative to dense layers.
- [RNN/LSTM](../15_rnn_lstm/README.md) — the sequential model that Transformers replace.
  Comparison: $O(n)$ memory but sequential, vs $O(n^2)$ memory but parallel.
- [Information Theory](https://github.com/hien078/applied-mathematics-foundation) — cross-entropy loss
  used in language modelling; softmax as a probability distribution.
- **Attention Variants (further reading):** FlashAttention (memory-efficient exact
  attention), Sparse Attention (Longformer, BigBird), Linear Attention (Performers),
  Sliding Window Attention.

---

## 10. References

- **Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017).** Attention is all you need. *Advances in Neural Information Processing Systems (NIPS)*, 30, 5998–6008.
- **Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019).** BERT: Pre-training of deep bidirectional transformers for language understanding. *NAACL-HLT*, 4171–4186.
- **Radford, A., Narasimhan, K., Salimans, T., & Sutskever, I. (2018).** Improving language understanding by generative pre-training. *OpenAI Technical Report*.
- **Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., & Liu, Y. (2024).** RoFormer: Enhanced transformer with rotary position embedding. *Neurocomputing*, 568, 127063.
- **Goodfellow, I., Bengio, Y., & Courville, A. (2016).** *Deep Learning*. MIT Press.

