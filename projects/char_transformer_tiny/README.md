# char_transformer_tiny

Capstone integration project: a character-level mini-transformer language model in **pure
NumPy**, trained end-to-end on an embedded public-domain corpus (Shakespeare sonnets).
Since library v0.3.0 the layer math — forward **and manual backward** — lives in
`ml_first_principles.transformer_core` (`Embedding`, `CausalSelfAttention`,
`TransformerBlock`, `softmax_cross_entropy`); this project assembles those layers and
keeps only the project-specific glue in `src/ct_model.py`: weight tying, corpus handling,
batching, and generation. (The original hand-derived implementation was written here
first, then promoted into the library.)

## Architecture

Single block, single head, weights tied between token embedding and output projection:

```
ids (B, T)                             T = block_size = 32
  │
  ├─ Wte[ids]  (token embedding, V×D)  D = d_model = 48, V = 58 (corpus charset)
  ├─ Wpe[:T]   (learned positional embedding, T×D)
  ▼
x0 = Wte[ids] + Wpe[:T]                              (B, T, D)
  │
  │   Q = x0 Wq   K = x0 Wk   V = x0 Wv              (B, T, D)
  │   A = softmax( mask( Q Kᵀ / √D ) )               (B, T, T)  causal mask
  ▼
x1 = x0 + (A V) Wo                                   residual around attention
  │
  │   h = ReLU(x1 W1 + b1)                           (B, T, F)  F = d_ff = 96
  ▼
x2 = x1 + h W2 + b2                                  residual around MLP
  │
  ▼
logits = x2 Wteᵀ                                     (B, T, V)  tied projection
loss   = softmax cross-entropy vs next character
```

No LayerNorm — a deliberate simplification that keeps the backward pass compact; at this
scale the residual stream trains fine without it. The library `TransformerBlock` matches
this choice (a standalone `LayerNorm` layer exists in `transformer_core` for models that
want it).

Layer-to-library mapping:

| Piece | Implementation |
|---|---|
| Token + positional embeddings | `transformer_core.Embedding` (positions via a broadcast `(1, T)` id lookup) |
| Attention + MLP with residuals | `transformer_core.TransformerBlock` (contains `CausalSelfAttention`) |
| Loss | `transformer_core.softmax_cross_entropy` |
| Tied output projection `x2 Wteᵀ` | **local** in `CharTransformer` — the library `Embedding` cannot share its weight with a projection, so the tied matmul and its extra gradient term stay in the project |
| Optimizer | `optimizers.Adam` (step-based, dict of named tensors) |

## Math summary

Causal attention with scale $s = 1/\sqrt{D}$ and lower-triangular mask $M$:

$$A = \operatorname{softmax}\big(M \odot s\,QK^\top\big), \qquad \text{ctx} = AV.$$

Softmax is computed max-shifted (log-space stability); masked positions are $-\infty$
before the shift, so they contribute exactly zero attention and zero gradient.

Cross-entropy over $N = BT$ positions with targets $y$:

$$\mathcal{L} = -\frac{1}{N}\sum_{n} \log p_{n,y_n}, \qquad
\frac{\partial \mathcal{L}}{\partial z_n} = \frac{1}{N}(p_n - e_{y_n}).$$

Key hand-derived backward steps (now implemented in
`ml_first_principles/transformer_core.py`, except the tied-projection term which lives in
`src/ct_model.py::CharTransformer.loss_and_grads`):

- Softmax rows: $dS = A \odot (dA - \langle dA, A\rangle)$ — masked entries have $A=0$,
  so their gradients vanish automatically.
- Tied embedding $W_{te}$ accumulates **two** gradient contributions: the output
  projection ($\sum_{b,t} d\ell_{bt}\, x2_{bt}^\top$, local) and the embedding scatter
  (`np.add.at` over token ids, library `Embedding.backward`).
- Residuals split gradients additively: $dx_1 = dx_2 + W_1$-path, $dx_0 = dx_1 + QKV$-paths.

Correctness gate: a central finite-difference gradient check of **every entry of every
parameter** of the full model (`tests/test_ct_model.py::test_gradient_check_full_model`,
`atol=1e-7`, `rtol=1e-4`).

## How to run

From the repo root (uses the root environment; no extra dependencies):

```bash
source .venv/bin/activate
python projects/char_transformer_tiny/src/ct_train.py   # ~10 s, deterministic (seed 42)
python -m pytest projects/char_transformer_tiny/tests   # 6 tests, < 1 s
```

Training (400 Adam steps, batch 64, context 32, 22 896 parameters) writes
[reports/training_report.md](reports/training_report.md) with the config, loss trace,
wall time, and seeded sample generations.

## Sample output

Loss falls from 4.06 (≈ the uniform baseline $\ln 58$) to ≈ 2.2 in ~10 s. Greedy
decoding from the prompt `"Shall I "`:

```text
Shall I ther the thare than than the thar the thar the thar ...
```

Temperature 0.8 sampling produces sonnet-shaped pseudo-English with learned line breaks
and apostrophes — exactly what a 23 k-parameter character model should give.

## Library reuse and friction

The v0.2.0 friction reports from this project (no attention layer, no `LayerNorm`, no
step-based optimizer) drove the v0.3.0 `transformer_core` module and the step-based
`optimizers.Adam`; the project now consumes both. Remaining friction:

| Piece | Outcome |
|---|---|
| `transformer_core` (Embedding / CausalSelfAttention / TransformerBlock / softmax_cross_entropy) | **Used** — all layer forward/backward math comes from the library; the project only assembles and ties. |
| `optimizers.Adam` (step-based) | **Used** — `optimizer.step(model.params, grads)` over the flat prefixed dict. |
| Weight tying | **Still local** (v0.3.x friction): the library `Embedding` cannot share its weight matrix with an output projection, so `logits = x2 Wteᵀ` and the extra $W_{te}$ gradient term stay in `CharTransformer`. |
| `llm_models.BPETokenizer` | Not needed: the model is character-level by design (vocab = corpus charset). |

## Future work

- `--full` mode: train on the complete Shakespeare corpus. Requires downloading external
  data, which is **out of scope** and needs explicit approval per repo policy
  (see [data/README.md](data/README.md)). Not implemented.
- Multi-head attention, LayerNorm in the block, and stacking blocks.
- Library support for weight tying (would remove the last local gradient term).
