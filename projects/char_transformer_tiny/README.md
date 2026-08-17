# char_transformer_tiny

Capstone integration project: a character-level mini-transformer language model in **pure
NumPy**, trained end-to-end on an embedded public-domain corpus (Shakespeare sonnets). The
repo library (`ml_first_principles`) has no attention layer, so the transformer block —
forward **and manual backward** — lives in this project's `src/`.

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

No LayerNorm — a deliberate simplification that keeps the manual backward pass compact;
at this scale the residual stream trains fine without it.

## Math summary

Causal attention with scale $s = 1/\sqrt{D}$ and lower-triangular mask $M$:

$$A = \operatorname{softmax}\big(M \odot s\,QK^\top\big), \qquad \text{ctx} = AV.$$

Softmax is computed max-shifted (log-space stability); masked positions are $-\infty$
before the shift, so they contribute exactly zero attention and zero gradient.

Cross-entropy over $N = BT$ positions with targets $y$:

$$\mathcal{L} = -\frac{1}{N}\sum_{n} \log p_{n,y_n}, \qquad
\frac{\partial \mathcal{L}}{\partial z_n} = \frac{1}{N}(p_n - e_{y_n}).$$

Key hand-derived backward steps (implemented in `src/ct_model.py::loss_and_grads`):

- Softmax rows: $dS = A \odot (dA - \langle dA, A\rangle)$ — masked entries have $A=0$,
  so their gradients vanish automatically.
- Tied embedding $W_{te}$ accumulates **two** gradient contributions: the output
  projection ($\sum_{b,t} d\ell_{bt}\, x2_{bt}^\top$) and the embedding scatter
  (`np.add.at` over token ids).
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

| Piece | Outcome |
|---|---|
| `optimizers.adam` | **Not reusable**: it drives its own full optimization loop over one flat vector (`gradient_fn`, `max_iter`, `tol`-based stopping, per-iterate history copies) — incompatible with minibatch training over a dict of parameter tensors. A local per-tensor `Adam` class (same update rule, bias-corrected) lives in `ct_model.py`. |
| `nn_core` (Dense/ReLU/Sequential) | Not used: layers assume 2-D inputs and own their weights, which conflicts with the tied-embedding and (B, T, D) batched-sequence gradient flow. |
| `llm_models.BPETokenizer` | Not needed: the model is character-level by design (vocab = corpus charset). |

Missing library pieces worth adding later: an attention layer with backward, a
`LayerNorm`, and a step-based optimizer interface (`opt.step(params, grads)`).

## Future work

- `--full` mode: train on the complete Shakespeare corpus. Requires downloading external
  data, which is **out of scope** and needs explicit approval per repo policy
  (see [data/README.md](data/README.md)). Not implemented.
- Multi-head attention, LayerNorm, and stacking blocks.
