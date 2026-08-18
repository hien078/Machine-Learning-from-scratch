# Digits Autoencoder Capstone

Capstone project tying together three threads of the curriculum on sklearn's bundled
`load_digits` dataset (1797 grayscale 8x8 digit images — nothing is downloaded):

1. **Classification** — an MLP built from `ml_first_principles.nn_core` layers
   (`Dense`/`ReLU`/`Sigmoid`/`Sequential`) and trained with
   `ml_first_principles.optimizers.adam`, compared against a sklearn
   `LogisticRegression` baseline.
2. **Representation learning** — an undercomplete autoencoder
   (64 -> 8 -> 2 -> 8 -> 64) whose 2-D latent space is visualized colored by digit
   class, side by side with a from-scratch PCA projection (`numpy.linalg.eigh`).
3. **Compression quality** — test-set reconstruction error of the autoencoder vs
   rank-2 PCA through the same 2-D bottleneck.

The interesting engineering piece is the parameter bridge: `nn_core` layers update
themselves inside `backward`, so `da_experiment.loss_and_grad` calls
`backward(gradient, learning_rate=0.0)` to *collect* gradients without stepping, and
hands a flattened parameter vector to the standalone `adam` optimizer, which owns
the update. This turns the per-sample educational training loop into a fast,
fully vectorized full-batch one.

## Layout

| Path | Content |
|---|---|
| `src/da_experiment.py` | Everything: models, training, PCA, plotting, report writing. |
| `tests/` | 6 fast checks: shapes, loss decrease, PCA vs sklearn/analytic, determinism. |
| `reports/` | Generated `report.md`, `latent_space.png`, `reconstructions.png`. |

## How to run

From the repository root (uses the root environment, no extra dependencies):

```bash
source .venv/bin/activate
python projects/digits_autoencoder/src/da_experiment.py   # ~10 s, deterministic (seed 42)
pytest projects/digits_autoencoder/tests                   # ~2 s
```

The script rewrites everything under `reports/`.

## Findings (see [reports/report.md](reports/report.md))

- From-scratch MLP: **97.1 %** test accuracy vs **96.2 %** for sklearn
  `LogisticRegression` — full-batch Adam on plain MSE/one-hot is enough to edge past
  the linear baseline.
- Through the same 2-D bottleneck the nonlinear autoencoder reconstructs slightly
  better than rank-2 PCA (test MSE **0.0521** vs **0.0531**, ratio 0.98), because its
  ReLU layers can bend the 2-D manifold while PCA is stuck with the best flat plane.
- Both 2-D embeddings cleanly isolate distinctive digits (0, 4, 6) and both confuse
  the loopy 3/8/9 group — with only 2 latent dimensions, some overlap is unavoidable.
