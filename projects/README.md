# Projects — Applied Capstones

End-to-end projects that exercise the [`ml_first_principles`](../src/ml_first_principles/)
library on real tasks. Each follows the standard subproject layout
(`README.md`, `requirements.txt`, `data/`, `notebooks/`, `src/`, `tests/`, `reports/`),
trains in under 30 seconds with fixed seeds, uses no downloads (bundled/embedded data
only), and commits its generated report.

| Project | Exercises | Headline result |
|---|---|---|
| [tabular_benchmark](tabular_benchmark/README.md) | Linear, tree, ensemble, distance, probabilistic models vs sklearn | Closed-form models match sklearn to 4dp; speed is the honest gap |
| [char_transformer_tiny](char_transformer_tiny/README.md) | NumPy causal transformer with manual backprop (topics 13, 16, 21) | Loss 4.06 → 2.23 in 400 steps; full finite-difference gradient check |
| [digits_autoencoder](digits_autoencoder/README.md) | `nn_core` MLP + autoencoder vs PCA (topics 13, 17, 10) | 97.1% accuracy; AE beats rank-2 PCA reconstruction |
| [rl_gridworld](rl_gridworld/README.md) | Q-learning vs value iteration (topic 18) | Q-learning reaches the exact optimal return (9.50) |

Run any project from the repo root, e.g.:

```bash
python projects/tabular_benchmark/src/tb_benchmark.py
pytest projects   # fast test suite for all projects
```

API friction discovered here is filed in [CHANGELOG.md](../CHANGELOG.md) `[Unreleased]`
and drives the library's v0.2.0 scope ([ROADMAP.md](../ROADMAP.md) Phase 5).
