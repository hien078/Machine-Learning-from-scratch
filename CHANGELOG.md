# Changelog

All notable changes to the `ml-first-principles` library and this workspace.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [SemVer](https://semver.org/).

## [Unreleased]

### Changed
- Workspace de-duplication: every fact that existed in 2–4 hand-synced copies now
  has one owner. `CLAUDE.md` imports `AGENTS.md` instead of mirroring it; the
  22-topic list lives only in `INDEX.md`; `ROADMAP.md` merged into
  `CONTRIBUTING.md`; `_template_first_principles.ipynb` moved to
  `topics/_template.ipynb`.
- Site navigation is generated from the directory tree by `mkdocs-awesome-nav`
  (new `docs` extra). `mkdocs.yml` drops from 211 to 60 lines; the hand-written
  parts are `docs/.nav.yml`, `topics/.nav.yml`, `projects/.nav.yml`, and topic
  18's notebook order.

### Added
- `scripts/check.py`: runs all six quality gates in CI order. CI invokes it, so
  the local and CI contracts cannot drift.
- Tests asserting `__version__` matches `pyproject.toml` and that
  `requirements.txt` pins exactly the declared runtime dependency set.

### Removed
- `scripts/check_environment.py`: unreferenced, and a third copy of the
  dependency list.

## [0.3.0] - 2026-08-17

Sequence-model building blocks — the layers `projects/char_transformer_tiny`
had to implement locally are now first-class library citizens.

### Added
- `transformer_core` module: `Embedding`, `LayerNorm`, `CausalSelfAttention`
  (single-head, causal, max-shifted softmax), `TransformerBlock` (residual
  attention + residual ReLU MLP), and `softmax_cross_entropy` (log-space, mean
  loss + gradient). Uniform layer interface — `params`/`grads` dicts,
  `forward`/`backward` — with flat prefixed keys that drive the step-based
  optimizers directly. Every parameter of every layer is covered by a central
  finite-difference gradient check in the test suite.
- `optimizers.SGD`: step-based SGD class with optional classical momentum,
  mirroring `Adam`'s interface and validation.

### Changed
- `projects/char_transformer_tiny` refactored to consume the library layers
  (dogfooding); its regenerated training report is bit-identical to the
  v0.2.0 report — same loss trace and generations — confirming a faithful port.

### Deferred
- Weight tying: `Embedding` cannot share its matrix with an output projection,
  so the tied-embedding logits path stays local to the project.
- `nn_core` remains 2-D by design; sequence layers live in `transformer_core`.

## [0.2.0] - 2026-08-17

The ecosystem release: verification infrastructure, hardened curriculum, docs
site, applied projects, and the library API fixes those projects surfaced.

### Added (library API — all additive, driven by project friction)
- `optimizers.Adam`: step-based optimizer class (`step(params, grads)` over
  dicts of tensors) for external training loops; `adam(..., keep_history=False)`
  to opt out of the full iterate history.
- `nn_core` layers: `backward(grad, learning_rate=None)` computes and stores
  gradients without updating parameters (gradient-collection mode).
- `linear_models.LogisticRegression(l2=...)`: optional L2 penalty (intercept
  never penalized; default 0.0 identical to previous behavior).
- `tree_models.DecisionTreeClassifier.max_features` now accepts
  `"sqrt"`/`"log2"`/float like `RandomForestClassifier` (int > n_features now
  raises instead of silently clamping, matching the ensemble).
- `data_utils.standardize(X, mean=..., std=...)`: apply train statistics to a
  test fold.
- `metrics.mse`/`rmse`/`mae`/`r2_score` accept same-shape N-D arrays
  (reconstruction losses no longer need manual `.ravel()`).
- `rl_models.GridWorldEnv`: `dynamics()` accessor sharing `step()`'s logic,
  `num_states`/`num_actions` properties, optional `max_steps` truncation;
  `QLearningAgent.greedy_action(state)` for evaluation.
- `svm_models.LinearSVC` docstring documents that `C` scales the mean hinge
  loss (vs sklearn's sum) with the conversion factor.
- CI: mypy type checking and an enforced coverage floor (85%); ruff rule set
  expanded with `B` (bugbear) and `UP` (pyupgrade).
- `CONTRIBUTING.md` (setup, quality gates, release procedure).

### Added
- Four applied capstone projects under `projects/` (standard subproject layout,
  <30s seeded training, bundled data only, committed generated reports, fast
  test suites wired into CI): `tabular_benchmark` (scratch vs sklearn),
  `char_transformer_tiny` (NumPy causal transformer with manual backprop and a
  full finite-difference gradient check), `digits_autoencoder` (MLP +
  autoencoder vs PCA), `rl_gridworld` (Q-learning vs value iteration).
- `ROADMAP.md` with the five-phase ecosystem plan and decisions log.
- `py.typed` marker so type hints are exported to consumers (PEP 561).
- `ml_first_principles.__version__`.
- `scripts/execute_all_notebooks.py --write` / `--only`: the canonical (and only
  sanctioned) producer of committed notebook outputs.
- `scripts/normalize_notebooks.py --clear-outputs`: output clearing is now
  opt-in; committed fresh outputs are legitimate.
- CI: ruff format check, coverage report, notebook format gate on every push;
  weekly full notebook execution job.

- Documentation site at <https://hien078.github.io/Machine-Learning-from-scratch/>:
  MkDocs Material + mkdocs-jupyter (renders committed notebook outputs without
  executing), MathJax via arithmatex (existing ```math fences render unchanged),
  native mermaid; built `--strict` and deployed to GitHub Pages on every push
  (`.github/workflows/docs.yml`); `docs` extra in pyproject.
- Three new synthesis documents covering curriculum phases 3–5:
  `deep_learning_building_blocks.md`, `sequence_models_and_attention.md`,
  `generative_and_self_supervised.md`.

### Changed
- NOTEBOOK_STANDARDS.md §7/§10/§11: outputs policy shifted from "no outputs in
  source control" to "no hand-run stale outputs" — committed outputs must come
  from a fresh-kernel `execute_all_notebooks.py --write` run.
- Rebuilt 8 below-standard exercises notebooks to the §8 exercise standard
  (topics 01, 11, 14, 15, 16, 17, 19, 22): 4–10 cells each → 16–18 cells with
  hand derivations, deterministic checks, and failure-analysis questions.
- Expanded thin theory to peer depth: `topics/10_pca/theory.md` (6.5K → 19K),
  `topics/13_neural_networks/theory.md` (7.3K → 19K).
- Rewrote the 6 stub synthesis docs (~2K each) into full documents (~8K each).
- All 22 topics now carry the Verified maturity rung in INDEX.md.

### Deferred (known gaps, intentionally out of 0.2.0)
- `nn_core` still assumes 2-D inputs and has no attention, LayerNorm, softmax,
  or embedding layer (the char-transformer project implements these locally).
- `optimizers.sgd` remains a whole-loop driver (only Adam gained a step-based
  class).

### Fixed
- Dead links in `topics/01_linear_regression/theory.md` to the removed local
  `foundations/` directory — now point at the sister repo
  applied-mathematics-foundation. Topic README links into `src/` made absolute
  so they resolve both on GitHub and on the docs site.
- Wrong hand-computed expected values in two old exercises notebooks, hidden
  behind commented-out asserts: RNN second-step forward values (topic 15) and
  the 2-D cross-correlation example (topic 14). Both now assert-checked.

## [0.1.0] - 2026-08-17

Baseline of the existing workspace.

### Added
- 22-topic first-principles curriculum (`topics/`), each with `theory.md`,
  `first_principles.ipynb`, `exercises.ipynb`; topic 18 adds four focused
  notebooks.
- 8 cross-topic synthesis documents (`synthesis/`).
- `ml_first_principles` library: 17 modules, ~2.5k LOC — linear models, trees,
  ensembles, KNN/K-Means, Naive Bayes, SVM, neural network core, optimizers,
  metrics, data utilities, visualization, and phase-5 modules (RL, VAE/GAN,
  GNN, BPE/LoRA/DPO, SSL).
- 62 pytest tests with 1:1 module coverage and sklearn-oracle checks.
- Notebook tooling: environment check, fresh-kernel execution validator,
  format normalizer.
- CI: ruff lint + pytest on Python 3.12.
