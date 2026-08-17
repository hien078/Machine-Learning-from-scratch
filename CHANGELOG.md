# Changelog

All notable changes to the `ml-first-principles` library and this workspace.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [SemVer](https://semver.org/).

## [Unreleased]

### Added
- `ROADMAP.md` with the five-phase ecosystem plan and decisions log.
- `py.typed` marker so type hints are exported to consumers (PEP 561).
- `ml_first_principles.__version__`.
- `scripts/execute_all_notebooks.py --write` / `--only`: the canonical (and only
  sanctioned) producer of committed notebook outputs.
- `scripts/normalize_notebooks.py --clear-outputs`: output clearing is now
  opt-in; committed fresh outputs are legitimate.
- CI: ruff format check, coverage report, notebook format gate on every push;
  weekly full notebook execution job.

### Changed
- NOTEBOOK_STANDARDS.md §7/§10/§11: outputs policy shifted from "no outputs in
  source control" to "no hand-run stale outputs" — committed outputs must come
  from a fresh-kernel `execute_all_notebooks.py --write` run.

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
