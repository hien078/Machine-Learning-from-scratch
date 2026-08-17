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
