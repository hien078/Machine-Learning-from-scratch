# Ecosystem Roadmap

> From a 22-topic first-principles curriculum to a complete learning ecosystem:
> verified curriculum, reusable library, applied projects, published docs.

Per-topic maturity is tracked in [INDEX.md](INDEX.md); this file tracks phases and decisions.

| Phase | Pillar | Goal | Status |
|---|---|---|---|
| 1 | Infrastructure | Verification foundation | ✅ Done |
| 2 | Curriculum | Curriculum hardening | 🟡 Content done, CI streak pending |
| 3 | Publishing | Docs site | ✅ Done |
| 4 | Projects | Applied capstone projects | ✅ Done |
| 5 | Library | v0.2.0 + release | ✅ Done |

---

## Phase 1 — Verification Foundation

Make the [NOTEBOOK_STANDARDS.md](NOTEBOOK_STANDARDS.md) §9 *Verified* gate passable and enforced.

- [x] Resolve outputs policy: committed outputs are legitimate, produced only by `scripts/execute_all_notebooks.py --write`
- [x] `normalize_notebooks.py`: output-clearing behind opt-in `--clear-outputs`
- [x] `execute_all_notebooks.py`: `--write` and `--only` flags
- [x] NOTEBOOK_STANDARDS.md §8/§10 updated to match
- [x] Library hygiene: `py.typed`, `__version__`, `CHANGELOG.md`
- [x] CI: ruff format check, coverage report, notebook format gate; weekly full notebook execution job
- [x] INDEX.md maturity column; one canonical `--write` sweep; first topics marked Verified

**Done when:** the outputs contradiction is gone from scripts *and* standards; CI enforces lint + tests + notebook format on every push and full notebook execution weekly; per-topic maturity is visible in INDEX.md.

## Phase 2 — Curriculum Hardening

- [x] Rebuild the 8 weak `exercises.ipynb` (worst first): 16_transformer, 17_autoencoder, 01_linear_regression, 11_clustering, 14_cnn, 15_rnn_lstm, 19_generative_models, 22_self_supervised_learning — each to the §8 exercise standard, re-executed via `--write --only`
- [x] Expand thin theory to peer parity: `topics/10_pca/theory.md`, `topics/13_neural_networks/theory.md`
- [x] Rewrite the 6 stub synthesis docs (~2K each) into substantive documents
- [x] Add synthesis coverage for curriculum phases 3–5 (deep learning building blocks; sequence models & attention; generative & self-supervised)

**Done when:** all 22 topics Verified in INDEX.md; no exercises notebook below the §8 standard; synthesis/ covers all 5 curriculum phases; weekly notebook CI green 3 consecutive weeks.

## Phase 3 — Docs Site

- [x] MkDocs Material + `mkdocs-jupyter` (`execute: false` — renders committed outputs), arithmatex generic mode + superfences `math` fence so existing ```math fences render unchanged, native mermaid
- [x] Nav mirrors INDEX.md; sister repo [applied-mathematics-foundation](https://github.com/hien078/applied-mathematics-foundation) linked as prerequisites
- [x] `docs` extra in pyproject; GitHub Pages deploy workflow, `mkdocs build --strict` in CI

**Done when:** site live on Pages; math + mermaid render correctly; theory.md files unmodified (still GitHub-renderable).

## Phase 4 — Applied Projects

Each project follows the subproject layout (`README.md`, `requirements.txt`, `data/`, `notebooks/`, `src/`, `tests/`, `reports/`); default configs train < 30 s; downloads/heavy modes documented behind ask-first gates.

- [x] `projects/tabular_benchmark` (core): from-scratch linear/tree/ensemble/distance/probabilistic models vs sklearn on built-in datasets; metrics + timing report
- [x] `projects/char_transformer_tiny` (core): char-level mini-transformer via `nn_core`/`llm_models`/`optimizers`; embedded small corpus by default
- [x] `projects/digits_autoencoder` (optional): MLP + autoencoder on `load_digits`, latent-space visualization
- [x] `projects/rl_gridworld` (optional): tabular Q-learning learning-curve report
- [x] Project tests in CI fast path

**Done when:** core projects reproduce from fresh clone with one command; `pytest projects/` green in CI; API friction filed into CHANGELOG `[Unreleased]`.

## Phase 5 — Library v0.2.0 + Release

- [x] mypy in CI (incremental ratchet), coverage floor from Phase 1 measurements (`--cov-fail-under`), expanded ruff rules (`B`, `UP`)
- [x] API fixes surfaced by Phase 4 projects
- [x] `CONTRIBUTING.md`; CHANGELOG `[0.2.0]`
- [x] Bump to 0.2.0, tag `v0.2.0`, GitHub Release with sdist/wheel; optional one-time TestPyPI upload as a learning exercise

**Done when:** CI enforces types + coverage floor; release artifacts attached to the tag; CHANGELOG accurate.

---

## Decisions Log

- **Infra before content** (2026-08): the Verified gate was undefined while the standards demanded cleared outputs and the tree deliberately kept them; the expensive full re-execution sweep should happen exactly once, after the policy is settled.
- **Outputs stay committed** (2026-08): GitHub browsing is self-contained and the docs site can render notebooks without executing them at build time. The invariant is not *no outputs* but *no hand-run stale outputs* — `execute_all_notebooks.py --write` is the only legitimate producer.
- **MkDocs Material over Jupyter Book/Quarto** (2026-08): both alternatives require converting the ```math fences that were just adopted for GitHub rendering; Material renders them with config only.
- **Flat library namespace retained** (2026-08): 17 modules with a curated `__all__` is within flat-namespace comfort; revisit past ~25 modules.
- **No PyPI** (2026-08): solo learning repo — GitHub Releases carry the artifacts; one TestPyPI upload allowed as a learning exercise, not institutionalized.
