# Contributing / Process Notes

This is a solo learning workspace, so this file is primarily a process contract
with future-me (and any curious visitor). The authoritative rules live in
[CLAUDE.md](https://github.com/hien078/Machine-Learning-from-scratch/blob/master/CLAUDE.md)/[AGENTS.md](https://github.com/hien078/Machine-Learning-from-scratch/blob/master/AGENTS.md)
and [NOTEBOOK_STANDARDS.md](NOTEBOOK_STANDARDS.md); this is the short version.

## Setup

See [README.md](README.md), "Quick Start & Installation" — `requirements.txt` pins the
notebook runtime, `pip install -e ".[dev]"` adds the library plus test/lint/type tooling.

## Quality gates (all must pass before pushing)

```bash
python scripts/check.py
```

Runs lint, format, notebook format, types, library tests with the coverage floor,
and project tests — in that order, continuing past failures and naming every gate
that failed. CI runs this exact script, so green locally means green in CI.

Notebook content changes additionally require a fresh-kernel pass:

```bash
python scripts/execute_all_notebooks.py --only <path> --write   # canonical outputs
python scripts/execute_all_notebooks.py                         # full read-only sweep
```

Committed notebook outputs may ONLY come from `execute_all_notebooks.py --write`
— never from an interactive kernel (NOTEBOOK_STANDARDS §7/§11).

## Code conventions

General Python style — `from __future__ import annotations`, type hints on signatures,
Google-style docstrings on public API — is [AGENTS.md](https://github.com/hien078/Machine-Learning-from-scratch/blob/master/AGENTS.md) §8. Where content
belongs (theory vs notebook vs `src/`) is [NOTEBOOK_STANDARDS.md](NOTEBOOK_STANDARDS.md) §2.

Specific to this library:

- NumPy `NDArray` typing on array parameters and returns.
- `ValueError` for bad arguments, `RuntimeError` for not-fitted state.
- New public names go into the alphabetized `__all__` in
  `src/ml_first_principles/__init__.py` (a test enforces it resolves).
- Additive, non-breaking API changes preferred; breaking changes need a
  CHANGELOG entry and a version bump discussion.

## Release procedure (maintainer)

1. Move CHANGELOG `[Unreleased]` into a new `[x.y.z]` section (Keep a Changelog).
2. Bump `version` in `pyproject.toml` AND `__version__` in
   `src/ml_first_principles/__init__.py` (a test fails if they diverge).
3. `python -m build` → sdist + wheel in `dist/`.
4. `git tag vx.y.z && git push origin master --tags`.
5. `gh release create vx.y.z dist/* --title vx.y.z --notes-file <notes>`.
6. No PyPI (see the decisions log below); TestPyPI once as a learning exercise
   is allowed.

---

# Ecosystem Roadmap

> From a 22-topic first-principles curriculum to a complete learning ecosystem:
> verified curriculum, reusable library, applied projects, published docs.

Per-topic maturity lives in [INDEX.md](INDEX.md); what each release shipped lives in
[CHANGELOG.md](CHANGELOG.md).

| Phase | Pillar | Goal | Status |
|---|---|---|---|
| 1 | Infrastructure | Verification foundation | ✅ Done |
| 2 | Curriculum | Curriculum hardening | 🟡 Content done, CI streak pending |
| 3 | Publishing | Docs site | ✅ Done |
| 4 | Projects | Applied capstone projects | ✅ Done |
| 5 | Library | v0.2.0 + release | ✅ Done |

Per-phase task checklists are finished work; they stay in git history rather than
here, and their outcome is recorded in CHANGELOG.md.

## Open

- **Phase 2 exit criterion — CI streak.** All 22 topics are Verified in INDEX.md and
  `topics/synthesis/` covers all 5 curriculum phases. What remains is the weekly notebook
  execution job (`.github/workflows/ci.yml`, `notebooks`) staying green 3 consecutive
  weeks.

## Decisions Log

- **Infra before content** (2026-08): the Verified gate was undefined while the standards demanded cleared outputs and the tree deliberately kept them; the expensive full re-execution sweep should happen exactly once, after the policy is settled.
- **Outputs stay committed** (2026-08): GitHub browsing is self-contained and the docs site can render notebooks without executing them at build time. The invariant is not *no outputs* but *no hand-run stale outputs* — `execute_all_notebooks.py --write` is the only legitimate producer.
- **Inline matplotlib backend for `--write`** (2026-08): the headless sweep originally forced `MPLBACKEND=Agg`, which is non-interactive — every `plt.show()` committed a `FigureCanvasAgg` warning and no image, leaving the tree with zero figures. `module://matplotlib_inline.backend_inline` is equally headless and deterministic but captures figures.
- **MkDocs Material over Jupyter Book/Quarto** (2026-08): both alternatives require converting the ```math fences that were just adopted for GitHub rendering; Material renders them with config only.
- **Flat library namespace retained** (2026-08): 17 modules with a curated `__all__` is within flat-namespace comfort; revisit past ~25 modules.
- **No PyPI** (2026-08): solo learning repo — GitHub Releases carry the artifacts; one TestPyPI upload allowed as a learning exercise, not institutionalized.
- **Single source per fact** (2026-08): the topic list, the agent guidelines, and the dependency set each had 2–4 hand-synced copies. Each now has one owner — INDEX.md, AGENTS.md, and pyproject.toml respectively — and the others link to it.
- **Process docs merged** (2026-08): ROADMAP.md folded into this file. Both were process contracts with future-me, and a roadmap whose phases are all done is not a second document. Kept at repo root rather than `.github/` because `docs/` is a symlink farm rooted at the repo root — a `.github/` location would break relative links in either the GitHub view or the docs site.
