# Contributing / Process Notes

This is a solo learning workspace, so this file is primarily a process contract
with future-me (and any curious visitor). The authoritative rules live in
[CLAUDE.md](CLAUDE.md)/[AGENTS.md](AGENTS.md) and
[NOTEBOOK_STANDARDS.md](NOTEBOOK_STANDARDS.md); this is the short version.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # pinned notebook runtime
pip install -e ".[dev]"           # library + test/lint/type tooling
```

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

- `from __future__ import annotations`, full type hints on signatures,
  NumPy `NDArray` typing, Google-style docstrings on public API.
- `ValueError` for bad arguments, `RuntimeError` for not-fitted state.
- New public names go into the alphabetized `__all__` in
  `src/ml_first_principles/__init__.py` (a test enforces it resolves).
- Additive, non-breaking API changes preferred; breaking changes need a
  CHANGELOG entry and a version bump discussion.
- Docs: theory in `theory.md` (markdown only, ```math fences), computation in
  notebooks, reusable code in `src/` — see NOTEBOOK_STANDARDS §2.

## Release procedure (maintainer)

1. Move CHANGELOG `[Unreleased]` into a new `[x.y.z]` section (Keep a Changelog).
2. Bump `version` in `pyproject.toml` AND `__version__` in
   `src/ml_first_principles/__init__.py` (a test fails if they diverge).
3. `python -m build` → sdist + wheel in `dist/`.
4. `git tag vx.y.z && git push origin master --tags`.
5. `gh release create vx.y.z dist/* --title vx.y.z --notes-file <notes>`.
6. No PyPI (see ROADMAP decisions log); TestPyPI once as a learning exercise
   is allowed.
