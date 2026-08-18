# Ecosystem Roadmap

> From a 22-topic first-principles curriculum to a complete learning ecosystem:
> verified curriculum, reusable library, applied projects, published docs.

This file tracks phases and the decisions behind them. Per-topic maturity lives in
[INDEX.md](INDEX.md); what each release shipped lives in [CHANGELOG.md](CHANGELOG.md).

| Phase | Pillar | Goal | Status |
|---|---|---|---|
| 1 | Infrastructure | Verification foundation | ✅ Done |
| 2 | Curriculum | Curriculum hardening | 🟡 Content done, CI streak pending |
| 3 | Publishing | Docs site | ✅ Done |
| 4 | Projects | Applied capstone projects | ✅ Done |
| 5 | Library | v0.2.0 + release | ✅ Done |

Per-phase task checklists are finished work; they stay in git history rather than
here, and their outcome is recorded in CHANGELOG.md.

---

## Open

- **Phase 2 exit criterion — CI streak.** All 22 topics are Verified in INDEX.md and
  `synthesis/` covers all 5 curriculum phases. What remains is the weekly notebook
  execution job (`.github/workflows/ci.yml`, `notebooks`) staying green 3 consecutive
  weeks.

---

## Decisions Log

- **Infra before content** (2026-08): the Verified gate was undefined while the standards demanded cleared outputs and the tree deliberately kept them; the expensive full re-execution sweep should happen exactly once, after the policy is settled.
- **Outputs stay committed** (2026-08): GitHub browsing is self-contained and the docs site can render notebooks without executing them at build time. The invariant is not *no outputs* but *no hand-run stale outputs* — `execute_all_notebooks.py --write` is the only legitimate producer.
- **Inline matplotlib backend for `--write`** (2026-08): the headless sweep originally forced `MPLBACKEND=Agg`, which is non-interactive — every `plt.show()` committed a `FigureCanvasAgg` warning and no image, leaving the tree with zero figures. `module://matplotlib_inline.backend_inline` is equally headless and deterministic but captures figures.
- **MkDocs Material over Jupyter Book/Quarto** (2026-08): both alternatives require converting the ```math fences that were just adopted for GitHub rendering; Material renders them with config only.
- **Flat library namespace retained** (2026-08): 17 modules with a curated `__all__` is within flat-namespace comfort; revisit past ~25 modules.
- **No PyPI** (2026-08): solo learning repo — GitHub Releases carry the artifacts; one TestPyPI upload allowed as a learning exercise, not institutionalized.
- **Single source per fact** (2026-08): the topic list, the agent guidelines, and the dependency set each had 2–3 hand-synced copies. Each now has one owner — INDEX.md, AGENTS.md, and pyproject.toml respectively — and the others link to it.
