# AGENTS.md v2.6

Supplementary rules that override/extend Antigravity defaults. Does not repeat `<guidelines>`, `<communication_style>`, `<web_application_development>`. On conflict → read sections in order §1 → §11.

> **Single source for §1–§10.** [CLAUDE.md](CLAUDE.md) imports this file and overrides only §11; there is no second copy of the shared rules to keep in sync.

## 1. Stop — Ask Before

- `rm`, `git reset --hard`, force-push, deleting files/folders.
- Installing new dependencies (`pip install`, `npm install`).
- Edits > 500 lines/call, or files outside the project root.
- Training/download > 30s wall-time. Never auto-set `SafeToAutoRun=true` for these.
- Dataset download > 100MB, or any write/delete inside `data/`.

## 2. Project Context

- **Purpose:** Educational ML workspace — learning, experimentation, implementation from scratch.
- **Stage:** Learning phase (notebooks/scripts OK, production rules apply when project > 1 file).
- **Tech:** Python 3.12+, NumPy, PyTorch, Matplotlib, Jupyter.
- **Subproject layout (capstone or any multi-file project):** `README.md`, `requirements.txt`, `src/`, `tests/`, `reports/`; add `data/` or `notebooks/` only when they hold real content — no placeholder directories or READMEs.
- **Repo organization:** `topics/` (22 algorithm stories) · `topics/synthesis/` (cross-topic comparisons) · `src/ml_first_principles/` (reusable implementations) · `tests/` · `scripts/` (notebook validation/normalization). Math prerequisites live in the sister repo [applied-mathematics-foundation](https://github.com/hien078/applied-mathematics-foundation). Each topic folder follows the first-principles file pattern below.
- **Map files (read first to navigate):** [README.md](README.md), [INDEX.md](INDEX.md) (22-topic curriculum index), [NOTEBOOK_STANDARDS.md](NOTEBOOK_STANDARDS.md) (quality contract), [topics/_template.ipynb](topics/_template.ipynb) (notebook boilerplate).
- **Web (edge case):** dark mode + HSL accent. WCAG 2.1 AA.

**First-principles file pattern** — every topic folder. Each file has a distinct role; do not duplicate content across files.

| File | Role | Content |
|---|---|---|
| `README.md` | Navigation | Scope, prerequisites, maturity, file order, related synthesis/projects. |
| `theory.md` | WHY + WHAT | Motivation, assumptions, notation, derivations. **Markdown only — no code.** |
| `first_principles.ipynb` | HOW + BUILD + VERIFY | From-scratch NumPy implementation, library comparison, experiments, failure cases. |
| `exercises.ipynb` | Practice | Hand derivation, coding task with deterministic check, conceptual question. |

Large topics may add focused notebooks (e.g. `variants.ipynb`, `geometry_and_sparsity.ipynb`). The boundary between Markdown, notebooks, and reusable Python is defined in [NOTEBOOK_STANDARDS.md](NOTEBOOK_STANDARDS.md).

**First-principles checklist** — before writing any topic content, identify:

1. **Problem**: what real-world or mathematical problem requires this method?
2. **Assumptions**: what simplifications make the model tractable?
3. **Variables & parameters**: what changes ($x$, $y$) vs what is fixed ($\lambda$, $\alpha$)?
4. **Objective**: what quantity is being minimized/maximized/estimated?
5. **Governing principle**: optimization, probabilistic inference, geometric separation, information compression, …
6. **Formulation**: convert assumptions into equations — do not introduce formulas without justification.
7. **Verification**: simple cases, limiting behavior, numerical comparison with a trusted library.

**Commands (bash):** `source .venv/bin/activate` · `pip install -r requirements.txt` (ask per §1) · `jupyter lab`. Verification = notebook runs clean top-to-bottom on a fresh kernel (§9). On Windows PowerShell: `.venv\Scripts\Activate.ps1`.

## 3. Language

- **Default: English for everything** — prose, code, identifiers, commits, docstrings, logs.
- **Vietnamese only when explicitly requested** by the user (e.g. "trả lời tiếng Việt", "giải thích bằng VIE").
- Math: LaTeX MUST be used for all mathematical symbols and formulas (both inline $...$ and block $$...$$).
- Terminology (when VIE requested): Vietnamese explanation + `English term` on first use, then English term only.
- Never translate: function names, library names, error messages, flags.

## 4. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

- State assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 5. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it **before** showing to user.

## 6. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it — don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: every changed line should trace directly to the user's request.

## 7. Math & ML

> **Learning phase:** notebooks/scripts for learning → `print()` OK, simple code OK, no need to split folders or use config YAML. Production rules below apply when project > 1 file.

**Learning:** explain concepts → intuition + visual imagery first, rigorous math second. Illustrate with short code examples. Encourage implementing from scratch (NumPy) before using libraries.

**Derivation:** show work, no skipping steps; cite theorem/rule at each step; conclude with **Result:** (expression). Multiple approaches → pick shortest, mention alternatives in 1 line.

**Reproducibility:** seed `random`/`numpy`/`torch`(+`cuda`)/`tf`. Pin `requirements.txt`. Log: hparams, metrics, git hash, dataset hash, seed.

**Layer separation:** `data/` (loading only, no model imports). `models/` (pure architecture, no hardcoded hparams). `train.py`/`evaluate.py` read config YAML. Checkpoints include `{epoch, loss, config_hash, git_sha, timestamp}`.

**Numerical:** inference with `torch.no_grad()` / `@inference_mode()`. Float comparison via `np.isclose`/`torch.allclose` + explicit `atol`. Log-space for probabilities (`logsumexp`, `log_softmax`). Accuracy to 4dp, loss scale-dependent.

**Jupyter:** first cell = imports + `%load_ext autoreload` + seed. Split `theory.md` (derivations) / `first_principles.ipynb` (implementation + verification). Clear outputs before commit. Edit by cell index, never overwrite entire file.

**LaTeX:** `pdflatex` default, `xelatex` when Unicode needed. Clean compile (0 errors) before reporting done. `biblatex+biber`, no mixing `natbib`. Gitignore `*.aux *.log *.out *.bbl`. Filename: snake_case, no spaces/diacritics.

## 8. Code

> **Learning phase exemption:** standalone notebook or single-file script (not imported elsewhere) → relaxed: `print()` OK, hardcode paths OK, skip type hints / `__future__` import for one-shot cells. Rules below apply to `src/` library code or any project > 1 file.

**Python (library/multi-file):** `from __future__ import annotations` for new modules. Type hints required for signatures. `pathlib.Path` over `os.path`. Logger: `logger.info("loss %.4f", loss)` (lazy %, no f-string). `dataclass(frozen=True)` for immutable schemas. No bare `except Exception:` without specific handling.

**Never (library/multi-file):**
- `print()` in library code → use `logging`.
- Hardcode `D:\...` → use env var / config.
- `pickle` for long-term artifacts → `safetensors`/`joblib`/JSON.
- Magic numbers in training loops → constants at file top / config.

**Docstring:** public API → full Google style (Args/Returns/Raises). Private/helper → only when WHY is non-obvious. No docstrings for trivial getters/setters.

**Error recovery:** on failure → read full error, diagnose root cause, then fix. NO blind retries, NO `try/except` to hide errors, NO random fixes "just to make it run". Stuck > 2 failures with same symptom → stop and summarize for user. Do NOT switch to a different broken approach to escape the original — that just multiplies the problem.

## 9. Plan & Verify

Transform vague tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"

| Task | Plan | Verify |
|---|---|---|
| Short exercise | skip | run code / compare with expected |
| Pure derivation | skip | check boundaries / special cases |
| Explain concept / answer question | skip | cite source or show derivation |
| New model (>1 file) | `task.md` | sanity check 1-2 epochs |
| Bug with repro | skip | regression test pass |
| Refactor >3 files | plan + list files | tests pass after each step |
| LaTeX doc | brief outline | compile + review PDF |

Never say "done" without seeing actual output. If unable to run → state "not run because X".

## 10. Response Style

**Match length to question depth.** One-liner question → 1-2 sentence answer. Bug fix → diff + 1 line of why. Concept explanation → as long as needed, zero padding.

- **Plan first when:** task falls in §9 "Plan" column (≥ 1 entry). Otherwise jump in.
- **Tables vs. bullets:** table when comparing ≥ 2 things across the same dimensions. Bullets for flat lists.
- **Show, don't tell:** prefer code + actual output over prose. "Tested: 3/3 pass" beats "I tested it carefully".
- **Headers:** only for responses with ≥ 3 distinct sections. Don't header a 3-sentence answer.
- **Stop signal:** wrote > 300 words of prose for a coding task without showing code → you are over-explaining.

## 11. Tools (Antigravity)

Output overflow prevention: `-n 20`, `--oneline`, `head -50`, `--max-count`, `PAGER=cat`. Async commands → `command_status` to check, `send_command_input` for stdin / kill.

- **Artifact vs KI:** Artifact = long-form text for user (report, derivation, plan). Code files → write directly. KI = durable facts about repo (architecture, schema, conventions). No KI for debug / todo / facts already in code.
- **Web UI:** when building web UI, follow `<web_application_development>` defaults. Visual rules already in §2.
- **Auto-run:** never auto-set `SafeToAutoRun=true` for the actions listed in §1.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
