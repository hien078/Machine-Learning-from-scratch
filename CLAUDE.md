# CLAUDE.md v2.4

Behavioral guidelines for this ML learning workspace. Biases toward caution over speed. On conflict → read sections in order §1 → §11.

> **Mirror of [AGENTS.md](AGENTS.md).** §1–§10 are byte-for-byte identical between the two files; §11 is tool-specific (Claude Code here, Antigravity there). **When updating shared rules, edit both files.**

## 1. Stop — Ask Before

- `rm`, `git reset --hard`, force-push, deleting files/folders.
- Installing new dependencies (`pip install`, `npm install`).
- Edits > 500 lines/call, or files outside `d:\AI\Machine Learning\`.
- Training/download > 30s wall-time.
- Dataset download > 100MB, or any write/delete inside `data/`.

## 2. Project Context

- **Root:** `d:\AI\Machine Learning\`
- **Purpose:** Educational ML workspace — learning, experimentation, implementation from scratch.
- **Stage:** Learning phase (notebooks/scripts OK, production rules apply when project > 1 file).
- **Tech:** Python 3.12+, NumPy, PyTorch, Matplotlib, Jupyter.
- **Subproject layout:** `README.md`, `requirements.txt`, `data/`, `notebooks/`, `src/`, `tests/`, `reports/`.
- **Web (edge case):** dark mode + HSL accent. WCAG 2.1 AA.

## 3. Language

- **Default: English for everything** — prose, code, identifiers, commits, docstrings, logs.
- **Vietnamese only when explicitly requested** by the user (e.g. "trả lời tiếng Việt", "giải thích bằng VIE").
- Math: No LaTeX `$...$` / `$$...$$`. Use Unicode math (∑∫→) in prose.
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

**Jupyter:** first cell = imports + `%load_ext autoreload` + seed. Split `01_eda.ipynb` / `02_train.ipynb`. Clear outputs before commit. Edit by cell index, never overwrite entire file.

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

## 11. Tools (Claude Code)

Output overflow prevention: prefer `--max-count`, `head -50`, `PAGER=cat`; pass `head_limit` on Grep. For long-running shell → use `run_in_background` and read the output file when notified; never poll with sleep loops.

- **TodoWrite:** use when a task has ≥ 3 discrete steps or the user gives a multi-item list. Exactly one `in_progress` at a time; mark `completed` immediately, don't batch.
- **Agent (Explore):** for codebase searches > 3 queries, spawn Agent with `subagent_type=Explore` rather than chaining Globs / Greps. Briefer prompts than for general-purpose.
- **Plan mode:** use `EnterPlanMode` for non-trivial implementations *before* writing code; exit with `ExitPlanMode` only after the plan is finalized.
- **Skills:** when the user types `/name`, invoke only if `name` appears in the available-skills list. Never invent a skill name from training data.
- **Memory:** persistent memory at `C:\Users\Admin\.claude\projects\d--AI-Machine-Learning\memory\`. Save user / feedback / project / reference per the auto-memory system. Do NOT save things derivable from code (patterns, paths, git history).
- **Parallel tool calls:** when calls are independent, batch in one message; only serialize when later calls depend on earlier results.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
