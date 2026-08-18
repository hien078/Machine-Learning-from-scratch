# CLAUDE.md v2.6

Behavioral guidelines for this ML learning workspace. Biases toward caution over speed. On conflict → read sections in order §1 → §11.

> **§1–§10 live in [AGENTS.md](AGENTS.md) and are imported below — edit shared rules there, never here.** This file adds only §11 (Claude Code tooling), which supersedes the Antigravity §11 in the imported text.

@AGENTS.md

---

## 11. Tools (Claude Code)

*Supersedes §11 of the imported AGENTS.md.*

Output overflow prevention: prefer `--max-count`, `head -50`, `PAGER=cat`; pass `head_limit` on Grep. For long-running shell → use `run_in_background` and read the output file when notified; never poll with sleep loops.

- **TodoWrite:** use when a task has ≥ 3 discrete steps or the user gives a multi-item list. Exactly one `in_progress` at a time; mark `completed` immediately, don't batch.
- **Agent (Explore):** for codebase searches > 3 queries, spawn Agent with `subagent_type=Explore` rather than chaining Globs / Greps. Briefer prompts than for general-purpose.
- **Plan mode:** use `EnterPlanMode` for non-trivial implementations *before* writing code; exit with `ExitPlanMode` only after the plan is finalized.
- **Skills:** when the user types `/name`, invoke only if `name` appears in the available-skills list. Never invent a skill name from training data.
- **Memory:** persistent memory managed by Claude Code's auto-memory system. Do NOT save things derivable from code (patterns, paths, git history).
- **Parallel tool calls:** when calls are independent, batch in one message; only serialize when later calls depend on earlier results.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
