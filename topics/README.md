# Topics — First Principles ML Stories

Each topic tells a complete reasoning story: **WHY → WHAT → HOW → BUILD → VERIFY → CONNECTIONS**

22 topics, numbered `01`–`22`. Open any folder and read its `README.md` first.

## Navigation

[**INDEX.md**](../INDEX.md) is the single source of truth for the curriculum: the topic
matrix with per-topic maturity, the prerequisite graph, the math-to-algorithm mapping,
and the five curriculum phases.

## What is in a topic folder

The four-file layout and what each file must contain are defined in
[NOTEBOOK_STANDARDS.md](../NOTEBOOK_STANDARDS.md) §2–§3. In short: `README.md` navigates,
`theory.md` derives, `first_principles.ipynb` implements and verifies, `exercises.ipynb`
drills.

Large topics add focused notebooks — topic 18 splits into `bandits_and_exploration.ipynb`,
`monte_carlo_and_td.ipynb`, `policy_gradient_methods.ipynb`, and `deep_rl_advances.ipynb`.

Topics 14–17 compare against PyTorch, which is an optional dependency
(`pip install torch`); every from-scratch implementation runs without it.

## Quality gate

Maturity levels and the *Verified* gate are defined in
[NOTEBOOK_STANDARDS.md](../NOTEBOOK_STANDARDS.md) §9–§10 and enforced by
`python scripts/check.py` plus `python scripts/execute_all_notebooks.py`.
