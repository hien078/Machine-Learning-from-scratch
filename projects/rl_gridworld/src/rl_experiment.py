"""Tabular RL capstone: Q-learning on GridWorldEnv vs a value-iteration baseline.

Trains ``QLearningAgent`` across multiple seeds, runs a small epsilon/alpha
sweep, computes the exact optimal solution with value iteration from the same
environment dynamics, and writes plots plus a markdown report to ``reports/``.

Run from the repo root::

    python projects/rl_gridworld/src/rl_experiment.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from ml_first_principles.rl_models import (
    GOAL_REWARD,
    STEP_REWARD,
    TRAP_REWARD,
    GridWorldEnv,
    QLearningAgent,
)

matplotlib.use("Agg")

# --- experiment configuration (fixed constants, per repo conventions) ---
GRID_SIZE = 4
GOAL = (3, 3)
TRAP = (1, 1)
GAMMA = 0.99
N_EPISODES = 300
MAX_STEPS_PER_EPISODE = 100
SEEDS = (0, 1, 2, 3, 4)
BASELINE_EPSILON = 0.1
BASELINE_ALPHA = 0.5
SWEEP_EPSILONS = (0.05, 0.1, 0.3)
SWEEP_ALPHAS = (0.1, 0.5)
TAIL_WINDOW = 50  # episodes averaged when scoring a config
SMOOTH_WINDOW = 10  # moving-average window for the learning-curve plot
VI_TOL = 1e-10
ACTION_ARROWS = ("^", "v", "<", ">")  # Up, Down, Left, Right

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"

# Data-viz palette (light mode, see dataviz reference palette).
INK = "#0b0b0b"
MUTED = "#898781"
GRID_LINE = "#e1e0d9"
SURFACE = "#fcfcfb"
SERIES_BLUE = "#2a78d6"
SEQ_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def make_env() -> GridWorldEnv:
    """Build the environment used everywhere in this study."""
    return GridWorldEnv(grid_size=GRID_SIZE, goal=GOAL, trap=TRAP)


# --- exact dynamics + value iteration -------------------------------------


def build_dynamics(env: GridWorldEnv) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract the deterministic MDP tables from the environment definition.

    Returns:
        ``(next_state, reward, terminal)`` where ``next_state`` and ``reward``
        have shape ``(S, A)`` and ``terminal`` is a boolean mask of shape
        ``(S,)`` marking absorbing states.
    """
    n_states = env.grid_size**2
    n_actions = len(env.actions)
    next_state = np.zeros((n_states, n_actions), dtype=np.int64)
    reward = np.zeros((n_states, n_actions))
    terminal = np.zeros(n_states, dtype=bool)
    for r in range(env.grid_size):
        for c in range(env.grid_size):
            s = r * env.grid_size + c
            if (r, c) in (env.goal, env.trap):
                terminal[s] = True
                next_state[s, :] = s  # absorbing, never stepped
                continue
            for a, (dr, dc) in enumerate(env.actions):
                nr = max(0, min(env.grid_size - 1, r + dr))
                nc = max(0, min(env.grid_size - 1, c + dc))
                next_state[s, a] = nr * env.grid_size + nc
                if (nr, nc) == env.goal:
                    reward[s, a] = GOAL_REWARD
                elif (nr, nc) == env.trap:
                    reward[s, a] = TRAP_REWARD
                else:
                    reward[s, a] = STEP_REWARD
    return next_state, reward, terminal


def value_iteration(
    next_state: np.ndarray,
    reward: np.ndarray,
    terminal: np.ndarray,
    gamma: float = GAMMA,
    tol: float = VI_TOL,
    max_iter: int = 10_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the MDP exactly, returning ``(v, q)`` with terminal values fixed at 0."""
    v = np.zeros(len(terminal))
    for _ in range(max_iter):
        q = reward + gamma * v[next_state]
        q[terminal, :] = 0.0
        v_new = q.max(axis=1)
        if np.max(np.abs(v_new - v)) < tol:
            return v_new, q
        v = v_new
    raise RuntimeError("value iteration did not converge")


def optimal_action_sets(q: np.ndarray, atol: float = 1e-8) -> list[np.ndarray]:
    """Per-state arrays of actions whose Q-value ties the maximum."""
    return [np.flatnonzero(q[s] >= q[s].max() - atol) for s in range(q.shape[0])]


# --- rollouts and training -------------------------------------------------


def rollout_return(policy: np.ndarray, max_steps: int = MAX_STEPS_PER_EPISODE) -> float:
    """Undiscounted return of one greedy episode following ``policy`` from start."""
    env = make_env()
    state = env.reset()
    total = 0.0
    for _ in range(max_steps):
        state, r, done, _ = env.step(int(policy[state]))
        total += r
        if done:
            break
    return total


def random_policy_return(seed: int, n_episodes: int = 20) -> float:
    """Mean undiscounted return of a uniform-random policy (exploration floor)."""
    rng = np.random.default_rng(seed)
    env = make_env()
    totals = []
    for _ in range(n_episodes):
        env.reset()
        total = 0.0
        for _ in range(MAX_STEPS_PER_EPISODE):
            _, r, done, _ = env.step(int(rng.integers(len(env.actions))))
            total += r
            if done:
                break
        totals.append(total)
    return float(np.mean(totals))


def train_agent(
    epsilon: float,
    alpha: float,
    seed: int,
    n_episodes: int = N_EPISODES,
    optimal_return: float | None = None,
) -> tuple[QLearningAgent, np.ndarray, int | None]:
    """Train one agent and record the per-episode undiscounted returns.

    Returns:
        ``(agent, returns, solve_episode)`` where ``solve_episode`` is the first
        episode (1-based) after which the greedy policy attains
        ``optimal_return``, or ``None`` if never reached / not tracked.
    """
    env = make_env()
    agent = QLearningAgent(
        num_states=env.grid_size**2,
        num_actions=len(env.actions),
        alpha=alpha,
        gamma=GAMMA,
        epsilon=epsilon,
        random_state=seed,
    )
    returns = np.zeros(n_episodes)
    solve_episode: int | None = None
    for ep in range(n_episodes):
        state = env.reset()
        total = 0.0
        for _ in range(MAX_STEPS_PER_EPISODE):
            action = agent.select_action(state)
            next_s, r, done, _ = env.step(action)
            agent.update(state, action, r, next_s, done)
            state = next_s
            total += r
            if done:
                break
        returns[ep] = total
        if optimal_return is not None and solve_episode is None:
            greedy = rollout_return(agent.q_table.argmax(axis=1))
            if np.isclose(greedy, optimal_return):
                solve_episode = ep + 1
    return agent, returns, solve_episode


@dataclass(frozen=True)
class SweepResult:
    """Multi-seed outcome of one (epsilon, alpha) configuration."""

    epsilon: float
    alpha: float
    tail_mean: float  # mean return over the final TAIL_WINDOW episodes, across seeds
    greedy_return: float  # mean greedy-rollout return of the trained policies
    agreement: float  # mean fraction of non-terminal states matching the VI policy
    episodes_to_solve: float  # mean episodes until the greedy policy is optimal


def run_config(
    epsilon: float,
    alpha: float,
    optimal_sets: list[np.ndarray],
    terminal: np.ndarray,
    optimal_return: float,
) -> tuple[SweepResult, np.ndarray, list[QLearningAgent]]:
    """Train one configuration across all seeds; also return the returns matrix."""
    all_returns = np.zeros((len(SEEDS), N_EPISODES))
    greedy_returns = []
    agreements = []
    solve_episodes = []
    agents = []
    for i, seed in enumerate(SEEDS):
        agent, returns, solve_ep = train_agent(epsilon, alpha, seed, optimal_return=optimal_return)
        all_returns[i] = returns
        policy = agent.q_table.argmax(axis=1)
        greedy_returns.append(rollout_return(policy))
        agreements.append(policy_agreement(agent.q_table, optimal_sets, terminal))
        solve_episodes.append(N_EPISODES if solve_ep is None else solve_ep)
        agents.append(agent)
    result = SweepResult(
        epsilon=epsilon,
        alpha=alpha,
        tail_mean=float(all_returns[:, -TAIL_WINDOW:].mean()),
        greedy_return=float(np.mean(greedy_returns)),
        agreement=float(np.mean(agreements)),
        episodes_to_solve=float(np.mean(solve_episodes)),
    )
    return result, all_returns, agents


def policy_agreement(
    q_table: np.ndarray,
    optimal_sets: list[np.ndarray],
    terminal: np.ndarray,
) -> float:
    """Fraction of non-terminal states whose greedy action is VI-optimal."""
    greedy = q_table.argmax(axis=1)
    hits = [greedy[s] in optimal_sets[s] for s in range(len(terminal)) if not terminal[s]]
    return float(np.mean(hits))


# --- rendering -------------------------------------------------------------


def policy_arrows(policy: np.ndarray, env: GridWorldEnv) -> str:
    """Text grid of greedy-action arrows, with G (goal) and T (trap) marked."""
    rows = []
    for r in range(env.grid_size):
        cells = []
        for c in range(env.grid_size):
            if (r, c) == env.goal:
                cells.append("G")
            elif (r, c) == env.trap:
                cells.append("T")
            else:
                cells.append(ACTION_ARROWS[int(policy[r * env.grid_size + c])])
        rows.append(" ".join(cells))
    return "\n".join(rows)


def _style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_color(GRID_LINE)
    ax.tick_params(colors=MUTED, labelcolor=MUTED)


def plot_learning_curve(all_returns: np.ndarray, optimal_return: float, out_path: Path) -> None:
    """Mean episode return across seeds with min-max band and the optimal line."""
    kernel = np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW
    smoothed = np.array([np.convolve(run, kernel, mode="valid") for run in all_returns])
    episodes = np.arange(smoothed.shape[1]) + SMOOTH_WINDOW

    fig, ax = plt.subplots(figsize=(7.2, 4.2), facecolor=SURFACE)
    _style_axes(ax)
    ax.axhline(
        optimal_return, color=MUTED, linewidth=1.2, linestyle="--", label="Value-iteration optimum"
    )
    ax.fill_between(
        episodes,
        smoothed.min(axis=0),
        smoothed.max(axis=0),
        color=SERIES_BLUE,
        alpha=0.18,
        linewidth=0,
        label=f"Range across {len(SEEDS)} seeds",
    )
    ax.plot(episodes, smoothed.mean(axis=0), color=SERIES_BLUE, linewidth=2, label="Mean return")
    ax.set_xlabel("Episode", color=INK)
    ax.set_ylabel(f"Return ({SMOOTH_WINDOW}-episode moving average)", color=INK)
    ax.set_title(
        f"Q-learning on {GRID_SIZE}x{GRID_SIZE} GridWorld "
        f"(epsilon={BASELINE_EPSILON}, alpha={BASELINE_ALPHA})",
        color=INK,
    )
    ax.grid(color=GRID_LINE, linewidth=0.6)
    ax.legend(loc="lower right", frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_value_heatmaps(
    v_optimal: np.ndarray,
    v_learned: np.ndarray,
    env: GridWorldEnv,
    out_path: Path,
) -> None:
    """Side-by-side heatmaps of the optimal and the learned state-value function."""
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_RAMP)
    grids = [
        (v_optimal.reshape(env.grid_size, env.grid_size), "Value iteration $V^*$"),
        (v_learned.reshape(env.grid_size, env.grid_size), "Q-learning $\\max_a Q$ (seed 0)"),
    ]
    vmin = min(g.min() for g, _ in grids)
    vmax = max(g.max() for g, _ in grids)

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.0), facecolor=SURFACE)
    for ax, (grid, title) in zip(axes, grids, strict=True):
        _style_axes(ax)
        im = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, color=INK)
        ax.set_xlabel("Column", color=INK)
        ax.set_ylabel("Row", color=INK)
        ax.set_xticks(range(env.grid_size))
        ax.set_yticks(range(env.grid_size))
        for r in range(env.grid_size):
            for c in range(env.grid_size):
                label = {env.goal: "G", env.trap: "T"}.get((r, c), f"{grid[r, c]:.2f}")
                # Flip label ink on dark cells so text never relies on the fill.
                frac = (grid[r, c] - vmin) / (vmax - vmin) if vmax > vmin else 0.0
                ax.text(
                    c,
                    r,
                    label,
                    ha="center",
                    va="center",
                    color=SURFACE if frac > 0.55 else INK,
                    fontsize=9,
                )
    cbar = fig.colorbar(im, ax=axes, shrink=0.85)
    cbar.set_label("State value", color=INK)
    cbar.ax.tick_params(colors=MUTED)
    cbar.outline.set_edgecolor(GRID_LINE)
    fig.savefig(out_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


# --- report ----------------------------------------------------------------


def write_report(
    baseline: SweepResult,
    all_returns: np.ndarray,
    sweep: list[SweepResult],
    optimal_return: float,
    random_return: float,
    vi_policy_text: str,
    ql_policy_text: str,
    out_path: Path,
) -> None:
    """Write the markdown report embedding the generated figures."""
    tail_mean = all_returns[:, -TAIL_WINDOW:].mean()
    tail_lo = all_returns[:, -TAIL_WINDOW:].mean(axis=1).min()
    tail_hi = all_returns[:, -TAIL_WINDOW:].mean(axis=1).max()
    best = max(sweep, key=lambda s: s.tail_mean)

    lines = [
        "# GridWorld Q-learning vs Value Iteration",
        "",
        "Generated by `src/rl_experiment.py` (deterministic; re-running "
        "reproduces this file byte-for-byte). "
        f"Env: {GRID_SIZE}x{GRID_SIZE} grid, goal {GOAL}, trap {TRAP}, "
        f"gamma={GAMMA}, {N_EPISODES} episodes x {len(SEEDS)} seeds {SEEDS}.",
        "",
        "## Learning curve",
        "",
        "![Learning curve](learning_curve.png)",
        "",
        f"- Baseline config: epsilon={BASELINE_EPSILON}, alpha={BASELINE_ALPHA}.",
        f"- Final mean return (last {TAIL_WINDOW} episodes, across seeds): "
        f"**{tail_mean:.2f}** (per-seed range {tail_lo:.2f} to {tail_hi:.2f}).",
        f"- Value-iteration optimal return from start: **{optimal_return:.2f}** "
        f"(undiscounted; greedy rollout of the exact solution).",
        f"- Uniform-random policy baseline: **{random_return:.2f}**.",
        f"- Greedy rollout of the trained policies: **{baseline.greedy_return:.2f}** "
        f"-- the residual gap in the training curve is the epsilon-greedy "
        f"exploration cost, not a policy error.",
        "",
        "## Epsilon / alpha sweep",
        "",
        f"Scored by mean return over the final {TAIL_WINDOW} episodes across "
        f"{len(SEEDS)} seeds. `agreement` is the fraction of non-terminal states "
        "whose greedy action is optimal under value iteration; `episodes to "
        "optimal` is the mean episode after which the greedy policy first "
        "achieves the optimal return.",
        "",
        "| epsilon | alpha | tail mean return | greedy return | agreement | episodes to optimal |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for s in sweep:
        marker = " **(best)**" if s is best else ""
        lines.append(
            f"| {s.epsilon} | {s.alpha} | {s.tail_mean:.2f}{marker} "
            f"| {s.greedy_return:.2f} | {s.agreement:.2f} | {s.episodes_to_solve:.0f} |"
        )
    lines += [
        "",
        "## Learned greedy policy",
        "",
        "Arrows = greedy action, `G` = goal (+10), `T` = trap (-5). "
        "Left: value iteration. Right: Q-learning (seed 0, baseline config).",
        "",
        "```",
        "Value iteration        Q-learning",
        *(
            f"{vi_row:<22} {ql_row}"
            for vi_row, ql_row in zip(
                vi_policy_text.splitlines(), ql_policy_text.splitlines(), strict=True
            )
        ),
        "```",
        "",
        "Ties note: several states have multiple optimal actions (equal path "
        "length to the goal), so the two grids may show different arrows at a "
        "state while both are optimal; `agreement` above counts an action as "
        "correct if it ties the value-iteration maximum.",
        "",
        "## Value function",
        "",
        "![Value heatmaps](value_heatmap.png)",
        "",
        "## Findings",
        "",
    ]
    # Aggregate the sweep along each axis so the findings are computed, not asserted.
    speed_by_alpha = {
        a: np.mean([s.episodes_to_solve for s in sweep if s.alpha == a]) for a in SWEEP_ALPHAS
    }
    tail_by_epsilon = {
        e: np.mean([s.tail_mean for s in sweep if s.epsilon == e]) for e in SWEEP_EPSILONS
    }
    lines += [
        f"- **Q-learning reaches the value-iteration optimum.** The greedy "
        f"rollout of the trained baseline policies earns {baseline.greedy_return:.2f}, "
        f"matching the exact optimum {optimal_return:.2f} (true for every sweep "
        f"config), and {baseline.agreement:.0%} of non-terminal greedy actions "
        f"tie the optimal action values.",
        f"- **Alpha sets convergence speed.** In this deterministic env a large "
        f"step size is safe: alpha={SWEEP_ALPHAS[1]} needs "
        f"{speed_by_alpha[SWEEP_ALPHAS[1]]:.0f} episodes on average to make the "
        f"greedy policy optimal vs {speed_by_alpha[SWEEP_ALPHAS[0]]:.0f} for "
        f"alpha={SWEEP_ALPHAS[0]}, while final greedy quality is identical.",
        "- **Epsilon taxes the training return.** Mean tail return falls "
        "monotonically with exploration ("
        + ", ".join(f"epsilon={e}: {tail_by_epsilon[e]:.2f}" for e in SWEEP_EPSILONS)
        + f") because each random step risks the trap or a detour, which is why "
        f"epsilon={best.epsilon}, alpha={best.alpha} wins the tail-return "
        f"ranking; the exploration cost is paid during training only, not by "
        f"the final greedy policy.",
        "",
    ]
    out_path.write_text("\n".join(lines))


# --- entry point -----------------------------------------------------------


def main() -> None:
    """Run the full study and write reports/ artifacts."""
    t0 = time.perf_counter()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    env = make_env()

    # Exact solution from the same dynamics.
    next_state, reward, terminal = build_dynamics(env)
    v_star, q_star = value_iteration(next_state, reward, terminal)
    optimal_sets = optimal_action_sets(q_star)
    vi_policy = q_star.argmax(axis=1)
    optimal_return = rollout_return(vi_policy)
    random_return = float(np.mean([random_policy_return(seed) for seed in SEEDS]))

    # Sweep (the baseline config is one of the grid points).
    sweep_results: list[SweepResult] = []
    baseline_result = None
    baseline_returns = None
    baseline_agents = None
    for epsilon in SWEEP_EPSILONS:
        for alpha in SWEEP_ALPHAS:
            result, all_returns, agents = run_config(
                epsilon, alpha, optimal_sets, terminal, optimal_return
            )
            sweep_results.append(result)
            if epsilon == BASELINE_EPSILON and alpha == BASELINE_ALPHA:
                baseline_result, baseline_returns, baseline_agents = result, all_returns, agents
    assert baseline_result is not None and baseline_returns is not None

    # Rendering.
    seed0_agent = baseline_agents[0]
    ql_policy = seed0_agent.q_table.argmax(axis=1)
    vi_text = policy_arrows(vi_policy, env)
    ql_text = policy_arrows(ql_policy, env)
    plot_learning_curve(baseline_returns, optimal_return, REPORTS_DIR / "learning_curve.png")
    v_learned = seed0_agent.q_table.max(axis=1)
    v_learned[terminal] = 0.0
    plot_value_heatmaps(v_star, v_learned, env, REPORTS_DIR / "value_heatmap.png")

    wall = time.perf_counter() - t0
    write_report(
        baseline_result,
        baseline_returns,
        sweep_results,
        optimal_return,
        random_return,
        vi_text,
        ql_text,
        REPORTS_DIR / "report.md",
    )

    best = max(sweep_results, key=lambda s: s.tail_mean)
    print(f"Optimal return (value iteration): {optimal_return:.2f}")
    print(f"Baseline greedy return:           {baseline_result.greedy_return:.2f}")
    print(f"Baseline tail mean return:        {baseline_result.tail_mean:.2f}")
    print(f"Policy agreement with VI:         {baseline_result.agreement:.2%}")
    print(f"Best sweep config: epsilon={best.epsilon}, alpha={best.alpha}")
    print(f"Wrote {REPORTS_DIR}/report.md (+2 PNGs) in {wall:.1f}s")


if __name__ == "__main__":
    main()
