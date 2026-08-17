"""First-principles pure NumPy implementations of reinforcement learning algorithms."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

GOAL_REWARD = 10.0
TRAP_REWARD = -5.0
STEP_REWARD = -0.1
START_STATE = (0, 0)


class GridWorldEnv:
    """A discrete grid environment for reinforcement learning experiments.

    The agent starts at ``(0, 0)`` and moves on a square grid; reaching the
    goal or the trap ends the episode. Terminal states are absorbing: calling
    ``step`` after the episode has ended raises instead of silently resuming.
    """

    def __init__(
        self,
        grid_size: int = 4,
        goal: tuple[int, int] = (3, 3),
        trap: tuple[int, int] = (1, 1),
        max_steps: int | None = None,
    ) -> None:
        """Validate the layout and place the agent at the start state.

        Args:
            grid_size: Side length of the square grid.
            goal: Terminal cell with reward ``GOAL_REWARD``.
            trap: Terminal cell with reward ``TRAP_REWARD``.
            max_steps: Optional episode-length cap. When set, ``step`` reports
                ``done=True`` once the episode reaches this many steps; reward
                semantics are unchanged. When ``None`` (the default) behavior
                is identical to an uncapped environment.

        Raises:
            ValueError: If ``grid_size`` is smaller than 2, if ``goal`` or
                ``trap`` lies outside the grid, if start, goal, and trap are
                not pairwise distinct, or if ``max_steps`` is not positive.
        """
        if grid_size < 2:
            raise ValueError("grid_size must be at least 2")
        for name, cell in (("goal", goal), ("trap", trap)):
            if not (0 <= cell[0] < grid_size and 0 <= cell[1] < grid_size):
                raise ValueError(f"{name} {cell} lies outside the {grid_size}x{grid_size} grid")
        if goal == trap or goal == START_STATE or trap == START_STATE:
            raise ValueError("start, goal, and trap must be pairwise distinct cells")
        if max_steps is not None and max_steps < 1:
            raise ValueError("max_steps must be at least 1 when set")
        self.grid_size = grid_size
        self.goal = goal
        self.trap = trap
        self.max_steps = max_steps
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
        self.state = START_STATE
        self._done = False
        self._step_count = 0

    @property
    def num_states(self) -> int:
        """Number of discrete states (``grid_size ** 2``)."""
        return self.grid_size**2

    @property
    def num_actions(self) -> int:
        """Number of discrete actions (Up, Down, Left, Right)."""
        return len(self.actions)

    def reset(self) -> int:
        """Reset the environment and return the initial state index."""
        self.state = START_STATE
        self._done = False
        self._step_count = 0
        return self._state_to_idx(self.state)

    def _state_to_idx(self, state: tuple[int, int]) -> int:
        return state[0] * self.grid_size + state[1]

    def _transition(
        self, state: tuple[int, int], action_idx: int
    ) -> tuple[tuple[int, int], float, bool]:
        """Pure transition function shared by ``step`` and ``dynamics``.

        Args:
            state: Current ``(row, col)`` cell.
            action_idx: Index into ``self.actions``.

        Returns:
            ``(next_cell, reward, terminal)`` where moves off the grid are
            clipped to the border and ``terminal`` marks arrival at the goal
            or the trap.
        """
        dr, dc = self.actions[action_idx]
        nr = max(0, min(self.grid_size - 1, state[0] + dr))
        nc = max(0, min(self.grid_size - 1, state[1] + dc))
        next_cell = (nr, nc)
        if next_cell == self.goal:
            return next_cell, GOAL_REWARD, True
        if next_cell == self.trap:
            return next_cell, TRAP_REWARD, True
        return next_cell, STEP_REWARD, False

    def step(self, action_idx: int) -> tuple[int, float, bool, dict[str, Any]]:
        """Apply one action and return ``(next_state, reward, done, info)``.

        Args:
            action_idx: Index into ``self.actions`` (Up, Down, Left, Right).

        Returns:
            The next state index, the reward, whether the episode ended, and an
            empty info dictionary. With ``max_steps`` set, ``done`` is also
            ``True`` once the episode reaches that many steps (the reward for
            the truncating step is unchanged).

        Raises:
            RuntimeError: If called after the episode has ended without an
                intervening ``reset``.
        """
        if self._done:
            raise RuntimeError("episode has ended; call reset() before stepping again")
        self.state, reward, done = self._transition(self.state, action_idx)
        self._step_count += 1
        if self.max_steps is not None and self._step_count >= self.max_steps:
            done = True
        self._done = done
        return self._state_to_idx(self.state), reward, done, {}

    def dynamics(self) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        """Return the exact deterministic MDP tables ``(P, R)``.

        Both tables are derived from the same transition function ``step``
        uses, so they agree with ``step`` on every non-terminal ``(state,
        action)`` pair. The environment is deterministic: each entry is the
        single possible outcome, not a distribution. ``max_steps`` truncation
        is an episode-level device and does not appear in these tables.

        Returns:
            ``P`` of shape ``(num_states, num_actions)`` with dtype ``int64``,
            mapping ``(state, action)`` to the next-state index, and ``R`` of
            the same shape with dtype ``float64`` holding the reward. The goal
            and trap rows are absorbing: ``P[s, a] = s`` and ``R[s, a] = 0.0``
            (``step`` never runs from a terminal state).
        """
        p = np.zeros((self.num_states, self.num_actions), dtype=np.int64)
        r = np.zeros((self.num_states, self.num_actions), dtype=np.float64)
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                s = self._state_to_idx((row, col))
                if (row, col) in (self.goal, self.trap):
                    p[s, :] = s
                    continue
                for a in range(self.num_actions):
                    next_cell, reward, _ = self._transition((row, col), a)
                    p[s, a] = self._state_to_idx(next_cell)
                    r[s, a] = reward
        return p, r


class QLearningAgent:
    """Tabular Q-learning agent with an epsilon-greedy policy."""

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        random_state: int | None = None,
    ) -> None:
        """Initialize the Q-table and the exploration generator.

        Args:
            num_states: Number of discrete states.
            num_actions: Number of discrete actions.
            alpha: Learning rate in ``(0, 1]``.
            gamma: Discount factor in ``[0, 1]``.
            epsilon: Exploration probability in ``[0, 1]``.
            random_state: Seed for the isolated random generator.

        Raises:
            ValueError: If any hyperparameter lies outside its valid range.
        """
        if num_states < 1 or num_actions < 1:
            raise ValueError("num_states and num_actions must be positive")
        if not 0.0 < alpha <= 1.0 or not 0.0 <= gamma <= 1.0 or not 0.0 <= epsilon <= 1.0:
            raise ValueError("alpha must be in (0, 1]; gamma and epsilon must be in [0, 1]")
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((num_states, num_actions))
        self._rng = np.random.default_rng(random_state)

    def select_action(self, state: int) -> int:
        """Select an action epsilon-greedily, breaking value ties uniformly.

        Args:
            state: Current state index.

        Returns:
            The chosen action index.
        """
        if self._rng.random() < self.epsilon:
            return int(self._rng.integers(self.num_actions))
        values = self.q_table[state]
        best_actions = np.flatnonzero(values == values.max())
        return int(self._rng.choice(best_actions))

    def greedy_action(self, state: int) -> int:
        """Return the greedy action for ``state``, breaking value ties uniformly.

        Ties are broken with the same seeded generator used for exploration,
        so the sequence of choices is deterministic under ``random_state``.
        No epsilon-exploration is applied.

        Args:
            state: State index.

        Returns:
            An action index maximizing ``q_table[state]``.
        """
        values = self.q_table[state]
        best_actions = np.flatnonzero(values == values.max())
        return int(self._rng.choice(best_actions))

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> float:
        """Apply one Q-learning update toward the Bellman optimality target.

        Args:
            state: State where ``action`` was taken.
            action: Action index that was taken.
            reward: Observed reward.
            next_state: Resulting state index.
            done: Whether the episode ended, which drops the bootstrap term.

        Returns:
            The temporal-difference error before the learning-rate scaling.
        """
        target = reward if done else reward + self.gamma * np.max(self.q_table[next_state])
        td_error = target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error
        return float(td_error)
