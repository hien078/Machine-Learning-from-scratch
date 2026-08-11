"""First-principles pure NumPy implementations of reinforcement learning algorithms."""

from __future__ import annotations

from typing import Any

import numpy as np

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
    ) -> None:
        """Validate the layout and place the agent at the start state.

        Args:
            grid_size: Side length of the square grid.
            goal: Terminal cell with reward ``GOAL_REWARD``.
            trap: Terminal cell with reward ``TRAP_REWARD``.

        Raises:
            ValueError: If ``grid_size`` is smaller than 2, if ``goal`` or
                ``trap`` lies outside the grid, or if start, goal, and trap are
                not pairwise distinct.
        """
        if grid_size < 2:
            raise ValueError("grid_size must be at least 2")
        for name, cell in (("goal", goal), ("trap", trap)):
            if not (0 <= cell[0] < grid_size and 0 <= cell[1] < grid_size):
                raise ValueError(f"{name} {cell} lies outside the {grid_size}x{grid_size} grid")
        if goal == trap or goal == START_STATE or trap == START_STATE:
            raise ValueError("start, goal, and trap must be pairwise distinct cells")
        self.grid_size = grid_size
        self.goal = goal
        self.trap = trap
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
        self.state = START_STATE
        self._done = False

    def reset(self) -> int:
        """Reset the environment and return the initial state index."""
        self.state = START_STATE
        self._done = False
        return self._state_to_idx(self.state)

    def _state_to_idx(self, state: tuple[int, int]) -> int:
        return state[0] * self.grid_size + state[1]

    def step(self, action_idx: int) -> tuple[int, float, bool, dict[str, Any]]:
        """Apply one action and return ``(next_state, reward, done, info)``.

        Args:
            action_idx: Index into ``self.actions`` (Up, Down, Left, Right).

        Returns:
            The next state index, the reward, whether the episode ended, and an
            empty info dictionary.

        Raises:
            RuntimeError: If called after the episode has ended without an
                intervening ``reset``.
        """
        if self._done:
            raise RuntimeError("episode has ended; call reset() before stepping again")
        dr, dc = self.actions[action_idx]
        nr = max(0, min(self.grid_size - 1, self.state[0] + dr))
        nc = max(0, min(self.grid_size - 1, self.state[1] + dc))
        self.state = (nr, nc)

        if self.state == self.goal:
            self._done = True
            return self._state_to_idx(self.state), GOAL_REWARD, True, {}
        if self.state == self.trap:
            self._done = True
            return self._state_to_idx(self.state), TRAP_REWARD, True, {}
        return self._state_to_idx(self.state), STEP_REWARD, False, {}


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
