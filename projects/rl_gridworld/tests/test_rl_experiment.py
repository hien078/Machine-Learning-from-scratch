"""Tests for the GridWorld Q-learning vs value-iteration experiment."""

from __future__ import annotations

import numpy as np
import pytest
from rl_experiment import (
    GAMMA,
    build_dynamics,
    make_env,
    optimal_action_sets,
    policy_agreement,
    random_policy_return,
    rollout_return,
    train_agent,
    value_iteration,
)


@pytest.fixture(scope="module")
def exact_solution() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(v_star, q_star, terminal) for the study environment."""
    next_state, reward, terminal = build_dynamics(make_env())
    v_star, q_star = value_iteration(next_state, reward, terminal)
    return v_star, q_star, terminal


def test_value_iteration_satisfies_bellman_optimality(exact_solution) -> None:
    v_star, _, terminal = exact_solution
    next_state, reward, _ = build_dynamics(make_env())
    backup = (reward + GAMMA * v_star[next_state]).max(axis=1)
    backup[terminal] = 0.0
    np.testing.assert_allclose(v_star, backup, atol=1e-8)
    assert np.all(v_star[terminal] == 0.0)


def test_value_iteration_policy_earns_optimal_return(exact_solution) -> None:
    _, q_star, _ = exact_solution
    # Shortest path on the 4x4 grid: 6 moves -> 5 step penalties + goal reward.
    expected = 10.0 - 5 * 0.1
    assert np.isclose(rollout_return(q_star.argmax(axis=1)), expected, atol=1e-12)


def test_qlearning_improves_over_random_policy() -> None:
    agent, returns, _ = train_agent(epsilon=0.1, alpha=0.5, seed=0, n_episodes=150)
    random_return = random_policy_return(seed=0)
    assert returns[-20:].mean() > random_return + 1.0
    assert rollout_return(agent.q_table.argmax(axis=1)) > random_return + 1.0


def test_converged_greedy_policy_matches_value_iteration(exact_solution) -> None:
    _, q_star, terminal = exact_solution
    optimal_sets = optimal_action_sets(q_star)
    agent, _, _ = train_agent(epsilon=0.1, alpha=0.5, seed=0, n_episodes=300)
    assert policy_agreement(agent.q_table, optimal_sets, terminal) >= 0.85


def test_training_is_deterministic_under_fixed_seed() -> None:
    agent_a, returns_a, _ = train_agent(epsilon=0.1, alpha=0.5, seed=3, n_episodes=100)
    agent_b, returns_b, _ = train_agent(epsilon=0.1, alpha=0.5, seed=3, n_episodes=100)
    np.testing.assert_array_equal(returns_a, returns_b)
    np.testing.assert_array_equal(agent_a.q_table, agent_b.q_table)
