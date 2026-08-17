# Topic 18: Reinforcement Learning

> **Phase:** 5 | **Status:** ✅ Complete | **Prerequisites:** [02 Gradient Descent](../02_gradient_descent/README.md), [13 Neural Networks](../13_neural_networks/README.md), Probability & Statistics

## Overview

Reinforcement Learning (RL) addresses how agents ought to take actions in an environment to maximize cumulative reward. Unlike supervised learning, which relies on labeled examples, RL relies on learning from interaction, dealing with delayed rewards, and balancing exploration of new strategies with the exploitation of known strategies. This module covers Markov Decision Processes (MDPs), Bellman equations, tabular methods like Value Iteration and Q-Learning, and introduces the concepts behind Deep Q-Networks and Policy Gradients.

## Scope

- **In scope:** MDPs, Bellman equations, bandits and exploration, Monte Carlo and TD learning, tabular Q-Learning/SARSA, REINFORCE and actor-critic policy gradients, DQN and Double DQN in pure NumPy on toy environments.
- **Out of scope (described but not implemented):** SAC, large-scale continuous control, distributed RL, and RLHF systems engineering.

## Recommended Reading Order

1. **Theory & Foundations:** Start with `theory.md` to understand the mathematics.
2. **Core Algorithms:** Move to `first_principles.ipynb` and `bandits_and_exploration.ipynb`.
3. **Advanced Value Methods:** Read `monte_carlo_and_td.ipynb`.
4. **Policy Methods:** Proceed to `policy_gradient_methods.ipynb`.
5. **Deep RL:** Study `deep_rl_advances.ipynb`.
6. **Practice:** Finish with `exercises.ipynb` to verify your understanding.

## Contents

| File | Description |
|------|-------------|
| [theory.md](theory.md) | First-principles derivation of MDPs, Bellman equations, tabular methods, Policy Gradients, PPO, continuous control, RLHF, and more. |
| [first_principles.ipynb](first_principles.ipynb) | From-scratch NumPy implementation of GridWorld, Value Iteration, Q-Learning, Deep Q-Network (DQN), and Deadly Triad failure case. |
| [bandits_and_exploration.ipynb](bandits_and_exploration.ipynb) | Exploration foundation including epsilon-greedy and UCB. |
| [monte_carlo_and_td.ipynb](monte_carlo_and_td.ipynb) | Implementation of Monte Carlo methods and Temporal Difference learning (TD(λ)). |
| [policy_gradient_methods.ipynb](policy_gradient_methods.ipynb) | Implementation of REINFORCE, Advantage Actor-Critic (A2C), and PPO. |
| [deep_rl_advances.ipynb](deep_rl_advances.ipynb) | Deep RL advances including Double DQN and DDPG, with a conceptual outlook on Soft Actor-Critic (SAC). |
| [exercises.ipynb](exercises.ipynb) | Pen-and-paper calculations, implementation exercises (SARSA, Double DQN), and conceptual analysis. |

The reusable, unit-tested reference implementations live in [`src/ml_first_principles/rl_models.py`](https://github.com/hien078/Machine-Learning-from-scratch/blob/master/src/ml_first_principles/rl_models.py) (`GridWorldEnv`, `QLearningAgent`), covered by `tests/test_phase5_models.py`.

## Connections

**Prerequisites**:
- [02 Gradient Descent](../02_gradient_descent/theory.md): Foundation for policy gradients and function approximation.
- [13 Neural Networks](../13_neural_networks/theory.md): Required for understanding Deep Q-Networks (DQN).
- Probability & Statistics: Expectations, stochastic processes, distributions, and Monte Carlo sampling.
- Calculus & Optimization: Chain rule for policy gradients and advantage estimation.

**Related Synthesis**:
- [Optimization Methods Compared](../../synthesis/optimization_methods_compared.md)

**Next Topics**:
- [19 Generative Models](../19_generative_models/README.md)
- [21 LLM Engineering](../21_llm_engineering/README.md): RLHF and DPO connections
