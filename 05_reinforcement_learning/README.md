# Reinforcement Learning

An agent learns to make decisions by interacting with an environment: taking actions, observing states, and receiving rewards. The goal is to learn a **policy π(a|s)** that maximizes cumulative reward.

---

## Prerequisites

### Math (from [math_for_ai_roadmap.md](../00_foundations/01_math_essentials/math_for_ai_roadmap.md))
- **(3) Probability & Statistics:** expectation, conditional probability, sampling, importance sampling.
- **(4) Optimization:** SGD, policy gradient, constrained optimization (for TRPO/PPO).
- **(7) Discrete Math & Graphs:** dynamic programming, Bellman equations, tree search.
- **(B) RL Math** *(extra, beyond the main pillars)*: MDP, Bellman equation, policy gradient theorem, TD learning, GAE.

### Code
- Python, NumPy.
- Gymnasium (OpenAI Gym successor) for environments.
- PyTorch (for deep RL: DQN, PPO, A2C).

### Recommended prior modules
- `03_deep_learning/01_multi_layer_perceptron/` (neural network fundamentals for function approximation).

---

## Subprojects

### [`00_markov_decision_process/`](00_markov_decision_process/)
The mathematical framework that every RL algorithm solves a variant of.

| Topic | What to implement | Key math |
|---|---|---|
| MDP definition | (S, A, P, R, γ) | State / action / transition / reward / discount |
| Bellman equations | Value iteration on GridWorld | V*(s) = max_a [R(s,a) + γ Σ P(s'|s,a) V*(s')] |
| Policy iteration | Alternate eval + improve | Convergence to π* |
| Partially observable (POMDP) | Belief state | Bayesian filtering |

### [`01_q_learning/`](01_q_learning/) & [`02_dqn/`](02_dqn/) & [`03_double_dqn/`](03_double_dqn/)
Value-based methods: Q-Learning, Deep Q-Network (DQN), Double DQN.

| Topic | What to implement | Key math |
|---|---|---|
| Tabular Q-Learning | Q-table update on GridWorld | Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') − Q(s,a)] |
| ε-greedy exploration | Exploration vs exploitation | Decay schedule |
| DQN | Neural network Q-function | Experience replay, target network |
| Double DQN | Reduce overestimation bias | Separate selection and evaluation |

### [`04_policy_gradient/`](04_policy_gradient/)
REINFORCE — the on-policy starting point.

| Topic | What to implement | Key math |
|---|---|---|
| REINFORCE | Monte Carlo policy gradient | ∇J = E[∇ log π(a|s) · Gₜ] |
| Baseline | Variance reduction | b(s) ≈ V(s), advantage Â = G − b |
| GAE | Generalized Advantage Estimation | Âₜ = Σ (γλ)ˡ δₜ₊ₗ |

### [`05_a2c/`](05_a2c/) & [`06_ddpg/`](06_ddpg/)
Actor-Critic family: A2C (on-policy, discrete) and DDPG (off-policy, continuous).

| Topic | What to implement | Key math |
|---|---|---|
| A2C | Shared actor + critic network | Actor: ∇ log π · Â; Critic: minimize TD error |
| DDPG | Continuous action space | Deterministic policy gradient, soft update |
| TD(0) / TD(λ) | Temporal difference learning | V(s) ← V(s) + α[r + γV(s') − V(s)] |

### [`07_ppo/`](07_ppo/)
Proximal Policy Optimization — the on-policy workhorse of modern RL.

| Topic | What to implement | Key math |
|---|---|---|
| Clipped objective | Trust region via clipping | min(rₜ Âₜ, clip(rₜ, 1−ε, 1+ε) Âₜ) |
| KL penalty variant | Adaptive KL coefficient | β · KL(π_old ‖ π_new) |
| GAE integration | Advantage estimation | Âₜ via Σ (γλ)ˡ δₜ₊ₗ |
| Mini-batch updates | Multiple epochs per rollout | SGD on saved trajectories |

### [`08_sac/`](08_sac/)
Soft Actor-Critic — off-policy maximum-entropy RL for continuous control.

| Topic | What to implement | Key math |
|---|---|---|
| Entropy-regularized RL | Reward + temperature · entropy | J(π) = E[Σ (rₜ + α H(π(·|sₜ)))] |
| Twin Q-networks | Reduce overestimation | min(Q₁, Q₂) for target |
| Reparameterized policy | Gradient through stochastic sampling | a = μ + σ ⊙ ε, then tanh |
| Automatic α tuning | Learn entropy temperature | Target entropy heuristic |

---

## Learning Objectives

After completing this module, you should be able to:

- [ ] Implement tabular Q-Learning from scratch and solve a GridWorld / FrozenLake.
- [ ] Derive the policy gradient theorem from first principles.
- [ ] Implement REINFORCE with baseline on CartPole.
- [ ] Implement PPO (clipped objective) and train on a Gymnasium environment.
- [ ] Explain the bias-variance tradeoff in TD(0) vs Monte Carlo returns.
- [ ] Explain why experience replay and target networks stabilize DQN.

---

## Key References

- Sutton & Barto — *Reinforcement Learning: An Introduction*, 2nd ed. (free PDF at [incompleteideas.net](http://incompleteideas.net/book/the-book.html)).
- Schulman et al. (2017) — *Proximal Policy Optimization Algorithms*.
- Mnih et al. (2015) — *Human-level control through deep reinforcement learning* (DQN).
- Lillicrap et al. (2016) — *Continuous control with deep reinforcement learning* (DDPG).
- OpenAI Spinning Up — [spinningup.openai.com](https://spinningup.openai.com/).
- CleanRL — [github.com/vwxyzjn/cleanrl](https://github.com/vwxyzjn/cleanrl) (single-file implementations).

---

## Subproject Layout

Each subproject should follow:
```
algorithm_name/
├── data/           # Saved trajectories, replay buffers (gitignored if large)
├── notebooks/      # Reward curves, environment rendering, analysis
├── src/            # From-scratch implementation + PyTorch version
├── tests/          # Unit tests (Bellman update, gradient check)
└── reports/        # Training curves, hyperparameter comparisons
```
