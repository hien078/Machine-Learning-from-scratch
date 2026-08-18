# Reinforcement Learning

## 0. Notation

| Symbol | Type | Meaning |
|--------|------|---------|
| $\mathcal{S}$ | Set | State space |
| $\mathcal{A}$ | Set | Action space |
| $P(s' \mid s, a)$ | Function | Transition probability from state $s$ to $s'$ given action $a$ |
| $R(s, a, s')$ | Function | Expected reward for transitioning $s \to s'$ via $a$ |
| $\gamma$ | Scalar | Discount factor, $\gamma \in [0, 1)$ |
| $\pi(a \mid s)$ | Function | Policy: probability of taking action $a$ in state $s$ |
| $V^\pi(s)$ | Function | State-value function under policy $\pi$ |
| $Q^\pi(s, a)$ | Function | Action-value function under policy $\pi$ |
| $G_t$ | Scalar (RV) | Return (cumulative discounted reward) from time $t$ |
| $\alpha$ | Scalar | Learning rate for value updates |
| $\epsilon$ | Scalar | Exploration rate for $\epsilon$-greedy policies |
| $\theta$ | Vector | Parameters of the policy in Policy Gradient methods |
| $N(a)$ | Integer | Count of how many times action $a$ has been taken |
| $\lambda$ | Scalar | Trace decay parameter for eligibility traces |
| $\delta_t$ | Scalar | TD error at time $t$ |
| $A(s, a)$ | Function | Advantage function: $Q(s, a) - V(s)$ |
| $\mathcal{H}(\pi)$ | Scalar | Entropy of policy $\pi$ |

## 1. WHY: Motivation and Problem Statement

Reinforcement Learning (RL) addresses the **sequential decision-making** problem. 

**Contrast with Supervised Learning:**
In supervised learning, an algorithm is provided with an explicit dataset of inputs and optimal output targets (labels). In RL, the agent is never explicitly told the optimal action. Instead, it must discover which actions yield the highest reward by interacting with an environment.

**Key Challenges in RL:**
1. **Delayed Rewards:** An action taken at time $t$ might not yield a positive reward until time $t+k$. This requires the agent to reason about long-term consequences.
2. **Credit Assignment Problem:** When a reward is finally received, determining which past actions were responsible for this outcome is difficult.
3. **Exploration vs. Exploitation Trade-off:** To maximize rewards, the agent must *exploit* its current knowledge of good actions. But to discover better actions, it must *explore* new, untried actions.

RL mathematically formalizes these concepts, providing algorithms to solve them from first principles.

## 2. WHAT: Markov Decision Processes

The environment in RL is typically formulated as a **Markov Decision Process (MDP)**, which is defined by a 5-tuple:

```math
\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma)
```

**The Markov Property:**
An environment satisfies the Markov property if the future state depends only on the current state and action, not on the sequence of events that preceded it.

```math
\mathbb{P}(S_{t+1} = s', R_{t+1} = r \mid S_t = s, A_t = a, S_{t-1}, A_{t-1}, \dots, S_0, A_0) = \mathbb{P}(S_{t+1} = s', R_{t+1} = r \mid S_t = s, A_t = a)
```

**Policy Definition:**
A policy defines the agent's behavior. 
- **Deterministic Policy:** $a = \pi(s)$, maps states to a single specific action.
- **Stochastic Policy:** $\pi(a \mid s) = \mathbb{P}(A_t = a \mid S_t = s)$, maps states to a probability distribution over actions.

**Objective Function:**
The goal of RL is to find an optimal policy $\pi^\ast$ that maximizes the expected cumulative discounted reward from any initial state.

## 2.5 Multi-Armed Bandits

The simplest setting of RL removes states entirely, focusing purely on actions and rewards. This is the **$k$-armed bandit problem**.
- We have $k$ actions (arms).
- Each action $a$ yields a reward drawn from a stationary distribution with mean $\mu_a$.
- The goal is to maximize total reward over $T$ time steps.

**Regret:**
We define the optimal mean reward as $\mu^\ast = \max_a \mu_a$. The performance is measured by **Regret** $R_T$, the expected loss from not picking the optimal action every time:

```math
R_T = T \cdot \mu^\ast - \sum_{t=1}^T \mu_{a_t}
```

**Action-Value Estimation:**
The agent estimates the value of each action using the sample average of observed rewards:

```math
Q_t(a) = \frac{\sum_{i=1}^{t-1} R_i \cdot \mathbb{I}(A_i = a)}{\sum_{i=1}^{t-1} \mathbb{I}(A_i = a)}
```

By the law of large numbers $Q_t(a)$ approaches $\mu_a$ as the arm is pulled more often, so an arm tried only a few times still carries a very uncertain estimate.

## 2.6 Exploration Strategies

To minimize regret, the agent must balance exploration and exploitation.

**$\epsilon$-greedy:**
- Exploit: Choose $\text{argmax}_a Q(a)$ with probability $1 - \epsilon$.
- Explore: Choose a random action with probability $\epsilon$.
While simple, $\epsilon$-greedy yields linear regret in the limit because it never stops exploring sub-optimal actions uniformly.

**Upper Confidence Bound (UCB1):**
Instead of random exploration, UCB uses optimism in the face of uncertainty. It relies on the **Hoeffding Inequality**, which bounds the probability that the sample mean $Q(a)$ deviates from the true mean $\mu_a$ by more than $U$:

```math
\mathbb{P}(\mu_a > Q(a) + U) \leq e^{-2 N(a) U^2}
```

Setting this probability to $p = t^{-4}$ and solving for $U$ yields $U = \sqrt{\frac{2 \ln t}{N(a)}}$. 
**Result (UCB1 Rule):**

```math
a_t = \text{argmax}_a \left[ Q_t(a) + c \sqrt{\frac{\ln t}{N_t(a)}} \right]
```

where $c$ controls the degree of exploration. UCB achieves logarithmic regret.

**Thompson Sampling:**
A Bayesian approach. For Bernoulli bandits (rewards $\in \lbrace0, 1\rbrace$), we model the probability of success for each arm $a$ as a Beta distribution: $\theta_a \sim \text{Beta}(\alpha_a, \beta_a)$.
1. Sample $\hat{\theta}_a \sim \text{Beta}(\alpha_a, \beta_a)$ for all $a$.
2. Choose $a_t = \text{argmax}_a \hat{\theta}_a$.
3. Update posterior: if reward is 1, $\alpha_a \gets \alpha_a + 1$; if 0, $\beta_a \gets \beta_a + 1$.

**Boltzmann (Softmax) Exploration:**
Select actions with probability proportional to exponentiated values:

```math
\mathbb{P}(a_t = a) = \frac{e^{Q(a)/\tau}}{\sum_{a'} e^{Q(a')/\tau}}
```

where $\tau$ is a temperature parameter controlling randomness.

## 3. HOW: Returns and Value Functions

**The Return ($G_t$):**
The total discounted reward from time step $t$ is:

```math
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}
```

The discount factor $\gamma$ ensures the sum is finite and weights near-term rewards higher than distant ones.

**State-Value Function $(V^\pi(s))$:**
The expected return starting from state $s$ and following policy $\pi$:

```math
V^\pi(s) = \mathbb{E}_\pi [ G_t \mid S_t = s ]
```

**Action-Value Function $(Q^\pi(s, a))$:**
The expected return starting from state $s$, taking action $a$, and then following policy $\pi$:

```math
Q^\pi(s, a) = \mathbb{E}_\pi [ G_t \mid S_t = s, A_t = a ]
```

Fixing the first action instead of drawing it from $\pi$ is what makes $Q^\pi$ the object to compare candidate actions in a state, while $V^\pi$ averages over them.

## 4. HOW: Bellman Equations

The Bellman equations express the recursive relationship between the value of a state and the values of its successor states.

### Derivation of the Bellman Expectation Equation for $V^\pi(s)$

1. **Definition:**

   $$V^\pi(s) = \mathbb{E}_ \pi [ G_t \mid S_t = s ]$$

2. **Expand the Return (Recursive substitution):**

   $$V^\pi(s) = \mathbb{E}_ \pi [ R_{t+1} + \gamma G_{t+1} \mid S_t = s ]$$

3. **Linearity of Expectation:**

   $$V^\pi(s) = \mathbb{E}_ \pi [ R_{t+1} \mid S_t = s ] + \gamma \mathbb{E}_ \pi [ G_{t+1} \mid S_t = s ]$$

4. **Law of Total Expectation (condition on $A_t$ and $S_{t+1}$):**

   $$V^\pi(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \sum_{s' \in \mathcal{S}} P(s' \mid s, a) \left[ R(s, a, s') + \gamma \mathbb{E}_ \pi [ G_{t+1} \mid S_{t+1} = s' ] \right]$$

5. **Substitute the definition of $V^\pi(s')$:**

   $$V^\pi(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \sum_{s' \in \mathcal{S}} P(s' \mid s, a) \left[ R(s, a, s') + \gamma V^\pi(s') \right]$$

**Result (Bellman Expectation Equation):**

```math
V^\pi(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) Q^\pi(s, a)
```

where the action-value is

```math
Q^\pi(s, a) = \sum_{s' \in \mathcal{S}} P(s' \mid s, a) [ R(s, a, s') + \gamma V^\pi(s') ]
```

The two identities together unroll $V^\pi$ by one step: average over the actions the policy takes, then over the states the environment moves to.

### Bellman Optimality Equations

For the optimal policy $\pi^\ast$, the value functions satisfy:

```math
V^\ast(s) = \max_{a} Q^\ast(s, a) = \max_{a} \sum_{s'} P(s' \mid s, a) [ R(s, a, s') + \gamma V^\ast(s') ]
```

```math
Q^\ast(s, a) = \sum_{s'} P(s' \mid s, a) \left[ R(s, a, s') + \gamma \max_{a'} Q^\ast(s', a') \right]
```

**Convergence (Contraction Mapping Theorem):**
The Bellman optimality operator $B$ applied to a value function $V$ is defined as

```math
(BV)(s) = \max_a \sum_{s'} P(s' \mid s, a) [R + \gamma V(s')]
```

It can be shown that $B$ is a $\gamma$-contraction in the max norm:

```math
\Vert BV_1 - BV_2\Vert_\infty \leq \gamma \Vert V_1 - V_2\Vert_\infty
```

By the Banach fixed-point theorem, repeated application of $B$ converges to a unique fixed point $V^\ast$.

## 5. HOW: Dynamic Programming

When the MDP model ($P$ and $R$) is completely known, we can compute optimal policies using Dynamic Programming (DP).

**Value Iteration:**
Repeatedly apply the Bellman optimality operator to find $V^\ast$.
1. Initialize $V_0(s) = 0$ for all $s$.
2. Update: $V_{k+1}(s) = \max_{a} \sum_{s'} P(s' \mid s, a) [ R(s, a, s') + \gamma V_k(s') ]$
3. Stop when $\max_s |V_{k+1}(s) - V_k(s)| < \theta$.
4. Extract deterministic policy: $\pi^\ast(s) = \text{argmax}_ a \sum_{s'} P(s' \mid s, a) [ R(s, a, s') + \gamma V^\ast(s') ]$.

**Policy Iteration:**
Alternates between Policy Evaluation (finding $V^\pi$ for a fixed $\pi$) and Policy Improvement (updating $\pi$ greedily with respect to $V^\pi$). Guaranteed to converge in finite steps for finite MDPs.

## 5.5 Monte Carlo Methods

When the MDP model is unknown but we can sample episodes, **Monte Carlo (MC)** methods learn value functions directly from episodes of experience.

**MC Prediction:**
To evaluate a policy $\pi$, we estimate $V^\pi(s)$ by averaging the returns observed after visiting state $s$.
- **First-visit MC:** Averages returns only for the *first* time $s$ is visited in an episode. Unbiased estimator of $V^\pi(s)$.
- **Every-visit MC:** Averages returns for *every* time $s$ is visited. Biased but consistent, often has lower variance.

**MC Control:**
MC can optimize a policy using Generalized Policy Iteration. Since we lack the transition model, we must estimate $Q(s, a)$ instead of $V(s)$. To ensure exploration, MC Control often uses **Exploring Starts** (every state-action pair has a non-zero probability of starting an episode) or an $\epsilon$-greedy policy.

**MC vs. TD:**
- **Bias:** MC is unbiased. TD has bias (due to bootstrapping from initial estimates).
- **Variance:** MC has high variance (depends on a full sequence of random actions, transitions, and rewards). TD has lower variance (only depends on one random transition).

**Off-Policy MC via Importance Sampling:**
To learn a target policy $\pi$ from behavior policy $b$, we weight returns by the importance sampling ratio (the relative probability of the trajectory):

```math
\rho_{t:T-1} = \prod_{k=t}^{T-1} \frac{\pi(A_k \mid S_k)}{b(A_k \mid S_k)}
```

The value update becomes $V(S_t) \gets V(S_t) + \alpha (\rho_{t:T-1} G_t - V(S_t))$.

## 5.6 n-step TD & TD($\lambda$)

TD (1-step) and MC (infinite-step) are endpoints of a spectrum. **$n$-step TD** bridges the gap.

**$n$-step Return:**

```math
G_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k R_{t+k+1} + \gamma^n V(S_{t+n})
```

The update rule is $V(S_t) \gets V(S_t) + \alpha (G_t^{(n)} - V(S_t))$.

**TD($\lambda$) and Eligibility Traces:**
Instead of a single $n$-step return, TD($\lambda$) uses an exponentially weighted average of all $n$-step returns, called the **$\lambda$-return** (the forward view):

```math
G_t^\lambda = (1 - \lambda) \sum_{n=1}^\infty \lambda^{n-1} G_t^{(n)}
```

The **backward view** provides an online, incremental mechanism to achieve the same result using **eligibility traces** $E(s)$. 
When a state is visited, its trace is incremented; otherwise, it decays by $\gamma \lambda$:

```math
E_t(s) = \gamma \lambda E_{t-1}(s) + \mathbb{I}(S_t = s)
```

On each step, all states are updated by the TD error $\delta_t$ scaled by their trace:

```math
V(s) \gets V(s) + \alpha \delta_t E_t(s)
```

**Unifying Spectrum:**
- $\lambda = 0$: TD(0), purely bootstraps.
- $\lambda = 1$: Monte Carlo, purely uses actual returns.

## 6. HOW: Model-Free Methods (Tabular)

When the environment dynamics ($P$, $R$) are unknown, the agent must learn directly from experience (sampled trajectories).

**Temporal Difference (TD) Learning:**
TD methods bootstrap: they update estimates based on other learned estimates, without waiting for the episode to end.

**SARSA (On-Policy TD Control):**
Learns the value of the policy being followed. The agent experiences $(S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1})$.
Update rule:

```math
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ \underbrace{R_{t+1} + \gamma Q(S_{t+1}, A_{t+1})}_{\text{TD Target}} - Q(S_t, A_t) \right]
```

**Q-Learning (Off-Policy TD Control):**
Learns the optimal policy regardless of the agent's actual exploratory actions.
Update rule:

```math
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ \underbrace{R_{t+1} + \gamma \max_{a} Q(S_{t+1}, a)}_{\text{TD Target}} - Q(S_t, A_t) \right]
```

**Exploration-Exploitation ($\epsilon$-greedy):**
To ensure all state-action pairs are visited infinitely often (a requirement for convergence), Q-learning typically uses an $\epsilon$-greedy policy during training:
- With probability $1 - \epsilon$, choose $a = \text{argmax}_a Q(s, a)$ (Exploit).
- With probability $\epsilon$, choose a random action $a \in \mathcal{A}$ (Explore).

**Convergence:** Q-Learning converges to $Q^\ast$ with probability 1 if all $(s,a)$ pairs are visited infinitely often and the learning rate $\alpha$ decays appropriately ($\sum \alpha_t = \infty, \sum \alpha_t^2 < \infty$).

## 7. HOW: Policy Gradients

Instead of learning value functions and deriving a policy, Policy Gradient methods parameterize the policy $\pi_\theta(a \mid s)$ directly and optimize the parameters $\theta$ via gradient ascent on the expected return $J(\theta) = \mathbb E_{\pi_\theta}[G_0]$.

### Derivation of the Policy Gradient Theorem
1. **Objective:** $\nabla_\theta J(\theta) = \nabla_\theta \mathbb E_{\tau \sim \pi_\theta}[R(\tau)] = \nabla_\theta \int P(\tau; \theta) R(\tau) d\tau$
2. **Log-Derivative Trick:** $\nabla_\theta P(\tau; \theta) = P(\tau; \theta) \nabla_\theta \log P(\tau; \theta)$
3. **Substitute:** $\nabla_\theta J(\theta) = \int P(\tau; \theta) \nabla_\theta \log P(\tau; \theta) R(\tau) d\tau = \mathbb E_{\tau \sim \pi_\theta} [ \nabla_\theta \log P(\tau; \theta) R(\tau) ]$
4. **Trajectory Probability Expansion:** $P(\tau; \theta) = P(s_0) \prod_{t=0}^T \pi_\theta(a_t \mid s_t) P(s_{t+1} \mid s_t, a_t)$
5. **Log and Derivative:** The dynamics $P$ do not depend on $\theta$, so their gradients are zero.

   $$\nabla_\theta \log P(\tau; \theta) = \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t \mid s_t)$$

**Result (Policy Gradient Theorem):**

```math
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t \mid s_t) G_t \right]
```

**REINFORCE Algorithm:** A Monte Carlo policy gradient algorithm that samples trajectories and updates $\theta \leftarrow \theta + \alpha G_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)$.

**Variance Reduction & Advantage:**
The variance of Monte Carlo returns $G_t$ is high. We can subtract a baseline $b(s)$ without changing the expected gradient. Typically, $b(s) = V(s)$.

The **Advantage function** is $A(s, a) = Q(s, a) - V(s)$.

Using $A(s,a)$ leads to **Actor-Critic** architectures: an Actor updates the policy, and a Critic estimates the value function to compute the advantage.

## 7.5 PPO & Trust Region Methods

Standard policy gradients suffer from unstable training: a large update to $\theta$ can collapse policy performance. 

**TRPO (Trust Region Policy Optimization):**
TRPO constrains the policy update so the new policy doesn't deviate too far from the old one, measured by KL divergence:

```math
\max_\theta \mathbb{E} \left[ \frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)} A(s,a) \right] \quad \text{s.t.} \quad \text{KL}(\pi_{\theta_{old}} || \pi_\theta) \leq \delta
```

**PPO (Proximal Policy Optimization):**
PPO approximates TRPO's constraint using a clipped surrogate objective, making it simpler and faster.
Let the probability ratio be

```math
r_t(\theta) = \frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)}
```

**Result (PPO Clipped Objective):**

```math
L^{CLIP}(\theta) = \mathbb{E} \left[ \min(r_t(\theta)A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)A_t) \right]
```

This penalizes changes to the policy that move $r_t(\theta)$ away from 1 if that change improves the objective, preventing excessively large updates.

**Generalized Advantage Estimation (GAE):**
To compute the advantage $A_t$ with a good bias-variance tradeoff, GAE uses an exponentially-weighted sum of TD errors $\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$:

```math
A_t^{GAE(\gamma,\lambda)} = \sum_{l=0}^\infty (\gamma\lambda)^l \delta_{t+l}
```

The parameter $\lambda$ selects the tradeoff: at $\lambda = 0$ only the one-step TD error survives (low variance, high bias), while $\lambda \to 1$ recovers the full Monte Carlo advantage.

## 7.6 Actor-Critic Architecture

Policy gradient methods often use an Actor-Critic architecture to reduce variance.
- **Actor:** Parameterized by $\theta$, it defines the policy $\pi_\theta(a|s)$.
- **Critic:** Parameterized by $w$, it estimates the value function $V_w(s)$.

**Loss Functions:**
- Actor Loss (Policy Gradient with Advantage): $L_{actor} = -\mathbb{E}[\log \pi_\theta(a|s) \cdot A(s,a)]$
- Critic Loss (MSE against Returns): $L_{critic} = \mathbb{E}[(V_w(s) - G_t)^2]$

**A2C vs. A3C:**
- **A3C (Asynchronous Advantage Actor-Critic):** Multiple worker agents interact with their own environments in parallel and asynchronously update a global network.
- **A2C (Advantage Actor-Critic):** A synchronous, deterministic version of A3C. It waits for all workers to finish their segments, calculates gradients, and averages them to update the network synchronously.

## 8. Failure Cases

1. **The Deadly Triad:** When combining (1) Function approximation (e.g., neural networks), (2) Bootstrapping (e.g., TD learning), and (3) Off-policy training (e.g., Q-learning), training frequently diverges and becomes unstable.
2. **Sparse Rewards:** If the environment only provides a reward at the very end (e.g., winning/losing a game of chess), random exploration will rarely discover a successful trajectory, causing learning to stall entirely.
3. **Non-stationarity:** If the environment dynamics $P(s' \mid s, a)$ change over time (or in multi-agent RL where opponents change policies), past experience becomes invalid, breaking the Markov assumption and destabilizing learned Q-values.
4. **Catastrophic Forgetting:** When exploring a new part of the state space, neural-network-based agents (like DQN) often aggressively update weights, "forgetting" how to behave in previously mastered regions. Experience replay is required to mitigate this.
5. **Exploration Failure:** Simple $\epsilon$-greedy exploration is extremely inefficient in high-dimensional state spaces. The agent acts randomly and fails to purposefully explore promising but uncertain regions.
6. **Reward Hacking / Misspecification:** If the reward function does not perfectly align with human intent, the agent will find unintended, trivial, or destructive ways to maximize the metric.
7. **Overestimation Bias:** Q-learning's $\max_a Q(s',a)$ operator leads to systematic overestimation of action values due to noise. (Mitigated by Double DQN).
8. **Sample Inefficiency:** Model-free methods (like PPO or DQN) require massive amounts of environment interactions to learn, often millions of frames, which is impractical in real-world robotics.

## 9. Deep RL Advances

To stabilize and improve deep Q-learning, several extensions were developed on top of DQN.

**Double DQN:**
Fixes the overestimation bias of standard DQN. It decouples action selection from action evaluation by using the online network to select the action and the target network to evaluate it:

```math
Y_t = R_{t+1} + \gamma Q_{target}(S_{t+1}, \text{argmax}_a Q_{online}(S_{t+1}, a))
```

**Dueling DQN:**
Splits the Q-network into two streams: one estimates state-value $V(s)$ and the other estimates advantage $A(s,a)$. They are combined at the final layer:

```math
Q(s,a) = V(s) + A(s,a) - \frac{1}{|\mathcal{A}|} \sum_{a'} A(s,a')
```

Subtracting the mean advantage ensures identifiability of $V(s)$.

**Prioritized Experience Replay (PER):**
Instead of sampling transitions uniformly from the replay buffer, PER samples transitions with high expected learning progress.
The priority is typically proportional to the absolute TD error: $p_i = |\delta_i| + \epsilon$. 
To correct the bias introduced by non-uniform sampling, importance sampling weights are applied to the loss.

## 10. Continuous Action Spaces

For continuous control (e.g., robotics), discrete action spaces are impractical.

**Gaussian Policy Parameterization:**
Instead of outputting probabilities for discrete actions, the neural network outputs the parameters of a continuous distribution (typically Gaussian) from which actions are sampled:

```math
\pi_\theta(a \mid s) = \mathcal{N}(\mu_\theta(s), \sigma_\theta(s))
```

**Deterministic Policy Gradient (DPG):**
Instead of a stochastic policy, DPG optimizes a deterministic policy $\mu_\theta(s)$. 
**Result (DPG Theorem):** (Silver, 2014)

```math
\nabla_\theta J(\theta) = \mathbb{E}_{s \sim \rho^\mu} [ \nabla_\theta \mu_\theta(s) \nabla_a Q^\mu(s, a) |_{a=\mu_\theta(s)} ]
```

**DDPG (Deep Deterministic Policy Gradient):**
An off-policy actor-critic algorithm for continuous spaces that uses the DPG theorem. It uses target networks (for both actor and critic) and adds exploratory noise (e.g., Ornstein-Uhlenbeck noise) to the deterministic action.

**Soft Actor-Critic (SAC):**
An off-policy maximum entropy RL algorithm. It augments the standard RL objective to maximize both expected return and policy entropy:

```math
J(\pi) = \mathbb{E}_{\pi} \left[ \sum_{t} \gamma^t \left( R_t + \alpha \mathcal{H}(\pi(\cdot \mid S_t)) \right) \right]
```

where $\alpha$ is a temperature parameter. SAC is highly sample-efficient and robust due to its entropy maximization, which encourages exploration.

## 11. Model-Based RL

Model-based RL attempts to learn the transition dynamics $P(s' \mid s, a)$ and reward function $R(s,a)$ directly to form a model of the environment.

**Dyna-Q:**
An architecture that integrates learning and planning. 
1. Interact with the environment to collect real experience.
2. Update the Q-values (direct RL).
3. Update the learned model.
4. **Planning:** Sample states and actions from the model to generate simulated experience, and perform additional Q-learning updates.

**World Models:**
Advanced MBRL systems (like Dreamer) learn latent-space representations of the environment (a "world model"). The agent can then hallucinate rollouts entirely within the latent model, allowing it to learn behaviors without costly environment interactions.

**Planning vs. Learning Tradeoff:**
Model-based methods are highly sample-efficient but suffer from model bias (compounding errors when the learned model is inaccurate). Model-free methods are sample-inefficient but directly optimize the policy without compounding model errors.

## 12. RLHF (Reinforcement Learning from Human Feedback)

When the desired reward function is too complex to specify mathematically (e.g., "be helpful and harmless" for LLMs), RLHF learns it from human preferences.

**Reward Modeling:**
1. Collect pairs of agent trajectories (or text completions) $(y_1, y_2)$ given context $x$.
2. Humans label which one is preferred: $y_1 \succ y_2$.
3. Train a reward model $r_\phi(x, y)$ using the **Bradley-Terry preference model**:

   $$P(y_1 \succ y_2) = \frac{\exp(r_\phi(x, y_1))}{\exp(r_\phi(x, y_1)) + \exp(r_\phi(x, y_2))}$$

**KL-Constrained Policy Optimization:**
Once $r_\phi$ is trained, use PPO to optimize the policy $\pi_\theta$, adding a KL divergence penalty to prevent the policy from diverging too far from the original reference model $\pi_{ref}$ (which prevents "reward hacking" the learned model):

```math
\max_\theta \mathbb{E}_{x, y \sim \pi_\theta} \left[ r_\phi(x, y) \right] - \beta \cdot \text{KL}(\pi_\theta(\cdot|x) || \pi_{ref}(\cdot|x))
```

*(Note: See [Module 21](../21_llm_engineering/theory.md) for Direct Preference Optimization (DPO), which bypasses the explicit reward modeling step.)*

## 13. Multi-Agent RL (MARL)

When multiple agents interact in the same environment.

**Settings:**
- **Cooperative:** Agents share a common reward function.
- **Competitive:** Zero-sum games (e.g., chess) where one's gain is another's loss.
- **Mixed:** Elements of both (e.g., self-driving cars navigating traffic).

**Nash Equilibrium:**
A joint policy configuration where no agent can improve its expected return by unilaterally changing its own policy. MARL algorithms often seek to converge to a Nash equilibrium rather than a single optimal policy.

**Non-stationarity Challenge:**
From the perspective of any single agent, the environment includes the other agents. As they learn and change their policies, the environment dynamics appear non-stationary, breaking the standard Markov assumption.

**Self-Play:**
In competitive settings, an agent can learn robust strategies by playing against past versions of itself, naturally establishing a curriculum of increasing difficulty.

## 14. Connections

- **[02 Gradient Descent](../02_gradient_descent/theory.md):** The bedrock of Policy Gradient optimization and neural value function approximation.
- **[13 Neural Networks](../13_neural_networks/theory.md):** Used as non-linear function approximators for $Q(s, a; w)$ in DQN and $\pi(a \mid s; \theta)$ in Policy Gradients, replacing tabular representations to handle continuous state spaces.
- **[Optimization Methods Compared](../synthesis/optimization_methods_compared.md):** RL can be viewed as stochastic optimization over policy spaces, often requiring specialized optimizers like Adam to handle noisy, non-stationary TD targets.

## 15. References

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
- Mnih, V., et al. (2015). "Human-level control through deep reinforcement learning". *Nature*, 518, 529–533.
- Williams, R. J. (1992). "Simple statistical gradient-following algorithms for connectionist reinforcement learning". *Machine Learning*, 8(3-4), 229–256. (REINFORCE)
- Schulman, J., et al. (2017). "Proximal Policy Optimization Algorithms". *arXiv preprint arXiv:1707.06347*.
- Silver, D., et al. (2014). "Deterministic Policy Gradient Algorithms". *ICML*.
- Haarnoja, T., et al. (2018). "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor". *ICML*.
- Christiano, P. F., et al. (2017). "Deep Reinforcement Learning from Human Preferences". *NeurIPS*.
- Sutton, R. S., et al. (1999). "Policy Gradient Methods for Reinforcement Learning with Function Approximation". *NIPS*.
- Auer, P., et al. (2002). "Finite-time Analysis of the Multiarmed Bandit Problem". *Machine Learning*, 47, 235-256.
