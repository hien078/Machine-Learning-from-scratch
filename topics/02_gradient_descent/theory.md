# Gradient Descent — Theory

> **Purpose:** Pure theory, definitions, and conceptual understanding.
> Read this first, then open `first_principles.ipynb` for computation and experiments.

## Prerequisites

- [01 Linear Regression](../01_linear_regression/README.md) — uses GD as an alternative to the normal equation
- [Foundations: Calculus & Optimization](https://github.com/hien078/applied-mathematics-foundation)

---

## 0. Notation

| Symbol | Type | Meaning |
|---|---|---|
| $\theta$ | vector $\in \mathbb{R}^d$ | parameters (weights) to optimize |
| $\theta^*$ | vector $\in \mathbb{R}^d$ | a minimizer of $L$ |
| $L(\theta)$ | scalar function | loss (objective) function; $L : \mathbb{R}^d \to \mathbb{R}$ |
| $\nabla L(\theta)$ | vector $\in \mathbb{R}^d$ | gradient of $L$ at $\theta$ |
| $\nabla^2 L(\theta)$ | matrix $\in \mathbb{R}^{d \times d}$ | Hessian of $L$ at $\theta$ |
| $\alpha$ (or $\eta$) | scalar $> 0$ | learning rate (step size) |
| $M$ (or $L_s$) | scalar $> 0$ | Lipschitz smoothness constant |
| $\mu$ | scalar $\ge 0$ | strong convexity constant |
| $\kappa$ | scalar $\ge 1$ | condition number; $\kappa = M / \mu$ |
| $v_t$ | vector $\in \mathbb{R}^d$ | velocity / momentum accumulator at step $t$ |
| $\beta, \beta_1, \beta_2$ | scalars $\in [0, 1)$ | momentum / exponential decay coefficients |
| $m_t$ | vector $\in \mathbb{R}^d$ | first moment estimate (Adam) |
| $s_t$ | vector $\in \mathbb{R}^d$ | second moment estimate (Adam) |
| $n$ | scalar $\in \mathbb{N}$ | number of training samples |
| $B$ | scalar $\in \mathbb{N}$ | mini-batch size |
| $k$ (or $t$) | scalar $\in \mathbb{N}$ | iteration counter |

**Conventions.**

- All vectors are column vectors.
- Subscripts $t$ or $k$ index iterations; subscripts $i$ index data points.
- $\|\cdot\|$ denotes the Euclidean ($\ell^2$) norm unless stated otherwise.

---

## 1. WHY — Optimization Without a Closed Form

Many ML objectives have no useful closed-form minimizer. Linear regression with the normal equation is an exception; logistic regression, neural networks, and most modern models are not. We need a general iterative method that:

- Uses only **local information** (gradient at the current point),
- **Scales** to millions of parameters,
- Works for **any differentiable** loss function.

Gradient descent is the foundational algorithm for all of the above.

---

## 2. WHAT — The Gradient Descent Update

### 2.1 Definition (gradient descent)

Given a differentiable objective $L : \mathbb{R}^d \to \mathbb{R}$, the **gradient descent** update starting from $\theta_0$ is:

$$\theta_{t+1} = \theta_t - \alpha \nabla L(\theta_t)$$

The negative gradient $-\nabla L(\theta_t)$ is the direction of steepest descent under the Euclidean norm. The scalar $\alpha > 0$ controls the step size.

### 2.2 Intuition — local linear approximation

At any point $\theta_t$, the first-order Taylor expansion is:

$$L(\theta) \approx L(\theta_t) + \nabla L(\theta_t)^\top (\theta - \theta_t)$$

Moving along $-\nabla L(\theta_t)$ decreases this linear approximation the fastest per unit step. The learning rate $\alpha$ limits how far we trust this local approximation.

---

## 3. HOW — Convergence Theory

### 3.1 Smoothness assumption

> **Definition.** $L$ is **$M$-smooth** if $\nabla L$ is $M$-Lipschitz continuous:
>
> $$\|\nabla L(\theta) - \nabla L(\phi)\| \le M \|\theta - \phi\| \quad \forall\, \theta, \phi$$

Equivalently, for twice-differentiable $L$: $\|\nabla^2 L(\theta)\| \le M$ for all $\theta$ (the largest eigenvalue of the Hessian is at most $M$).

### 3.2 Descent lemma

> **Lemma.** If $L$ is $M$-smooth, then for any $\theta, \phi$:
>
> $$L(\phi) \le L(\theta) + \nabla L(\theta)^\top (\phi - \theta) + \frac{M}{2}\|\phi - \theta\|^2$$

Setting $\phi = \theta - \alpha \nabla L(\theta)$ (the GD step):

$$L(\theta_{t+1}) \le L(\theta_t) - \alpha\|\nabla L(\theta_t)\|^2 + \frac{M\alpha^2}{2}\|\nabla L(\theta_t)\|^2$$

$$= L(\theta_t) - \alpha\left(1 - \frac{M\alpha}{2}\right)\|\nabla L(\theta_t)\|^2$$

For guaranteed decrease we need $1 - \frac{M\alpha}{2} > 0$, i.e.:

$$\boxed{\alpha < \frac{2}{M}}$$

**Result:** Gradient descent guarantees a decrease in $L$ at every step when $\alpha < 2/M$. The optimal fixed step size is $\alpha = 1/M$, which gives the largest per-step decrease.

### 3.3 Convergence rate — convex case

> **Theorem (sublinear rate).** If $L$ is convex and $M$-smooth, and $\alpha = 1/M$, then after $T$ steps:
>
> $$L(\theta_T) - L(\theta^*) \le \frac{M\|\theta_0 - \theta^*\|^2}{2T}$$

**Proof sketch.** From the descent lemma with $\alpha = 1/M$: each step gives $L(\theta_{t+1}) \le L(\theta_t) - \frac{1}{2M}\|\nabla L(\theta_t)\|^2$. Summing telescopically and using convexity ($L(\theta_t) - L(\theta^*) \le \nabla L(\theta_t)^\top(\theta_t - \theta^*)$), the bound follows by averaging.

**Result:** Convex, smooth GD converges at rate $O(1/T)$. To halve the error, double the iterations.

### 3.4 Convergence rate — strongly convex case

> **Definition.** $L$ is **$\mu$-strongly convex** ($\mu > 0$) if:
>
> $$L(\phi) \ge L(\theta) + \nabla L(\theta)^\top(\phi - \theta) + \frac{\mu}{2}\|\phi - \theta\|^2 \quad \forall\, \theta, \phi$$

> **Theorem (linear rate).** If $L$ is $\mu$-strongly convex and $M$-smooth, with $\alpha = 1/M$:
>
> $$\|\theta_T - \theta^*\|^2 \le \left(1 - \frac{\mu}{M}\right)^T \|\theta_0 - \theta^*\|^2$$

**Result:** Strong convexity yields a **linear (exponential) convergence rate**. The convergence factor is $1 - 1/\kappa$ where $\kappa = M/\mu$ is the condition number. Ill-conditioned problems ($\kappa \gg 1$) converge slowly.

### 3.5 Quadratic example

For the quadratic $L(\theta) = \frac{1}{2}\theta^\top A\theta - b^\top\theta$ with symmetric positive definite $A$:

- $\nabla L(\theta) = A\theta - b$
- $M = \lambda_{\max}(A)$, $\mu = \lambda_{\min}(A)$, $\kappa = \lambda_{\max}/\lambda_{\min}$
- Convergence requires $0 < \alpha < 2/\lambda_{\max}$
- At $\alpha = 2/\lambda_{\max}$, the component along the top eigenvector oscillates without convergence

---

## 4. Stochastic Variants

### 4.1 Full-batch vs stochastic vs mini-batch

| Variant | Gradient estimate | Cost per step | Convergence |
|---|---|---|---|
| **Full-batch GD** | $\nabla L(\theta) = \frac{1}{n}\sum_{i=1}^n \nabla \ell_i(\theta)$ | $O(nd)$ | Deterministic, smooth |
| **SGD** | $\nabla \ell_i(\theta)$ for random $i$ | $O(d)$ | Noisy, needs decaying $\alpha$ |
| **Mini-batch SGD** | $\frac{1}{\vert B\vert}\sum_{i \in B} \nabla \ell_i(\theta)$ | $O(Bd)$ | Reduced variance vs SGD |

### 4.2 SGD convergence

For convex $L$ with bounded gradient variance $\sigma^2$, SGD with $\alpha_t = c/\sqrt{t}$ achieves:

$$\mathbb{E}[L(\bar{\theta}_T)] - L(\theta^*) = O\!\left(\frac{1}{\sqrt{T}}\right)$$

This is slower than full-batch $O(1/T)$, but each iteration costs $O(d)$ instead of $O(nd)$. For large $n$, SGD reaches a given accuracy faster in wall-clock time.

---

## 5. Momentum Methods

### 5.1 Polyak heavy-ball momentum

Standard GD zig-zags on elongated loss surfaces (high condition number). Momentum adds an exponentially weighted memory of past gradients:

$$v_{t+1} = \beta v_t - \alpha \nabla L(\theta_t)$$
$$\theta_{t+1} = \theta_t + v_{t+1}$$

where $\beta \in [0, 1)$ is the momentum coefficient (typically $\beta = 0.9$).

**Intuition:** The velocity $v_t$ accumulates components that consistently point in the same direction and cancels out oscillations across the narrow valley. For the quadratic case, optimal $\beta$ reduces the convergence factor from $(\kappa - 1)/(\kappa + 1)$ to $(\sqrt{\kappa} - 1)/(\sqrt{\kappa} + 1)$.

**Result:** Momentum accelerates convergence on ill-conditioned problems by smoothing out oscillations.

### 5.2 Nesterov accelerated gradient (NAG)

Nesterov's key insight: compute the gradient at the **lookahead** point $\theta_t + \beta v_t$ instead of $\theta_t$:

$$v_{t+1} = \beta v_t - \alpha \nabla L(\theta_t + \beta v_t)$$
$$\theta_{t+1} = \theta_t + v_{t+1}$$

> **Theorem (Nesterov).** For convex, $M$-smooth $L$, NAG achieves:
>
> $$L(\theta_T) - L(\theta^*) = O\!\left(\frac{1}{T^2}\right)$$

This $O(1/T^2)$ rate is provably optimal among first-order methods with access only to gradients (Nemirovski & Yudin, 1983).

**Result:** NAG achieves the optimal first-order convergence rate for smooth convex problems, improving from $O(1/T)$ to $O(1/T^2)$.

---

## 6. Adaptive Methods

### 6.1 AdaGrad

AdaGrad adapts the learning rate per-parameter based on the history of squared gradients:

$$s_t = s_{t-1} + (\nabla L(\theta_t))^2 \quad \text{(element-wise square)}$$
$$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{s_t} + \epsilon} \odot \nabla L(\theta_t)$$

- **Strength:** Automatically reduces the step for frequently updated parameters. Excellent for sparse features.
- **Weakness:** $s_t$ grows monotonically, so the effective learning rate decays to zero, potentially stopping learning prematurely.

### 6.2 RMSProp

RMSProp fixes AdaGrad's monotonic decay by using an exponential moving average:

$$s_t = \beta_2 \, s_{t-1} + (1 - \beta_2)(\nabla L(\theta_t))^2$$
$$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{s_t} + \epsilon} \odot \nabla L(\theta_t)$$

This keeps a sliding window of recent gradient magnitudes instead of accumulating all history.

### 6.3 Adam (Adaptive Moment Estimation)

Adam combines momentum (first moment) with RMSProp (second moment), adding **bias correction**:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) \nabla L(\theta_t) \quad \text{(first moment)}$$
$$s_t = \beta_2 s_{t-1} + (1 - \beta_2) (\nabla L(\theta_t))^2 \quad \text{(second moment)}$$

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{s}_t = \frac{s_t}{1 - \beta_2^t} \quad \text{(bias correction)}$$

$$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{\hat{s}_t} + \epsilon} \odot \hat{m}_t$$

Default hyperparameters: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$.

**Why bias correction?** Since $m_0 = s_0 = 0$, the exponential moving averages are biased toward zero in early iterations. Dividing by $(1 - \beta^t)$ corrects this; $\hat{m}_t$ is an unbiased estimate of $\mathbb{E}[\nabla L]$ and $\hat{s}_t$ is an unbiased estimate of $\mathbb{E}[(\nabla L)^2]$.

**Result:** Adam combines the benefits of momentum (acceleration) and adaptive step sizes (per-parameter scaling), making it robust to hyperparameter choices in practice.

---

## 7. Learning Rate Schedules

A fixed learning rate is often suboptimal. Common schedules:

### 7.1 Step decay

Multiply the learning rate by a factor $\gamma < 1$ every $k$ epochs:

$$\alpha_t = \alpha_0 \cdot \gamma^{\lfloor t/k \rfloor}$$

### 7.2 Cosine annealing

Smoothly decrease the learning rate following a cosine curve:

$$\alpha_t = \alpha_{\min} + \frac{1}{2}(\alpha_0 - \alpha_{\min})\left(1 + \cos\!\left(\frac{\pi t}{T}\right)\right)$$

This allows the optimizer to explore early (large $\alpha$) and converge later (small $\alpha$).

### 7.3 Warm-up

Start with a very small learning rate and linearly increase to $\alpha_0$ over the first $T_w$ steps. This stabilizes early training, especially for large models with adaptive optimizers.

### 7.4 $1/t$ decay (for SGD)

$$\alpha_t = \frac{\alpha_0}{1 + \gamma t}$$

This is the classical schedule required for SGD convergence guarantees: $\sum_t \alpha_t = \infty$ (can reach any point) and $\sum_t \alpha_t^2 < \infty$ (noise averages out).

---

## 8. Failure Cases

### 8.1 Learning rate too large → divergence

When $\alpha > 2/M$, the GD update **overshoots** the minimum. For quadratics, the iterates along the top eigenvector direction grow geometrically. In practice, the loss explodes to $+\infty$ or NaN.

### 8.2 Learning rate too small → stalling

An overly conservative $\alpha$ makes GD converge, but so slowly that it appears stuck. The number of iterations to reach $\epsilon$-accuracy scales as $O(1/\alpha)$.

### 8.3 Ill-conditioning → zig-zagging

When $\kappa = M/\mu \gg 1$, the loss surface is an elongated ellipsoid. GD oscillates across the narrow direction while making slow progress along the long axis. Momentum and adaptive methods alleviate this.

### 8.4 Saddle points (non-convex)

In high-dimensional non-convex problems (e.g., neural networks), saddle points are more common than local minima. At a saddle, $\nabla L = 0$ but the Hessian has both positive and negative eigenvalues. Standard GD can get stuck near saddle points because the gradient is nearly zero. Adding noise (SGD) or second-order information helps escape.

### 8.5 Adam convergence issues

Adam does not converge on all convex problems. Reddi et al. (2018) showed a simple counterexample. Fixes include AMSGrad (using the maximum of past second moments). In practice, Adam usually works but may generalize worse than SGD with momentum on some tasks.

### 8.6 Small gradient ≠ small loss gap

A small $\|\nabla L(\theta)\|$ does not guarantee that $L(\theta)$ is close to $L(\theta^*)$. On flat plateaus, the gradient is small but the function value may be far from optimal.

---

## 9. Summary of Convergence Rates

| Method | Convex, smooth | Strongly convex, smooth |
|---|---|---|
| **GD** | $O(1/T)$ | $O((1 - 1/\kappa)^T)$ |
| **NAG** | $O(1/T^2)$ | $O((1 - 1/\sqrt{\kappa})^T)$ |
| **SGD** | $O(1/\sqrt{T})$ | $O(1/T)$ (with $\alpha_t \propto 1/t$) |

---

## 10. Connections

- **[01 Linear Regression](../01_linear_regression/README.md)** — GD is an alternative to the normal equation; the MSE Hessian $\frac{2}{n}X^\top X$ determines the smoothness constant.
- **[03 Regularization](../03_regularization/README.md)** — Regularization modifies the loss landscape, improving the condition number $\kappa$.
- **[04 Logistic Regression](../04_logistic_regression/README.md)** — No closed form; GD (or variants) is the standard solver.
- **[13 Neural Networks](../13_neural_networks/README.md)** — Backpropagation computes gradients; SGD/Adam are the default optimizers.
- **[Optimization Methods Compared](../../synthesis/optimization_methods_compared.md)** — Cross-topic comparison of optimization approaches.

---

## 11. References

- **Boyd, S., & Vandenberghe, L. (2004).** *Convex Optimization*. Cambridge University Press. Chapter 9: *Unconstrained Minimization*.
- **Nesterov, Y. (1983).** A method for solving the convex programming problem with convergence rate $O(1/k^2)$. *Doklady Akademii Nauk SSSR*, 269(3), 543–547.
- **Polyak, B. T. (1964).** Some methods of speeding up the convergence of iteration methods. *USSR Computational Mathematics and Mathematical Physics*, 4(5), 1–17.
- **Kingma, D. P., & Ba, J. (2014).** Adam: A method for stochastic optimization. *arXiv preprint arXiv:1412.6980*.
- **Reddi, S. J., Kale, S., & Kumar, S. (2018).** On the convergence of Adam and beyond. *International Conference on Learning Representations (ICLR)*.

