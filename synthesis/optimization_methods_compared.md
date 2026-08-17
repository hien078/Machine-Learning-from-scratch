# Optimization Methods Compared — Cross-Topic Synthesis

> How optimization methods connect across ML models.
> See [INDEX.md](../INDEX.md) for the full curriculum index.

---

## Overview

Every ML model that can't be solved in closed-form needs an optimization algorithm.
The choice of optimizer affects convergence speed, memory usage, and solution quality.

---

## Method Comparison Table

| Method | Type | Per-iteration Cost | Convergence | Memory | Used By |
|---|---|---|---|---|---|
| Normal Equation | Direct solve | $O(nd^2+d^3)$ | One numerical solve | $O(d^2)$ | Linear Regression (small d) |
| Gradient Descent | First-order | $O(nd)$ | $O(1/\epsilon)$ | $O(d)$ | All differentiable models |
| SGD | First-order | $O(d)$ | $O(1/\epsilon^2)$ | $O(d)$ | Large-scale, Neural Networks |
| Mini-batch SGD | First-order | $O(bd)$ | Between GD/SGD | $O(d)$ | Deep Learning |
| Momentum | First-order | $O(nd)$ | Accelerated | $O(d)$ | Neural Networks |
| Adam | Adaptive | $O(nd)$ | Adaptive lr | $O(2d)$ | Deep Learning (default) |
| Newton's Method | Second-order | $O(d^3)$ | Locally quadratic under regularity conditions | $O(d^2)$ | Logistic Regression (IRLS) |
| Coordinate Descent | Coordinate-wise | $O(n)$ per coordinate | Problem dependent | $O(d)$ | Lasso |
| EM Algorithm | Latent-variable | Varies | Monotone likelihood; local rate is problem dependent | Varies | GMM, HMM |

---

## Closed-Form Solutions

When available, closed-form solutions are optimal — no iteration, no hyperparameters.

### Normal Equation (Linear Regression)

$$\hat{\mathbf{w}} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}$$

- **When to use:** $d < 10{,}000$, $\mathbf{X}^\top\mathbf{X}$ well-conditioned
- **When NOT to use:** Large $d$, ill-conditioned, need regularization path
- **Numerical issue:** Use `numpy.linalg.lstsq` (SVD-based), not `numpy.linalg.inv`

### Ridge Closed-Form

$$\hat{\mathbf{w}}_{\text{ridge}} = (\mathbf{X}^\top \mathbf{X} + \lambda \mathbf{I})^{-1} \mathbf{X}^\top \mathbf{y}$$

- Adding $\lambda\mathbf{I}$ improves the condition number: $\kappa_{\text{ridge}} \le \kappa_{\text{OLS}}$
- Always invertible for $\lambda > 0$

---

## First-Order Methods (Gradient-Based)

### Gradient Descent

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \nabla \mathcal{L}(\mathbf{w}_t)$$

**Convergence theory:**
- Convex + L-Lipschitz gradient: $\mathcal{L}(\mathbf{w}_T) - \mathcal{L}^\ast \le \frac{L \Vert\mathbf{w}_0 - \mathbf{w}^\ast\Vert^2}{2T}$
- Strongly convex: linear convergence rate
- Learning rate: $\eta < \frac{2}{L}$ for convergence

**Failure modes:**
1. Too large $\eta$ → divergence
2. Too small $\eta$ → slow convergence
3. Ill-conditioned Hessian → zigzag path
4. Non-convex → local minima, saddle points

### SGD (Stochastic Gradient Descent)

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta_t \nabla \mathcal{L}_i(\mathbf{w}_t)$$

- Uses one random sample instead of full dataset
- Noisy gradient but cheaper per step
- Learning rate schedule: $\eta_t = \frac{\eta_0}{1 + \alpha t}$ (must decay)
- Noise helps escape shallow local minima

### Mini-batch SGD

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \frac{1}{|B|} \sum_{i \in B} \nabla \mathcal{L}_i(\mathbf{w}_t)$$

- Batch size $|B|$: trade-off between noise and speed
- Sweet spot: $B = 32$ to $256$ for deep learning
- Vectorized: GPU-efficient

### Momentum

```math
\mathbf{v}_{t+1} = \beta \mathbf{v}_t + \nabla \mathcal{L}(\mathbf{w}_t)
```

```math
\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \mathbf{v}_{t+1}
```

- Accumulates velocity in consistent gradient direction
- Dampens oscillation in ravine-shaped landscapes
- Typical: $\beta = 0.9$

### Adam (Adaptive Moment Estimation)

```math
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \quad \text{(first moment)}
```

```math
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \quad \text{(second moment)}
```

```math
\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
```

- Adaptive per-parameter learning rates
- Default: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$
- Often works well for deep learning, but it is not universally superior
- Some convergence counterexamples motivate variants such as AMSGrad; AdamW instead
  addresses decoupled weight decay

---

## Second-Order Methods

### Newton's Method

$$\mathbf{w}_{t+1} = \mathbf{w}_t - [\nabla^2 \mathcal{L}(\mathbf{w}_t)]^{-1} \nabla \mathcal{L}(\mathbf{w}_t)$$

- Uses Hessian (curvature information)
- Quadratic convergence near optimum
- **Cost:** $O(d^3)$ per step (Hessian inversion)
- **Used by:** Logistic Regression (IRLS), small-scale problems
- **NOT used for:** Deep learning (Hessian too large)

---

## Special-Purpose Methods

### Coordinate Descent (Lasso)

For $\frac{1}{2n}\lVert y-Xw\rVert_2^2+\lambda\lVert w\rVert_1$,

$$w_j^{(t+1)} = \frac{S_{\lambda}\left(\frac{1}{n}x_j^\top r^{(-j)}\right)}{\frac{1}{n}\lVert x_j\rVert_2^2}.$$

where $S_\lambda$ is the soft-thresholding operator.

- Updates one coordinate at a time
- Natural for L1 penalty (handles non-differentiability)
- Convergence: linear rate for strongly convex
- **Used by:** Lasso, Elastic Net

---

## Decision Guide

```
Can you solve it analytically?
├── YES → Use closed-form (Normal Eq, Ridge)
└── NO → Is d small enough for Hessian?
    ├── YES → Newton / IRLS (Logistic Regression)
    └── NO → First-order methods
        ├── n < 10,000 → Full-batch GD
        └── n ≥ 10,000 → SGD / Mini-batch
            ├── Simple model → SGD + momentum
            └── Deep network → Adam (or AdamW)
```

---

## Connections

- **Topics:** [02 Gradient Descent](../topics/02_gradient_descent/README.md), [01 Linear Regression](../topics/01_linear_regression/README.md), [03 Regularization](../topics/03_regularization/README.md), [04 Logistic Regression](../topics/04_logistic_regression/README.md), [13 Neural Networks](../topics/13_neural_networks/README.md)
- **Foundations:** [Calculus & Optimization](https://github.com/hien078/applied-mathematics-foundation)
- **Related synthesis:** [Loss Functions Map](loss_functions_map.md)
- **Maps:** [INDEX.md](../INDEX.md)
