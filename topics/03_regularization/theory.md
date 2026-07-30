# Regularization — Theory

## 1. WHY — Unstable Flexible Models

When predictors are numerous, correlated, or weakly identified, small changes in the
sample can produce large coefficient changes. A degree-15 polynomial fitted by OLS may
have coefficients of order $10^6$ even when the target range is $[-1, 1]$. Those huge
coefficients cancel on training points but not on new data. Regularization trades some
fit for lower effective complexity and often lower prediction variance.

## 2. WHAT — Objectives and Notation

| Symbol | Type | Meaning |
|---|---:|---|
| $X \in \mathbb{R}^{n \times p}$ | matrix | design matrix |
| $y \in \mathbb{R}^n$ | vector | targets |
| $\theta \in \mathbb{R}^p$ | vector | coefficients |
| $\lambda \ge 0$ | scalar | regularization strength |
| $I_p$ | matrix | identity of size $p$ |
| $\sigma_j$ | scalar | $j$-th singular value of $X$ |

### 2.1 Ridge (L2 penalty)

$$
L_{\text{ridge}}(\theta)
= \frac{1}{n}\|X\theta - y\|_2^2 + \lambda\|\theta\|_2^2
$$

### 2.2 Lasso (L1 penalty)

$$
L_{\text{lasso}}(\theta)
= \frac{1}{n}\|X\theta - y\|_2^2 + \lambda\|\theta\|_1
$$

### 2.3 Elastic Net

$$
L_{\text{EN}}(\theta)
= \frac{1}{n}\|X\theta - y\|_2^2
  + \lambda_1\|\theta\|_1
  + \lambda_2\|\theta\|_2^2
$$

Elastic Net combines L1 sparsity with L2 grouping. When $\lambda_1 = 0$ it reduces to
Ridge; when $\lambda_2 = 0$ it reduces to Lasso.

**Convention note.** Multiplicative constants between the data term and penalty differ
across implementations. sklearn's `Ridge(alpha=α)` minimizes $\|X\theta - y\|_2^2 + \alpha\|\theta\|_2^2$,
so $\alpha = n\lambda$ in our convention. sklearn's `Lasso(alpha=α)` minimizes
$\frac{1}{2n}\|X\theta - y\|_2^2 + \alpha\|\theta\|_1$, so $\alpha = \lambda/2$.

**Intercept convention.** The intercept is usually not penalized. Standard implementation:
center $X$ and $y$, fit ridge/lasso on centered data without an intercept column, then
recover $\hat\theta_0 = \bar{y} - \bar{x}^\top\hat\theta$.

## 3. HOW — Ridge: Closed Form and SVD Shrinkage

### 3.1 Gradient and strict convexity

$$
\nabla L_{\text{ridge}}(\theta)
= \frac{2}{n}X^\top(X\theta - y) + 2\lambda\theta
$$

$$
\nabla^2 L_{\text{ridge}}(\theta)
= \frac{2}{n}(X^\top X + n\lambda I_p)
$$

For any $v \neq 0$ and $\lambda > 0$:

$$
v^\top(X^\top X + n\lambda I_p)v
= \|Xv\|_2^2 + n\lambda\|v\|_2^2 > 0
$$

**Result:** $X^\top X + n\lambda I_p$ is positive definite for any $X$ and any $\lambda > 0$.
Ridge is strictly convex and has a unique minimizer.

### 3.2 Closed form

Setting the gradient to zero:

$$
\hat\theta_{\text{ridge}}
= (X^\top X + n\lambda I_p)^{-1}X^\top y
$$

**Result:** no rank assumption on $X$ is needed when $\lambda > 0$.

### 3.3 SVD shrinkage

Let $X = U_r\Sigma_r V_r^\top$ be the compact SVD with $r = \operatorname{rank}(X)$.

$$
\hat\theta_{\text{ridge}}
= \sum_{j=1}^{r}
  \frac{\sigma_j}{\sigma_j^2 + n\lambda}
  \langle u_j, y\rangle\, v_j
$$

The prediction smoother has eigenvalues (shrinkage factors):

$$
\rho_j(\lambda)
= \frac{\sigma_j^2}{\sigma_j^2 + n\lambda} \in (0, 1)
$$

Small singular values shrink most. This is the right behavior: low-$\sigma_j$ directions
also have high variance in $\hat\theta_{\text{OLS}}$.

### 3.4 Effective degrees of freedom

$$
\operatorname{df}(\lambda)
= \sum_{j=1}^{r} \frac{\sigma_j^2}{\sigma_j^2 + n\lambda}
$$

This decreases monotonically from $r$ to $0$ as $\lambda$ grows.

## 4. HOW — Lasso: Subgradients and Soft-Thresholding

### 4.1 No closed form

The L1 norm $\|\theta\|_1$ is not differentiable at any $\theta$ with a zero coordinate.
Since the Lasso solution typically has many zeros, the standard "set gradient to zero"
approach does not apply.

### 4.2 Subdifferential of the L1 norm

The subdifferential factorizes coordinate-wise:

$$
\partial|z| =
\begin{cases}
\{+1\} & \text{if } z > 0 \\
\{-1\} & \text{if } z < 0 \\
[-1, +1] & \text{if } z = 0
\end{cases}
$$

### 4.3 First-order optimality (KKT)

A point $\theta^*$ minimizes $L_{\text{lasso}}$ if and only if:

- If $\theta_j^* > 0$: $(2/n)x_j^\top r = -\lambda$ where $r = X\theta^* - y$
- If $\theta_j^* < 0$: $(2/n)x_j^\top r = +\lambda$
- If $\theta_j^* = 0$: $|x_j^\top r| \le n\lambda/2$

The last condition explains sparsity: features with small correlation to the residual
are set to exactly zero.

### 4.4 Soft-thresholding

The one-dimensional Lasso has a closed form:

$$
S_\lambda(z)
= \operatorname{sign}(z)\max(|z| - \lambda, 0)
$$

**Result:** soft-thresholding shrinks $z$ toward zero by exactly $\lambda$ and clips
anything closer than $\lambda$ to zero. This formula drives coordinate descent: the
$p$-dimensional Lasso reduces to a sequence of one-dimensional problems.

## 5. Geometry — L1 vs L2

$\ell_2$ constraint sets ($\|\theta\|_2^2 \le t$) have smooth boundaries. The loss
contour is tangent at a generic point — continuous shrinkage, no exact zeros.

$\ell_1$ constraint sets ($\|\theta\|_1 \le t$) are diamonds with axis-aligned corners.
The loss contour very often touches a corner, where some coordinates equal exactly zero.

| | Constraint shape | Tangency | Sparsity |
|---|---|---|---|
| Ridge | Smooth sphere | Generic point | No exact zeros |
| Lasso | Diamond with corners | Often at corners | Promotes sparsity |

**Result:** the geometric difference between the smooth L2 ball and the cornered L1
diamond is the entire reason Lasso produces sparse solutions.

## 6. Statistical and Bayesian Views

### 6.1 Bias–variance tradeoff

Ridge and Lasso introduce bias but can reduce variance. For Ridge, a local
parameter-risk theorem guarantees that a small positive $\lambda$ reduces total risk
when $X$ has full column rank.

### 6.2 Bayesian interpretation

| Estimator | Likelihood | Prior on $\theta$ | $\lambda$ maps to |
|---|---|---|---|
| OLS | Gaussian noise | flat (improper) | — |
| Ridge | Gaussian noise | $\mathcal{N}(0, \tau^2 I)$ | $\lambda = \sigma^2/(n\tau^2)$ |
| Lasso | Gaussian noise | $\operatorname{Laplace}(0, b)$ | $\lambda = 2\sigma^2/(nb)$ |

**Ridge as Gaussian MAP.** Taking $-\log$ of the posterior under Gaussian likelihood
and Gaussian prior yields the ridge objective.

**Lasso as Laplace MAP.** The Laplace density has a cusp at zero and heavier tails than
the Gaussian. Taking $-\log$ of the posterior yields the lasso objective. Note: the
Laplace prior is continuous, so draws from the posterior are not themselves exactly
sparse; exact zeros arise from the MAP optimization.

## 7. Failure Cases

- Features must be placed on comparable scales before penalty strengths are compared.
- Lasso selection can be unstable among highly correlated features.
- Excessive regularization underfits (all coefficients shrunk to zero).
- Selecting $\lambda$ on test data leaks information; use cross-validation.
- Ridge never produces exact zeros; if sparsity is needed, use Lasso or Elastic Net.
- Lasso may select at most $\min(n, p)$ features when $p > n$.

## 8. Connections

- [Bias–Variance](../../synthesis/bias_variance_tradeoff.md)
- [Geometry of ML](../../synthesis/geometry_of_ml.md)
- [Regularization Across Models](../../synthesis/regularization_across_models.md)
- [Linear Regression](../01_linear_regression/README.md)
- [Gradient Descent](../02_gradient_descent/README.md)
