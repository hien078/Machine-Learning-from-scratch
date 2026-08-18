# Regularization — Theory

## 1. WHY — Unstable Flexible Models

When predictors are numerous, correlated, or weakly identified, small changes in the
sample can produce large coefficient changes. A degree-15 polynomial fitted by OLS may
have coefficients of order $10^6$ even when the target range is $[-1, 1]$. Those huge
coefficients cancel on training points but not on new data.

Regularization trades some fit for lower effective complexity and often lower
prediction variance.

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

```math
L_{\text{ridge}}(\theta)
= \frac{1}{n}\|X\theta - y\|_2^2 + \lambda\|\theta\|_2^2
```

Adding the squared norm makes the objective strictly convex, so a unique minimizer
always exists and every coefficient is pulled toward zero (§3).

### 2.2 Lasso (L1 penalty)

```math
L_{\text{lasso}}(\theta)
= \frac{1}{n}\|X\theta - y\|_2^2 + \lambda\|\theta\|_1
```

Swapping the squared norm for the L1 norm keeps the objective convex but destroys
differentiability at zero, which is what lets coefficients be set exactly to zero (§4).

### 2.3 Elastic Net

```math
L_{\text{EN}}(\theta)
= \frac{1}{n}\|X\theta - y\|_2^2
+ \lambda_1\|\theta\|_1
+ \lambda_2\|\theta\|_2^2
```

Elastic Net combines L1 sparsity with L2 grouping. When $\lambda_1 = 0$ it reduces to
Ridge; when $\lambda_2 = 0$ it reduces to Lasso.

**Grouping effect.** Lasso treats a group of highly correlated features as interchangeable:
it picks one member (near-arbitrarily, and unstably across resamples) and zeroes the rest.

The added L2 term makes the objective strictly convex in $\theta$, which forces nearly
identical features to receive nearly identical coefficients.

Zou & Hastie (2005) prove that for standardized features with correlation $\rho$, the
gap between two same-sign coefficients is bounded by a term proportional to
$\sqrt{2(1-\rho)}\,/\,\lambda_2$, which vanishes as $\rho \to 1$: the group enters or
leaves the model together.

**Capacity.** The L2 term also removes Lasso's saturation: Elastic Net can select more
than $n$ features when $p > n$ (see the corresponding failure case in Section 7).

**sklearn convention.** `ElasticNet(alpha=α, l1_ratio=r)` minimizes
$\frac{1}{2n}\Vert X\theta - y\Vert_2^2 + \alpha r\Vert\theta\Vert_1 + \frac{\alpha(1-r)}{2}\Vert\theta\Vert_2^2$,
so $\lambda_1 = 2\alpha r$ and $\lambda_2 = \alpha(1-r)$ in our convention. Beware:
`ElasticNet` and `Ridge` do **not** share an alpha scale —
`ElasticNet(alpha=α, l1_ratio=0)` matches `Ridge(alpha=nα)`, not `Ridge(alpha=α)`.

**Convention note.** Multiplicative constants between the data term and penalty differ
across implementations.

sklearn's `Ridge(alpha=α)` minimizes $\Vert X\theta - y\Vert_2^2 + \alpha\Vert\theta\Vert_2^2$,
so $\alpha = n\lambda$ in our convention. sklearn's `Lasso(alpha=α)` minimizes
$\frac{1}{2n}\Vert X\theta - y\Vert_2^2 + \alpha\Vert\theta\Vert_1$, so $\alpha = \lambda/2$.

**Intercept convention.** The intercept is usually not penalized. Standard implementation:
center $X$ and $y$, fit ridge/lasso on centered data without an intercept column, then
recover $\hat\theta_0 = \bar{y} - \bar{x}^\top\hat\theta$.

*Why exclude it?* The penalty is meant to price model complexity, and the intercept
carries none — it is the baseline level of $y$.

Penalizing $\theta_0$ would make the fit depend on the arbitrary origin of the targets:
adding a constant to every $y_i$ (e.g., switching temperature units from °C to K) would
change *all* fitted coefficients instead of merely shifting the intercept.

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

```math
v^\top(X^\top X + n\lambda I_p)v
= \|Xv\|_2^2 + n\lambda\|v\|_2^2 > 0
```

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

```math
\hat\theta_{\text{ridge}}
= \sum_{j=1}^{r}
\frac{\sigma_j}{\sigma_j^2 + n\lambda}
\langle u_j, y\rangle\, v_j
```

Predictions are linear in $y$: $\hat{y} = X\hat\theta_{\text{ridge}} = H(\lambda)\,y$, where

$$
H(\lambda)
= X(X^\top X + n\lambda I_p)^{-1}X^\top
= \sum_{j=1}^{r} \rho_j(\lambda)\, u_j u_j^\top
$$

is the **prediction smoother**. Its nonzero eigenvalues are the shrinkage factors:

$$
\rho_j(\lambda)
= \frac{\sigma_j^2}{\sigma_j^2 + n\lambda} \in (0, 1)
$$

Small singular values shrink most. This is the right behavior: directions with small
$\sigma_j$ are exactly the directions in which $\hat\theta_{\text{OLS}}$ has the highest
variance (made precise in Section 6.1).

### 3.4 Effective degrees of freedom

$$
\operatorname{df}(\lambda)
= \operatorname{tr} H(\lambda)
= \sum_{j=1}^{r} \frac{\sigma_j^2}{\sigma_j^2 + n\lambda}
= \sum_{j=1}^{r} \rho_j(\lambda)
$$

This decreases monotonically from $r$ to $0$ as $\lambda$ grows. As $\lambda \to 0^+$,
$H(\lambda)$ tends to the orthogonal projection onto the column space of $X$ — the OLS
hat matrix, whose eigenvalues are all $0$ or $1$ — and $\operatorname{df} \to r$. Ridge
replaces those hard $0/1$ eigenvalues with soft factors $\rho_j \in (0,1)$; hence
*effective* degrees of freedom.

## 4. HOW — Lasso: Subgradients and Soft-Thresholding

### 4.1 No closed form

The L1 norm $\Vert\theta\Vert_1$ is not differentiable at any $\theta$ with a zero coordinate.
Since the Lasso solution typically has many zeros, the standard "set gradient to zero"
approach does not apply.

### 4.2 Subdifferential of the L1 norm

The subdifferential factorizes coordinate-wise:

```math
\partial|z| =
\begin{cases}
\{+1\} & \text{if } z > 0 \\
\{-1\} & \text{if } z < 0 \\
[-1, +1] & \text{if } z = 0
\end{cases}
```

Away from zero the subgradient is just the sign of $z$; at zero it widens into a whole
interval, and that slack is what allows a coordinate to rest exactly at zero.

### 4.3 First-order optimality (KKT)

A point $\theta^\ast$ minimizes $L_{\text{lasso}}$ if and only if:

- If $\theta_j^\ast > 0$: $(2/n)x_j^\top r = -\lambda$ where $r = X\theta^\ast - y$
- If $\theta_j^\ast < 0$: $(2/n)x_j^\top r = +\lambda$
- If $\theta_j^\ast = 0$: $|x_j^\top r| \le n\lambda/2$

The last condition explains sparsity: features with small correlation to the residual
are set to exactly zero.

### 4.4 Soft-thresholding

Consider the one-dimensional Lasso problem $\min_\theta \frac{1}{2}(z - \theta)^2 + \lambda |\theta|$. The subgradient optimality condition is:

$$0 \in \theta - z + \lambda \partial |\theta|$$

Analyzing the subdifferential $\partial |\theta|$ by cases:
- If $\theta > 0$: $\theta - z + \lambda = 0 \implies \theta = z - \lambda$ (valid when $z > \lambda$).
- If $\theta < 0$: $\theta - z - \lambda = 0 \implies \theta = z + \lambda$ (valid when $z < -\lambda$).
- If $\theta = 0$: $|z - 0| \le \lambda \implies |z| \le \lambda$.

Combining these three cases yields the **soft-thresholding operator**:

$$
S_\lambda(z)
= \operatorname{sign}(z)\max(|z| - \lambda, 0)
$$

**Result:** Soft-thresholding shrinks $z$ toward zero by exactly $\lambda$ and clips anything closer than $\lambda$ to zero. This closed-form solution drives coordinate descent: the $p$-dimensional Lasso reduces to a sequence of one-dimensional soft-thresholding problems.

**Numerical example** ($\lambda = 1.5$):

| $z$ | $S_{1.5}(z)$ | what happened |
|---:|---:|---|
| $3$ | $3 - 1.5 = 1.5$ | shrunk toward zero by exactly $1.5$ |
| $1$ | $0$ | inside the dead zone $[-1.5,\, 1.5]$: clipped |
| $-2$ | $-2 + 1.5 = -0.5$ | shrunk toward zero by exactly $1.5$ |

Sketch — a dead zone of width $2\lambda$, slope $1$ outside:

```text
      S_λ(z)
        |              /
        |             /
        |            /
  ──────●━━━━━━━━━━━●──────→  z
       /−λ     0    +λ
      /
     /
```

### 4.5 Coordinate descent

Fix every coordinate except $\theta_j$ and collect the **partial residual**

$$
r_{-j} = y - \sum_{k \neq j} x_k\theta_k
$$

($r_{-j}$: the part of $y$ not yet explained by the other features). As a function of
$\theta_j$ alone, $L_{\text{lasso}}$ is exactly the one-dimensional problem of 4.4, and
the same case analysis gives the closed-form coordinate update

$$
\theta_j \;\leftarrow\; \frac{S_{n\lambda/2}\big(x_j^\top r_{-j}\big)}{\|x_j\|_2^2}
$$

- $x_j^\top r_{-j}$: correlation between feature $j$ and the not-yet-explained residual
- $S_{n\lambda/2}$: zero the coordinate when that correlation is below the KKT threshold of 4.3, shrink it otherwise
- $\|x_j\|_2^2$: converts the correlation back to coefficient units

Cycling this update over $j = 1, \dots, p$ until convergence is the algorithm behind
sklearn's `Lasso` and `glmnet`.

### 4.6 Where the path starts: $\lambda_{\max}$

Apply the zero-condition of 4.3 to the candidate solution $\theta = 0$ (so $r = -y$):
it is optimal iff $|x_j^\top y| \le n\lambda/2$ for every $j$, i.e. iff

$$
\lambda \;\ge\; \lambda_{\max} \;=\; \frac{2}{n}\,\|X^\top y\|_\infty
$$

- $\|X^\top y\|_\infty$: the largest absolute correlation between any single feature and the target

**Result:** for $\lambda \ge \lambda_{\max}$ the Lasso solution is identically zero, and
just below $\lambda_{\max}$ the first feature to enter the model is the one most
correlated with $y$. Practical solvers compute the whole regularization path on a grid
from $\lambda_{\max}$ down to a small fraction of it (e.g. $10^{-3}\lambda_{\max}$).

### 4.7 Worked comparison: orthonormal design

When $X^\top X = I_p$, both estimators act coordinate-wise on
$\hat\theta_{\text{OLS}} = X^\top y$, and the Ridge/Lasso contrast becomes pure algebra.

From the closed form in 3.2:

$$
\hat\theta_{\text{ridge},j}
= \frac{\hat\theta_{\text{OLS},j}}{1 + n\lambda}
$$

*Multiplicative* shrinkage — every coefficient is scaled by the same factor, so a nonzero
coefficient never becomes exactly zero.

The Lasso objective separates into $p$ copies of the 4.4 problem with threshold $n\lambda/2$:

$$
\hat\theta_{\text{lasso},j}
= S_{n\lambda/2}\big(\hat\theta_{\text{OLS},j}\big)
$$

*Subtractive* shrinkage — every coefficient moves toward zero by the same amount, and
anything closer than the threshold is deleted.

**Numbers.** Take $\hat\theta_{\text{OLS}} = (3,\ 1,\ -2)$, with $n\lambda = 1$ for Ridge
and $n\lambda/2 = 1.5$ for Lasso:

| $\hat\theta_{\text{OLS},j}$ | Ridge: divide by $1 + n\lambda = 2$ | Lasso: $S_{1.5}$ |
|---:|---:|---:|
| $3$ | $1.5$ | $1.5$ |
| $1$ | $0.5$ | $0$ |
| $-2$ | $-1$ | $-0.5$ |

Ridge keeps all three coefficients alive; Lasso deletes the small one. This is the
algebraic content of the geometric picture in Section 5.

## 5. Geometry — L1 vs L2

$\ell_2$ constraint sets ($\Vert\theta\Vert_2^2 \le t$) have smooth boundaries. The loss
contour is tangent at a generic point — continuous shrinkage, no exact zeros.

$\ell_1$ constraint sets ($\Vert\theta\Vert_1 \le t$) are diamonds with axis-aligned corners.
The loss contour very often touches a corner, where some coordinates equal exactly zero.

| | Constraint shape | Tangency | Sparsity |
|---|---|---|---|
| Ridge | Smooth sphere | Generic point | No exact zeros |
| Lasso | Diamond with corners | Often at corners | Promotes sparsity |

**Result:** the geometric difference between the smooth L2 ball and the cornered L1
diamond is the entire reason Lasso produces sparse solutions. Section 4.7 is the
algebraic counterpart of this picture: the smooth ball yields multiplicative shrinkage
$1/(1 + n\lambda)$, the cornered diamond yields subtractive shrinkage with a dead zone.

## 6. Statistical and Bayesian Views

### 6.1 Bias–variance tradeoff

Ridge and Lasso introduce bias but can reduce variance. For Ridge this can be made
fully explicit.

Assume $y = X\theta^\star + \varepsilon$ with
$\operatorname{Var}(\varepsilon) = \sigma^2 I$ and $X$ of full column rank. Decomposing
along the right singular vectors $v_j$:

$$
\operatorname{Var}\!\big(\hat\theta_{\text{ridge}}\big)\ \text{along}\ v_j
= \frac{\sigma^2}{\sigma_j^2}\,\rho_j(\lambda)^2,
\qquad
\text{Bias along}\ v_j
= -\big(1 - \rho_j(\lambda)\big)\,\langle v_j, \theta^\star\rangle
$$

- $\sigma^2/\sigma_j^2$: the OLS variance in direction $v_j$ — it explodes when $\sigma_j$ is small
- $\rho_j^2 \in (0, 1)$: ridge multiplies that variance by the *squared* shrinkage factor of 3.3
- $(1 - \rho_j)\langle v_j, \theta^\star\rangle$: the price — the true signal in that direction is under-estimated by the shrunk fraction

The same factor $\rho_j$ from Section 3.3 controls both effects, and shrinkage is
strongest exactly where OLS variance is worst. Summing over directions, the total
parameter risk is

$$
\mathbb{E}\,\big\|\hat\theta_{\text{ridge}} - \theta^\star\big\|_2^2
= \sum_{j=1}^{p}\left[
\frac{\sigma^2}{\sigma_j^2}\,\rho_j^2
+ (1 - \rho_j)^2\langle v_j, \theta^\star\rangle^2
\right]
$$

At $\lambda = 0^+$ the variance term decreases at rate
$-2n\sigma^2\sum_j \sigma_j^{-4} < 0$, while the bias term — quadratic in
$(1 - \rho_j)$ — starts flat. The derivative of the risk at $\lambda = 0^+$ is therefore
strictly negative: **some** $\lambda > 0$ always beats OLS (Hoerl & Kennard, 1970).

The theorem does not say *which* $\lambda$; that is what cross-validation is for.

### 6.2 Bayesian interpretation

| Estimator | Likelihood | Prior on $\theta$ | $\lambda$ maps to |
|---|---|---|---|
| OLS | Gaussian noise | flat (improper) | — |
| Ridge | Gaussian noise | $\mathcal{N}(0, \tau^2 I)$ | $\lambda = \sigma^2/(n\tau^2)$ |
| Lasso | Gaussian noise | $\operatorname{Laplace}(0, b)$ | $\lambda = 2\sigma^2/(nb)$ |

**Ridge as Gaussian MAP.** Taking $-\log$ of the posterior under Gaussian likelihood
and Gaussian prior yields the ridge objective.

**Lasso as Laplace MAP.** The Laplace density has a cusp at zero and heavier tails than
the Gaussian. Taking $-\log$ of the posterior yields the lasso objective.

Note: the Laplace prior is continuous, so draws from the posterior are not themselves
exactly sparse; exact zeros arise from the MAP optimization.

## 7. Failure Cases

- Features must be placed on comparable scales before penalty strengths are compared:
  the penalty treats all coefficients as commensurable, but a coefficient's size depends
  on its feature's units.
- Rescaling $x_j$ from km to m divides $\theta_j$ by $10^3$ and quietly divides its
  share of the penalty too.
- Lasso selection can be unstable among highly correlated features; Elastic Net's
  grouping effect (Section 2.3) is the standard remedy.
- Excessive regularization underfits (all coefficients shrunk to zero — for Lasso this
  happens for every $\lambda \ge \lambda_{\max}$, Section 4.6).
- Selecting $\lambda$ on test data leaks information; use cross-validation.
- Ridge never produces exact zeros; if sparsity is needed, use Lasso or Elastic Net.
- When $p > n$, Lasso selects at most $n$ features; Elastic Net removes this cap
  (Section 2.3).

## 8. Connections

- [Bias–Variance](../synthesis/bias_variance_tradeoff.md)
- [Geometry of ML](../synthesis/geometry_of_ml.md)
- [Regularization Across Models](../synthesis/regularization_across_models.md)
- [Linear Regression](../01_linear_regression/README.md)
- [Gradient Descent](../02_gradient_descent/README.md)

---

## 9. References

- **Hastie, T., Tibshirani, R., & Friedman, J. (2009).** *The Elements of Statistical Learning* (2nd ed.). Springer. Chapter 3.4: *Shrinkage Methods*.
- **Hoerl, A. E., & Kennard, R. W. (1970).** Ridge regression: Biased estimation for nonorthogonal problems. *Technometrics*, 12(1), 55–67.
- **Tibshirani, R. (1996).** Regression shrinkage and selection via the lasso. *Journal of the Royal Statistical Society: Series B (Methodological)*, 58(1), 267–288.
- **Zou, H., & Hastie, T. (2005).** Regularization and variable selection via the elastic net. *Journal of the Royal Statistical Society: Series B (Statistical Methodology)*, 67(2), 301–320.

