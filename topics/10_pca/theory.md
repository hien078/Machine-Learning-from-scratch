# PCA — Theory

## 1. WHY — Redundant Directions

### 1.1 Problem

Correlated features contain redundant directions. A $d$-dimensional data cloud may
effectively live on a much lower-dimensional subspace: two sensors measuring nearly
the same quantity trace a thin ellipse, not a full-rank blob. PCA finds the linear
subspace that preserves the maximum centered squared variation — equivalently (§4),
the subspace that loses the least under squared-error reconstruction.

**Practical needs:**
- Reduce storage and computation.
- Remove multicollinearity before supervised learning.
- Visualize high-dimensional data in 2D or 3D.
- Pre-processing for noise removal (truncated reconstruction).

### 1.2 Assumptions

1. **Centered data.** Features are mean-centered ($\mu = 0$); if not, center before analysis.
2. **Linearity.** The latent structure is a linear subspace (curved manifolds require kernel PCA, t-SNE, or UMAP).
3. **Variance ≈ information.** Directions of maximum variance carry the most signal. Fails when noise variance dominates signal variance.
4. **Comparable scales.** Features should be on comparable scales, or standardized first — otherwise the largest-unit feature dominates (see §9).

### 1.3 Governing principle

PCA is an **optimization over subspaces**: choose an orthonormal basis $W_k$ to
maximize projected variance (§3), which the Pythagorean decomposition (§4) shows is
the same as minimizing squared reconstruction error. Both objectives are quadratic in
the data, which is why the solution is spectral (eigenvectors) rather than iterative.

A probabilistic reading exists — probabilistic PCA, §10 — but is not needed to derive
the method.

## 2. WHAT — Notation and Setup

| Symbol | Type | Meaning |
|---|---:|---|
| $X \in \mathbb{R}^{n \times d}$ | matrix | data matrix, $n$ observations, $d$ features |
| $\mu = \frac{1}{n}X^\top\mathbf{1}$ | vector | feature-wise mean |
| $X_c = X - \mathbf{1}\mu^\top$ | matrix | centered data |
| $S = \frac{1}{n}X_c^\top X_c$ | matrix | empirical covariance (symmetric PSD) |
| $v_j \in \mathbb{R}^d$ | vector | $j$-th principal direction (unit eigenvector of $S$) |
| $\lambda_j$ | scalar | $j$-th eigenvalue of $S$ ($\lambda_1 \ge \lambda_2 \ge \cdots \ge 0$) |
| $W_k = [v_1, \ldots, v_k]$ | matrix | $d \times k$ projection matrix, $W_k^\top W_k = I_k$ |
| $Z = X_c W_k$ | matrix | $n \times k$ score matrix (low-dim coordinates) |
| $U \Sigma V^\top$ | matrices | thin SVD of $X_c$ (§5) |
| $\sigma_j$ | scalar | $j$-th singular value of $X_c$, $\lambda_j = \sigma_j^2 / n$ |

**Variables vs parameters.** The data $X$ is given and fixed. The quantities *learned*
from it are $\mu$, $W_k$, and the spectrum $\lbrace \lambda_j \rbrace$.

The only free *hyperparameter* is the number of retained components $k$ (§6) — no
learning rate, no regularization strength, no iterative fitting in the exact form.

**Objective.** Find an orthonormal $W_k$ maximizing $\operatorname{tr}(W_k^\top S W_k)$,
the total variance of the projected data; §3 derives this form and its solution.

## 3. HOW — Variance Maximization

### 3.1 One-component problem

Project each centered row $x_i$ onto a candidate direction $v \in \mathbb{R}^d$. The
scalar coordinate of the projection is $x_i^\top v$.

Because the data are centered, the coordinates have zero mean:

```math
\frac{1}{n}\sum_i x_i^\top v = \bigl(\frac{1}{n}\sum_i x_i\bigr)^\top v = 0
```

by linearity of the inner product. The sample variance of the projected data is therefore

```math
\text{Var}_v = \frac{1}{n}\sum_{i=1}^{n} (x_i^\top v)^2
= \frac{1}{n}\|X_c v\|_2^2
= \frac{1}{n} v^\top X_c^\top X_c v
= v^\top S v,
```

using, in order: the definition of sample variance (zero mean), the definition of the
Euclidean norm, and the definition of $S$.

The objective $v^\top S v$ scales like $\Vert v \Vert_2^2$, so without a constraint it
is unbounded — the question is about *direction* only. We therefore solve

```math
v^\ast = \arg\max_{v \in \mathbb{R}^d} \; v^\top S v
\quad \text{subject to} \quad v^\top v = 1.
```

**Existence.** The unit sphere is compact and $v \mapsto v^\top S v$ is continuous, so
a maximizer exists (Weierstrass extreme value theorem).

**Lagrangian.** $\mathcal{L}(v, \alpha) = v^\top S v - \alpha(v^\top v - 1)$.

By the Lagrange multiplier theorem, any constrained extremum is a stationary point of
$\mathcal{L}$. Using the matrix-calculus identities $\nabla_v (v^\top S v) = 2Sv$
(valid because $S$ is symmetric) and $\nabla_v (v^\top v) = 2v$:

```math
\nabla_v \mathcal{L} = 2Sv - 2\alpha v = 0 \iff Sv = \alpha v.
```

Every stationary point is an eigenvector of $S$. At an eigenpair $(v_j, \lambda_j)$ the
objective evaluates to $v_j^\top S v_j = \lambda_j v_j^\top v_j = \lambda_j$, so the
best stationary point is the one with the largest eigenvalue.

**Result:** The direction of maximum projected variance is the top eigenvector of $S$;
the variance captured is $\lambda_1$.

### 3.2 Global optimality via the spectral theorem

The Lagrangian argument finds stationary points; the spectral theorem confirms the
global maximum directly.

$S$ is symmetric, so it admits an orthonormal eigenbasis
$v_1, \ldots, v_d$ with real eigenvalues $\lambda_1 \ge \cdots \ge \lambda_d \ge 0$
(PSD because $v^\top S v = \frac{1}{n}\|X_c v\|_2^2 \ge 0$). Expand any unit vector as
$v = \sum_j c_j v_j$ with $\sum_j c_j^2 = 1$. Then

```math
v^\top S v = \sum_{j=1}^{d} \lambda_j c_j^2 \le \lambda_1 \sum_{j=1}^{d} c_j^2 = \lambda_1,
```

a weighted average of eigenvalues, with equality iff all weight sits on the top
eigenspace. No second-order check is needed.

### 3.3 Multiple components

For component $j$, maximize $v^\top S v$ subject to $v^\top v = 1$ and orthogonality to
the previous directions, $v^\top v_i = 0$ for $i < j$. The Lagrangian gains one
multiplier $\beta_i$ per orthogonality constraint:

```math
\mathcal{L} = v^\top S v - \alpha (v^\top v - 1) - \sum_{i<j} \beta_i\, v^\top v_i,
\qquad
\nabla_v \mathcal{L} = 2Sv - 2\alpha v - \sum_{i<j} \beta_i v_i = 0.
```

Left-multiply the stationarity condition by $v_i^\top$ (for $i < j$): the terms
$v_i^\top v = 0$ and $v_i^\top v_{i'} = \delta_{ii'}$ (orthonormality) leave
$2 v_i^\top S v = \beta_i$. But $v_i^\top S v = (S v_i)^\top v = \lambda_i v_i^\top v = 0$
(symmetry of $S$, then the eigen-relation, then the constraint).

Hence every $\beta_i = 0$ and the condition reduces to $Sv = \alpha v$ again — an
eigenvector problem restricted to the orthogonal complement of
$\operatorname{span}(v_1, \ldots, v_{j-1})$, whose best eigenvalue is $\lambda_j$.

Equivalently, for all $k$ components at once: maximize
$\operatorname{tr}(W_k^\top S W_k)$ subject to $W_k^\top W_k = I_k$. By the
**Rayleigh–Ritz (Ky Fan) theorem**, the maximum is $\sum_{j \le k} \lambda_j$, attained
at $W_k = [v_1, \ldots, v_k]$.

**Result:** the first $k$ principal directions are the $\text{top-}k$ eigenvectors of
the covariance matrix, and the $j$-th component captures variance $\lambda_j$.

## 4. Equivalence — Reconstruction View

### 4.1 Projection and reconstruction

Given $W_k$ with orthonormal columns, the $\text{rank-}k$ approximation of $X_c$ is

```math
\hat{X}_c = X_c W_k W_k^\top.
```

Each row is projected onto $\operatorname{col}(W_k)$, then embedded back into
$\mathbb{R}^d$. The matrix $P = W_k W_k^\top$ is an **orthogonal projector**: it is
symmetric ($P^\top = P$, transpose rule) and idempotent
($P^2 = W_k (W_k^\top W_k) W_k^\top = W_k I_k W_k^\top = P$).

### 4.2 Pythagorean decomposition

Split any row $x$ as $x = Px + (I - P)x$. The cross term vanishes:

```math
(Px)^\top (I - P)x = x^\top P (I - P) x = x^\top (P - P^2) x = 0,
```

using symmetry then idempotence of $P$. So $\|x\|_2^2 = \|Px\|_2^2 + \|(I-P)x\|_2^2$
(Pythagorean theorem). Summing over all rows:

```math
\|X_c\|_F^2 = \underbrace{\|X_c W_k W_k^\top\|_F^2}_{\text{captured}}
\; + \; \underbrace{\|X_c - X_c W_k W_k^\top\|_F^2}_{\text{residual}}.
```

The left side is a constant of the data:

```math
\|X_c\|_F^2 = n \operatorname{tr}(S) = n \sum_{j=1}^{d} \lambda_j
```

The trace equals the sum of eigenvalues, by the spectral theorem. The captured term
equals the projected-variance objective of §3.3:

```math
\|X_c W_k W_k^\top\|_F^2
= \operatorname{tr}\bigl(W_k W_k^\top X_c^\top X_c W_k W_k^\top\bigr)
= \operatorname{tr}\bigl(W_k^\top X_c^\top X_c W_k\bigr)
= n \operatorname{tr}(W_k^\top S W_k),
```

using the cyclic property of the trace and $W_k^\top W_k = I_k$.

Since total = captured + residual with a fixed total, **minimizing the residual over
orthonormal $W_k$ is the same problem as maximizing captured variance**. At the
optimum (§3.3, Ky Fan) the captured term is $n \sum_{j \le k} \lambda_j$, so

```math
\min_{W_k^\top W_k = I_k} \|X_c - X_c W_k W_k^\top\|_F^2
= n \sum_{j=k+1}^{d} \lambda_j.
```

The unavoidable reconstruction error is exactly the variance carried by the discarded
directions, summed over all $n$ points — so a fast-decaying spectrum makes truncation
cheap.

### 4.3 Eckart–Young

Stronger still: among *all* $\text{rank-}k$ matrices — not only projections of the
form $X_c W W^\top$ — the best Frobenius-norm approximation of $X_c$ is the truncated
SVD $U_k \Sigma_k V_k^\top$ (**Eckart–Young–Mirsky theorem**), which coincides with
the PCA reconstruction (§5). PCA is optimal among all linear $\text{rank-}k$ compressions.

**Result:** variance maximization and squared-error reconstruction are the same
optimization; the discarded error is exactly $n$ times the sum of trailing eigenvalues.

## 5. SVD Connection

### 5.1 Thin SVD of centered data

```math
X_c = U \Sigma V^\top,
```

where $U \in \mathbb{R}^{n \times r}$, $\Sigma = \operatorname{diag}(\sigma_1, \ldots, \sigma_r)$,
$V \in \mathbb{R}^{d \times r}$, $r = \operatorname{rank}(X_c)$, with
$U^\top U = V^\top V = I_r$ and $\sigma_1 \ge \cdots \ge \sigma_r > 0$.

### 5.2 Covariance in terms of SVD

```math
S = \frac{1}{n} X_c^\top X_c
= \frac{1}{n} V \Sigma U^\top U \Sigma V^\top
= V \frac{\Sigma^2}{n} V^\top.
```

This is already an eigendecomposition: $V$ contains eigenvectors, and
$\lambda_j = \sigma_j^2 / n$. The scores also come for free:
$Z = X_c W_k = U_k \Sigma_k$ — the left singular vectors scaled by singular values.

**Result:** PCA directions are the right singular vectors of $X_c$; eigenvalues are
squared singular values over $n$. There is no need to form $S$ explicitly.

### 5.3 Why SVD beats forming the covariance — conditioning

Forming $S = \frac{1}{n} X_c^\top X_c$ **squares the condition number**:

```math
\kappa_2(S) = \frac{\lambda_1}{\lambda_r} = \frac{\sigma_1^2}{\sigma_r^2} = \kappa_2(X_c)^2.
```

Consequence in floating point: a direction with
$\sigma_j / \sigma_1 < \sqrt{\varepsilon_{\text{mach}}}$ (about $10^{-8}$ in double
precision) contributes $\lambda_j / \lambda_1 < \varepsilon_{\text{mach}}$ to $S$ —
below the relative rounding error of the dominant entries.

Once $S$ is formed, that component is numerically indistinguishable from zero and no
eigensolver can recover it: the damage happens in the matrix product, before any
decomposition runs. A backward-stable SVD of $X_c$ works with the singular values
directly and resolves ratios down to $\varepsilon_{\text{mach}} \approx 10^{-16}$.

Trade-off: when $d$ is small and conditioning mild, forming the $d \times d$
covariance costs $O(nd^2)$ once and a symmetric eigendecomposition suffices. When
$n < d$, the same spectrum comes from the $n \times n$ Gram matrix
$\frac{1}{n} X_c X_c^\top$, whose eigenvectors are the left singular vectors.

## 6. Choosing $k$ — Explained Variance

### 6.1 Explained variance ratio

```math
\text{EVR}_j = \frac{\lambda_j}{\sum_{m=1}^{d} \lambda_m}
= \frac{\sigma_j^2}{\sum_{m=1}^{d} \sigma_m^2}.
```

Because $\lambda_j$ is exactly the variance captured by component $j$ (§3) and the
denominator is the total variance $\operatorname{tr}(S)$ (§4.2), the cumulative EVR
$\sum_{j \le k} \text{EVR}_j$ is precisely the fraction of $\|X_c\|_F^2$ preserved by
a $\text{rank-}k$ reconstruction. Common rule: retain components until the cumulative
EVR exceeds a threshold (e.g. 95%).

### 6.2 Caveats

- **Thresholds are conventions**, not statistics — 95% has no optimality property.
- **Scree/elbow reading.** Under signal-plus-isotropic-noise, trailing eigenvalues
  flatten near the noise variance; the elbow marks the signal/noise split. Anisotropic
  noise blurs the elbow.
- **Variance is not task relevance.** A component with tiny EVR can carry all the
  class-discriminative information (§9). EVR guides *compression*, not *prediction*.
- **Small-sample bias.** Top eigenvalues are overestimated, trailing ones
  underestimated (§9), so cumulative EVR is optimistic when $n$ is not much larger than $d$.

## 7. Whitening

Projection decorrelates the scores but leaves them with unequal variances
$\lambda_1, \ldots, \lambda_k$. **Whitening** additionally rescales each score axis to
unit variance:

```math
Z_{\text{white}} = X_c W \Lambda^{-1/2}, \qquad \Lambda = \operatorname{diag}(\lambda_1, \ldots, \lambda_d).
```

Verification by substitution, using $W^\top S W = \Lambda$ (eigendecomposition):

```math
\frac{1}{n} Z_{\text{white}}^\top Z_{\text{white}}
= \Lambda^{-1/2}\, W^\top S W\, \Lambda^{-1/2}
= \Lambda^{-1/2} \Lambda \Lambda^{-1/2} = I.
```

**ZCA whitening** rotates back to the original axes:
$X_{\text{zca}} = X_c W \Lambda^{-1/2} W^\top = X_c S^{-1/2}$. Any rotation of white
data stays white; among all whitening transforms, ZCA is the one closest to the
original data in least squares — useful when coordinates should stay interpretable
(e.g. image pixels).

**Caution.** Whitening divides by $\sqrt{\lambda_j}$; near-zero eigenvalues amplify
noise unboundedly. Regularize with $\lambda_j + \epsilon$ or whiten only the top-$k$
components.

**Result:** whitening = rotate to principal axes, then rescale each axis by
$1/\sqrt{\lambda_j}$; ZCA adds a rotation back.

## 8. Identifiability

Eigenvectors are unique only up to sign: if $v$ is an eigenvector, so is $-v$.
When eigenvalues coincide ($\lambda_j = \lambda_{j+1}$), any rotation within the
eigenspace is valid. **Consequence:** compare PCA solutions by subspace alignment, not
by raw loading values.

## 9. Failure Cases

1. **Scale dominance.** If one feature is measured in meters and another in millimeters,
   the large-unit feature dominates variance. PCA without standardization reflects
   units, not structure.

2. **Standardization is a choice.** Standardizing first (correlation-matrix PCA) is a
   modeling choice, not a fix-all: it also equalizes genuinely different signal
   strengths.

3. **Outlier sensitivity.** PCA uses $\ell_2$ (squared) distances. A single extreme
   observation can rotate the first PC toward itself. Robust variants (e.g. robust
   covariance estimates) trade efficiency for resistance.

4. **Linearity.** PCA finds a linear subspace. If the data lie on a curved manifold
   (e.g. a Swiss roll), PCA may fail to capture the intrinsic low-dimensional structure.
   Nonlinear extensions: kernel PCA, t-SNE, UMAP, autoencoders (§10).

5. **Variance ≠ relevance.** PCA is unsupervised. The directions of maximum variance
   need not align with class-discriminative directions — a low-variance direction may
   separate the classes perfectly. LDA (topic 12) optimizes separation instead.

6. **Sample size.** With $n < d$, at most $n - 1$ eigenvalues are nonzero. Estimated
   eigenvalues are biased: the largest are overestimated, the smallest underestimated,
   and the estimated directions fluctuate strongly when adjacent eigenvalues are close
   (§8).

## 10. Connections

- **[Autoencoders](../17_autoencoder/README.md).** A *linear* autoencoder with squared
  reconstruction loss learns exactly the principal subspace: Baldi & Hornik (1989)
  showed its loss surface has no spurious local minima and every minimum projects onto
  $\operatorname{span}(v_1, \ldots, v_k)$.
- **Basis and nonlinear extensions.** The learned basis need not be orthonormal or
  eigenvalue-ordered. Nonlinear autoencoders generalize the reconstruction view of §4
  beyond linear subspaces.
- **[LDA and t-SNE](../12_dimensionality_reduction/README.md).** LDA replaces
  "maximize variance" with "maximize between-class over within-class scatter" —
  supervised where PCA is unsupervised. t-SNE abandons linearity and global structure
  to preserve local neighborhoods.
- **Kernel PCA.** The Gram-matrix route of §5.3 with kernel evaluations instead of
  inner products: eigendecompose the centered kernel matrix to run PCA implicitly in a
  nonlinear feature space.
- **Probabilistic PCA.** Tipping & Bishop (1999): the latent linear-Gaussian model
  $x = Wz + \mu + \varepsilon$ with $z \sim \mathcal{N}(0, I_k)$,
  $\varepsilon \sim \mathcal{N}(0, \sigma^2 I_d)$ has a maximum-likelihood solution
  whose $W$ spans the principal subspace.
- **Zero-noise limit.** Classical PCA is the $\sigma^2 \to 0$ limit. This bridges PCA
  to [generative models](../19_generative_models/README.md).
- **[Regularization](../03_regularization/README.md).** Truncated reconstruction is a
  spectral filter; principal-components regression and ridge shrink along the same
  eigen-directions.
- [Linear Algebra Foundations](https://github.com/hien078/applied-mathematics-foundation) ·
  [Geometry of ML](../synthesis/geometry_of_ml.md)

## 11. Verification Pointers

Numerical checks live in [first_principles.ipynb](first_principles.ipynb) — theory
predicts, the notebook confirms:

- the three solver routes (covariance eigendecomposition, thin SVD, power iteration)
  agree up to sign (§5, §8);
- scratch PCA matches scikit-learn's components and explained-variance ratios under
  sign-invariant comparison (§6, §8);
- reconstruction error decreases as $k$ grows and vanishes at full rank, matching the
  trailing-eigenvalue formula (§4);
- orthonormality of $W_k$ and zero-mean scores hold to machine precision (§3);
- scale and outlier sensitivity demos, and stability of the estimated first PC as $n$
  grows (§9).

---

## 12. References

- **Pearson, K. (1901).** On lines and planes of closest fit to systems of points in space. *The London, Edinburgh, and Dublin Philosophical Magazine and Journal of Science*, 2(11), 559–572.
- **Hotelling, H. (1933).** Analysis of a complex of statistical variables into principal components. *Journal of Educational Psychology*, 24(6), 417–441.
- **Eckart, C., & Young, G. (1936).** The approximation of one matrix by another of lower rank. *Psychometrika*, 1(3), 211–218.
- **Baldi, P., & Hornik, K. (1989).** Neural networks and principal component analysis: Learning from examples without local minima. *Neural Networks*, 2(1), 53–58.
- **Tipping, M. E., & Bishop, C. M. (1999).** Probabilistic principal component analysis. *Journal of the Royal Statistical Society: Series B*, 61(3), 611–622.
- **Jolliffe, I. T. (2002).** *Principal Component Analysis* (2nd ed.). Springer Series in Statistics.
- **Hastie, T., Tibshirani, R., & Friedman, J. (2009).** *The Elements of Statistical Learning: Data Mining, Inference, and Prediction* (2nd ed.). Springer. Chapter 14.5: *Principal Components Analysis*.
- **Golub, G. H., & Van Loan, C. F. (2013).** *Matrix Computations* (4th ed.). Johns Hopkins University Press. Chapters 2.4, 8.6: conditioning and SVD algorithms.
