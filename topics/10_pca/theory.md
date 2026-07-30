# PCA — Theory

## 1. WHY — Redundant Directions

Correlated features contain redundant directions. A $d$-dimensional data cloud may
effectively live on a much lower-dimensional subspace. PCA finds a linear subspace
that preserves the maximum centered squared variation.

**Practical needs:**
- Reduce storage and computation.
- Remove multicollinearity before supervised learning.
- Visualize high-dimensional data in 2D or 3D.
- Pre-processing for noise removal (truncated reconstruction).

### Assumptions

1. **Centered data.** Features are mean-centered ($\mu = 0$); if not, center before analysis.
2. **Linearity.** The latent structure is a linear subspace (curved manifolds require kernel PCA, t-SNE, or UMAP).
3. **Variance ≈ information.** Directions of maximum variance carry the most signal. Fails when noise variance dominates signal variance.
4. **Comparable scales.** Features should be on comparable scales, or standardized first — otherwise the largest-unit feature dominates (see §7).

## 2. WHAT — Notation and Setup

| Symbol | Type | Meaning |
|---|---:|---|
| $X \in \mathbb{R}^{n \times d}$ | matrix | data matrix, $n$ observations, $d$ features |
| $\mu = \frac{1}{n}X^\top\mathbf{1}$ | vector | feature-wise mean |
| $X_c = X - \mathbf{1}\mu^\top$ | matrix | centered data |
| $S = \frac{1}{n}X_c^\top X_c$ | matrix | empirical covariance |
| $v_j \in \mathbb{R}^d$ | vector | $j$-th principal direction (unit eigenvector of $S$) |
| $\lambda_j$ | scalar | $j$-th eigenvalue of $S$ ($\lambda_1 \ge \lambda_2 \ge \cdots \ge 0$) |
| $W_k = [v_1, \ldots, v_k]$ | matrix | $d \times k$ projection matrix |
| $Z = X_c W_k$ | matrix | $n \times k$ score matrix (low-dim coordinates) |

## 3. HOW — Variance Maximization

### 3.1 One-component problem

Project $X_c$ onto a unit vector $v \in \mathbb{R}^d$ ($\|v\|_2 = 1$). The sample
variance of the projected data is:

$$
\text{Var}_v = \frac{1}{n}\|X_c v\|_2^2 = v^\top S v
$$

We seek $v$ maximizing $v^\top S v$ subject to $v^\top v = 1$.

**Lagrangian.** $\mathcal{L}(v, \alpha) = v^\top S v - \alpha(v^\top v - 1)$

Taking the gradient and setting to zero:

$$
\nabla_v \mathcal{L} = 2Sv - 2\alpha v = 0 \implies Sv = \alpha v
$$

This is an eigenvector equation. The objective at the solution is $v^\top S v = \alpha$.

**Result:** The direction of maximum projected variance is the eigenvector of $S$
corresponding to the largest eigenvalue $\lambda_1$. The variance captured is $\lambda_1$.

### 3.2 Multiple components

For $k$ components, maximize $\operatorname{tr}(W_k^\top S W_k)$ subject to
$W_k^\top W_k = I_k$.

By the **Rayleigh–Ritz theorem**, the solution is $W_k = [v_1, \ldots, v_k]$, the top
$k$ eigenvectors of $S$.

**Result:** the first $k$ principal directions are the top-$k$ eigenvectors of the
covariance matrix, and the $j$-th component captures variance $\lambda_j$.

## 4. Reconstruction View

### 4.1 Projection and reconstruction

Given $W_k$, the rank-$k$ approximation of $X_c$ is:

$$
\hat{X}_c = X_c W_k W_k^\top
$$

Each row is projected onto $\operatorname{col}(W_k)$, then embedded back into
$\mathbb{R}^d$.

### 4.2 Reconstruction error

$$
\|X_c - \hat{X}_c\|_F^2
= \sum_{j=k+1}^{d} \lambda_j \cdot n
$$

Minimizing squared reconstruction error is equivalent to maximizing captured variance.

**Result:** PCA provides the best linear rank-$k$ reconstruction under squared error.

## 5. SVD Connection

### 5.1 Thin SVD of centered data

$$
X_c = U \Sigma V^\top
$$

where $U \in \mathbb{R}^{n \times r}$, $\Sigma = \operatorname{diag}(\sigma_1, \ldots, \sigma_r)$,
$V \in \mathbb{R}^{d \times r}$, $r = \operatorname{rank}(X_c)$.

### 5.2 Covariance in terms of SVD

$$
S = \frac{1}{n}X_c^\top X_c
  = V \frac{\Sigma^2}{n} V^\top
$$

This is already an eigendecomposition: $V$ contains eigenvectors, and $\lambda_j = \sigma_j^2/n$.

**Result:** PCA directions are the right singular vectors of $X_c$. There is no need
to form $S$ explicitly — computing the SVD of $X_c$ directly is numerically more stable.

### 5.3 Explained variance ratio

$$
\text{EVR}_j = \frac{\lambda_j}{\sum_{m=1}^{d} \lambda_m}
= \frac{\sigma_j^2}{\sum_{m=1}^{d} \sigma_m^2}
$$

The cumulative EVR guides the choice of $k$: retain components until the total
exceeds a threshold (e.g. 95%).

## 6. Identifiability

Eigenvectors are unique only up to sign: if $v$ is an eigenvector, so is $-v$.
When eigenvalues coincide ($\lambda_j = \lambda_{j+1}$), any rotation within the
eigenspace is valid. **Consequence:** compare PCA solutions by subspace alignment, not
by raw loading values.

## 7. Failure Cases

1. **Scale dominance.** If one feature is measured in meters and another in millimeters,
   the large-unit feature dominates variance. PCA without standardization reflects
   units, not structure.

2. **Outlier sensitivity.** PCA uses $\ell_2$ (squared) distances. A single extreme
   observation can rotate the first PC toward itself.

3. **Linearity.** PCA finds a linear subspace. If the data lie on a curved manifold
   (e.g. a Swiss roll), PCA may fail to capture the intrinsic low-dimensional structure.
   Nonlinear extensions: kernel PCA, t-SNE, UMAP.

4. **No label information.** PCA is unsupervised. The directions of maximum variance
   need not align with class-discriminative directions.

5. **Sample size.** With $n < d$, at most $n - 1$ eigenvalues are nonzero. Estimated
   eigenvalues are biased: the largest are overestimated, the smallest underestimated.

## 8. Connections

- [Linear Algebra Foundations](../../foundations/linear_algebra/)
- [Geometry of ML](../../synthesis/geometry_of_ml.md)
- [Autoencoders](../17_autoencoder/README.md) (nonlinear PCA analogy)
- [Regularization](../03_regularization/README.md) (PCA used for denoising / pre-processing)

---

## 9. References

- **Pearson, K. (1901).** On lines and planes of closest fit to systems of points in space. *The London, Edinburgh, and Dublin Philosophical Magazine and Journal of Science*, 2(11), 559–572.
- **Hotelling, H. (1933).** Analysis of a complex of statistical variables into principal components. *Journal of Educational Psychology*, 24(6), 417–441.
- **Jolliffe, I. T. (2002).** *Principal Component Analysis* (2nd ed.). Springer Series in Statistics.
- **Hastie, T., Tibshirani, R., & Friedman, J. (2009).** *The Elements of Statistical Learning: Data Mining, Inference, and Prediction* (2nd ed.). Springer. Chapter 14.5: *Principal Components Analysis*.

