# Dimensionality Reduction Beyond PCA — Theory

## 1. WHY — Supervised Separation and Nonlinear Visualization

PCA finds linear directions of maximum variance — it ignores labels entirely. Two
common situations where PCA falls short:

1. **Supervised classification.** Class-discriminative directions may not align with
   high-variance directions. A feature with large variance can be pure noise across
   classes, while a low-variance feature perfectly separates them.

2. **Nonlinear manifolds.** Real data often lies on curved surfaces. PCA can only find
   flat subspaces, so it distorts local neighborhoods on a manifold.

**LDA** (Linear Discriminant Analysis) addresses (1) by maximizing between-class
separation relative to within-class spread. **t-SNE** addresses (2) by preserving local
pairwise similarities through a nonlinear, non-parametric embedding.

## 2. WHAT — Notation and Setup

### 2.1 Common Notation

| Symbol | Type | Meaning |
|---|---:|---|
| $X \in \mathbb{R}^{n \times d}$ | matrix | data matrix, $n$ samples, $d$ features |
| $y_i \in \lbrace1, \ldots, C\rbrace$ | scalar | class label of sample $i$ |
| $n_k$ | scalar | number of samples in class $k$ |
| $\mu$ | vector | global mean $\frac{1}{n}\sum_i x_i$ |
| $\mu_k$ | vector | mean of class $k$: $\frac{1}{n_k}\sum_{i: y_i=k} x_i$ |
| $S_W$ | matrix | within-class scatter matrix |
| $S_B$ | matrix | between-class scatter matrix |
| $w \in \mathbb{R}^d$ | vector | projection direction |

### 2.2 LDA Assumptions

1. **Labeled data** with $C \geq 2$ known classes.
2. **Gaussian class-conditional distributions** $x \mid y=k \sim \mathcal{N}(\mu_k, \Sigma_k)$.
3. **Equal covariance** across classes: $\Sigma_1 = \cdots = \Sigma_C = \Sigma$ (homoscedasticity).
4. **Non-singular** $S_W$ (requires $n > d$ or regularization).

### 2.3 t-SNE Notation

| Symbol | Type | Meaning |
|---|---:|---|
| $p_{j \mid i}$ | scalar | conditional similarity of $x_j$ to $x_i$ in high-d |
| $p_{ij}$ | scalar | symmetrized joint similarity in high-d |
| $q_{ij}$ | scalar | joint similarity in low-d (Student-t kernel) |
| $z_i \in \mathbb{R}^m$ | vector | low-d embedding of point $i$ (typically $m = 2$) |
| $\sigma_i$ | scalar | bandwidth of Gaussian kernel around $x_i$ |
| $\text{Perp}$ | scalar | perplexity parameter (effective number of neighbors) |

## 3. HOW — Fisher's Linear Discriminant Analysis

### 3.1 Scatter Matrices

**Within-class scatter** measures how spread out each class is internally:

```math
S_W = \sum_{k=1}^{C} \sum_{i:\, y_i = k} (x_i - \mu_k)(x_i - \mu_k)^\top
```

**Between-class scatter** measures how separated the class means are:

$$S_B = \sum_{k=1}^{C} n_k (\mu_k - \mu)(\mu_k - \mu)^\top$$

**Total scatter** decomposes as $S_T = S_W + S_B$, where
$S_T = \sum_{i=1}^{n} (x_i - \mu)(x_i - \mu)^\top$.

### 3.2 Fisher's Criterion

Project all data onto direction $w$. The projected class mean is $\tilde{\mu}_k = w^\top \mu_k$
and the projected within-class variance is $\tilde{s}_k^2 = w^\top S_W^{(k)} w$.
Fisher seeks the direction that maximizes class separation relative to within-class spread:

$$J(w) = \frac{w^\top S_B w}{w^\top S_W w}$$

This is a **generalized Rayleigh quotient**. Taking the gradient and setting to zero:

```math
\nabla_w J = 0 \implies S_B w = J(w)\, S_W w
```

which is the generalized eigenvalue problem:

$$S_W^{-1} S_B w = \lambda w$$

**Result:** The optimal LDA directions are the eigenvectors of $S_W^{-1} S_B$
corresponding to the largest eigenvalues. At most $\min(d, C-1)$ directions are
non-trivial because $\text{rank}(S_B) \leq C - 1$.

### 3.3 Two-Class Special Case

For $C = 2$, $S_B = n_1 n_2 / n \cdot (\mu_1 - \mu_2)(\mu_1 - \mu_2)^\top$ has rank 1.
The single discriminant direction (up to scale) is:

$$w^\ast = S_W^{-1}(\mu_1 - \mu_2)$$

**Result:** In the two-class case, LDA reduces to a single closed-form direction —
no eigendecomposition needed.

### 3.4 Regularized LDA

When $d > n$ or features are highly collinear, $S_W$ is singular. Add regularization:

$$\tilde{S}_W = S_W + \alpha I$$

for a small $\alpha > 0$. This is equivalent to assuming a small isotropic noise floor
in each class.

## 4. HOW — t-SNE

### 4.1 High-Dimensional Similarities

For each pair $(i, j)$, define a conditional probability that $x_j$ is a neighbor of $x_i$
under a Gaussian centered at $x_i$:

```math
p_{j \mid i} = \frac{\exp\bigl(-\|x_i - x_j\|^2 / 2\sigma_i^2\bigr)}{\sum_{k \neq i} \exp\bigl(-\|x_i - x_k\|^2 / 2\sigma_i^2\bigr)}
```

Set $p_{i \mid i} = 0$. The bandwidth $\sigma_i$ is chosen so that the entropy of the
conditional distribution $P_i$ matches $\log(\text{Perp})$, where $\text{Perp}$ is the
user-specified perplexity (typically 5–50). Higher perplexity → larger effective
neighborhood → more global structure preserved.

Symmetrize: $p_{ij} = \frac{p_{j \mid i} + p_{i \mid j}}{2n}$.

### 4.2 Low-Dimensional Similarities (Student-t Kernel)

In the embedding space, use a Student-t distribution with 1 degree of freedom
(i.e., a Cauchy kernel):

```math
q_{ij} = \frac{(1 + \|z_i - z_j\|^2)^{-1}}{\sum_{k \neq l} (1 + \|z_k - z_l\|^2)^{-1}}
```

**Why Student-t?** In high dimensions, moderate-distance points become nearly equidistant
(concentration of measure). In low dimensions, there is less "room" to place moderately
distant points — this is the **crowding problem**. The heavy tails of the Student-t
distribution allow moderately distant high-d neighbors to be placed further apart in
low-d without incurring a large cost, while still keeping true neighbors close.

### 4.3 KL Divergence Objective

t-SNE minimizes:

```math
C = D_{\text{KL}}(P \| Q) = \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}}
```

Because KL divergence is asymmetric:
- **Large $p_{ij}$, small $q_{ij}$** (nearby points mapped far apart): **high cost** —
  local structure is preserved.
- **Small $p_{ij}$, large $q_{ij}$** (distant points mapped close): **low cost** —
  global distances are not enforced.

This is why t-SNE preserves local neighborhoods but not global geometry.

### 4.4 Gradient and Attraction/Repulsion Forces

The gradient of $C$ with respect to $z_i$ is:

```math
\frac{\partial C}{\partial z_i} = 4 \sum_{j \neq i} (p_{ij} - q_{ij})(1 + \|z_i - z_j\|^2)^{-1}(z_i - z_j)
```

This decomposes into two forces on each embedded point:

- **Attraction** (when $p_{ij} > q_{ij}$): point $z_j$ pulls $z_i$ closer — neighbors
  in high-d that are too far apart in low-d.
- **Repulsion** (when $p_{ij} < q_{ij}$): point $z_j$ pushes $z_i$ away — non-neighbors
  in high-d that are too close in low-d.

**Result:** t-SNE performs gradient descent on the KL divergence. At convergence,
attractions and repulsions balance: nearby high-d points cluster together in low-d,
while distant points spread apart.

### 4.5 Perplexity

Perplexity controls the effective number of neighbors:

$$\text{Perp}(P_i) = 2^{H(P_i)}, \quad H(P_i) = -\sum_{j} p_{j \mid i} \log_2 p_{j \mid i}$$

- **Low perplexity** (5–10): focus on very local structure; may fragment clusters.
- **High perplexity** (30–50): more global context; clusters become rounder.
- **Too high perplexity** ($\geq n$): similarities become nearly uniform; embedding
  becomes meaningless.

## 5. Comparison: PCA vs LDA vs t-SNE

| Property | PCA | LDA | t-SNE |
|---|---|---|---|
| Supervision | Unsupervised | Supervised | Unsupervised* |
| Objective | Max variance | Max class separation | Preserve local neighborhoods |
| Linearity | Linear | Linear | Nonlinear |
| Parametric | Yes ($W_k$) | Yes ($W$) | No (no explicit mapping) |
| Max components | $\min(n, d)$ | $C - 1$ | Any (usually 2–3) |
| Preserves | Global variance | Between-class / within-class ratio | Local pairwise distances |
| Invertible | Yes (reconstruction) | Approximately | No |

\* t-SNE can use labels for coloring but does not optimize over them.

## 6. Failure Cases

### 6.1 LDA Failures

1. **Non-Gaussian classes.** LDA assumes Gaussian class-conditionals. If classes are
   multi-modal or have different shapes, the scatter matrices misrepresent the structure.

2. **Singular $S_W$.** When $n < d$ or features are linearly dependent, $S_W$ is rank-
   deficient and $S_W^{-1}$ does not exist. **Fix:** regularize ($S_W + \alpha I$) or
   use PCA pre-processing to reduce to $n - 1$ dimensions first.

3. **At most $C - 1$ directions.** For binary classification, LDA gives exactly one
   discriminant direction — insufficient for complex visualization.

4. **Homoscedasticity violation.** If class covariances differ substantially, the pooled
   $S_W$ is a poor summary and LDA may give suboptimal separation.

### 6.2 t-SNE Failures

1. **Stochastic output.** Different random initializations yield different embeddings.
   Conclusions must be robust across runs.

2. **Perplexity sensitivity.** The visual appearance of clusters depends heavily on
   perplexity. Always try multiple values (e.g., 5, 15, 30, 50).

3. **No global structure.** Distances between clusters and relative cluster sizes in a
   t-SNE plot are not meaningful. Two clusters far apart in the plot may or may not be
   far apart in the original space.

4. **Crowding problem.** Without the heavy-tailed kernel, moderate-distance points in
   high-d crush together in low-d. The Student-t kernel mitigates but does not eliminate
   this for very high-dimensional data.

5. **Non-parametric.** t-SNE does not produce a mapping function. New points cannot be
   projected without re-running the full optimization (or using parametric extensions).

6. **Computational cost.** Naive t-SNE is $O(n^2)$ per iteration (pairwise distances).
   Barnes–Hut approximation reduces this to $O(n \log n)$.

## 7. Connections

- **[PCA](../10_pca/README.md):** Unsupervised linear baseline; LDA extends PCA with
  label information; t-SNE provides a nonlinear alternative.
- **[Information Theory](https://github.com/hien078/applied-mathematics-foundation):** t-SNE
  minimizes KL divergence; perplexity is defined via entropy.
- **[Clustering](../11_clustering/README.md):** t-SNE is often used to visualize
  cluster structure (but does not itself cluster).
- **[Autoencoder](../17_autoencoder/README.md):** Nonlinear parametric dimensionality
  reduction; can be seen as a learnable t-SNE alternative.
- **[Geometry of ML](../../synthesis/geometry_of_ml.md):** Subspace projection, manifold
  embedding, and metric preservation are geometric themes shared across PCA, LDA, and t-SNE.

---

## 8. References

- **Fisher, R. A. (1936).** The use of multiple measurements in taxonomic problems. *Annals of Eugenics*, 7(2), 179–188.
- **van der Maaten, L., & Hinton, G. (2008).** Visualizing data using t-SNE. *Journal of Machine Learning Research*, 9, 2579–2605.
- **Hastie, T., Tibshirani, R., & Friedman, J. (2009).** *The Elements of Statistical Learning: Data Mining, Inference, and Prediction* (2nd ed.). Springer. Chapter 4.3: *Linear Discriminant Analysis*.

