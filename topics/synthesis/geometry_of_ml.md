# Geometry of Machine Learning — Cross-Topic Synthesis

> One curriculum, one picture: subspaces, distances, margins, norm balls, and cells.
> See [INDEX.md](../../INDEX.md) for the full curriculum index.

---

## Overview

Most algorithms in this repo are geometric statements in disguise: least squares is a
projection, regularization is a constraint region, an SVM is a distance maximizer, PCA
is a best-fit subspace, KNN and K-Means tile space into cells, and kernels relocate all
of this into an implicit feature space. This page collects the pictures; the topic
`theory.md` files carry the full derivations.

## Shared Geometric Objects

| Object | Algorithms | Interpretation |
|---|---|---|
| Orthogonal projection | Linear Regression, PCA | Closest point in a linear subspace |
| Norm ball | Ridge, Lasso, SVM | Feasible or penalized parameter geometry |
| Pairwise distance | KNN, K-Means, DBSCAN | Local similarity in feature space |
| Hyperplane and margin | Logistic Regression, SVM | Linear decision boundary and confidence geometry |
| Voronoi / piecewise regions | Trees, KNN, K-Means | Partition of input space into cells |
| Inner product (kernel) | Kernel SVM, Kernel PCA | Geometry of an implicit feature space |
| Learned representation | Neural Networks, Autoencoders | Geometry induced by a parameterized map |

---

## Least Squares as Projection

OLS finds the point in the column space of $X$ closest to $y$:
$\hat{y}=X\hat{w}$ is the orthogonal projection of $y$ onto
$\operatorname{col}(X)$. The projector is the hat matrix

```math
H=X(X^\top X)^{-1}X^\top,\qquad \hat{y}=Hy,
```

with $H^2=H$ and $H^\top=H$: projecting twice changes nothing, and the residual
$r=y-\hat{y}=(I-H)y$ is orthogonal to every column of $X$ — that orthogonality *is* the
normal equations. Two useful corollaries:

- $\operatorname{tr}(H)=d$ counts the model's degrees of freedom; ridge shrinks this
  trace below $d$, which is one way to see ridge as capacity control.
- Adding a column to $X$ enlarges the subspace, so training error can only decrease —
  the geometric root of overfitting in [Bias–Variance](bias_variance_tradeoff.md).

Recall: [Linear Regression theory](../01_linear_regression/theory.md).

---

## Regularization as Constrained Regions

The penalized problem $\min_w L(w)+\lambda R(w)$ corresponds (under regularity
conditions) to minimizing $L$ over the region $\lbrace w : R(w)\le t\rbrace$. The
solution sits where the elliptical level sets of $L$ first touch that region — so the
region's *shape* decides the solution's character:

| Penalty | Region | First contact | Consequence |
|---|---|---|---|
| $\ell_2$: $\Vert w\Vert_2\le t$ | Ball (smooth, rotationally symmetric) | Generic tangency point | All coordinates shrink, none exactly zero |
| $\ell_1$: $\Vert w\Vert_1\le t$ | Cross-polytope (diamond) with corners on axes | Often a corner or edge | Coordinates at the corner are exactly zero → sparsity |
| Elastic Net | Rounded diamond | Between the two | Sparsity with grouped selection |

Corners are low-dimensional faces where some $w_j=0$; because they protrude toward the
loss contours, they are touched with positive probability — the entire geometric story
of Lasso's sparsity. Full picture with figures:
[Regularization theory](../03_regularization/theory.md).

---

## SVM Margins as Distance Geometry

The signed distance from a point $x$ to the hyperplane $w^\top x+b=0$ is
$(w^\top x+b)/\Vert w\Vert_2$. Fixing the canonical scale
$y_i(w^\top x_i+b)\ge 1$ makes the margin width $2/\Vert w\Vert_2$, so *maximizing the
margin is minimizing a norm* — the same object as ridge's penalty, deployed as distance
rather than shrinkage. Support vectors are exactly the points achieving the minimal
distance; the boundary is determined by them alone, which is why the solution is a
sparse combination of training points. Recall: [SVM theory](../09_svm/theory.md).

---

## PCA as Best-Fit Subspace

PCA finds the $k$-dimensional subspace minimizing squared reconstruction error of the
centered data — equivalently, maximizing the variance of the projection:

```math
\min_{\substack{V\in\mathbb{R}^{d\times k}\\ V^\top V=I}}
\sum_{i=1}^{n}\Vert x_i-VV^\top x_i\Vert_2^2
\quad\Longleftrightarrow\quad
\max_{V^\top V=I}\operatorname{tr}\!\left(V^\top \Sigma V\right).
```

The equivalence is Pythagoras: $\Vert x\Vert^2 = \Vert VV^\top x\Vert^2 + \Vert x-VV^\top x\Vert^2$,
so minimizing the residual is maximizing the projection. Same projection machinery as
least squares, but the roles differ: regression projects the *target* $y$ onto
$\operatorname{col}(X)$ (errors measured vertically), PCA projects the *rows* of $X$
onto a subspace of feature space (errors measured perpendicular to the subspace).
Recall: [PCA theory](../10_pca/theory.md).

---

## KNN and K-Means: Voronoi Geometry

Both methods tile feature space into cells of the form
$\lbrace x : \Vert x-c_j\Vert \le \Vert x-c_l\Vert \ \forall l\rbrace$:

- **1-NN:** every training point owns its Voronoi cell; the decision boundary is the
  union of cell faces between opposite-class points — piecewise linear, arbitrarily
  wiggly, zero training error (the high-variance extreme of
  [Bias–Variance](bias_variance_tradeoff.md)).
- **K-Means:** the assignment step is exactly Voronoi assignment to centroids; cells
  are convex polyhedra, which is why K-Means can only carve convex clusters and fails
  on rings and moons. The update step moves each centroid to its cell's mean — Lloyd's
  algorithm alternates geometry (assign) and averaging (update).

Distance-based methods inherit every decision made *before* distance is measured:
units, scaling, and irrelevant dimensions reshape the Voronoi diagram without touching
the algorithm. Recall: [KNN](../07_knn/theory.md),
[Clustering](../11_clustering/theory.md).

---

## Kernels: Implicit Feature-Space Geometry

A kernel $k(x,x')=\langle\phi(x),\phi(x')\rangle$ gives inner products in a feature
space without visiting it. Since inner products determine all Euclidean geometry,
distances and angles come for free:

```math
\Vert\phi(x)-\phi(x')\Vert^2 = k(x,x)-2k(x,x')+k(x',x'),
```

so any algorithm expressible through distances or inner products — SVM, PCA, K-Means,
ridge — can run in the implicit space. A linear hyperplane there pulls back to a curved
boundary in input space: the kernel trick is a *geometry transplant*, not a new
algorithm. The RBF kernel's width $\gamma$ sets how fast similarity decays, i.e. the
resolution of the implicit geometry. Recall: [SVM theory](../09_svm/theory.md),
[Dimensionality Reduction](../12_dimensionality_reduction/theory.md) (kernel PCA).

---

## Curse of Dimensionality: Distances Concentrate

In high dimension, i.i.d.-feature geometry becomes uniform. For random points with
independent coordinates, pairwise distances concentrate around their mean:

```math
\frac{\max_i \operatorname{dist}(x, x_i) - \min_i \operatorname{dist}(x, x_i)}
     {\min_i \operatorname{dist}(x, x_i)} \;\longrightarrow\; 0
\quad\text{as } d\to\infty,
```

so "nearest" neighbor loses meaning — every point is nearly equidistant. Companion
facts: nearly all the volume of a high-dimensional ball lies in a thin shell near its
surface, and a hypercube's mass hides in its corners.

| Method | Failure mode in high $d$ |
|---|---|
| KNN | Neighbors no closer than strangers; $k$ cannot fix it |
| K-Means | Cluster distances wash out; inertia uninformative |
| RBF kernels | $k(x,x')$ nearly constant → Gram matrix close to $I$ |
| Density estimation | Exponentially many samples needed per unit volume |

The practical escape is dimensionality reduction — projecting onto the low-dimensional
structure real data usually has ([PCA](../10_pca/README.md),
[Autoencoders](../17_autoencoder/README.md)) — or models that do not rely on raw
distances (trees, linear models).

---

## Connections

- **Topics:** [01 Linear Regression](../01_linear_regression/README.md), [03 Regularization](../03_regularization/README.md), [07 KNN](../07_knn/README.md), [09 SVM](../09_svm/README.md), [10 PCA](../10_pca/README.md), [11 Clustering](../11_clustering/README.md), [12 Dimensionality Reduction](../12_dimensionality_reduction/README.md)
- **Foundations:** [Linear Algebra](https://github.com/hien078/applied-mathematics-foundation)
- **Related synthesis:** [Bias–Variance Trade-off](bias_variance_tradeoff.md), [Regularization Across Models](regularization_across_models.md)
- **Maps:** [INDEX.md](../../INDEX.md)
