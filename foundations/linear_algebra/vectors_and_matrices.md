# Vectors and Matrices for Machine Learning

## Vectors, Spaces, and Dimensions

A feature vector $x\in\mathbb R^d$ has $d$ coordinates relative to a chosen basis. A
dataset with $n$ rows and $d$ features is represented by $X\in\mathbb R^{n\times d}$.
Writing dimensions is not cosmetic: it prevents invalid products and exposes whether a
formula acts across samples or features.

For $A\in\mathbb R^{m\times n}$, the column space is

$$\operatorname{col}(A)=\{Ax:x\in\mathbb R^n\}\subseteq\mathbb R^m,$$

and the null space is

$$\operatorname{null}(A)=\{x\in\mathbb R^n:Ax=0\}.$$

The rank–nullity theorem gives
$\operatorname{rank}(A)+\dim\operatorname{null}(A)=n$.

## Matrix Operations

**Transpose.** $(A^\top)_{ij} = A_{ji}$. Key identity: $(AB)^\top = B^\top A^\top$. In ML,
transposing switches between row-vectors and column-vectors, and appears in every
gradient formula involving matrix products.

**Trace.** $\operatorname{tr}(A) = \sum_i A_{ii}$. Useful identities:
- $\operatorname{tr}(AB) = \operatorname{tr}(BA)$ (cyclic property)
- $x^\top A x = \operatorname{tr}(Axx^\top)$ — converts scalar quadratic forms to matrix traces

The trace appears in Frobenius norm ($\lVert A \rVert_F^2 = \operatorname{tr}(A^\top A)$)
and in loss functions like $\operatorname{tr}(\Sigma)$ for total variance in PCA.

**Determinant.** $\det(A)$ measures the volume scaling of the linear map $A$.
- $\det(A) = 0$ means $A$ is singular (not invertible)
- $\det(AB) = \det(A)\det(B)$
- $\det(A^\top) = \det(A)$

In ML, determinants appear in Gaussian density functions and in the log-determinant
for covariance matrix estimation.

**Inverse.** $A^{-1}$ exists only when $\det(A) \neq 0$ (i.e., $A$ is square and full rank).
Key identity: $(AB)^{-1} = B^{-1}A^{-1}$.

> **Practical rule:** never compute $A^{-1}b$ explicitly. Use `numpy.linalg.solve(A, b)`
> instead — it is faster and numerically more stable.

## Inner Products and Orthogonality

The Euclidean inner product is $\langle x,y\rangle=x^\top y$. Vectors are orthogonal
when $x^\top y=0$. For nonzero vectors,

$$\cos\theta=\frac{x^\top y}{\lVert x\rVert_2\lVert y\rVert_2}.$$

This is **cosine similarity**, widely used in NLP and recommendation systems to
measure the angle between feature vectors regardless of their magnitude.

Orthogonality is central to least-squares residuals, PCA components, and stable matrix
factorizations.

## Special Matrix Types

| Type | Definition | Key property | ML usage |
|---|---|---|---|
| Symmetric | $A = A^\top$ | Real eigenvalues, orthogonal eigenvectors | Covariance matrices, Hessians |
| Diagonal | $A_{ij} = 0$ for $i \neq j$ | Eigenvalues = diagonal entries | Feature scaling, variance |
| Orthogonal | $Q^\top Q = I$ | Preserves lengths and angles | QR factorization, rotations |
| Positive definite | $x^\top A x > 0$ for all $x \neq 0$ | All eigenvalues $> 0$ | Convex loss, unique minimum |

**Symmetric matrices** are the most important in ML: every covariance matrix
$\Sigma = \frac{1}{n}X^\top X$ and every Hessian $\nabla^2 f$ is symmetric.

**Orthogonal matrices** preserve geometry: if $Q$ is orthogonal,
$\lVert Qx \rVert = \lVert x \rVert$. This is why QR decomposition is numerically stable.

## Linear Systems and Least Squares

The system $Ax=b$ has a unique solution only when the relevant linear map is invertible.
If $b\notin\operatorname{col}(A)$, least squares finds

$$x^*\in\arg\min_x\lVert Ax-b\rVert_2^2.$$

The residual $r=b-Ax^*$ is orthogonal to every column of $A$, so
$A^\top r=0$. This gives the **normal equations** $A^\top A x^* = A^\top b$,
but a numerical implementation should use
QR or SVD through `numpy.linalg.lstsq` rather than explicitly forming $(A^\top A)^{-1}$.

> **Why avoid the inverse?** When $A$ is ill-conditioned ($\kappa(A) \gg 1$), forming
> $A^\top A$ squares the condition number, amplifying numerical errors catastrophically.

## Eigenvalues and Eigenvectors

For a square matrix $A \in \mathbb{R}^{n \times n}$, a scalar $\lambda$ and nonzero
vector $v$ satisfying

$$Av = \lambda v$$

are called an **eigenvalue** and **eigenvector** of $A$.

**Geometric intuition:** $A$ stretches the direction $v$ by factor $\lambda$.
- $\lambda > 0$: same direction
- $\lambda < 0$: reversed direction
- $|\lambda| > 1$: expansion
- $|\lambda| < 1$: contraction

**Spectral theorem:** Every real symmetric matrix $A$ has a decomposition
$A = Q\Lambda Q^\top$, where $Q$ is orthogonal (eigenvectors as columns) and
$\Lambda = \operatorname{diag}(\lambda_1, \ldots, \lambda_n)$.

**ML significance:**
- **PCA:** eigenvectors of the covariance matrix = principal components; eigenvalues = explained variance
- **Gradient descent:** eigenvalues of the Hessian control convergence speed (see [gradient_descent.ipynb](../calculus_optimization/gradient_descent.ipynb))
- **Condition number:** $\kappa(A) = \lambda_{\max}/\lambda_{\min}$ — measures how sensitive a linear system is to perturbations

## Positive Semidefinite Matrices

A symmetric matrix $M$ is **positive semidefinite** (PSD, written $M \succeq 0$)
when $x^\top Mx\ge0$ for every $x$. Equivalently, all eigenvalues of $M$ are $\geq 0$.

A matrix is **positive definite** ($M \succ 0$) when $x^\top Mx > 0$ for all $x \neq 0$,
equivalently all eigenvalues are strictly positive.

| Property | PSD ($M \succeq 0$) | PD ($M \succ 0$) |
|---|---|---|
| Eigenvalues | $\lambda_i \geq 0$ | $\lambda_i > 0$ |
| Invertible? | Not necessarily | Always |
| Quadratic $f(x) = x^\top M x$ | Convex | Strictly convex |
| Minimum of $f$? | May have many | Unique |

**Key examples in ML:**
- Gram matrices $X^\top X$ are always PSD
- Covariance matrices $\Sigma$ are PSD
- The Hessian $\nabla^2 f$ at a local minimum is PSD
- Positive definiteness of $X^\top X$ guarantees a unique least-squares solution

## ML Connections

- [Linear Regression](../../topics/01_linear_regression/README.md): projections and rank
- [KNN](../../topics/07_knn/README.md): norms and feature geometry
- [SVM](../../topics/09_svm/README.md): hyperplanes and margins
- [PCA](../../topics/10_pca/README.md): eigenvectors and SVD
- [Neural Networks](../../topics/13_neural_networks/README.md): matrix transformations
