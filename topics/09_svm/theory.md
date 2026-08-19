# Support Vector Machines — Theory

## 0. Notation

| Symbol | Type | Meaning |
|---|---|---|
| $n$ | scalar | number of training examples |
| $p$ | scalar | number of features |
| $x_i$ | vector of length $p$ | feature vector of the $i$-th example |
| $X$ | matrix of size $n \times p$ | design matrix; row $i$ is $x_i^T$ |
| $y_i$ | scalar in $\lbrace-1, +1\rbrace$ | binary label of the $i$-th example |
| $w$ | vector of length $p$ | weight vector (normal to the separating hyperplane) |
| $b$ | scalar | bias (offset) term |
| $f(x)$ | scalar | decision function: $f(x) := w^T x + b$ |
| $\xi_i$ | scalar $\ge 0$ | slack variable for example $i$ (soft-margin) |
| $C$ | scalar $> 0$ | regularization parameter controlling margin–violation trade-off |
| $\alpha_i$ | scalar $\ge 0$ | dual variable (Lagrange multiplier) for example $i$ |
| $K(x, x')$ | scalar | kernel function evaluating the inner product in feature space |
| $\phi(x)$ | vector | feature map induced by a kernel |
| $\lambda$ | scalar $\ge 0$ | regularization strength in the hinge-loss formulation $(\lambda = 1/(2nC))$ |

**Label convention.** SVM uses $y_i \in \lbrace-1, +1\rbrace$, not $\lbrace0, 1\rbrace$.

The *functional margin* of example $i$ is $y_i f(x_i) = y_i (w^T x_i + b)$. Correct
classification means $y_i f(x_i) > 0$.

---

## 1. WHY — Maximum Margin Principle

Given linearly separable data, infinitely many hyperplanes separate the classes. Which
one should we choose?

**Intuition.** A hyperplane that barely squeezes between two close points from different
classes is fragile — a small perturbation in the data moves it. A hyperplane with a *wide
gap* (margin) on both sides is robust to noise and generalizes better.

**Geometric margin.** The hyperplane $w^T x + b = 0$ divides space into two half-spaces.
The signed distance from a point $x_i$ to the hyperplane is

```math
d_i = \frac{w^T x_i + b}{\|w\|_2}.
```

The *margin* is the distance from the hyperplane to the nearest point on either side.
For correctly classified data ($y_i f(x_i) > 0$), the geometric margin is

```math
\gamma = \min_{i=1,\dots,n} \frac{y_i(w^T x_i + b)}{\|w\|_2}.
```

SVM maximizes this quantity.

---

## 2. WHAT — Hard-Margin SVM

### 2.1 Scale invariance and canonical form

Since $w$ and $b$ can be rescaled arbitrarily (multiplying both by a constant $k > 0$ does
not change the hyperplane), we fix the scale by requiring

$$\min_{i=1,\dots,n} y_i(w^T x_i + b) = 1.$$

Under this convention, the closest points satisfy $y_i(w^T x_i + b) = 1$ and lie on the
*margin boundaries* $w^T x + b = +1$ and $w^T x + b = -1$. The distance between these
two parallel hyperplanes is

```math
\text{margin} = \frac{2}{\|w\|_2}.
```

**Proof.** Pick $x_+$ on $w^T x + b = +1$ and $x_-$ on $w^T x + b = -1$.

Their difference projected onto the unit normal $w / \Vert w\Vert_2$ gives
$(w^T(x_+ - x_-)) / \Vert w\Vert_2 = 2 / \Vert w\Vert_2$. $\blacksquare$

### 2.2 Optimization problem

Maximizing $2/\Vert w\Vert_2$ is equivalent to minimizing $\frac{1}{2}\Vert w\Vert_2^2$.

**Result:** The hard-margin SVM solves

```math
\min_{w, b} \quad \frac{1}{2} \|w\|_2^2 \quad \text{s.t.} \quad y_i(w^T x_i + b) \ge 1, \quad i = 1, \dots, n. \qquad (2.1)
```

This is a *convex quadratic program* (QP) — the objective is convex quadratic and the
constraints are linear.

**Assumption:** The data must be perfectly linearly separable for feasibility.

---

## 3. WHAT — Soft-Margin SVM

Real data is rarely perfectly separable. The soft-margin SVM introduces *slack variables*
$\xi_i \ge 0$ to allow violations:

```math
\min_{w, b, \xi} \quad \frac{1}{2} \|w\|_2^2 + C \sum_{i=1}^n \xi_i \quad \text{s.t.} \quad y_i(w^T x_i + b) \ge 1 - \xi_i, \quad \xi_i \ge 0. \qquad (3.1)
```

**Interpretation of $\xi_i$:**
- $\xi_i = 0$: point is on or beyond the correct margin boundary.
- $0 < \xi_i < 1$: point is inside the margin but on the correct side.
- $\xi_i \ge 1$: point is misclassified.

**Role of $C$:**
- Large $C$ → penalizes violations heavily → narrow margin, fewer violations.
- Small $C$ → allows more violations → wider margin, more regularization.

---

## 4. HOW — Dual Formulation

### 4.1 Lagrangian

Introduce Lagrange multipliers $\alpha_i \ge 0$ for the margin constraints and $\mu_i \ge 0$
for $\xi_i \ge 0$:

```math
\mathcal{L}(w, b, \xi, \alpha, \mu) = \frac{1}{2}\|w\|^2 + C\sum_i \xi_i - \sum_i \alpha_i \big[y_i(w^T x_i + b) - 1 + \xi_i\big] - \sum_i \mu_i \xi_i.
```

Each constraint of (3.1) is now priced by its own multiplier, so the constrained problem
becomes the search for a stationary point of $\mathcal{L}$.

### 4.2 Stationarity conditions

Set the partial derivatives to zero:

**w.r.t. $w$:**

```math
\frac{\partial \mathcal{L}}{\partial w} = w - \sum_i \alpha_i y_i x_i = 0 \implies w = \sum_{i=1}^n \alpha_i y_i x_i. \qquad (4.1)
```

**w.r.t. $b$:**

```math
\frac{\partial \mathcal{L}}{\partial b} = -\sum_i \alpha_i y_i = 0 \implies \sum_{i=1}^n \alpha_i y_i = 0. \qquad (4.2)
```

**w.r.t. $\xi_i$:**

```math
\frac{\partial \mathcal{L}}{\partial \xi_i} = C - \alpha_i - \mu_i = 0 \implies \alpha_i + \mu_i = C. \qquad (4.3)
```

Since $\mu_i \ge 0$, equation (4.3) gives $0 \le \alpha_i \le C$.

### 4.3 Dual problem

Substitute (4.1), (4.2), (4.3) back into the Lagrangian. After simplification:

```math
\|w\|^2 = \left(\sum_i \alpha_i y_i x_i\right)^T \left(\sum_j \alpha_j y_j x_j\right) = \sum_i \sum_j \alpha_i \alpha_j y_i y_j x_i^T x_j.
```

The Lagrangian becomes the *Wolfe dual*:

$$\max_\alpha \quad W(\alpha) = \sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j y_i y_j x_i^T x_j$$

$$\text{s.t.} \quad \sum_{i=1}^n \alpha_i y_i = 0, \quad 0 \le \alpha_i \le C, \quad i = 1, \dots, n. \qquad (4.4)$$

**Result:** The dual (4.4) is a convex QP in $n$ variables. The data enters *only* through
inner products $x_i^T x_j$ — this is the gateway to the kernel trick (§7).

---

## 5. KKT Conditions and Support Vectors

The *Karush–Kuhn–Tucker* (KKT) conditions for the soft-margin SVM are:

1. **Primal feasibility:** $y_i(w^T x_i + b) \ge 1 - \xi_i$, $\xi_i \ge 0$.
2. **Dual feasibility:** $0 \le \alpha_i \le C$.
3. **Complementary slackness:**
   - $\alpha_i [y_i(w^T x_i + b) - 1 + \xi_i] = 0$
   - $\mu_i \xi_i = (C - \alpha_i) \xi_i = 0$

### 5.1 Three types of points

From the KKT conditions, every training point falls into exactly one category:

| Condition | $\alpha_i$ | $\xi_i$ | Location |
|---|---|---|---|
| $\alpha_i = 0$ | 0 | 0 | Beyond or on the margin — **non-support vector** |
| $0 < \alpha_i < C$ | $\in (0, C)$ | 0 | Exactly on the margin boundary — **free support vector** |
| $\alpha_i = C$ | $C$ | $> 0$ | Inside margin or misclassified — **bounded support vector** |

**Support vectors** are points with $\alpha_i > 0$. Only these contribute to $w$ via (4.1).

**Key insight:** If you remove a non-support-vector point, the solution does not change.
The decision boundary depends only on the support vectors.

### 5.2 Recovering $b$

For any free support vector ($0 < \alpha_i < C$, so $\xi_i = 0$):

$$y_i(w^T x_i + b) = 1 \implies b = y_i - w^T x_i.$$

In practice, average over all free support vectors for numerical stability.

---

## 6. Hinge Loss View

The soft-margin SVM (3.1) has an equivalent *unconstrained* formulation using the
**hinge loss**:

```math
\min_{w, b} \quad \frac{1}{2}\|w\|_2^2 + C \sum_{i=1}^n \max(0, 1 - y_i f(x_i)). \qquad (6.1)
```

**Proof of equivalence.** In (3.1), at optimality $\xi_i = \max(0, 1 - y_i f(x_i))$
because the $\xi_i$ constraint is tight when active and zero otherwise.

Substituting eliminates $\xi_i$. $\blacksquare$

### 6.1 Hinge loss properties

The hinge loss $\ell(m) = \max(0, 1 - m)$ where $m = y_i f(x_i)$ is the functional margin:

- Zero when $m \ge 1$ (point is on the correct side of the margin).
- Linear with slope $-1$ when $m < 1$.
- Non-differentiable at $m = 1$ (use subgradients).

### 6.2 Regularized risk form

Dividing (6.1) by $n$ and defining $\lambda = 1/(2nC)$:

```math
\min_{w, b} \quad \frac{1}{n} \sum_{i=1}^n \max(0, 1 - y_i f(x_i)) + \lambda \|w\|_2^2. \qquad (6.2)
```

This is the standard *regularized empirical risk minimization* form: data loss + regularizer.

### 6.3 Subgradient

The subgradient of the hinge loss w.r.t. $w$ for example $i$:

```math
\frac{\partial \ell_i}{\partial w} = \begin{cases} 0 & \text{if } y_i f(x_i) \ge 1, \\ -y_i x_i & \text{if } y_i f(x_i) < 1. \end{cases}
```

**Result:** The full subgradient of (6.1) w.r.t. $w$ is

```math
g_w = w - C \sum_{i:\, y_i f(x_i) < 1} y_i x_i. \qquad (6.3)
```

This is the basis for subgradient descent (Pegasos algorithm).

### 6.4 Comparison with logistic loss

| Loss | Formula | At margin $m=0$ | Differentiable? |
|---|---|---|---|
| Hinge | $\max(0, 1-m)$ | 1 | No (kink at $m=1$) |
| Logistic | $\log(1 + e^{-m})$ | $\log 2 \approx 0.69$ | Yes (smooth) |

Both penalize negative margins (misclassifications). The hinge loss is exactly zero for
$m \ge 1$, producing sparse $\alpha$ (support vectors). The logistic loss is always
positive — every point influences the fit.

---

## 7. Kernel Trick

### 7.1 Motivation

The dual (4.4) and the prediction function depend on data only through inner products:

$$w^T x = \sum_i \alpha_i y_i (x_i^T x), \qquad W(\alpha) = \sum_i \alpha_i - \frac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j (x_i^T x_j).$$

**Idea:** Replace the inner product $x_i^T x_j$ with a *kernel function*
$K(x_i, x_j) = \phi(x_i)^T \phi(x_j)$ that computes the inner product in a
higher-dimensional feature space *without explicitly computing* $\phi(x)$.

### 7.2 Mercer's condition

A function $K: \mathbb{R}^p \times \mathbb{R}^p \to \mathbb{R}$ is a valid kernel if and
only if the Gram matrix $G_{ij} = K(x_i, x_j)$ is positive semidefinite for every finite
set of points $\lbrace x_1, \dots, x_n\rbrace$.

### 7.3 Common kernels

| Kernel | $K(x, x')$ | Parameters | Feature space |
|---|---|---|---|
| Linear | $x^T x'$ | — | Original space |
| Polynomial | $(x^T x' + c)^d$ | degree $d$, offset $c \ge 0$ | All monomials up to degree $d$ |
| RBF (Gaussian) | $\exp(-\gamma \Vert x - x'\Vert^2)$ | $\gamma > 0$ | Infinite-dimensional |
| Sigmoid | $\tanh(\kappa \thinspace x^T x' + c)$ | $\kappa > 0$, $c$ | Not always valid (not PSD for all $\kappa, c$) |

**RBF intuition.** Two points close together ($\Vert x - x'\Vert$ small) → $K \approx 1$.
Far apart → $K \approx 0$.

Each training point acts like a localized "bump." High $\gamma$ → sharp bumps (complex
boundary), low $\gamma$ → smooth bumps (simple boundary).

### 7.4 Kernelized prediction

$$f(x) = \sum_{i=1}^n \alpha_i y_i K(x_i, x) + b.$$

Only the support vectors ($\alpha_i > 0$) contribute to the sum.

---

## 8. Failure Cases

1. **Sensitive to feature scaling.** SVM uses distances (via $\Vert w\Vert$ and inner products).
   Features on different scales dominate the margin. **Cure:** Always standardize features
   before training.

2. **Choice of $C$.** Too large → overfitting (memorizing noise). Too small → underfitting
   (wide margin ignores structure). **Cure:** Cross-validation.

3. **Choice of kernel and hyperparameters.** RBF with large $\gamma$ overfits; small
   $\gamma$ underfits. Polynomial with high $d$ is expensive and prone to overfitting.
   **Cure:** Grid search or Bayesian optimization over $(C, \gamma)$ or $(C, d)$.

4. **Computational cost of kernel SVM.** Training requires the $n \times n$ Gram matrix →
   $O(n^2)$ memory. Solving the dual QP is $O(n^2)$ to $O(n^3)$ time.

5. **Impractical for $n > 10^4\text{–}10^5$.** **Cure:** Use linear SVM with stochastic
   subgradient descent, or approximate kernel methods (random Fourier features, Nyström).

6. **No probabilistic output.** SVM produces a decision function, not probabilities.
   Probability calibration (Platt scaling) can be added post-hoc but is not native.

7. **Multi-class.** SVM is inherently binary. Multi-class requires one-vs-one ($\binom{K}{2}$
   classifiers) or one-vs-rest ($K$ classifiers).

---

## 9. Connections

- **[Logistic Regression](../04_logistic_regression/README.md).** Same linear decision
  boundary $w^T x + b = 0$. Logistic uses smooth log-loss; SVM uses kinked hinge loss.
  Both can be viewed as regularized empirical risk minimization with different losses.

- **Consequences of the loss choice.** At the margin, hinge loss = 1, log-loss ≈ 0.69.
  SVM produces sparse support vectors; logistic regression uses all points.

- **[Regularization](../03_regularization/README.md).** The $\frac{1}{2}\Vert w\Vert^2$ term is
  L2 regularization. The SVM objective (6.2) is structurally identical to ridge-regularized
  logistic regression, just with a different loss function.

- **[Geometry of ML](../synthesis/geometry_of_ml.md).** SVM is the canonical geometric
  classifier — the margin is a geometric quantity (distance), and the dual reveals that
  only boundary points (support vectors) determine the solution.

- **[Neural Networks](../13_neural_networks/README.md).** A single-layer network with
  hinge loss is a linear SVM. Kernel SVM can be seen as a two-layer network where the
  first layer is fixed (kernel features) and only the output weights are learned.

- **[Optimization](https://github.com/hien078/applied-mathematics-foundation).** SVM is a showcase
  for constrained optimization: Lagrangians, duality, KKT conditions. The Pegasos
  algorithm demonstrates subgradient methods for non-smooth optimization.

---

## 10. References

- **Cortes, C., & Vapnik, V. (1995).** Support-vector networks. *Machine Learning*, 20(3), 273–297.
- **Schölkopf, B., & Smola, A. J. (2002).** *Learning with Kernels: Support Vector Machines, Regularization, Optimization, and Beyond*. MIT Press.
- **Boyd, S., & Vandenberghe, L. (2004).** *Convex Optimization*. Cambridge University Press. Chapter 5: *Duality*.
- **Hastie, T., Tibshirani, R., & Friedman, J. (2009).** *The Elements of Statistical Learning: Data Mining, Inference, and Prediction* (2nd ed.). Springer. Chapter 12: *Support Vector Machines and Flexible Discriminants*.

