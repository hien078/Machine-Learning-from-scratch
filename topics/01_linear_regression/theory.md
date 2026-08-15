# Linear Regression — Theory

> **Purpose:** Pure theory, definitions, and conceptual understanding.
> Read this first, then open `first_principles.ipynb` for computation and experiments.

## Prerequisites

- [Foundations: Linear Algebra](../../foundations/linear_algebra/README.md)
- [Foundations: Calculus & Optimization](../../foundations/calculus_optimization/README.md)

---

## 0. Notation

Every symbol used later, defined once.

| Symbol | Type | Meaning |
|---|---|---|
| $n$ | scalar $\in \mathbb{N}$ | number of training examples |
| $p$ | scalar $\in \mathbb{N}$ | number of features (including the intercept after the bias trick) |
| $x_i$ | vector $\in \mathbb{R}^p$ | feature vector of the $i$-th example |
| $y_i$ | scalar $\in \mathbb{R}$ | target of the $i$-th example |
| $X$ | matrix $\in \mathbb{R}^{n \times p}$ | design matrix; row $i$ is $x_i^\top$ |
| $y$ | vector $\in \mathbb{R}^n$ | target vector with entries $y_i$ |
| $\theta$ | vector $\in \mathbb{R}^p$ | model parameters (weights) |
| $\hat{y}_i$ | scalar | model prediction for example $i$; $\hat{y}_i = x_i^\top \theta$ |
| $\hat{y}$ | vector $\in \mathbb{R}^n$ | prediction vector; $\hat{y} = X \theta$ |
| $r_i$ | scalar | residual for example $i$; $r_i = \hat{y}_i - y_i$ |
| $r^*$ | vector $\in \mathbb{R}^n$ | residual vector at the optimum; $r^* = \hat{y}^* - y$ |
| $L(\theta)$ | scalar function | the MSE loss; $L : \mathbb{R}^p \to \mathbb{R}^+$ |
| $\nabla L(\theta)$ | vector $\in \mathbb{R}^p$ | gradient of $L$ at $\theta$ |
| $\nabla^2 L(\theta)$ | matrix $\in \mathbb{R}^{p \times p}$ | Hessian of $L$ at $\theta$ |
| $\theta^*$ | vector $\in \mathbb{R}^p$ | a minimiser of $L$ |
| $H$ | matrix $\in \mathbb{R}^{n \times n}$ | hat matrix; $H := X (X^\top X)^{-1} X^\top$ |
| $X^+$ | matrix $\in \mathbb{R}^{p \times n}$ | Moore–Penrose pseudoinverse of $X$ |
| $\varepsilon_i$ | scalar (random) | noise term in the probabilistic model |
| $\sigma^2$ | scalar $> 0$ | noise variance |
| $\|\cdot\|$ | scalar | Euclidean ($\ell^2$) norm |
| $\langle \cdot, \cdot \rangle$ | scalar | Euclidean inner product; $\langle a, b \rangle = a^\top b$ |
| $\text{Col}(X)$ | subspace $\subseteq \mathbb{R}^n$ | column space of $X$ (span of its columns) |
| $\text{Null}(X)$ | subspace $\subseteq \mathbb{R}^p$ | null space of $X$; $\{ v : Xv = 0 \}$ |
| $\text{rank}(X)$ | scalar $\in \mathbb{N}$ | $\dim \text{Col}(X)$; for $X \in \mathbb{R}^{n \times p}$, $\text{rank}(X) \le \min(n, p)$ |
| $\text{RSS}$ | scalar $\ge 0$ | residual sum of squares; $\text{RSS} := \|r^*\|^2 = n \, L(\theta^*)$ |
| $\hat{\sigma}^2$ | scalar $> 0$ | unbiased estimate of $\sigma^2$; $\hat{\sigma}^2 := \text{RSS}/(n-p)$ |
| $\text{SE}(\hat{\theta}_j)$ | scalar | standard error of $\hat{\theta}_j$; $\text{SE}(\hat{\theta}_j) := \hat{\sigma} \sqrt{[(X^\top X)^{-1}]_{jj}}$ |
| $h_{ii}$ | scalar $\in [0, 1]$ | leverage of example $i$; the $i$-th diagonal entry of $H$ |
| $x_0$ | vector $\in \mathbb{R}^p$ | feature vector of a new (unseen) point, leading $1$ included |

**Conventions.**

- All vectors are *column* vectors; lowercase letters denote vectors, uppercase letters matrices.
- $A^\top$ is the transpose of $A$; $A^{-1}$ its inverse (when it exists); $A^+$ its Moore–Penrose pseudoinverse.
- Subscripts index examples ($x_i$); commas separate row/column indices when both are needed ($X_{i,j}$).
- A matrix is **symmetric** if $A^\top = A$; **positive semi-definite (PSD)** if $v^\top A v \ge 0$ for all $v$; **positive definite (PD)** if $v^\top A v > 0$ for all $v \neq 0$.
- An $n \times n$ matrix $P$ is an **orthogonal projection** iff $P^\top = P$ and $P^2 = P$.
- **Residual sign.** This document defines $r := \hat{y} - y$ (prediction minus target). Many statistics texts (e.g. Hastie et al.) and most software report $e := y - \hat{y} = -r$. All squared and norm-based quantities (MSE, RSS, $R^2$, leverage, Cook's distance) agree under both conventions; only the signs of raw residuals flip.

---

## 1. The Problem — WHY

We have continuous target values $y \in \mathbb{R}$ that we believe depend on some input features $\mathbf{x} \in \mathbb{R}^d$. We want to find a function $f(\mathbf{x})$ that predicts $y$ as accurately as possible.

Why start with a linear function?
- **Simplicity:** It is mathematically tractable and easy to optimize.
- **Interpretability:** The coefficients directly tell us the effect of each feature.
- **Foundation:** Many complex models (like neural networks) are built upon linear models.

---

## 2. Core Idea — The Linear Model

### 2.1 Definition (linear model)

A **linear model** assumes the prediction is a linear combination of features. For each example $i \in \{1, \dots, n\}$ and an unknown parameter vector $\theta \in \mathbb{R}^p$:

$$\hat{y}_i = x_{i1} \theta_1 + x_{i2} \theta_2 + \dots + x_{ip} \theta_p = x_i^\top \theta$$

### 2.2 Matrix form

Stack the per-example equations row by row. Define the **design matrix** $X \in \mathbb{R}^{n \times p}$ as the matrix whose $i$-th row equals $x_i^\top$, the **target vector** $y \in \mathbb{R}^n$, and the **prediction vector** $\hat{y} \in \mathbb{R}^n$. Then:

$$\hat{y} = X\theta \in \mathbb{R}^n$$

### 2.3 Bias trick

A real model has an intercept (bias) term $\theta_0$:

$$\hat{y}_i = \theta_0 + x_{i1} \theta_1 + \dots + x_{id} \theta_d$$

Absorb $\theta_0$ into $\theta$ by prepending a column of 1s to the feature matrix. Let $\mathbb{1} := (1, 1, \dots, 1)^\top \in \mathbb{R}^n$. Then

$$\tilde{X} := [ \mathbb{1} \mid X ] \in \mathbb{R}^{n \times (d+1)}, \quad \tilde{\theta} := (\theta_0, \theta_1, \dots, \theta_d)^\top \in \mathbb{R}^{d+1}, \quad \hat{y} = \tilde{X} \tilde{\theta}$$

From here on we drop the tildes and assume $X$ already includes the bias column; $p$ denotes the total number of weights (including $\theta_0$).

### 2.4 Interpreting the coefficients

Because $\hat{y} = x^\top \theta$, the coefficient $\theta_j$ is exactly $\partial \hat{y} / \partial x_j$: the change in the *prediction* per unit increase in feature $j$, all other features held fixed. Three caveats keep this honest:

- **Model, not mechanism.** "Held fixed" is a statement about the fitted function, not about the world. On observational data, coefficients measure association; causal claims require randomisation or explicit causal assumptions.
- **Correlated features.** When columns of $X$ are strongly correlated, "move $x_j$ while fixing the rest" describes a region the data barely visits; individual coefficients become unstable even when predictions remain stable (§8, §9.2).
- **Units.** $\theta_j$ scales inversely with the units of $x_j$; comparing coefficient magnitudes across features is only meaningful after standardising them.

---

## 3. Assumptions

### 3.1 The Gauss–Markov assumptions

Classical regression theory is organised around five assumptions on $(X, \varepsilon)$. The first four are the **Gauss–Markov** assumptions; the fifth (normality) is needed only for exact distributional results.

| Label | Name | Statement |
|---|---|---|
| **A1** | Linearity in parameters | $y = X\theta + \varepsilon$ with $X \in \mathbb{R}^{n \times p}$ fixed, $\text{rank}(X) = p$. |
| **A2** | Zero-mean noise | $\mathbb{E}[\varepsilon] = 0$. |
| **A3** | Homoscedasticity | $\text{Var}(\varepsilon_i) = \sigma^2$ for every $i$. |
| **A4** | No autocorrelation | $\text{Cov}(\varepsilon_i, \varepsilon_j) = 0$ for $i \neq j$. |
| **A5** | Normality (optional) | $\varepsilon \sim \mathcal{N}(0, \sigma^2 I_n)$ |

**Remarks.**

- The rank condition in A1 implicitly requires $n \ge p$: column rank cannot exceed the number of rows. With $n < p$, only the singular machinery of §8 applies.
- In this fixed-design setting, A2 reduces to $\mathbb{E}[\varepsilon] = 0$. In the random-design setting the corresponding assumption is **strict exogeneity**, $\mathbb{E}[\varepsilon \mid X] = 0$; its failure mode — noise correlated with regressors, e.g. through an omitted variable — is the one diagnosed in §12.

**Compact restatement of A2 + A3 + A4:**

$$\mathbb{E}[\varepsilon] = 0, \quad \text{Var}(\varepsilon) := \mathbb{E}[\varepsilon \varepsilon^\top] = \sigma^2 I_n$$

---

## 4. Formulation — Mean Squared Error

### 4.1 Definition (residual and MSE)

The **residual** for example $i$ is

$$r_i(\theta) := \hat{y}_i - y_i = x_i^\top \theta - y_i$$

*(Sign convention: many texts define the residual with the opposite sign, $e_i := y_i - \hat{y}_i = -r_i$ — see the convention note in §0. All squared quantities are unaffected.)*

The **mean squared error** is:

$$\begin{aligned}
L(\theta) &= \frac{1}{n} \sum_i r_i(\theta)^2 & \text{(scalar form)} \\
&= \frac{1}{n} \cdot \|X\theta - y\|^2 & \text{(vector form)} \\
&= \frac{1}{n} \cdot \bigl(\theta^\top X^\top X \theta - 2\, y^\top X \theta + y^\top y\bigr) & \text{(expanded quadratic)}
\end{aligned}$$

**Ordinary least squares (OLS)** is the optimisation problem

$$\theta^* \in \arg\min_{\theta \in \mathbb{R}^p} L(\theta)$$

### 4.2 Theorem (MSE is the Gaussian negative log-likelihood)

> **Theorem.** Assume the data-generating process
>
> $$y_i = x_i^\top \theta + \varepsilon_i, \quad \varepsilon_i \sim \mathcal{N}(0, \sigma^2) \text{ i.i.d.}, \quad i = 1, \ldots, n.$$
>
> Then the maximum-likelihood estimator of $\theta$ given $(X, y)$ is identical to the OLS minimiser of $L(\theta)$.

**Proof.** Conditional on $X$, the noise terms $\varepsilon_i = y_i - x_i^\top \theta$ are i.i.d. $\mathcal{N}(0, \sigma^2)$, so the joint density factorises:

$$p(y \mid X, \theta) = \prod_i (2\pi \sigma^2)^{-1/2} \cdot \exp\!\left( -\frac{(y_i - x_i^\top \theta)^2}{2\sigma^2} \right)$$

Take logarithms:

$$\begin{aligned}
\ell(\theta) &:= \log p(y \mid X, \theta) \\
&= -\frac{n}{2} \cdot \log(2\pi \sigma^2) - \frac{1}{2\sigma^2} \cdot \sum_i (y_i - x_i^\top \theta)^2 \\
&= -\frac{n}{2} \cdot \log(2\pi \sigma^2) - \frac{n}{2\sigma^2} \cdot L(\theta)
\end{aligned}$$

The first term does not depend on $\theta$, and $n / (2\sigma^2) > 0$ is a positive constant. Therefore

$$\arg\max_\theta \ell(\theta) = \arg\min_\theta L(\theta) \quad \blacksquare$$

**Result:** Under i.i.d. Gaussian noise, minimizing MSE is equivalent to maximum likelihood estimation of $\theta$.

**Remarks.**

- Different noise models give different losses: $\varepsilon_i \sim \text{Laplace}(0, b)$ yields mean absolute error (MAE); a density proportional to $\exp[-\rho_{\delta}(\varepsilon)]$ yields the Huber loss, where $\rho_\delta(u) := \frac{1}{2} u^2$ for $|u| \le \delta$ and $\delta \bigl( |u| - \frac{\delta}{2} \bigr)$ otherwise.
- A single residual of magnitude 10 contributes the same to $L$ as 100 residuals of magnitude 1 — a quadratic penalty for outliers.

---

## 5. Derivation — Gradient, Hessian, and Convexity

### 5.1 Matrix calculus identities

Two identities power all of OLS calculus. For column vectors $\theta \in \mathbb{R}^p$:

**Identity (i) — linear form.** For a constant vector $b \in \mathbb{R}^p$,

$$\nabla_\theta (b^\top \theta) = b$$

**Identity (ii) — quadratic form.** For a constant matrix $A \in \mathbb{R}^{p \times p}$,

$$\nabla_\theta (\theta^\top A \theta) = (A + A^\top) \theta$$

*Special case.* When $A$ is symmetric, this reduces to $\nabla_\theta (\theta^\top A \theta) = 2 A \theta$.

### 5.2 Gradient of $L$

> **Theorem.** $\nabla L(\theta) = \frac{2}{n} \cdot X^\top (X \theta - y)$

**Proof.** Start from the expanded quadratic form of $L(\theta)$ and differentiate term by term:

- *Quadratic term.* Apply Identity (ii) with $A = X^\top X$. Since $X^\top X$ is symmetric, the gradient is $2 X^\top X \theta$.
- *Linear term.* Rewrite $-2\, y^\top X \theta = -2 (X^\top y)^\top \theta$ and apply Identity (i) with $b = X^\top y$, yielding $-2 X^\top y$.
- *Constant term.* $y^\top y$ does not depend on $\theta$, so its gradient is $0$.

Combining:

$$\nabla L(\theta) = \frac{1}{n} \cdot \bigl( 2 X^\top X \theta - 2 X^\top y \bigr) = \frac{2}{n} \cdot X^\top (X \theta - y) \quad \blacksquare$$

**Result:** $\nabla L(\theta) = \frac{2}{n} X^\top (X\theta - y)$

### 5.3 Hessian of $L$

> **Theorem.** $\nabla^2 L(\theta) = \frac{2}{n} \cdot X^\top X$, independent of $\theta$.

**Proof.** From the gradient, $\nabla L(\theta) = \frac{2}{n} \cdot (X^\top X \theta - X^\top y)$. The map $\theta \mapsto \frac{2}{n} X^\top X \theta$ is linear with constant Jacobian $\frac{2}{n} X^\top X$; the constant term has zero Jacobian. $\blacksquare$

**Result:** $\nabla^2 L(\theta) = \frac{2}{n} X^\top X$

### 5.4 Convexity of $L$

> **Theorem.**
>
> 1. $L$ is convex on $\mathbb{R}^p$.
> 2. $L$ is **strictly** convex if and only if $\text{rank}(X) = p$.

**Proof.** A twice-differentiable function on $\mathbb{R}^p$ is convex when its Hessian is PSD everywhere.

*(1) $X^\top X$ is PSD.* For any $v \in \mathbb{R}^p$,

$$v^\top (X^\top X) v = (Xv)^\top (Xv) = \|Xv\|^2 \ge 0$$

Hence $X^\top X$ is PSD, $\nabla^2 L$ is PSD, and $L$ is convex.

*(2) PD ⟺ full column rank.* $X^\top X$ is PD iff $\|Xv\|^2 > 0$ for every $v \neq 0$, iff $Xv = 0$ implies $v = 0$, iff $\text{Null}(X) = \{0\}$, iff $\text{rank}(X) = p$. $\blacksquare$

**Result:** $L$ is a convex quadratic; strictly convex iff $X$ has full column rank.

---

## 6. Normal Equations and Closed-Form Solution

### 6.1 First-order optimality

Since $L$ is convex, every critical point is a global minimiser. Setting $\nabla L(\theta) = 0$:

$$X^\top X \theta = X^\top y$$

This is the system of **normal equations** for OLS.

### 6.2 Existence and uniqueness

> **Theorem.** Let $X \in \mathbb{R}^{n \times p}$, $y \in \mathbb{R}^n$, and $L(\theta) = \frac{1}{n} \|X\theta - y\|^2$. Define $\Theta^* := \arg\min_\theta L(\theta)$. Then
>
> 1. **(Existence.)** $\Theta^*$ is non-empty.
> 2. **(Uniqueness.)** $|\Theta^*| = 1$ $\iff$ $\text{rank}(X) = p$ $\iff$ $X^\top X$ is invertible.
> 3. **(Closed form.)** When $\text{rank}(X) = p$, the unique minimiser is
>
> $$\theta^* = (X^\top X)^{-1} X^\top y$$
>
> When $\text{rank}(X) < p$, $\Theta^*$ is an affine subspace of $\mathbb{R}^p$ of dimension $p - \text{rank}(X)$. §8 selects its unique minimum-norm element via the pseudoinverse.

**Proof.**

*(1) Existence.* First, $\text{Null}(X^\top X) = \text{Null}(X)$: if $X^\top X v = 0$ then $\|Xv\|^2 = v^\top X^\top X v = 0$, so $Xv = 0$; the converse is immediate. Hence $\text{rank}(X^\top X) = \text{rank}(X) = \text{rank}(X^\top)$, and since $\text{Col}(X^\top X) \subseteq \text{Col}(X^\top)$ with equal dimensions, $\text{Col}(X^\top X) = \text{Col}(X^\top)$. As $X^\top y \in \text{Col}(X^\top)$, the normal equations are consistent.

*(2) Uniqueness.* $L$ is strictly convex iff $\text{rank}(X) = p$ (§5.4). A strictly convex function has at most one minimiser; combined with (1), exactly one.

*(3) Closed form.* Under $\text{rank}(X) = p$, $X^\top X$ is invertible, so the normal equations have the unique solution $\theta^* = (X^\top X)^{-1} X^\top y$. $\blacksquare$

**Result:** $\hat{\theta} = (X^\top X)^{-1} X^\top y$ when $X$ has full column rank.

---

## 7. Geometric Meaning — Hat Matrix and Orthogonal Projection

### 7.1 Residual orthogonality

Rewrite the normal equations as

$$X^\top (X \theta^* - y) = 0$$

i.e. $X^\top r^* = 0$, where $r^* := X \theta^* - y$. The residual is orthogonal to every column of $X$, hence to every vector in $\text{Col}(X)$. This is the linear-algebra signature of an **orthogonal projection**: $\hat{y}^* = X \theta^*$ is the unique vector in $\text{Col}(X)$ such that $y - \hat{y}^* \perp \text{Col}(X)$.

### 7.2 Definition (hat matrix)

Assume $\text{rank}(X) = p$. Substituting $\theta^* = (X^\top X)^{-1} X^\top y$ into $\hat{y}^* = X \theta^*$:

$$\hat{y}^* = X (X^\top X)^{-1} X^\top y = H y$$

where $H := X (X^\top X)^{-1} X^\top \in \mathbb{R}^{n \times n}$. $H$ is called the **hat matrix** — it puts the hat on $y$.

### 7.3 Properties of the hat matrix

> **Theorem.** The hat matrix $H$ satisfies
>
> 1. **Symmetry.** $H^\top = H$.
> 2. **Idempotence.** $H^2 = H$.
> 3. **Range.** $\text{Im}(H) = \text{Col}(X)$.
> 4. **Spectrum.** Every eigenvalue of $H$ is $0$ or $1$, and $\text{rank}(H) = \text{trace}(H) = p$.
>
> Properties (1)–(2) together characterise an orthogonal projection matrix in $\mathbb{R}^n$.

**Proof.**

*(1)* $H^\top = \bigl( X (X^\top X)^{-1} X^\top \bigr)^\top = X (X^\top X)^{-1} X^\top = H$.

*(2)* $H^2 = X (X^\top X)^{-1} X^\top X (X^\top X)^{-1} X^\top = X (X^\top X)^{-1} X^\top = H$.

*(3)* For any $y$, $Hy = X[(X^\top X)^{-1} X^\top y] \in \text{Col}(X)$. Conversely, $z = Xw \implies Hz = Xw = z$.

*(4)* If $Hv = \lambda v$ with $v \neq 0$, then $H^2 v = \lambda^2 v$; using $H^2 = H$: $\lambda(\lambda - 1)v = 0 \implies \lambda \in \{0, 1\}$. The multiplicity of eigenvalue 1 equals $\text{rank}(X) = p$. $\blacksquare$

### 7.4 Corollary (Pythagoras)

> $\|y\|^2 = \|\hat{y}^*\|^2 + \|r^*\|^2$

**Proof.** $r^* \perp \text{Col}(X)$ and $\hat{y}^* \in \text{Col}(X)$, so $\langle \hat{y}^*, r^* \rangle = 0$. Since $y = \hat{y}^* - r^*$:

$$\|y\|^2 = \|\hat{y}^*\|^2 - 2\langle \hat{y}^*, r^* \rangle + \|r^*\|^2 = \|\hat{y}^*\|^2 + \|r^*\|^2 \quad \blacksquare$$

### 7.5 Leverage

The diagonal entries of $H$ have a name and a job of their own. The **leverage** of example $i$ is

$$h_{ii} := H_{ii} = x_i^\top (X^\top X)^{-1} x_i$$

> **Theorem.**
>
> 1. $0 \le h_{ii} \le 1$ for every $i$.
> 2. $\sum_{i=1}^n h_{ii} = p$; the average leverage is $p/n$.
> 3. $\partial \hat{y}_i^* / \partial y_i = h_{ii}$ — leverage is the sensitivity of a fitted value to its own target.
> 4. If $\mathbb{1} \in \text{Col}(X)$ (intercept present), then $h_{ii} \ge 1/n$.

**Proof.**

*(1)* By symmetry and idempotence, $h_{ii} = (H^2)_{ii} = \sum_j H_{ij} H_{ji} = \sum_j H_{ij}^2 \ge 0$, and $h_{ii} = h_{ii}^2 + \sum_{j \neq i} H_{ij}^2 \ge h_{ii}^2$, which forces $h_{ii} \le 1$.

*(2)* $\sum_i h_{ii} = \text{trace}(H) = p$ (§7.3).

*(3)* $\hat{y}^* = Hy$ is linear in $y$ with Jacobian $H$.

*(4)* Let $P_{\mathbb{1}} := \frac{1}{n} \mathbb{1} \mathbb{1}^\top$, the orthogonal projection onto $\text{span}(\mathbb{1})$. Since $\mathbb{1} \in \text{Col}(X)$, $H P_{\mathbb{1}} = P_{\mathbb{1}}$, and taking transposes $P_{\mathbb{1}} H = P_{\mathbb{1}}$; hence $(H - P_{\mathbb{1}})^2 = H - P_{\mathbb{1}}$ and $H - P_{\mathbb{1}}$ is symmetric — an orthogonal projection, therefore PSD, therefore its diagonal $h_{ii} - \frac{1}{n}$ is non-negative. $\blacksquare$

**Result:** points with leverage well above the average $p/n$ (rule of thumb: $h_{ii} > 2p/n$) sit far from the bulk of the design and can single-handedly steer the fit. The statistical consequences — unequal residual variances and influence measures — are developed in §9.8.

---

## 8. Singular Case — SVD and Moore–Penrose Pseudoinverse

When $\text{rank}(X) < p$ (the *multicollinear* regime), $X^\top X$ is singular, the closed form is undefined, and $\Theta^*$ is an infinite affine subspace.

### 8.1 The singular value decomposition

> **Theorem (SVD).** Every $X \in \mathbb{R}^{n \times p}$ admits a factorisation
>
> $$X = U \Sigma V^\top$$
>
> where $U \in \mathbb{R}^{n \times n}$ and $V \in \mathbb{R}^{p \times p}$ are orthogonal and $\Sigma \in \mathbb{R}^{n \times p}$ is diagonal with non-negative entries $\sigma_1 \ge \sigma_2 \ge \cdots \ge \sigma_{\min(n,p)} \ge 0$.

### 8.2 Definition (Moore–Penrose pseudoinverse)

Given the SVD $X = U \Sigma V^\top$, build $\Sigma^+ \in \mathbb{R}^{p \times n}$ by inverting the non-zero singular values. The **Moore–Penrose pseudoinverse** is

$$X^+ := V \Sigma^+ U^\top \in \mathbb{R}^{p \times n}$$

### 8.3 Properties

> **Theorem.**
>
> 1. If $\text{rank}(X) = p$, then $X^+ = (X^\top X)^{-1} X^\top$. Hence $\theta^* = X^+ y$ agrees with the OLS closed form.
> 2. For arbitrary $X$, $\theta_{\text{minnorm}} := X^+ y$ is the unique element of $\Theta^*$ with smallest Euclidean norm.
> 3. The prediction $X X^+ y$ is the orthogonal projection of $y$ onto $\text{Col}(X)$, regardless of rank.

**Result:** When the design is multicollinear, the *parameters* $\theta$ are non-unique, but the *predictions* $\hat{y}$ are. The pseudoinverse picks the shortest coordinate vector that lands on the projection. Numerically, solve least squares with QR or SVD rather than forming the inverse.

---

## 9. Statistical Properties — Gauss–Markov

### 9.1 Unbiasedness

> **Theorem.** Under A1 + A2, $\mathbb{E}[\hat{\theta}] = \theta$.

**Proof.** From $\hat{\theta} = (X^\top X)^{-1} X^\top y = \theta + (X^\top X)^{-1} X^\top \varepsilon$:

$$\mathbb{E}[\hat{\theta}] = \theta + (X^\top X)^{-1} X^\top \cdot \mathbb{E}[\varepsilon] = \theta \quad \blacksquare$$

**Result:** The OLS estimator is unbiased.

### 9.2 Variance

> **Theorem.** Under A1 + A2 + A3 + A4,
> $$\text{Var}(\hat{\theta}) = \sigma^2 (X^\top X)^{-1}$$

**Proof.** $\hat{\theta} - \theta = A\varepsilon$ where $A := (X^\top X)^{-1} X^\top$. Then

$$\text{Var}(\hat{\theta}) = A \cdot \text{Var}(\varepsilon) \cdot A^\top = \sigma^2 A A^\top = \sigma^2 (X^\top X)^{-1} \quad \blacksquare$$

**Result:** $\text{Var}(\hat{\theta}) = \sigma^2 (X^\top X)^{-1}$

Three knobs control the variance:
- **$\sigma^2$**: more noise → more variance. Linear.
- **Sample size $n$** *(heuristic)*: if the rows of $X$ behave like i.i.d. draws with second-moment matrix $\Sigma$, then $X^\top X \approx n \Sigma$, so $(X^\top X)^{-1} \approx \Sigma^{-1}/n$ and standard errors shrink like $1/\sqrt{n}$. For a fixed design this is an intuition, not an identity.
- **Multicollinearity**: near-proportional columns make $(X^\top X)^{-1}$ have huge diagonal entries.

### 9.3 Gauss–Markov theorem

> **Theorem (Gauss–Markov).** Under A1 + A2 + A3 + A4, the OLS estimator $\hat{\theta}$ is the **Best Linear Unbiased Estimator (BLUE)** — it has the lowest variance among all linear unbiased estimators.

**Proof sketch.** Write any linear unbiased estimator as $\tilde{\theta} = Cy$ with $CX = I_p$. Set $C = C_{\text{OLS}} + D$ where $DX = 0$. Then

$$\text{Var}(\tilde{\theta}) = \text{Var}(\hat{\theta}) + \sigma^2 DD^\top$$

Since $DD^\top$ is PSD, $\text{Var}(\tilde{\theta}) \succeq \text{Var}(\hat{\theta})$ with equality iff $D = 0$. $\blacksquare$

**Result:** OLS is the unique BLUE. Biased or non-linear estimators (Ridge, Lasso) can have smaller variance at the price of bias.

### 9.4 Estimating $\sigma^2$

The variance formula of §9.2 involves the unknown $\sigma^2$; using it in practice requires an estimate.

> **Theorem.** Under A1 + A2 + A3 + A4, $\mathbb{E}[\text{RSS}] = (n - p) \, \sigma^2$. Consequently
>
> $$\hat{\sigma}^2 := \frac{\text{RSS}}{n - p}$$
>
> is an unbiased estimator of $\sigma^2$.

**Proof.** Since every column of $X$ lies in $\text{Col}(X)$, on which $H$ acts as the identity (§7.3), $HX = X$ and hence $(I_n - H)X = 0$. Therefore

$$r^* = Hy - y = -(I_n - H)\, y = -(I_n - H)(X\theta + \varepsilon) = -(I_n - H)\, \varepsilon$$

Because $I_n - H$ is symmetric and idempotent,

$$\text{RSS} = \|r^*\|^2 = \varepsilon^\top (I_n - H) \varepsilon$$

Take expectations with the trace trick, using $\mathbb{E}[\varepsilon \varepsilon^\top] = \sigma^2 I_n$ (A2–A4):

$$\mathbb{E}[\text{RSS}] = \mathbb{E}\bigl[ \text{trace}\bigl( (I_n - H)\, \varepsilon \varepsilon^\top \bigr) \bigr] = \text{trace}\bigl( (I_n - H)\, \sigma^2 I_n \bigr) = \sigma^2 \bigl( n - \text{trace}(H) \bigr) = \sigma^2 (n - p) \quad \blacksquare$$

**Result:** $\hat{\sigma}^2 = \text{RSS}/(n-p)$; the divisor $n - p$ (not $n$) accounts for the $p$ degrees of freedom absorbed by the fit. The **standard error** of a coefficient is

$$\text{SE}(\hat{\theta}_j) := \hat{\sigma} \sqrt{[(X^\top X)^{-1}]_{jj}}$$

Normality (A5) is *not* needed for any of this — only for the exact distributional results below.

### 9.5 Sampling distribution (adding A5)

Under A1–A5 ($\varepsilon \sim \mathcal{N}(0, \sigma^2 I_n)$):

$$\hat{\theta} \sim \mathcal{N}(\theta, \sigma^2 (X^\top X)^{-1}), \quad \text{RSS}/\sigma^2 \sim \chi^2(n-p)$$

with $\hat{\theta}$ and RSS independent. The t-pivot $({\hat{\theta}_j - \theta_j})/{\text{SE}(\hat{\theta}_j)} \sim t(n-p)$ gives confidence intervals:

$$\hat{\theta}_j \pm t_{1-\alpha/2}(n-p) \cdot \text{SE}(\hat{\theta}_j)$$

### 9.6 Goodness of fit — $R^2$ and the ANOVA decomposition

Assume the design contains the intercept column, $\mathbb{1} \in \text{Col}(X)$, and let $\bar{y} := \frac{1}{n} \mathbb{1}^\top y$.

> **Theorem (ANOVA decomposition).** With $\mathbb{1} \in \text{Col}(X)$,
>
> $$\underbrace{\|y - \bar{y} \mathbb{1}\|^2}_{\text{TSS}} = \underbrace{\|\hat{y}^* - \bar{y} \mathbb{1}\|^2}_{\text{ESS}} + \underbrace{\|r^*\|^2}_{\text{RSS}}$$

**Proof.** The row of the normal equations $X^\top r^* = 0$ corresponding to the ones column gives $\mathbb{1}^\top r^* = 0$; in particular the fitted values have the same mean as $y$. Write $y - \bar{y}\mathbb{1} = (\hat{y}^* - \bar{y}\mathbb{1}) - r^*$. The cross term vanishes:

$$\langle \hat{y}^* - \bar{y}\mathbb{1}, \; r^* \rangle = \langle \hat{y}^*, r^* \rangle - \bar{y} \, \langle \mathbb{1}, r^* \rangle = 0 - 0 = 0$$

since both $\hat{y}^*$ and $\mathbb{1}$ lie in $\text{Col}(X)$, which is orthogonal to $r^*$ (§7.1). Expanding the square gives the identity. $\blacksquare$

The **coefficient of determination** is

$$R^2 := \frac{\text{ESS}}{\text{TSS}} = 1 - \frac{\text{RSS}}{\text{TSS}} \in [0, 1]$$

the fraction of the centred variation in $y$ captured by the fit; equivalently, $R^2$ is the squared sample correlation between $y$ and $\hat{y}^*$.

**Remarks.**

- **No intercept ⇒ no decomposition.** Without $\mathbb{1} \in \text{Col}(X)$, $\mathbb{1}^\top r^* = 0$ can fail, $\text{TSS} \neq \text{ESS} + \text{RSS}$, the two expressions for $R^2$ above disagree, and $1 - \text{RSS}/\text{TSS}$ can even be negative. (§7.4 is the *uncentred* analogue, valid regardless.)
- **$R^2$ never decreases when features are added** — enlarging $\text{Col}(X)$ can only shrink RSS — so it cannot compare models of different sizes. The **adjusted** version penalises $p$:
  $$\bar{R}^2 := 1 - \frac{\text{RSS}/(n-p)}{\text{TSS}/(n-1)} = 1 - (1 - R^2) \, \frac{n-1}{n-p}$$
- Under A1–A5, the overall **F-statistic** $F = \dfrac{\text{ESS}/(p-1)}{\text{RSS}/(n-p)} \sim F(p-1, \, n-p)$ under the null hypothesis that all non-intercept coefficients are zero — the "F" referred to in §12.

### 9.7 Prediction at a new point

Let $x_0 \in \mathbb{R}^p$ be a new feature vector (leading $1$ included) and $\hat{y}_0 := x_0^\top \hat{\theta}$.

**Mean response.** Under A1–A4, $\mathbb{E}[\hat{y}_0] = x_0^\top \theta$ and $\text{Var}(\hat{y}_0) = \sigma^2 \, x_0^\top (X^\top X)^{-1} x_0$ (immediate from §9.1–9.2). Adding A5, a $1 - \alpha$ **confidence interval for the mean response** $x_0^\top \theta$ is

$$x_0^\top \hat{\theta} \; \pm \; t_{1-\alpha/2}(n-p) \cdot \hat{\sigma} \sqrt{x_0^\top (X^\top X)^{-1} x_0}$$

**New observation.** A fresh draw $y_0 = x_0^\top \theta + \varepsilon_0$, with $\varepsilon_0 \sim \mathcal{N}(0, \sigma^2)$ independent of the training noise, carries its own irreducible noise on top of the estimation error:

$$\text{Var}(y_0 - \hat{y}_0) = \sigma^2 \bigl( 1 + x_0^\top (X^\top X)^{-1} x_0 \bigr)$$

giving the wider **prediction interval**

$$x_0^\top \hat{\theta} \; \pm \; t_{1-\alpha/2}(n-p) \cdot \hat{\sigma} \sqrt{1 + x_0^\top (X^\top X)^{-1} x_0}$$

**Remarks.**

- As data accumulates and $x_0^\top (X^\top X)^{-1} x_0 \to 0$, the CI width shrinks to $0$ but the PI width tends to $2 \, z_{1-\alpha/2} \, \sigma$: no amount of data removes the noise in a single future observation.
- For a training point $x_i$, the quantity $x_i^\top (X^\top X)^{-1} x_i$ is exactly the leverage $h_{ii}$ of §7.5 — high-leverage points are those where the model is least certain about its own mean prediction.

### 9.8 Influence — studentized residuals and Cook's distance

Residuals do not share a common variance, even under homoscedastic noise:

> **Theorem.** Under A1–A4, $\text{Var}(r^*) = \sigma^2 (I_n - H)$; in particular $\text{Var}(r_i^*) = \sigma^2 (1 - h_{ii})$.

**Proof.** From §9.4, $r^* = -(I_n - H)\varepsilon$, so $\text{Var}(r^*) = (I_n - H) \, \sigma^2 I_n \, (I_n - H)^\top = \sigma^2 (I_n - H)$. $\blacksquare$

High-leverage points thus have *small* raw residuals by construction ($h_{ii} \to 1 \Rightarrow \text{Var}(r_i^*) \to 0$): the fit is dragged toward them. Two standard corrections:

- **Studentized residual.** $t_i := \dfrac{r_i^*}{\hat{\sigma} \sqrt{1 - h_{ii}}}$; observations with $|t_i| \gtrsim 2$ deserve a look. (This is the *internally* studentized version; the *externally* studentized one re-estimates $\hat{\sigma}$ without example $i$ and follows exactly $t(n-p-1)$ under A1–A5.)
- **Cook's distance.** $D_i := \dfrac{(r_i^*)^2}{p \, \hat{\sigma}^2} \cdot \dfrac{h_{ii}}{(1 - h_{ii})^2}$ measures how far *all* fitted values move when example $i$ is deleted. Common screens: investigate $D_i > 4/n$; $D_i$ near $1$ signals serious influence.

Influence = discrepancy × leverage: a point matters when it is both poorly fit *and* alone in feature space.

---

## 10. Polynomial Regression (Extension)

What if the relationship isn't linear? Apply a feature transformation $\phi(\mathbf{x})$ that adds polynomial terms (e.g., $x^2, x_1 x_2$):

$$\hat{y} = \theta^\top \phi(\mathbf{x})$$

This is **still a linear regression model** with respect to the parameters $\theta$. We just replace $X$ with $\Phi$ in the Normal Equation:

$$\hat{\theta} = \Phi^+ y$$

**Bias-Variance Tradeoff:** As we increase the polynomial degree, the model becomes more flexible (lower bias), but it starts to fit the noise in the training data (higher variance), leading to overfitting.

**Categorical features (dummy encoding).** A categorical feature with $k$ levels enters the design matrix as $k - 1$ indicator (dummy) columns, one level being dropped as the *reference*. Each dummy coefficient is then the predicted difference in $y$ between its level and the reference level, other features fixed. Including all $k$ indicators alongside an intercept is the **dummy-variable trap**: the $k$ columns sum to $\mathbb{1}$, so $\text{rank}(X) < p$ — exactly the singular regime of §8. Feature maps and dummy encodings compose freely; the model remains linear in $\theta$.

---

## 11. Computational Notes

- **Normal-equation route:** forming $X^\top X$ costs $O(np^2)$ and solving costs $O(p^3)$ with $O(p^2)$ memory. QR/SVD is more stable.
- **Gradient Descent:** $O(np)$ per iteration. Preferred when $n$ and $p$ are very large. Convergence requires $0 < \eta < 2/L_{\text{smooth}}$ where $L_{\text{smooth}} = \frac{2}{n}\sigma_1(X)^2$ is the largest eigenvalue of the Hessian $\nabla^2 L$.
- **Conditioning:** the speed of GD is governed by the condition number *of the Hessian*, $\kappa(\nabla^2 L) = \kappa(X^\top X) = (\sigma_1 / \sigma_p)^2 = \kappa(X)^2$, where $\kappa(X) := \sigma_1 / \sigma_p$ is the condition number of $X$ itself. Feature scaling typically shrinks $\kappa$ and speeds convergence.

---

## 12. When Assumptions Break

| Violation | Damage | Diagnostic |
|---|---|---|
| **A1 fails — nonlinearity** ($\mathbb{E}[y \mid x]$ not linear in $x$) | Bias; systematic patterns left in residuals | Residual-vs-fitted plots; add transformed features (§10) |
| **A1 fails — rank deficiency** ($\text{rank}(X) < p$) | $(X^\top X)^{-1}$ undefined; near failure inflates SEs | SVD/QR; condition number; VIF |
| **A2 fails** (omitted variable) | Bias in $\hat{\theta}$; inconsistent | Residual plots vs omitted predictors |
| **A3 fails** (heteroscedasticity) | OLS still unbiased but CIs and p-values wrong | Breusch–Pagan/White test; HC standard errors |
| **A4 fails** (autocorrelation) | SEs wrong | Durbin–Watson test; Newey–West SEs |
| **A5 fails** (non-normal noise) | $\hat{\theta}$ still correct mean & variance; exact t/F break | QQ-plot; rely on asymptotic CLT for large $n$ |

**Mental model:** A1 + A2 → unbiasedness. A3 + A4 → variance formula & Gauss–Markov. A5 → exact small-sample distribution.

---

## 13. Connections

- **Related models:** [Polynomial Regression (above)](#10-polynomial-regression-extension), [03 Regularization](../03_regularization/README.md)
- **Foundations used:** [Linear Algebra](../../foundations/linear_algebra/README.md) (Projection), [Calculus](../../foundations/calculus_optimization/README.md) (Derivatives)
- **Synthesis:** [Optimization Methods](../../synthesis/optimization_methods_compared.md)
- **Graph Map:** See [INDEX.md](../../INDEX.md)

---

## 14. References

- **Hastie, T., Tibshirani, R., & Friedman, J. (2009).** *The Elements of Statistical Learning: Data Mining, Inference, and Prediction* (2nd ed.). Springer. Chapter 3: *Linear Methods for Regression*.
- **Bishop, C. M. (2006).** *Pattern Recognition and Machine Learning*. Springer. Chapter 3: *Linear Models for Regression*.
- **Boyd, S., & Vandenberghe, L. (2004).** *Convex Optimization*. Cambridge University Press. Chapter 4: *Convex Optimization Problems* (Least-Squares).
- **Gauss, C. F. (1823).** *Theoria combinationis observationum erroribus minimis obnoxiae* (Theory of the Combination of Observations Least Subject to Errors). Gottingae: Dieterich.

