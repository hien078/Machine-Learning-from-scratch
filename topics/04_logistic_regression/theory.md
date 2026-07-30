# Logistic Regression — Theory

## 0. Notation

All symbols used below — defined once.

| Symbol | Type | Meaning |
|---|---|---|
| $n$ | scalar | number of training examples |
| $p$ | scalar | number of features (including bias if used) |
| $x_i$ | vector of length $p$ | feature vector of the $i$-th example (a column of $X^T$) |
| $X$ | matrix of size $n \times p$ | design matrix; row $i$ is $x_i^T$ |
| $y_i$ | scalar in $\{0, 1\}$ | binary label of the $i$-th example |
| $y$ | vector of length $n$ | stacked labels |
| $\theta$ | vector of length $p$ | model parameters |
| $z_i$ | scalar | linear score for example $i$: $z_i := x_i^T \theta$ |
| $\sigma(z)$ | scalar in $(0, 1)$ | **sigmoid** function: $\sigma(z) := 1 / (1 + e^{-z})$ |
| $p_i$ | scalar in $(0, 1)$ | model probability $P(y_i = 1 \mid x_i, \theta) = \sigma(z_i)$ |
| $L(\theta)$ | scalar | the cross-entropy / NLL loss; defined in §3 |
| $\lambda$ | scalar $\ge 0$ | regularisation strength |
| $I_p$ | matrix of size $p \times p$ | identity matrix |
| $W$ | diagonal matrix of size $n \times n$ | weights $W_{ii} := p_i (1 - p_i)$ used in the Hessian (§5) |
| $K$ | scalar | number of classes in the multi-class extension (§11) |

**Bias convention.** We prepend a constant column of ones to $X$ so the intercept is
absorbed into $\theta_0$. Penalties (when added in §10) act on $\theta_1, \dots, \theta_{p-1}$
only, not on the intercept.

**Vector convention.** All vectors are column vectors. Lowercase Latin / Greek = vector;
uppercase = matrix. Norms carry explicit subscripts: $\|\theta\|_1$, $\|\theta\|_2^2$.

---

## 1. WHY — From Linear Scores to Probabilities

A linear score $z = \theta^T x$ takes values on $(-\infty, +\infty)$. When the target is
a binary label $y \in \{0, 1\}$, using $z$ directly as a prediction fails:

- **Out-of-range predictions.** A prediction of $\hat{y} = 1.4$ has no meaning as a
  probability — there is no "140 percent chance."
- **Outlier sensitivity.** Observations far from the boundary distort the fitted line's
  slope, dragging the decision threshold in the wrong direction.
- **Bad probability model.** The truth is closer to: low $x$ → near-certain class 0
  (probability near 0), high $x$ → near-certain class 1 (probability near 1), with a
  *smooth transition* in between. A straight line cannot capture this saturation.

The fix: take the linear score and **squash it** through a function that maps
$(-\infty, +\infty) \to (0, 1)$. That function is the sigmoid.

---

## 1.1 Assumptions

1. **Bernoulli response.** Each $y_i \in \{0, 1\}$ is Bernoulli with success probability $p_i = \sigma(x_i^T \theta)$.
2. **Independence.** Training examples are i.i.d. given $\theta$.
3. **Linear log-odds.** The log-odds $\log(p_i / (1 - p_i)) = x_i^T \theta$ is a linear function of features.
4. **No perfect multicollinearity.** $X$ has full column rank (required for unique MLE and finite standard errors).
5. **No perfect separability.** The classes cannot be separated by a hyperplane (otherwise MLE doesn't exist; see §5.3).

---

## 2. The Sigmoid Function and Its Identities

### 2.1 Definition

The **sigmoid** (also called the *logistic function*) is

$$\sigma(z) := \frac{1}{1 + e^{-z}}, \qquad z \in \mathbb{R}.$$

Range $(0, 1)$, strictly increasing, infinitely differentiable, $\sigma(0) = 1/2$.

### 2.2 Identities

All three follow from direct calculation:

**(I1) Reflection.**   $\sigma(-z) = 1 - \sigma(z)$.

**(I2) Derivative.**   $\sigma'(z) = \sigma(z) (1 - \sigma(z))$.

**(I3) Logit inverse.**   If $p = \sigma(z)$ with $p \in (0, 1)$, then $z = \log(p / (1 - p))$.

**Proof of (I2).** Write $\sigma(z) = (1 + e^{-z})^{-1}$ and apply the chain rule:

$$\sigma'(z) = -(1 + e^{-z})^{-2} \cdot (-e^{-z}) = \frac{e^{-z}}{(1 + e^{-z})^2} = \sigma(z) \cdot \frac{e^{-z}}{1 + e^{-z}} = \sigma(z) (1 - \sigma(z)).$$

$\blacksquare$

**Reading.** Identity (I2) is the algebraic reason logistic regression has clean
closed-form gradients and Hessian — the derivative of the sigmoid is itself expressible
as a product of sigmoids. Identity (I3) is what lets us interpret $\theta^T x$ as a
*log-odds* (§8).

---

## 3. Likelihood and Cross-Entropy Loss

### 3.1 Model

Conditional on $x_i$ and $\theta$, model $y_i$ as a **Bernoulli** random variable with
success probability $p_i := \sigma(x_i^T \theta)$:

$$y_i \mid x_i, \theta \sim \text{Bernoulli}(p_i), \qquad i = 1, \dots, n.$$

Assume training examples are i.i.d. given $\theta$.

### 3.2 Likelihood

The probability mass function of a Bernoulli is $P(y_i \mid p_i) = p_i^{y_i} (1 - p_i)^{1 - y_i}$.
The full data likelihood is the product:

$$\mathcal{L}(\theta) = \prod_{i=1}^n p_i^{y_i} (1 - p_i)^{1 - y_i}.$$

### 3.3 Negative log-likelihood — the cross-entropy loss

Take the negative log of the likelihood and divide by $n$:

$$L(\theta) := -\frac{1}{n} \log \mathcal{L}(\theta) = -\frac{1}{n} \sum_{i=1}^n \Big[y_i \log p_i + (1-y_i)\log(1-p_i)\Big]. \qquad (3.1)$$

This is the **cross-entropy loss** for binary classification (also called the *binary
log-loss* or *logistic loss*).

Three key observations:

1. **Non-negative.** $p_i \in (0, 1)$ implies $\log p_i \le 0$ and $\log(1 - p_i) \le 0$.
   So $L(\theta) \ge 0$.
2. **Asymmetric per example.** For $y_i = 1$ only $\log p_i$ appears; for $y_i = 0$ only
   $\log(1 - p_i)$. The loss for one example is large exactly when the model puts low
   probability on the *true* class.
3. **Saturates badly at confident wrong predictions.** If $y_i = 1$ but $p_i \to 0$,
   then $-\log p_i \to +\infty$. Cross-entropy *aggressively* punishes confident-but-wrong
   predictions — precisely why softmax + cross-entropy is the default loss in deep learning.

### 3.4 Rewriting the loss with $z_i$ — the margin form

Replace $p_i = \sigma(z_i)$ in (3.1) and use $1 - \sigma(z) = \sigma(-z)$ from (I1):

$$L(\theta) = -\frac{1}{n} \sum_{i=1}^n \Big[ y_i \log \sigma(z_i) + (1 - y_i) \log \sigma(-z_i) \Big].$$

Use the identity $\log \sigma(z) = -\log(1 + e^{-z})$. The single-example loss simplifies
into a clean unified form:

$$\ell_i(\theta) = \log(1 + e^{-y_i' z_i}) \quad \text{where } y_i' := 2 y_i - 1 \in \{-1, +1\}. \qquad (3.2)$$

**Proof.** For $y_i = 1$: $-\log \sigma(z_i) = \log(1 + e^{-z_i}) = \log(1 + e^{-y_i' z_i})$
with $y_i' = +1$. For $y_i = 0$: $-\log \sigma(-z_i) = \log(1 + e^{z_i}) = \log(1 + e^{-y_i' z_i})$
with $y_i' = -1$. $\blacksquare$

**Result:** The cross-entropy loss is a smooth, monotone function of the *margin* $y_i' z_i$.
A large positive margin (correct sign, high magnitude) → small loss. A large negative
margin (wrong sign) → large loss. This is the same *margin* concept that drives SVMs, but
with the smooth log-loss instead of the kinked hinge loss.

---

## 4. Gradient of the Cross-Entropy Loss

### 4.1 Theorem (gradient)

> **Theorem 4.1.** The gradient of the cross-entropy loss (3.1) is
>
> $$\nabla L(\theta) = \frac{1}{n} \sum_{i=1}^n (p_i - y_i) \, x_i = \frac{1}{n} X^T (p - y), \qquad (4.1)$$
>
> where $p \in \mathbb{R}^n$ is the vector with entries $p_i = \sigma(x_i^T \theta)$.

**Proof.** Differentiate the single-example loss with respect to $\theta$. From (3.1):

$$\frac{\partial \ell_i}{\partial \theta} = -\Big[ y_i \cdot \frac{\sigma'(z_i)}{\sigma(z_i)} - (1 - y_i) \cdot \frac{\sigma'(z_i)}{1 - \sigma(z_i)} \Big] \cdot \frac{\partial z_i}{\partial \theta}.$$

Use (I2), $\sigma'(z_i) = \sigma(z_i)(1 - \sigma(z_i)) = p_i(1 - p_i)$, and
$\partial z_i / \partial \theta = x_i$:

$$\frac{\partial \ell_i}{\partial \theta} = -\Big[ y_i (1 - p_i) - (1 - y_i) p_i \Big] x_i = -\Big[ y_i - p_i \Big] x_i = (p_i - y_i) x_i.$$

Average over $i$ to get (4.1). $\blacksquare$

**Result:** $\nabla L(\theta) = \frac{1}{n} X^T (p - y)$.

### 4.2 Comparison with OLS

| Model | Gradient | Residual |
|---|---|---|
| Linear regression | $(2/n) \cdot X^T r$ | $r = X\theta - y$ |
| **Logistic regression** | $(1/n) \cdot X^T r$ | $r = p - y$, with $p_i = \sigma(x_i^T \theta)$ |

The structure is *identical* — sum over examples of (residual) $\cdot$ (feature). The only
change is that the residual is now $p_i - y_i$ (predicted probability minus label). This
is why every linear-regression algorithm (GD, SGD, coordinate descent) generalises straight
to logistic regression with almost no change.

### 4.3 First-order condition

Setting $\nabla L(\theta) = 0$ gives

$$X^T (p - y) = 0. \qquad (4.2)$$

This says the residual vector $p - y$ must be orthogonal to every column of $X$ — the
same geometric condition as OLS, but for the *probability* residual. The crucial difference:
$p$ depends on $\theta$ through the sigmoid, so (4.2) is a *non-linear* equation in $\theta$.

---

## 5. Hessian and Convexity

### 5.1 Theorem (Hessian)

> **Theorem 5.1.** The Hessian of the cross-entropy loss (3.1) is
>
> $$\nabla^2 L(\theta) = \frac{1}{n} X^T W X, \qquad (5.1)$$
>
> where $W \in \mathbb{R}^{n \times n}$ is the diagonal matrix with entries $W_{ii} = p_i (1 - p_i)$.

**Proof.** Differentiate (4.1) once more. Each gradient entry is
$\frac{1}{n} \sum_i (p_i - y_i) x_{ij}$. The derivative of $p_i$ with respect to
$\theta_k$ is $\sigma'(z_i) \cdot x_{ik} = p_i(1 - p_i) x_{ik}$ by (I2). So

$$\frac{\partial^2 L}{\partial \theta_j \partial \theta_k} = \frac{1}{n} \sum_i p_i(1 - p_i) x_{ij} x_{ik}.$$

In matrix form this is exactly $(1/n) \cdot X^T W X$. $\blacksquare$

**Result:** $\nabla^2 L(\theta) = \frac{1}{n} X^T W X$ with $W_{ii} = p_i(1 - p_i)$.

### 5.2 Theorem (convexity)

> **Theorem 5.2.** The cross-entropy loss $L(\theta)$ is convex on $\mathbb{R}^p$. If $X$
> has full column rank and at least one $W_{ii} > 0$, then $L$ is strictly convex.

**Proof.** A twice-differentiable function is convex iff its Hessian is positive
semidefinite everywhere. For any vector $v \in \mathbb{R}^p$:

$$v^T \nabla^2 L(\theta) v = \frac{1}{n} v^T X^T W X v = \frac{1}{n} (X v)^T W (X v) = \frac{1}{n} \sum_i W_{ii} (X v)_i^2.$$

Each $W_{ii} = p_i (1 - p_i) > 0$ for $p_i \in (0, 1)$, and $(X v)_i^2 \ge 0$. So the
sum is $\ge 0$ → positive semidefinite → convex.

If $X$ has full column rank, then $Xv = 0$ implies $v = 0$, so the sum is *strictly*
positive for $v \ne 0$ → positive definite → strictly convex. $\blacksquare$

**Result:** The cross-entropy loss is convex, with a unique global optimum when $X$ is
full rank. This is what separates logistic regression from neural networks (non-convex,
multiple local minima).

### 5.3 Failure case — perfect separability

If the two classes are *perfectly linearly separable*, then $L(\theta)$ has *no finite
minimiser*: shrinking the loss toward zero requires $\|\theta\| \to \infty$ to push the
sigmoid to 0/1 saturation on every example. The MLE does not exist.

**Cure:** Add *any* strictly convex regulariser (e.g. $\lambda \|\theta\|_2^2$, §10).
The regularised loss has a finite minimum even on perfectly separable data.

---

## 6. No Closed-Form Solution

For OLS we had $\hat{\theta}_{\text{OLS}} = (X^T X)^{-1} X^T y$ in closed form. Why does
this not work for logistic regression?

The first-order condition (4.2) is $X^T (p - y) = 0$, which expands to

$$\sum_{i=1}^n \Big( \sigma(x_i^T \theta) - y_i \Big) x_i = 0.$$

Each term contains $\sigma(x_i^T \theta)$ — a *non-linear* function of $\theta$. There is
no way to factor $\theta$ out and isolate it. The equation is *transcendental*.

We are forced to solve iteratively. The good news (Theorem 5.2): the loss is convex, so
any descent algorithm converges to the global optimum from any starting point.

---

## 7. Decision Boundary Geometry

In $d$ dimensions, the logistic model predicts

$$P(y = 1 \mid x) = \sigma(\theta_0 + \theta_1 x_1 + \cdots + \theta_d x_d).$$

The **decision boundary** is the set of points where the model is on the fence —
$P(y = 1 \mid x) = 0.5$ — which (by the reflection identity I1) is exactly

$$\theta_0 + \theta_1 x_1 + \cdots + \theta_d x_d = 0.$$

This is a *hyperplane*. So logistic regression is a **linear classifier**, even though its
probability output is a non-linear S-curve.

What varies across space is how *confident* the model is — not the *shape* of the boundary:

- Far from the hyperplane, on the "1" side, probability tends to 1.
- Far from the hyperplane, on the "0" side, probability tends to 0.
- Right *on* the hyperplane, probability is 0.5.
- The transition band where the probability swings from 0.1 to 0.9 has a width
  controlled by $\|\theta\|$ — a larger $\|\theta\|$ means a sharper transition.

**Practical consequence.** Logistic regression cannot separate XOR-shaped data with a
single boundary. When a non-linear classifier is needed, either engineer non-linear
features (polynomial trick) or use a non-linear model (decision trees, neural networks).

---

## 8. Log-Odds Interpretation

### 8.1 Odds and the logit

**Definition (odds).** The *odds* of an event with probability $p$ is
$\text{odds}(p) := p / (1 - p)$.

- $p = 0.5$ → odds = 1 ("50:50").
- $p = 0.9$ → odds = 9 ("9 to 1 in favour").
- $p = 0.1$ → odds $\approx$ 0.111 ("1 to 9").

The function $p \mapsto \log(p / (1 - p))$ is called the **logit** — and by (I3) it is
the *inverse* of the sigmoid. So if $p = \sigma(z)$, then $z = \log(p / (1 - p))$.

### 8.2 Coefficient interpretation

The model's linear score $z = \theta^T x$ is the **log-odds** of class 1.
Interpretation by coefficient:

- Increasing feature $x_j$ by 1 unit (other features fixed) multiplies the *odds* of
  class 1 by $e^{\theta_j}$.
- $\theta_j > 0$ → feature is associated with class 1.
- $\theta_j < 0$ → feature is associated with class 0.
- $|\theta_j|$ large → feature has a strong effect.

This is exactly the kind of interpretability that makes logistic regression a workhorse in
medicine, social science, and credit risk: the coefficients have a *direct* meaning on the
odds-ratio scale.

---

## 9. Optimisation Methods

Since no closed-form solution exists (§6), we must solve the first-order condition
iteratively. Four methods, in increasing sophistication.

### 9.1 Gradient descent

Update: $\theta_{k+1} \leftarrow \theta_k - \eta \cdot \nabla L(\theta_k)$ with
$\nabla L(\theta_k) = (1/n) X^T(p_k - y)$.

**Step size.** The Lipschitz constant of $\nabla L$ is bounded by

$$L_{\text{smooth}} \le \frac{1}{4 n} \, \lambda_{\max}(X^T X),$$

because $W_{ii} = p_i (1 - p_i) \le 1/4$ for every $p_i \in (0, 1)$ (maximum at
$p_i = 1/2$). With step $\eta = 1 / L_{\text{smooth}}$, convergence is

$$L(\theta_k) - L(\theta^*) \le \frac{\|\theta_0 - \theta^*\|_2^2}{2 \eta \cdot k} = O(1 / k).$$

**Result:** Sub-linear convergence — simple but slow.

### 9.2 Newton's method — IRLS

Newton's method replaces the constant step direction $-g_k$ by
$-(\nabla^2 L(\theta_k))^{-1} g_k$, using local curvature:

$$\theta_{k+1} = \theta_k - \Big(\frac{1}{n} X^T W_k X\Big)^{-1} \cdot \frac{1}{n} X^T(p_k - y).$$

**Why IRLS.** Define the "working response" $z_k := X \theta_k + W_k^{-1} (y - p_k)$.
The Newton update is equivalent to the *weighted least squares* solve:

$$\theta_{k+1} = (X^T W_k X)^{-1} X^T W_k z_k.$$

At each step we solve a *weighted OLS* problem with weights $W_k$ and response $z_k$. The
weights depend on $\theta_k$, so we *re-weight* and re-solve at every iteration — hence
**iteratively reweighted least squares** (IRLS).

**Convergence.** Quadratic: $\|\theta_{k+1} - \theta^*\|_2 \le C \cdot \|\theta_k - \theta^*\|_2^2$.
The number of correct digits *doubles* every iteration. 5–15 iterations reach machine
precision on typical problems.

**Cost per step.** $\Theta(n p^2 + p^3)$ — one $p \times p$ matrix solve per iteration.
Fine when $p \lesssim 10^3$.

**Result:** Newton/IRLS converges quadratically — vastly faster than GD per iteration, at
the cost of $O(p^3)$ per step.

### 9.3 Stochastic gradient descent (SGD)

Compute the gradient on a random *mini-batch* of size $b \ll n$, paying $O(b p)$ per step
instead of $O(n p)$. With a decaying step size $\eta_k \propto 1/\sqrt{k}$, SGD converges
at rate $O(1/\sqrt{k})$ in the convex case.

**When to use:** $n$ very large (millions of examples), online learning (data arrives
in a stream), or deep learning (logistic regression is the smallest such network).

### 9.4 L-BFGS — the practical default

**L-BFGS** (Limited-memory BFGS) is a quasi-Newton method that *approximates* the inverse
Hessian using only the last $m$ gradient differences (typically $m = 10$–$20$). It gets:

- Newton-like fast convergence (super-linear).
- GD-like memory cost ($O(mp)$, no $p \times p$ Hessian).
- No second derivatives needed.

This is the practical default. scikit-learn's `LogisticRegression(solver="lbfgs")` uses
L-BFGS.

**Result:** Use L-BFGS in practice; understand Newton/IRLS conceptually.

---

## 10. Regularised Logistic Regression

### 10.1 The penalised loss

Same recipe as Ridge/Lasso for regression: add a penalty on $\theta$ (excluding the
intercept):

$$L_{\text{ridge}}(\theta) := L(\theta) + \lambda \|\theta\|_2^2, \qquad L_{\text{lasso}}(\theta) := L(\theta) + \lambda \|\theta\|_1. \qquad (10.1)$$

Both are convex (sum of two convex functions). L2-regularised logistic regression is
*strictly* convex even on linearly separable data — the regulariser fixes the failure case
of §5.3.

### 10.2 Bayesian interpretation

> **Theorem 10.2.** Under the Bernoulli likelihood from §3 and the prior
>
> - $\theta_j \sim \text{Normal}(0, \tau^2)$ independently → MAP equals L2-regularised
>   logistic regression with $\lambda = 1 / (2 n \tau^2)$.
> - $\theta_j \sim \text{Laplace}(0, b)$ independently → MAP equals L1-regularised
>   logistic regression with $\lambda = 1 / (n b)$.

**Proof sketch.** Bayes' rule: $\log p(\theta \mid y, X) = \log \mathcal{L}(\theta) + \log p(\theta) + \text{const}$.
Negate and divide by $n$ — the data term is $L(\theta)$, and the prior adds
$\frac{1}{2 n \tau^2} \|\theta\|_2^2$ (Normal) or $\frac{1}{n b} \|\theta\|_1$ (Laplace).
Identifying the multiplier with $\lambda$ gives the claim. $\blacksquare$

### 10.3 Gradient of the regularised loss

- **L2.** $\nabla L_{\text{ridge}}(\theta) = \frac{1}{n} X^T (p - y) + 2 \lambda \theta$.
- **L1.** $\nabla L_{\text{lasso}}(\theta) \ni \frac{1}{n} X^T (p - y) + \lambda \, s$,
  where $s \in \partial \|\theta\|_1$ is a subgradient.

### 10.4 Algorithmic changes

For **L2**: gradient and Hessian pick up extra terms:

$$\nabla^2 L_{\text{ridge}}(\theta) = \frac{1}{n} X^T W X + 2 \lambda I_p.$$

The added $2\lambda I_p$ makes the system strictly positive definite even when $X$ is
rank-deficient — numerically stable Newton solve.

For **L1**: use the proximal-gradient (ISTA) recipe — alternate a gradient step on the
smooth data term with a soft-thresholding step for the L1 penalty. FISTA acceleration
applies unchanged.

**Result:** L1 logistic regression produces sparse coefficients (variable selection for
classification). L2 logistic regression shrinks coefficients smoothly and is the default
in most software (scikit-learn uses L2 with $C = 1/\lambda$).

---

## 11. Multi-Class Extension: Softmax Regression

For $K \ge 3$ classes, replace the sigmoid by the **softmax** function and the Bernoulli
by the **categorical** distribution.

### 11.1 Definition (softmax)

Given a vector of scores $z \in \mathbb{R}^K$:

$$\text{softmax}(z)_k = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}}, \qquad k = 1, \dots, K.$$

By construction $\text{softmax}(z)_k \in (0, 1)$ and $\sum_k \text{softmax}(z)_k = 1$.
When $K = 2$, the softmax reduces to the sigmoid (up to re-parameterisation $z = z_1 - z_2$).

### 11.2 Model

One parameter vector $\theta_k \in \mathbb{R}^p$ per class. Stack as $\Theta \in \mathbb{R}^{K \times p}$:

$$P(y_i = k \mid x_i, \Theta) = \text{softmax}(\Theta x_i)_k = \frac{e^{\theta_k^T x_i}}{\sum_{j=1}^K e^{\theta_j^T x_i}}.$$

### 11.3 Loss

Encode each label as a one-hot vector $y_i \in \{0, 1\}^K$. The **multinomial
cross-entropy** loss is:

$$L(\Theta) = -\frac{1}{n} \sum_{i=1}^n \sum_{k=1}^K y_{ik} \log P(y_i = k \mid x_i, \Theta). \qquad (11.1)$$

### 11.4 Gradient

Same shape as Theorem 4.1, per class:

$$\frac{\partial L}{\partial \theta_k} = \frac{1}{n} \sum_{i=1}^n (P(y_i = k \mid x_i, \Theta) - y_{ik}) \, x_i. \qquad (11.2)$$

**Result:** The multinomial gradient has exactly the binary structure $(1/n) X^T(P - Y)$,
repeated per class.

### 11.5 Identifiability

Softmax has a redundancy: shifting every $\theta_k$ by the same vector $c$ leaves the
probabilities unchanged. So $\Theta$ is identifiable only up to a constant shift — fixed
by either (i) setting one class as reference ($\theta_K = 0$), or (ii) adding an L2 penalty.

---

## 12. Statistical Properties

### 12.1 Consistency

Under mild regularity ($X$ full rank with high probability, $E[\|x_i\|^4] < \infty$), the
MLE is **consistent**: $\hat{\theta}_n \to \theta^*$ in probability as $n \to \infty$.

### 12.2 Asymptotic normality

$$\sqrt{n} (\hat{\theta}_n - \theta^*) \xrightarrow{d} \text{Normal}(0, \mathcal{I}(\theta^*)^{-1}),$$

where $\mathcal{I}(\theta^*)$ is the **Fisher information matrix**:

$$\mathcal{I}(\theta^*) := \mathbb{E}_x [\sigma(x^T \theta^*) (1 - \sigma(x^T \theta^*)) \, x x^T].$$

The empirical analogue is $(1/n) X^T W X$ — the Hessian from Theorem 5.1.

### 12.3 Standard errors and confidence intervals

$$\widehat{\text{Var}}(\hat{\theta}_n) \approx (X^T \widehat{W} X)^{-1},$$

where $\widehat{W}$ uses plug-in probabilities $\hat{p}_i = \sigma(x_i^T \hat{\theta}_n)$.
The $j$-th diagonal entry gives $\widehat{SE}(\hat{\theta}_j)$; a 95% CI is
$\hat{\theta}_j \pm 1.96 \cdot \widehat{SE}(\hat{\theta}_j)$.

### 12.4 Threshold selection

The model outputs $p(x) = P(y = 1 \mid x)$. To produce a class label, pick threshold $\tau$:

- **0-1 loss.** Bayes-optimal threshold is $\tau = 0.5$.
- **Asymmetric costs.** If false positive costs $c_{10}$ and false negative costs $c_{01}$:

$$\tau^* = \frac{c_{10}}{c_{10} + c_{01}}.$$

### 12.5 Calibration

A classifier is **calibrated** if $P(y = 1 \mid p(x) = q) = q$ for every $q \in (0, 1)$.
Logistic regression is usually well-calibrated when the model is correctly specified —
*because* its loss is the cross-entropy (Bernoulli likelihood).

### 12.6 ROC and AUC

Sweep $\tau$ from 1 to 0. At each $\tau$: TPR = $P(\hat{y}=1 \mid y=1)$,
FPR = $P(\hat{y}=1 \mid y=0)$. The ROC curve plots TPR vs FPR. **AUC** (Area Under the
Curve) is threshold-free and equals $P(p(x_+) > p(x_-))$ — the probability that the model
ranks a random positive higher than a random negative.

---

## 13. Connections

- **Linear regression.** Same gradient structure $(1/n) X^T r$; residual changes from
  $X\theta - y$ to $\sigma(X\theta) - y$.
- **Ridge / Lasso regression.** Same penalty terms, same Bayesian interpretation; only
  the data term changes from squared error to cross-entropy.
- **Smallest neural network.** A logistic regression model is exactly one linear layer →
  sigmoid activation → output. Stack two and you have a 2-layer MLP. The training
  algorithm (gradient descent on cross-entropy) is the *same* algorithm used to train
  every modern deep network.
- **Support vector machines.** Same margin concept ($y' z$), but with the smooth log-loss
  $\log(1 + e^{-y' z})$ instead of the kinked hinge loss $\max(0, 1 - y' z)$.
- **Generalised linear models (GLM).** IRLS generalises to any exponential family
  (Poisson regression, gamma regression) by varying $W_k$ and $z_k$.
- [Probability](../../foundations/probability_statistics/README.md)
- [Gradient Descent](../02_gradient_descent/README.md)
- [Loss Functions](../../synthesis/loss_functions_map.md)

---

## 14. Failure Modes

1. **Non-linear decision boundary.** Logistic regression's boundary is a hyperplane (§7).
   XOR-shaped data, images, high-order interactions → needs feature engineering or
   non-linear models.
2. **Perfect separability.** Unregularised MLE doesn't exist (§5.3). **Cure:** L2
   regularisation.
3. **Many irrelevant features.** Variance scales with $p/n$. **Cure:** L1 regularisation
   for variable selection.
4. **Class imbalance.** Default threshold 0.5 predicts majority class. **Cure:** tune
   threshold from cost ratio (§12.4), or use class weighting.
5. **Model mis-specification.** When the true decision function is far from linear,
   probabilities can be over-confident. **Cure:** post-hoc calibration or a more
   flexible model.
