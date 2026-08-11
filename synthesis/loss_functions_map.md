# Loss Functions Map — Cross-Topic Comparison

> How loss functions connect across ML models.
> See [INDEX.md](../INDEX.md) for the full curriculum index.

---

## Overview

Many supervised estimators minimize an explicit empirical loss. Others, such as KNN,
have no parameter-fitting objective, while trees greedily optimize split criteria. When
an explicit loss exists, its choice determines:
- What the model optimizes for
- What mathematical properties the solution has
- How the model handles outliers, class imbalance, and noise

---

## Regression Losses

### Mean Squared Error (MSE)

$$\mathcal{L}_{\text{MSE}} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

| Property | Value |
|---|---|
| Used by | Linear Regression, Polynomial Regression, Ridge, Neural Networks |
| Assumption | Gaussian noise on targets |
| Sensitivity | Sensitive to outliers (squared penalty) |
| Gradient | $\nabla_{\hat{\mathbf y}}\mathcal{L} = \frac{2}{n}(\hat{\mathbf y}-\mathbf y)$ |
| Probabilistic view | Equivalent to MLE under $y \sim \mathcal{N}(\hat{y}, \sigma^2)$ |
| Convexity | Convex in $\hat{y}$ |

### Mean Absolute Error (MAE)

$$\mathcal{L}_{\text{MAE}} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

| Property | Value |
|---|---|
| Used by | Robust regression; Lasso uses a separate $\ell_1$ parameter penalty |
| Assumption | Laplacian noise |
| Sensitivity | Robust to outliers |
| Gradient | Non-differentiable at 0 (use subgradient) |
| Probabilistic view | MLE under $y \sim \text{Laplace}(\hat{y}, b)$ |

### Huber Loss

$$\mathcal{L}_{\text{Huber}} = \begin{cases} \frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \le \delta \\ \delta |y - \hat{y}| - \frac{1}{2}\delta^2 & \text{otherwise} \end{cases}$$

Combines MSE (near zero) with MAE (far from zero). Differentiable everywhere.

---

## Classification Losses

### Binary Cross-Entropy (Log Loss)

$$\mathcal{L}_{\text{BCE}} = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(\hat{p}_i) + (1-y_i) \log(1-\hat{p}_i) \right]$$

| Property | Value |
|---|---|
| Used by | Logistic Regression, Neural Networks (binary) |
| Assumption | Bernoulli distribution on targets |
| Sensitivity | Penalizes confident wrong predictions heavily |
| Logit gradient | $\partial \ell/\partial z = \hat{p}-y$ for $\hat{p}=\sigma(z)$ |
| Probabilistic view | Negative log-likelihood of Bernoulli |
| Convexity | Convex in log-odds |

### Categorical Cross-Entropy

$$\mathcal{L}_{\text{CE}} = -\frac{1}{n} \sum_{i=1}^{n} \sum_{k=1}^{K} y_{ik} \log(\hat{p}_{ik})$$

| Property | Value |
|---|---|
| Used by | Softmax classifier, Neural Networks (multiclass) |
| Connection | Generalizes BCE to K classes |
| Related to | KL divergence: $\text{CE}(p, q) = H(p) + D_{\text{KL}}(p \mid q)$ |

### Hinge Loss

$$\mathcal{L}_{\text{hinge}} = \frac{1}{n} \sum_{i=1}^{n} \max(0, 1 - y_i \cdot f(x_i))$$

| Property | Value |
|---|---|
| Used by | SVM (linear and kernel) |
| Assumption | Maximum margin classifier |
| Sensitivity | Only penalizes violations of the margin |
| Gradient | Subgradient — not differentiable at $y \cdot f(x) = 1$ |
| Sparsity | The dual optimum often has many zero coefficients; primal weights need not be sparse |

---

## Regularization Penalties

| Penalty | Formula | Effect | Used by |
|---|---|---|---|
| L2 (Ridge) | $\lambda \Vert \mathbf{w}\Vert_2^2$ | Shrinks all weights toward zero | Ridge, Weight Decay |
| L1 (Lasso) | $\lambda \Vert \mathbf{w}\Vert_1$ | Drives some weights to exactly zero | Lasso, Sparse models |
| Elastic Net | $\lambda_1 \Vert \mathbf{w}\Vert_1 + \lambda_2 \Vert \mathbf{w}\Vert_2^2$ | Combines sparsity + shrinkage | Elastic Net |
| Dropout | Random activation masking during training | Stochastic capacity control | Neural Networks |

---

## Information-Theoretic Losses

### Entropy

$$H(p) = -\sum_k p_k \log p_k$$

- Used by: Decision Trees (splitting criterion)
- Measures: Impurity/uncertainty of a distribution

### Gini Impurity

$$G(p) = 1 - \sum_k p_k^2$$

- Used by: Decision Trees (alternative to entropy)
- Often produces splits similar to entropy, but this is data dependent

### KL Divergence

$$D_{\text{KL}}(p \mid q) = \sum_k p_k \log \frac{p_k}{q_k}$$

- Used by: VAE (ELBO), t-SNE, knowledge distillation
- Not symmetric, not a true metric
- Connection: $\text{CE}(p, q) = H(p) + D_{\text{KL}}(p \mid q)$

---

## The Big Picture

```
                    Regression                    Classification
                    ─────────                     ──────────────
    Target:         Continuous y                  Discrete y ∈ {0,1,...,K}
    
    Standard:       MSE ← Gaussian MLE            BCE ← Bernoulli MLE
                    ↓                              ↓
    Robust:         MAE ← Laplace MLE              Hinge ← Max margin
                    ↓                              ↓  
    Hybrid:         Huber                          Focal Loss
    
    Regularized:    + L2 (Ridge)                   + L2 (Weight Decay)
                    + L1 (Lasso)                   + Dropout
                    + ElasticNet                   + Early Stopping
```

---

## Connections

- **Topics:** [01 Linear Regression](../topics/01_linear_regression/README.md), [03 Regularization](../topics/03_regularization/README.md), [04 Logistic Regression](../topics/04_logistic_regression/README.md), [05 Decision Tree](../topics/05_decision_tree/README.md), [09 SVM](../topics/09_svm/README.md), [13 Neural Networks](../topics/13_neural_networks/README.md)
- **Foundations:** [Calculus & Optimization](https://github.com/hien078/applied-mathematics-foundation), [Information Theory](https://github.com/hien078/applied-mathematics-foundation)
- **Maps:** [INDEX.md](../INDEX.md)
