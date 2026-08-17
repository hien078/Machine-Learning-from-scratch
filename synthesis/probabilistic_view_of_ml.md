# Probabilistic View of ML — Cross-Topic Synthesis

> One assumption — the data came from a distribution — generates most of the objectives in this repo.
> See [INDEX.md](../INDEX.md) for the full curriculum index.

---

## Overview

Few losses in this repository are ad hoc. Almost every training objective follows from
two choices: a **distributional model** of the data and an **estimation principle**.
There are three estimation principles, forming a ladder of increasing Bayesian commitment:

| Level | Estimate | Objective | Repo appearance |
|---|---|---|---|
| MLE | Point $\hat\theta$ | $\arg\max_\theta \log p(\mathcal{D}\mid\theta)$ | Least squares, cross-entropy, GMM/EM |
| MAP | Point $\hat\theta$ + prior | $\arg\max_\theta \left[\log p(\mathcal{D}\mid\theta)+\log p(\theta)\right]$ | Ridge, Lasso, weight decay, Laplace smoothing |
| Bayesian | Posterior $p(\theta\mid\mathcal{D})$ | Integrate, don't optimize | VAE (variational, over latents) |

---

## MLE Generates the Standard Losses

Given independent observations $\lbrace(x_i,y_i)\rbrace_{i=1}^n$, maximum likelihood solves

```math
\hat{\theta}_{\mathrm{MLE}}=\arg\max_\theta\sum_{i=1}^n\log p(y_i\mid x_i,\theta).
```

Minimizing the negative log-likelihood (NLL) gives the training loss. The key cases:
Gaussian noise $y=f_\theta(x)+\varepsilon$ makes the NLL
$\frac{1}{2\sigma^2}\sum_i(y_i-f_\theta(x_i))^2$ plus constants — least squares *is*
Gaussian MLE ([Linear Regression](../topics/01_linear_regression/theory.md)); a
Bernoulli mean $\sigma(w^\top x)$ makes the NLL exactly BCE
([Logistic Regression](../topics/04_logistic_regression/theory.md)); a categorical
softmax makes it cross-entropy ([Neural Networks](../topics/13_neural_networks/theory.md)).

**Cross-entropy is KL minimization:** since $\mathrm{CE}(p,q)=H(p)+D_{\mathrm{KL}}(p\mid q)$
and $H(p)$ does not depend on $\theta$, minimizing CE against the empirical label
distribution minimizes the KL divergence from data to model. MLE itself is
asymptotically KL projection of the true distribution onto the model family.

| Distributional model | NLL, up to constants | Method |
|---|---|---|
| Gaussian with fixed variance | Squared error | Linear Regression |
| Laplace noise | Absolute error | Robust regression |
| Bernoulli with logistic mean | Binary cross-entropy | Logistic Regression |
| Categorical with softmax probabilities | Categorical cross-entropy | Multiclass neural classifier |
| Class-conditional factorized $p(x\mid y)p(y)$ | Joint NLL | Naive Bayes |
| Gaussian mixture with latent component | Marginal (log-sum) NLL | GMM with EM |

---

## MAP Generates Regularization

Maximum a posteriori estimation adds a log prior:

```math
\hat{\theta}_{\mathrm{MAP}}=\arg\min_\theta\left[-\log p(\mathcal{D}\mid\theta)-\log p(\theta)\right].
```

| Prior on $w$ | $-\log p(w)$ up to constants | Penalty | Method |
|---|---|---|---|
| Gaussian $\mathcal{N}(0,\tau^2 I)$ | $\frac{1}{2\tau^2}\Vert w\Vert_2^2$ | $\ell_2$ | Ridge, weight decay |
| Laplace, scale $b$ | $\frac{1}{b}\Vert w\Vert_1$ | $\ell_1$ | Lasso |
| Dirichlet on class-conditional counts | Additive pseudo-counts | Laplace smoothing | Naive Bayes |

With Gaussian noise variance $\sigma^2$, the ridge coefficient is $\lambda=\sigma^2/\tau^2$:
strong priors (small $\tau$) mean strong shrinkage. The coefficient depends on likelihood
normalization ($\frac{1}{n}$ vs $\frac{1}{2n}$ vs none), so objective conventions must
be stated explicitly. Note also that Lasso's exact zeros are a property of the MAP
*point estimate* (the Laplace prior is non-differentiable at 0); the full posterior mean
under the same prior is not sparse. See
[Regularization](../topics/03_regularization/theory.md) and
[Regularization Across Models](regularization_across_models.md).

---

## Generative vs Discriminative: the Classic Pairing

Naive Bayes and Logistic Regression solve the same classification problem from opposite
directions.

| | Naive Bayes ([08](../topics/08_naive_bayes/theory.md)) | Logistic Regression ([04](../topics/04_logistic_regression/theory.md)) |
|---|---|---|
| Models | $p(x\mid y)\,p(y)$, then Bayes' rule | $p(y\mid x)$ directly |
| Extra assumption | Conditional feature independence | None on $p(x)$ |
| Fit | Closed-form counts/moments | Iterative (convex) optimization |
| Small data | Often better (strong bias, low variance) | Needs more data |
| Large data | Asymptotic error limited by wrong independence assumption | Lower asymptotic error |
| Misspecification | Hurts twice: $p(x\mid y)$ and the induced boundary | Only the boundary can be wrong |

The connection is exact in one case: Gaussian Naive Bayes with shared class covariance
induces a posterior $p(y\mid x)$ that is *exactly* a logistic function of a linear score —
the same hypothesis class, estimated by different principles.

---

## Latent Variables: EM and the ELBO

When the model has unobserved variables $z$, the marginal likelihood
$\log p(x)=\log\sum_z p(x,z)$ puts a sum inside the log and blocks closed-form MLE.
Both EM and the VAE work with the same decomposition:

```math
\log p_\theta(x)=\underbrace{\mathbb{E}_{q(z)}\left[\log\frac{p_\theta(x,z)}{q(z)}\right]}_{\mathrm{ELBO}}+D_{\mathrm{KL}}\big(q(z)\mid p_\theta(z\mid x)\big).
```

- **EM (GMM):** the E-step sets $q(z)=p_\theta(z\mid x)$ exactly (the responsibilities),
  making the bound tight; the M-step maximizes the ELBO over $\theta$ in closed form.
  Monotone likelihood ascent follows ([Clustering](../topics/11_clustering/theory.md)).
- **VAE:** the exact posterior is intractable, so $q_\phi(z\mid x)$ is an encoder network
  and the ELBO is maximized jointly over $(\theta,\phi)$ by stochastic gradients — EM with
  an amortized, approximate E-step ([Autoencoder](../topics/17_autoencoder/theory.md),
  [Generative Models](../topics/19_generative_models/theory.md)).
- **K-Means** is the hard-assignment, $\sigma^2\to 0$ limit of EM on spherical Gaussians.

---

## The Map: Model → Likelihood → Prior → Objective

| Model | Likelihood | Prior | Objective (NLL form) | Topic |
|---|---|---|---|---|
| Linear Regression | Gaussian $y\mid x$ | Flat | MSE | [01](../topics/01_linear_regression/README.md) |
| Ridge | Gaussian $y\mid x$ | Gaussian on $w$ | MSE $+\lambda\Vert w\Vert_2^2$ | [03](../topics/03_regularization/README.md) |
| Lasso | Gaussian $y\mid x$ | Laplace on $w$ | MSE $+\lambda\Vert w\Vert_1$ | [03](../topics/03_regularization/README.md) |
| Logistic Regression | Bernoulli $y\mid x$ | Flat (or Gaussian) | BCE (+ $\ell_2$) | [04](../topics/04_logistic_regression/README.md) |
| Softmax classifier / NN | Categorical $y\mid x$ | Flat (Gaussian = weight decay) | Cross-entropy | [13](../topics/13_neural_networks/README.md) |
| Naive Bayes | Factorized $p(x\mid y)p(y)$ | Dirichlet (smoothing) | Joint NLL | [08](../topics/08_naive_bayes/README.md) |
| GMM | Mixture, latent $z$ | Flat | Marginal NLL via EM | [11](../topics/11_clustering/README.md) |
| K-Means | Spherical Gaussian, hard limit | — | Distortion | [11](../topics/11_clustering/README.md) |
| Probabilistic PCA | Linear-Gaussian latent | $z\sim\mathcal{N}(0,I)$ | Gaussian NLL | [10](../topics/10_pca/README.md) |
| VAE | $p_\theta(x\mid z)$, latent $z$ | $z\sim\mathcal{N}(0,I)$ | $-\mathrm{ELBO}$ | [17](../topics/17_autoencoder/README.md) |
| Language model | Categorical next token | Flat | Cross-entropy | [16](../topics/16_transformer/README.md) |

---

## Connections

- **Topics:** [01 Linear Regression](../topics/01_linear_regression/README.md), [03 Regularization](../topics/03_regularization/README.md), [04 Logistic Regression](../topics/04_logistic_regression/README.md), [08 Naive Bayes](../topics/08_naive_bayes/README.md), [11 Clustering](../topics/11_clustering/README.md), [17 Autoencoder](../topics/17_autoencoder/README.md), [19 Generative Models](../topics/19_generative_models/README.md)
- **Foundations:** [Probability and Statistics](https://github.com/hien078/applied-mathematics-foundation)
- **Related synthesis:** [Loss Functions Map](loss_functions_map.md), [Regularization Across Models](regularization_across_models.md)
- **Maps:** [INDEX.md](../INDEX.md)
