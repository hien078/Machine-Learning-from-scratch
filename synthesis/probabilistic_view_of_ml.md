# Probabilistic View of Machine Learning

## Purpose

Show how assumptions about a data-generating distribution produce familiar losses and
estimators.

## From Likelihood to Loss

Given independent observations $\{(x_i,y_i)\}_{i=1}^n$ and parameters $\theta$,
maximum likelihood solves

$$
\hat\theta_{\mathrm{MLE}}
=\arg\max_\theta\sum_{i=1}^n\log p(y_i\mid x_i,\theta).
$$

Minimizing negative log-likelihood gives the training loss.

| Distributional model | Negative log-likelihood, up to constants | Method |
|---|---|---|
| Gaussian with fixed variance | Squared error | Linear Regression |
| Bernoulli with logistic mean | Binary cross-entropy | Logistic Regression |
| Categorical with softmax probabilities | Categorical cross-entropy | Multiclass neural classifier |
| Class-conditional Gaussians | Gaussian log-density plus log prior | Gaussian Naive Bayes |
| Gaussian mixture with latent component | Log-sum likelihood | GMM with EM |

## MAP and Regularization

Maximum a posteriori estimation adds a log prior:

$$
\hat\theta_{\mathrm{MAP}}
=\arg\min_\theta\left[-\log p(\mathcal D\mid\theta)-\log p(\theta)\right].
$$

A zero-mean Gaussian prior yields an $\ell_2$ penalty; a Laplace prior yields an
$\ell_1$ penalty. The coefficient depends on the likelihood normalization and prior
scale, so repository-wide objective conventions must be stated explicitly.

## Generative and Discriminative Models

Naive Bayes models $p(x\mid y)p(y)$ and obtains $p(y\mid x)$ by Bayes' rule. Logistic
Regression models $p(y\mid x)$ directly. The generative model can exploit stronger
assumptions with little data; the discriminative model avoids modeling the feature
distribution.

## Connections

- [Probability and Statistics](../foundations/probability_statistics/README.md)
- [Logistic Regression](../topics/04_logistic_regression/README.md)
- [Naive Bayes](../topics/08_naive_bayes/README.md)
- [Clustering](../topics/11_clustering/README.md)
- [Regularization Across Models](regularization_across_models.md)
