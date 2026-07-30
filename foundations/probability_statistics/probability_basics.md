# Probability Theory for Machine Learning

## Events and Conditional Probability

For events $A$ and $B$ with $P(B)>0$,

$$P(A\mid B)=\frac{P(A\cap B)}{P(B)}.$$

Events are independent when $P(A\cap B)=P(A)P(B)$. Conditional independence is different:
$X_1\perp X_2\mid Y$ can hold even when $X_1$ and $X_2$ are marginally dependent.

## Bayes' Rule

$$P(Y\mid X)=\frac{P(X\mid Y)P(Y)}{P(X)}.$$

The likelihood $P(X\mid Y)$, prior $P(Y)$, evidence $P(X)$, and posterior $P(Y\mid X)$
play different roles. For classification, the evidence is shared across candidate classes
and can be omitted inside an $\arg\max$.

## Random Variables

A discrete random variable has probability mass function $p(x)=P(X=x)$. A continuous
random variable has density $f(x)$ satisfying
$P(a\le X\le b)=\int_a^b f(x)\,dx$. A density value is not itself a probability.

Expectation and variance are

$$\mathbb{E}[X]=\sum_x x p(x)\quad\text{or}\quad\int x f(x)\,dx,$$

$$\operatorname{Var}(X)=\mathbb{E}[(X-\mathbb{E}[X])^2].$$

Covariance measures linear co-variation:
$\operatorname{Cov}(X,Y)=\mathbb{E}[(X-\mathbb{E}[X])(Y-\mathbb{E}[Y])]$.

## Sampling and Estimation

An estimand is a population quantity, while an estimator is a random function of a
sample. Bias, variance, consistency, and uncertainty describe different estimator
properties. A low training loss alone establishes none of them.

## ML Connections

- [Linear Regression](../../topics/01_linear_regression/README.md): noise and inference
- [Logistic Regression](../../topics/04_logistic_regression/README.md): Bernoulli likelihood
- [Naive Bayes](../../topics/08_naive_bayes/README.md): conditional independence
- [Clustering](../../topics/11_clustering/README.md): mixture likelihoods
