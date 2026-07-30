# Entropy, Cross-Entropy, and Information

For a discrete distribution $p$ on outcomes $1,\ldots,K$, Shannon entropy is

$$H(p)=-\sum_{k=1}^Kp_k\log p_k.$$

The logarithm base determines units: base $2$ gives bits and the natural logarithm gives
nats. Entropy is nonnegative and is maximized by the uniform distribution on a fixed
finite support.

For a true distribution $p$ and model distribution $q$, cross-entropy is

$$H(p,q)=-\sum_{k=1}^Kp_k\log q_k.$$

It decomposes as

$$H(p,q)=H(p)+D_{\mathrm{KL}}(p\lVert q),$$

where

$$D_{\mathrm{KL}}(p\lVert q)=\sum_{k=1}^Kp_k\log\frac{p_k}{q_k}\ge0.$$

Therefore minimizing cross-entropy with fixed $p$ is equivalent to minimizing KL
divergence. KL divergence is asymmetric and is not a metric.

Decision trees use entropy of empirical class proportions as an impurity measure.
Logistic and neural classifiers minimize sample cross-entropy, which is the negative
log-likelihood of categorical targets.

## Connections

- [Decision Tree](../../topics/05_decision_tree/README.md)
- [Logistic Regression](../../topics/04_logistic_regression/README.md)
- [Dimensionality Reduction](../../topics/12_dimensionality_reduction/README.md)
- [Loss Functions](../../synthesis/loss_functions_map.md)
