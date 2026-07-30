# Floating-Point Stability

## Representation Error

Most real numbers cannot be represented exactly in binary floating point. For normalized
operations, a common model is

$$\operatorname{fl}(x\circ y)=(x\circ y)(1+\delta),\qquad |\delta|\lesssim u,$$

where $u$ is unit roundoff and $\circ$ is a basic arithmetic operation. Repeated
operations can amplify these local errors.

## Conditioning and Stability

Conditioning describes sensitivity of the mathematical problem; stability describes the
algorithm. A stable algorithm cannot recover information that an ill-conditioned problem
does not contain.

For a nonsingular matrix $A$, the $2$-norm condition number is

$$\kappa_2(A)=\lVert A\rVert_2\lVert A^{-1}\rVert_2
=\frac{\sigma_{\max}(A)}{\sigma_{\min}(A)}.$$

Forming $X^\top X$ squares the spectral condition number, which is why least squares
should normally use QR or SVD instead of explicit normal-equation inversion.

## Stable ML Patterns

- solve linear systems instead of multiplying by an inverse;
- use `eigh` for symmetric eigendecomposition;
- compute probabilities in log space;
- use `logsumexp` for logarithms of exponential sums;
- subtract the maximum logit before softmax;
- compare floats with scale-aware `atol` and `rtol`;
- gradient-check small differentiable components with central differences.

## Example: Stable Softmax

For logits $z\in\mathbb R^K$ and $m=\max_kz_k$,

$$\operatorname{softmax}(z)_k=\frac{e^{z_k-m}}{\sum_je^{z_j-m}}.$$

Subtracting $m$ leaves the mathematical ratio unchanged while preventing overflow.

## Connections

- [Linear Regression](../../topics/01_linear_regression/README.md)
- [Naive Bayes](../../topics/08_naive_bayes/README.md)
- [Neural Networks](../../topics/13_neural_networks/README.md)
