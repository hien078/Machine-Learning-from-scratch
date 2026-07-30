# Derivatives, Gradients, Jacobians, and Hessians

## Scalar Derivatives

The derivative of $f:\mathbb{R}\to\mathbb{R}$ at $x$ is

$$f'(x)=\lim_{h\to0}\frac{f(x+h)-f(x)}{h},$$

when the limit exists. It is the best local linear coefficient:
$f(x+h)=f(x)+f'(x)h+o(h)$.

## Gradients and Directional Derivatives

For $f:\mathbb{R}^d\to\mathbb{R}$, the gradient is

$$\nabla f(x)=\begin{bmatrix}\partial f/\partial x_1&\cdots&\partial f/\partial x_d\end{bmatrix}^\top.$$

For a unit direction $v$, the directional derivative is
$D_vf(x)=\nabla f(x)^\top v$. Cauchy–Schwarz shows this is largest when $v$ points along
$\nabla f(x)$, which explains why $-\nabla f(x)$ is steepest descent under the Euclidean
norm.

## Jacobians and Chain Rule

For $g:\mathbb{R}^d\to\mathbb{R}^m$, the Jacobian
$J_g(x)\in\mathbb{R}^{m\times d}$ contains first partial derivatives. If
$f:\mathbb{R}^m\to\mathbb{R}$, then

$$\nabla_x(f\circ g)(x)=J_g(x)^\top\nabla f(g(x)).$$

Backpropagation is repeated application of this vector chain rule while reusing
intermediate values.

## Hessian and Curvature

For twice-differentiable scalar $f$, the Hessian is
$H_f(x)=\nabla^2f(x)$. A symmetric Hessian with nonnegative eigenvalues everywhere is a
sufficient condition for convexity. Eigenvalue scale controls local curvature and the
stable learning-rate range of gradient descent on quadratics.

The **condition number** $\kappa=\lambda_{\max}/\lambda_{\min}$ of the Hessian measures
how elongated the local landscape is. For quadratics, gradient descent with optimal
step size reduces the error by a factor $(\kappa-1)/(\kappa+1)$ per iteration. Large
$\kappa$ means a narrow valley: fast oscillation across it, slow progress along it.
This motivates momentum, adaptive learning rates (Adam), and second-order methods.

## Matrix Derivative Example

For $L(w)=\frac1n\lVert Xw-y\rVert_2^2$,

$$\nabla_wL(w)=\frac{2}{n}X^\top(Xw-y),\qquad
\nabla_w^2L(w)=\frac{2}{n}X^\top X.$$

**Result:** least-squares curvature is positive semidefinite and becomes positive
definite when $X$ has full column rank.

## Matrix Calculus Identities

Common identities reused across ML derivations. $A$ is a constant matrix, $a$ a
constant vector, $x$ a variable vector.

| Expression | Gradient w.r.t. $x$ | Notes |
|---|---|---|
| $a^\top x$ | $a$ | Linear model prediction |
| $x^\top A x$ | $(A+A^\top)x$ | Simplifies to $2Ax$ when $A$ symmetric |
| $\lVert x\rVert_2^2$ | $2x$ | Special case $A=I$ |
| $\lVert Ax-b\rVert_2^2$ | $2A^\top(Ax-b)$ | Least-squares gradient |

For matrix-variable derivatives, see the [Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf)
for a comprehensive reference.

## Numerical Gradient Checking

The **central-difference** approximation verifies analytical gradients:

$$\frac{\partial f}{\partial x_i}\approx\frac{f(x+\epsilon e_i)-f(x-\epsilon e_i)}{2\epsilon},\qquad\epsilon\sim10^{-5}\text{–}10^{-7}.$$

Central difference has $O(\epsilon^2)$ error, versus $O(\epsilon)$ for the one-sided formula.
In practice, compare the analytical gradient $g$ against the numerical estimate $\hat{g}$
using relative error:

$$\frac{\lVert g-\hat{g}\rVert}{\lVert g\rVert+\lVert\hat{g}\rVert}<10^{-5}.$$

This is the primary verification tool when implementing gradients from scratch.

## Subgradients

Not all functions in ML are differentiable everywhere. ReLU is non-differentiable at
$x=0$; the $L^1$ norm $\lvert x\rvert$ has a kink at the origin.

A vector $s$ is a **subgradient** of convex $f$ at $x$ if for all $z$:

$$f(z)\ge f(x)+s^\top(z-x).$$

The set of all subgradients at $x$ is the **subdifferential** $\partial f(x)$. At
differentiable points, $\partial f(x)=\{\nabla f(x)\}$. Gradient descent still converges
on non-smooth convex problems when using any subgradient in place of the gradient,
though convergence may slow from $O(1/t)$ to $O(1/\sqrt{t})$.

In practice, frameworks like PyTorch assign $\text{ReLU}'(0)=0$ by convention and this
works well for training.

## Gradient Flow

Gradient descent $x_{t+1}=x_t-\eta\nabla f(x_t)$ is the **Euler method** applied to the
continuous ODE

$$\frac{dx}{dt}=-\nabla f(x),$$

called the **gradient flow**. The step size $\eta$ plays the role of the Euler time step
$h$: too large causes divergence (just as in numerical ODE solving), too small wastes
iterations. This viewpoint connects optimization convergence theory directly to ODE
stability analysis.

## Connections

- [Gradient Descent](../../topics/02_gradient_descent/README.md)
- [Logistic Regression](../../topics/04_logistic_regression/README.md)
- [Neural Networks](../../topics/13_neural_networks/README.md)
