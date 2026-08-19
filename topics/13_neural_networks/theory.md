# Neural Networks (MLP) — Theory

## 1. WHY — Beyond Linear Models

### 1.1 The problem

Linear and logistic regression fit a single hyperplane. Many real relationships —
XOR-style interactions, curved boundaries, feature interactions of unknown form — are
not representable by any hyperplane in the raw features.

The classical fix is manual feature engineering (products, polynomial terms, kernels). A
neural network instead **learns the features**: it composes simple parametric maps and
lets gradient descent choose them.

### 1.2 Why nonlinearity is necessary — composition of affine maps is affine

Suppose we stack two "layers" with no activation in between:

```math
f(x) = W_2\,(W_1 x + b_1) + b_2 .
```

Expand by distributivity of matrix multiplication over addition:

```math
f(x) = (W_2 W_1)\, x + (W_2 b_1 + b_2) = W' x + b' ,
\qquad W' := W_2 W_1,\; b' := W_2 b_1 + b_2 .
```

**Result:** by induction, any depth-$L$ stack of purely affine layers collapses to a
*single* affine map — exactly the expressive power of linear regression, regardless of
depth or width.

Inserting a nonlinear activation $\phi$ between layers breaks the collapse:
$W_2\,\phi(W_1 x + b_1) + b_2$ is no longer affine in $x$, and the class of representable
functions grows with width (§8).

### 1.3 The canonical example: XOR

Assign class 1 to points where $x_1 x_2 > 0$ and class 0 otherwise. No single hyperplane
separates the classes (Minsky & Papert, 1969), but a two-layer network with 2 hidden
units does: each hidden unit carves one half-plane, and the output layer combines the
two indicators.

The hidden layer *re-represents* the data so that the final linear layer succeeds — the
core mechanism of all deep learning.

### 1.4 First-principles summary

| Question | Answer for the MLP |
|---|---|
| Problem | learn nonlinear input–output maps without hand-crafted features |
| Assumptions | i.i.d. data; a.e.-differentiable activations; enough capacity |
| Variables | data $X$, $Y$ (given); activations $Z_\ell, H_\ell$ (computed) |
| Parameters | $W_\ell, b_\ell$ (learned); width $h$, depth $L$, rate $\eta$, decay $\lambda$ (fixed) |
| Objective | minimize cross-entropy $+$ weight decay over all $W_\ell, b_\ell$ |
| Governing principle | empirical risk minimization by gradient descent; gradients via the chain rule |

## 2. WHAT — Notation and Architecture

### 2.1 Data

| Symbol | Type | Meaning |
|---|---:|---|
| $X \in \mathbb{R}^{n \times d}$ | matrix | input data ($n$ samples, $d$ features) |
| $Y \in \lbrace0,1\rbrace^{n \times c}$ | matrix | one-hot targets ($c$ classes) |
| $y \in \lbrace0,\ldots,c-1\rbrace^n$ | vector | integer labels |

### 2.2 Two-layer classifier (one hidden layer)

| Layer | Parameters | Computation |
|---|---|---|
| Input | — | $X$ |
| Hidden pre-act. | $W_1 \in \mathbb{R}^{d \times h}$, $b_1 \in \mathbb{R}^h$ | $Z_1 = XW_1 + \mathbf{1}b_1^\top$ |
| Hidden act. | — | $H = \phi(Z_1)$ |
| Output logits | $W_2 \in \mathbb{R}^{h \times c}$, $b_2 \in \mathbb{R}^c$ | $S = HW_2 + \mathbf{1}b_2^\top$ |
| Output probs | — | $P = \text{softmax}(S)$ |

### 2.3 General $L$-layer recursion

With $H_0 := X$ and layer widths $d = h_0, h_1, \ldots, h_L = c$:

```math
Z_\ell = H_{\ell-1} W_\ell + \mathbf{1} b_\ell^\top, \qquad
H_\ell = \phi(Z_\ell) \;\; (\ell < L), \qquad
P = \text{softmax}(Z_L).
```

Everything below is stated for $L = 2$; the recursion extends it mechanically.

### 2.4 Stable softmax

```math
P_{ik} = \frac{\exp(S_{ik} - \max_m S_{im})}{\sum_{r=1}^{c} \exp(S_{ir} - \max_m S_{im})}
```

Subtracting the row maximum prevents overflow. This does not change the result
because the ratio is invariant to additive constants (the same log-sum-exp trick as in
[softmax regression](../04_logistic_regression/theory.md)).

## 3. HOW — Forward Pass

The data flow is:

```math
X \;\xrightarrow{W_1, b_1}\; Z_1 \;\xrightarrow{\phi}\; H
\;\xrightarrow{W_2, b_2}\; S \;\xrightarrow{\text{softmax}}\; P
```

**Shape flow** for a concrete example ($n = 100$, $d = 4$, $h = 8$, $c = 3$):

| Tensor | Shape | Description |
|---|---|---|
| $X$ | $(100, 4)$ | input |
| $W_1$ | $(4, 8)$ | first weight matrix |
| $Z_1 = XW_1 + b_1$ | $(100, 8)$ | pre-activations |
| $H = \phi(Z_1)$ | $(100, 8)$ | hidden activations |
| $W_2$ | $(8, 3)$ | second weight matrix |
| $S = HW_2 + b_2$ | $(100, 3)$ | logits |
| $P$ | $(100, 3)$ | predicted probabilities |

### 3.1 Loss

Cross-entropy on the data:

```math
\mathcal{L}_{\text{data}} = -\frac{1}{n}\sum_{i=1}^{n}\sum_{k=1}^{c} Y_{ik}\log P_{ik}
```

With weight decay (L2 regularization on weights, not biases):

```math
\mathcal{L} = \mathcal{L}_{\text{data}}
+ \frac{\lambda}{2}\bigl(\Vert W_1\Vert_F^2 + \Vert W_2\Vert_F^2\bigr)
```

This is the objective actually minimized during training: $\lambda$ sets how strongly
large weights are penalized against fit to the data, and $\lambda = 0$ recovers
$\mathcal{L}_{\text{data}}$.

## 4. Backpropagation

All gradients follow from the chain rule applied from output to input. Convention:
$G_A := \partial\mathcal{L}_{\text{data}}/\partial A$ has the **same shape as $A$** — a
rule that catches most shape bugs before they happen.

### 4.1 Output layer gradient

The key identity for softmax + cross-entropy:

```math
G_S = \frac{\partial\mathcal{L}_{\text{data}}}{\partial S}
    = \frac{1}{n}(P - Y) \in \mathbb{R}^{n \times c}
```

**Derivation.** For one sample $i$ with true class $k$, the loss is
$-\log P_{ik} = -S_{ik} + \log\sum_r e^{S_{ir}}$ (definition of softmax, log of a
quotient). Differentiate term by term:

```math
\frac{\partial(-\log P_{ik})}{\partial S_{ij}}
= -\mathbb{1}[j = k] + \frac{e^{S_{ij}}}{\sum_r e^{S_{ir}}}
= P_{ij} - Y_{ij},
```

citing the derivative of $\log$ (chain rule) and of $e^{S_{ij}}$ inside the sum.
Averaging over $n$ samples gives the factor $1/n$ — the same residual structure
($p_i - y_i$) as logistic regression.

### 4.2 Second layer: gradients of $W_2$, $b_2$

$S = HW_2 + \mathbf{1}b_2^\top$ is affine in $W_2$, so by the chain rule for matrix
products (and adding the weight-decay derivative $\lambda W_2$, linearity of
differentiation):

```math
\frac{\partial\mathcal{L}}{\partial W_2}
= H^\top G_S + \lambda W_2
\qquad
\frac{\partial\mathcal{L}}{\partial b_2}
= G_S^\top \mathbf{1}
```

**Dimension bookkeeping:** $H^\top G_S$ is $(h \times n)(n \times c) = h \times c$ — the
shape of $W_2$; $G_S^\top\mathbf{1}$ sums the rows of $G_S$ to shape $c$, matching $b_2$.
Transpose-and-multiply is the only combination that produces the required shape while
summing over the sample index — exactly what the multivariable chain rule prescribes.

### 4.3 Propagating to the hidden layer

$S$ depends on $H$ through $S = HW_2 + \mathbf{1}b_2^\top$; the chain rule gives

```math
G_H = G_S W_2^\top \in \mathbb{R}^{n \times h}
```

($(n \times c)(c \times h) = n \times h$ — the shape of $H$). The forward pass
multiplies by $W_2$; the backward pass multiplies by $W_2^\top$ — each layer's backward
step is the transpose of its forward step.

### 4.4 Through the activation

$H = \phi(Z_1)$ is applied element-wise, so the Jacobian is diagonal and the chain rule
reduces to an element-wise (Hadamard) product $\odot$:

```math
G_{Z_1} = G_H \odot \phi'(Z_1)
```

For **tanh**, $\phi'(z) = 1 - \tanh^2(z)$; for **ReLU**, $\phi'(z) = \mathbb{1}[z > 0]$
(see the trade-off table in §7).

### 4.5 First layer: gradients of $W_1$, $b_1$

By the identical affine-layer argument as §4.2, with $X$ playing the role of $H$:

```math
\frac{\partial\mathcal{L}}{\partial W_1}
= X^\top G_{Z_1} + \lambda W_1
\qquad
\frac{\partial\mathcal{L}}{\partial b_1}
= G_{Z_1}^\top \mathbf{1}
```

**Result:** backpropagation is repeated application of the chain rule, propagating the
error signal $G_S$ backward. Every layer contributes the same recipe —
$G_{W_\ell} = H_{\ell-1}^\top G_{Z_\ell}$, $G_{b_\ell} = G_{Z_\ell}^\top\mathbf{1}$,
$G_{H_{\ell-1}} = G_{Z_\ell} W_\ell^\top$, then $\odot\,\phi'$ — so extending to $L$
layers changes nothing conceptually; the backward pass costs roughly twice the forward.

**Verification pointer.** [first_principles.ipynb](first_principles.ipynb) checks every
gradient above against central finite differences and PyTorch autograd, with explicit
tolerances.

## 5. The Computational-Graph View

Backprop is a special case of **reverse-mode automatic differentiation**. View the
network as a directed acyclic graph of primitive operations (matmul, add, $\phi$,
softmax, log). The chain rule can be accumulated through the graph in two directions:

- **Forward mode** pushes derivatives from inputs to outputs: one sweep computes a
  Jacobian–vector product — the derivative along *one* input direction. The full
  gradient of $p$ parameters costs $p$ sweeps.
- **Reverse mode** pulls derivatives from outputs to inputs: one sweep computes a
  vector–Jacobian product — the sensitivity of *one* output to *all* inputs.

A training loss is a **scalar**: one output, millions of parameters. Reverse mode
delivers the whole gradient in one backward sweep at a small constant multiple of the
forward cost; forward mode would need one sweep per parameter. This asymmetry is why
deep learning is computationally feasible.

The price is **memory**: the backward sweep needs the forward intermediates ($Z_1$, $H$,
$P$ in §4) — why the from-scratch implementation returns a cache, and why frameworks
record a "tape" of operations.

## 6. Initialization

### 6.1 Why zeros fail — the symmetry argument

Initialize $W_1 = 0$. Every hidden unit then computes the same pre-activation
($Z_1 = \mathbf{1}b_1^\top$), so all columns of $H$ are identical; by §4.5 the gradient
$X^\top G_{Z_1}$ has identical columns too, so the units *remain* identical after every
gradient step (induction on the update rule).

The network behaves as if it had one hidden unit — width is wasted. The same holds
whenever two units start exactly equal.

**Random initialization exists to break this symmetry**; the scale of the randomness is
the next question.

### 6.2 Xavier (Glorot) initialization — for tanh/sigmoid

Model one pre-activation $z = \sum_{j=1}^{d_{\text{in}}} w_j x_j$ with independent
zero-mean weights and inputs. By the variance-of-a-sum rule for independent terms:

```math
\operatorname{Var}(z) = d_{\text{in}} \operatorname{Var}(w)\operatorname{Var}(x).
```

Keeping signal variance constant with depth, $\operatorname{Var}(z) =
\operatorname{Var}(x)$, forces $\operatorname{Var}(w) = 1/d_{\text{in}}$. The identical
argument on the backward pass (where the fan-in is $d_{\text{out}}$, by §4.3) forces
$\operatorname{Var}(w) = 1/d_{\text{out}}$. Glorot & Bengio's compromise satisfies both
approximately:

```math
W_{ij} \sim \mathcal{N}\!\left(0,\; \frac{2}{d_{\text{in}} + d_{\text{out}}}\right)
```

The derivation assumes the activation is roughly the identity near zero — true for tanh
and sigmoid in their linear regime, false for ReLU.

### 6.3 He initialization — for ReLU

ReLU zeroes the negative half of a symmetric pre-activation distribution, so
$\mathbb{E}[\phi(z)^2] = \tfrac{1}{2}\operatorname{Var}(z)$ — half the signal variance is
discarded at every layer. Doubling the weight variance compensates:

```math
W_{ij} \sim \mathcal{N}\!\left(0,\; \frac{2}{d_{\text{in}}}\right)
```

Using Xavier with deep ReLU networks shrinks activations by $\approx (1/2)^{L/2}$ over
$L$ layers — a silent vanishing-signal failure that He et al. (2015) diagnosed.

### 6.4 Biases

Biases are typically initialized to zero — the random weights already break symmetry,
and zero keeps activations centered at the start.

## 7. Activation Functions — Trade-offs

| Name | Formula | Derivative | Range | Main failure mode |
|---|---|---|---|---|
| sigmoid | $\sigma(z) = 1/(1+e^{-z})$ | $\sigma(z)(1-\sigma(z)) \le 1/4$ | $(0,1)$ | saturation both sides; not zero-centered |
| tanh | $\tanh(z)$ | $1 - \tanh^2(z) \le 1$ | $(-1,1)$ | saturation at $\pm 1$ |
| ReLU | $\max(0, z)$ | $\mathbb{1}[z > 0]$ | $[0,\infty)$ | dead units for $z < 0$ |
| Leaky ReLU | $\max(\alpha z, z)$, $\alpha \approx 0.01$ | $\alpha$ or $1$ | $\mathbb{R}$ | mostly fixes dead units |

Two opposite pathologies dominate the choice:

**Saturation (sigmoid, tanh).** Once $\vert z\vert \gg 1$ the derivative is
exponentially small, and by §4.4 the backward signal is multiplied by it — gradients
through saturated units vanish.

Worse, the sigmoid derivative is at most $1/4$ *everywhere*, so a deep sigmoid stack
shrinks gradients by at least $4^{-L}$: the classical vanishing-gradient problem. Tanh
is preferred for hidden layers because it is zero-centered and its derivative reaches
$1$.

**Dead units (ReLU).** For $z > 0$ the derivative is exactly $1$ — no saturation,
which is why deep ReLU networks train at all.

But for $z < 0$ it is exactly $0$: a unit whose pre-activations go negative for *every*
input (e.g. after one large gradient step) receives zero gradient forever and is
permanently dead. Leaky ReLU's small negative slope $\alpha$ lets such units recover.

The trade-off is asymmetric: dead ReLU units waste capacity, but saturated sigmoids
block learning through the *whole depth*. ReLU (or a smooth relative such as GELU, used
in [transformers](../16_transformer/theory.md)) is the default for deep networks; tanh
survives in shallow networks and gates ([LSTM](../15_rnn_lstm/theory.md)).

## 8. Universal Approximation

The universal approximation theorem (Cybenko 1989, Hornik 1991) states that a single
hidden layer with enough units can approximate any continuous function on a compact set
to arbitrary accuracy, for any non-polynomial activation.

**Intuition (no proof).** The difference of two shifted steep sigmoids is a localized
"bump"; scale and place enough bumps and any continuous function can be tiled to
accuracy $\varepsilon$, the way a Riemann sum tiles an integral. Width buys resolution.

**Caveats.** The theorem proves *existence*, not learnability: it bounds neither the
required width (possibly exponential in $d$), nor guarantees that gradient descent finds
the approximating weights, nor says anything about generalization from finite samples.

Depth matters in practice because some functions computable by a deep network of modest
width need exponentially many units in one hidden layer — depth compresses width.

## 9. Loss Surface and SGD

Unlike every convex model earlier in this curriculum (linear, logistic, SVM), the MLP
loss is **non-convex**: parameters enter through the composition $\phi(XW_1)W_2$, and
products of parameters destroy convexity.

Concretely, permuting the $h$ hidden units (with the matching rows of $W_2$) leaves the
function unchanged, so every minimum comes in at least $h!$ symmetric copies — and a
convex function cannot have multiple isolated minima.

- **No global guarantee.** Gradient descent finds a stationary point, not provably the
  best one; the convexity-based guarantees of
  [topic 02](../02_gradient_descent/theory.md) become local.
- **Saddles, not bad minima, dominate.** In high dimensions a stationary point with all
  curvature directions non-negative is combinatorially unlikely unless the loss there is
  already low; most obstacles are saddles with escape directions (Dauphin et al., 2014).
- **SGD's noise is a feature.** Mini-batch gradients are unbiased but noisy. The noise
  perturbs the iterate off saddles and out of sharp minima, biasing SGD toward flat
  basins — which correlate empirically with better generalization.
- Full-batch descent is, ironically, *more* prone to stalling at saddles.

The standard recipe — small random init (§6), mini-batch SGD, modest learning rate — is
not incidental: each ingredient targets a specific pathology of this surface.

## 10. Failure Cases

1. **No nonlinearity** $\to$ still affine (§1.2). Always use an activation between layers.

2. **Poor initialization scale** $\to$ vanishing or exploding gradients: large weights
   saturate tanh, small weights shrink signals with depth (§6.2–6.3).

3. **Zero or symmetric initialization** $\to$ hidden units never differentiate (§6.1);
   the network trains but behaves as width 1.

4. **Bad learning rate** $\to$ too large: loss diverges; too small: training stalls.

5. **Overfitting** with large hidden layers and no regularization — the model memorizes
   the training set. Weight decay, dropout, or early stopping mitigates this.

6. **Saturated activations / dead ReLU units** (§7) — such units stop learning and the
   effective capacity silently drops.

Each failure is demonstrated in [first_principles.ipynb](first_principles.ipynb)
(linear baseline on XOR, learning-rate sweep, capacity/decay experiments).

## 11. Connections

- [Gradient Descent](../02_gradient_descent/README.md) — the optimizer; §9 says which
  of its guarantees survive non-convexity.
- [Logistic Regression](../04_logistic_regression/README.md) — an MLP with **zero**
  hidden layers *is* softmax regression: same loss, same $\frac{1}{n}(P - Y)$ gradient
  (§4.1). Conversely, each hidden unit is a small logistic unit whose "labels" backprop
  invents.
- [Regularization](../03_regularization/README.md) — weight decay is L2 on the weights.
- [CNN](../14_cnn/README.md) — an MLP constrained by local connectivity and weight
  sharing across space.
- [RNN/LSTM](../15_rnn_lstm/README.md) — an MLP with weights shared across *time*; the
  vanishing-gradient analysis of §7 becomes the central obstacle.
- [Transformer](../16_transformer/README.md) — position-wise MLP blocks alternate with
  attention; initialization and residual connections manage depth.
- [Autoencoder](../17_autoencoder/README.md) — two MLPs trained end-to-end for
  compression instead of classification.
- [Numerical Computing](https://github.com/hien078/applied-mathematics-foundation) —
  log-sum-exp stability (§2.4).

---

## 12. References

- **Goodfellow, I., Bengio, Y., & Courville, A. (2016).** *Deep Learning*. MIT Press. Chapters 6 (*Deep Feedforward Networks*) and 8 (*Optimization for Training Deep Models*).
- **Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986).** Learning representations by back-propagating errors. *Nature*, 323, 533–536.
- **Cybenko, G. (1989).** Approximation by superpositions of a sigmoidal function. *Mathematics of Control, Signals and Systems*, 2(4), 303–314.
- **Hornik, K., Stinchcombe, M., & White, H. (1991).** Multilayer feedforward networks are universal approximators. *Neural Networks*, 4(2), 251–257.
- **Glorot, X., & Bengio, Y. (2010).** Understanding the difficulty of training deep feedforward neural networks. *Proceedings of the 13th International Conference on Artificial Intelligence and Statistics (AISTATS)*, 249–256.
- **He, K., Zhang, X., Ren, S., & Sun, J. (2015).** Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification. *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, 1026–1034.
- **Dauphin, Y., et al. (2014).** Identifying and attacking the saddle point problem in high-dimensional non-convex optimization. *Advances in Neural Information Processing Systems (NIPS)*, 27.
