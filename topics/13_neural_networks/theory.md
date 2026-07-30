# Neural Networks (MLP) — Theory

## 1. WHY — Beyond Linear Models

A composition of affine maps is still affine. Inserting a nonlinear activation
$\phi$ between layers breaks linearity and allows the model to represent
nonlinear decision boundaries. The canonical motivating example: an XOR pattern
cannot be separated by any single hyperplane but is trivially separated by a
two-layer network with 2 hidden units.

## 2. WHAT — Notation and Architecture

### 2.1 Data

| Symbol | Type | Meaning |
|---|---:|---|
| $X \in \mathbb{R}^{n \times d}$ | matrix | input data ($n$ samples, $d$ features) |
| $Y \in \{0,1\}^{n \times c}$ | matrix | one-hot targets ($c$ classes) |
| $y \in \{0,\ldots,c-1\}^n$ | vector | integer labels |

### 2.2 Two-layer classifier (one hidden layer)

| Layer | Parameters | Computation |
|---|---|---|
| Input | — | $X$ |
| Hidden pre-act. | $W_1 \in \mathbb{R}^{d \times h}$, $b_1 \in \mathbb{R}^h$ | $Z_1 = XW_1 + \mathbf{1}b_1^\top$ |
| Hidden act. | — | $H = \phi(Z_1)$ |
| Output logits | $W_2 \in \mathbb{R}^{h \times c}$, $b_2 \in \mathbb{R}^c$ | $S = HW_2 + \mathbf{1}b_2^\top$ |
| Output probs | — | $P = \text{softmax}(S)$ |

### 2.3 Activations

| Name | Formula | Derivative | Notes |
|---|---|---|---|
| tanh | $\tanh(z)$ | $1 - \tanh^2(z)$ | Saturates at $\pm 1$ |
| ReLU | $\max(0, z)$ | $\mathbb{1}[z > 0]$ | No saturation for $z > 0$ |

### 2.4 Stable softmax

$$
P_{ik} = \frac{\exp(S_{ik} - \max_m S_{im})}{\sum_{r=1}^{c} \exp(S_{ir} - \max_m S_{im})}
$$

Subtracting the row maximum prevents overflow. This does not change the result
because the ratio is invariant to additive constants.

## 3. HOW — Forward Pass

The data flow is:

$$
X \;\xrightarrow{W_1, b_1}\; Z_1 \;\xrightarrow{\phi}\; H
  \;\xrightarrow{W_2, b_2}\; S \;\xrightarrow{\text{softmax}}\; P
$$

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

$$
\mathcal{L}_{\text{data}} = -\frac{1}{n}\sum_{i=1}^{n}\sum_{k=1}^{c} Y_{ik}\log P_{ik}
$$

With weight decay (L2 regularization on weights, not biases):

$$
\mathcal{L} = \mathcal{L}_{\text{data}}
  + \frac{\lambda}{2}\bigl(\|W_1\|_F^2 + \|W_2\|_F^2\bigr)
$$

## 4. Backpropagation

All gradients are derived by applying the chain rule from output to input.

### 4.1 Output layer gradient

The key identity for softmax + cross-entropy:

$$
G_S = \frac{\partial\mathcal{L}_{\text{data}}}{\partial S}
    = \frac{1}{n}(P - Y) \in \mathbb{R}^{n \times c}
$$

**Derivation sketch.** For a single sample $i$ with true class $k$,
$\partial(-\log P_{ik})/\partial S_{ij} = P_{ij} - \mathbb{1}[j = k]$.
Averaging over $n$ samples gives the factor $1/n$.

### 4.2 Weight gradients

$$
\frac{\partial\mathcal{L}}{\partial W_2}
= H^\top G_S + \lambda W_2
\qquad
\frac{\partial\mathcal{L}}{\partial b_2}
= G_S^\top \mathbf{1}
$$

### 4.3 Hidden layer gradient

$$
G_H = G_S W_2^\top \in \mathbb{R}^{n \times h}
$$

### 4.4 Activation derivative

For **tanh**:

$$
G_{Z_1} = G_H \odot (1 - \tanh^2(Z_1))
$$

For **ReLU**:

$$
G_{Z_1} = G_H \odot \mathbb{1}[Z_1 > 0]
$$

### 4.5 First layer weight gradients

$$
\frac{\partial\mathcal{L}}{\partial W_1}
= X^\top G_{Z_1} + \lambda W_1
\qquad
\frac{\partial\mathcal{L}}{\partial b_1}
= G_{Z_1}^\top \mathbf{1}
$$

**Result:** backpropagation is repeated application of the chain rule, propagating
the error signal $G_S$ backward through each layer. Each layer adds one matrix
multiply and one element-wise operation. Extending to $L$ layers changes nothing
conceptually.

## 5. Initialization

Random initialization breaks symmetry. The scale matters for gradient flow.

### 5.1 Xavier (Glorot) initialization — for tanh

$$
W_{ij} \sim \mathcal{N}\!\left(0,\; \frac{2}{d_{\text{in}} + d_{\text{out}}}\right)
$$

Derived by requiring $\operatorname{Var}(\text{output}) \approx \operatorname{Var}(\text{input})$ under the assumption that activations
are in the linear regime near zero.

### 5.2 He initialization — for ReLU

$$
W_{ij} \sim \mathcal{N}\!\left(0,\; \frac{2}{d_{\text{in}}}\right)
$$

ReLU zeroes out half the activations on average, so the variance must be doubled
relative to Xavier.

### 5.3 Biases

Biases are typically initialized to zero.

## 6. Universal Approximation

The universal approximation theorem (Cybenko 1989, Hornik 1991) states that a single
hidden layer with enough units can approximate any continuous function on a compact
set to arbitrary accuracy. **Caveat:** the theorem proves existence, not that SGD
will find the approximating parameters, nor that the required width will be practical.

## 7. Failure Cases

1. **No nonlinearity** $\to$ still affine. Two linear layers $W_2(W_1 x + b_1) + b_2$
   simplify to $W'x + b'$. Always use an activation between layers.

2. **Poor initialization** $\to$ vanishing or exploding gradients. Large initial weights
   saturate tanh ($|z| \gg 1$, derivative $\approx 0$). Small initial weights cause
   vanishing signals in deep networks.

3. **Bad learning rate** $\to$ slow convergence or divergence. Too large: loss increases.
   Too small: training stalls.

4. **Overfitting** with large hidden layers and no regularization. The model memorizes
   the training set. Weight decay, dropout, or early stopping mitigates this.

5. **Saturated activations** (tanh at $\pm 1$). Once hidden units are permanently
   saturated, gradients vanish and those units stop learning.

## 8. Connections

- [Gradient Descent](../02_gradient_descent/README.md) — optimizer used in training
- [Logistic Regression](../04_logistic_regression/README.md) — special case: MLP with 0 hidden layers
- [Regularization](../03_regularization/README.md) — weight decay is L2 regularization
- [Numerical Computing](../../foundations/numerical_computing/README.md) — log-sum-exp stability
- [CNN / RNN / Transformer](../14_cnn/README.md) — architectures that generalize the MLP

---

## 9. References

- **Goodfellow, I., Bengio, Y., & Courville, A. (2016).** *Deep Learning*. MIT Press. Chapter 6: *Deep Feedforward Networks*.
- **Cybenko, G. (1989).** Approximation by superpositions of a sigmoidal function. *Mathematics of Control, Signals and Systems*, 2(4), 303–314.
- **Hornik, K., Stinchcombe, M., & White, H. (1991).** Multilayer feedforward networks are universal approximators. *Neural Networks*, 4(2), 251–257.
- **Glorot, X., & Bengio, Y. (2010).** Understanding the difficulty of training deep feedforward neural networks. *Proceedings of the 13th International Conference on Artificial Intelligence and Statistics (AISTATS)*, 249–256.
- **He, K., Zhang, X., Ren, S., & Sun, J. (2015).** Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification. *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, 1026–1034.

