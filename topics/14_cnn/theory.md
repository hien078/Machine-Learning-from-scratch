# Convolutional Neural Networks — Theory

## 0. Notation

| Symbol | Type | Meaning |
|---|---|---|
| $X \in \mathbb{R}^{H \times W}$ | matrix | single-channel input (height $H$, width $W$) |
| $X \in \mathbb{R}^{C_{\text{in}} \times H \times W}$ | tensor | multi-channel input ($C_{\text{in}}$ channels) |
| $K \in \mathbb{R}^{k_h \times k_w}$ | matrix | convolution kernel (filter) |
| $K \in \mathbb{R}^{C_{\text{out}} \times C_{\text{in}} \times k_h \times k_w}$ | tensor | full weight tensor for a conv layer |
| $b \in \mathbb{R}^{C_{\text{out}}}$ | vector | bias, one per output channel |
| $Y \in \mathbb{R}^{H_{\text{out}} \times W_{\text{out}}}$ | matrix | feature map (output of one filter) |
| $s$ | scalar | stride (default 1) |
| $p$ | scalar | padding width (number of zeros added per side) |
| $k_h, k_w$ | scalars | kernel height and width |
| $H_{\text{out}}, W_{\text{out}}$ | scalars | output spatial dimensions |

**Convention.** Deep-learning libraries implement **cross-correlation** and call it
"convolution." True mathematical convolution flips the kernel; cross-correlation does
not. We follow the library convention: the operation described below is
cross-correlation, written as $\star$.

---

## 1. WHY — Why Convolution for Images

A grayscale image of size $224 \times 224$ has 50 176 pixels. A dense (fully connected)
layer mapping this to just 256 hidden units needs $50\,176 \times 256 \approx 12.8$ million
parameters — for a single layer. Three problems arise:

1. **Too many parameters.** Millions of weights per layer → massive memory, slow training,
   and severe overfitting with limited data.
2. **No spatial awareness.** A dense layer treats pixel $(0, 0)$ and pixel $(200, 200)$
   identically — it cannot exploit the fact that nearby pixels are strongly correlated.
3. **No translation invariance.** A cat in the top-left corner activates completely different
   weights than the same cat in the bottom-right corner.

Convolution fixes all three by enforcing two structural priors:

- **Local connectivity.** Each output neuron connects only to a small spatial patch of the
  input (the *receptive field*), not to the entire image.
- **Parameter sharing (weight tying).** The same kernel is applied at every spatial
  location. A $3 \times 3$ kernel has 9 parameters regardless of the input size.

**Result:** These two constraints reduce parameters by orders of magnitude, inject useful
inductive bias (spatial locality), and make the network *equivariant* to translation:
shifting the input shifts the feature map by the same amount.

---

## 2. WHAT — The Convolution (Cross-Correlation) Operation

### 2.1 Single-channel, no padding, stride 1

Given input $X \in \mathbb{R}^{H \times W}$ and kernel $K \in \mathbb{R}^{k_h \times k_w}$,
the output ("valid" mode) is $Y \in \mathbb{R}^{H_{\text{out}} \times W_{\text{out}}}$ with

$$
Y_{i,j} = \sum_{m=0}^{k_h - 1} \sum_{n=0}^{k_w - 1} K_{m,n} \, X_{i+m,\, j+n},
\qquad
\begin{cases}
i = 0, \ldots, H - k_h \\
j = 0, \ldots, W - k_w
\end{cases}
$$

Output dimensions:

$$H_{\text{out}} = H - k_h + 1, \qquad W_{\text{out}} = W - k_w + 1.$$

This is an element-wise multiply-and-sum of the kernel with every overlapping patch of the
input — a *sliding dot product*.

### 2.2 Padding

| Mode | Padding $p$ | Output size | Use case |
|---|---|---|---|
| **Valid** | $p = 0$ | $(H - k_h + 1) \times (W - k_w + 1)$ | Default; output shrinks |
| **Same** | $p = \lfloor k_h / 2 \rfloor$ (each side) | $H \times W$ (if stride 1) | Preserve spatial size |

Padding adds $p$ rows/columns of zeros around the input before sliding the kernel.

### 2.3 Stride

With stride $s$, the kernel advances $s$ positions instead of 1:

$$H_{\text{out}} = \left\lfloor \frac{H + 2p - k_h}{s} \right\rfloor + 1, \qquad
  W_{\text{out}} = \left\lfloor \frac{W + 2p - k_w}{s} \right\rfloor + 1.$$

Stride $> 1$ downsamples the feature map spatially.

### 2.4 Multi-channel convolution

An RGB image has $C_{\text{in}} = 3$ channels. A single filter is now a 3D tensor
$K \in \mathbb{R}^{C_{\text{in}} \times k_h \times k_w}$. The output at position $(i, j)$
sums over all input channels:

$$Y_{i,j} = \sum_{c=0}^{C_{\text{in}} - 1} \sum_{m=0}^{k_h - 1} \sum_{n=0}^{k_w - 1}
  K_{c, m, n} \, X_{c,\, i+m,\, j+n} + b.$$

With $C_{\text{out}}$ such filters, the output is a tensor of shape
$C_{\text{out}} \times H_{\text{out}} \times W_{\text{out}}$. Each output channel is called
a **feature map**.

### 2.5 Parameter count — dense vs. conv

| Layer type | Input size | Output size | Parameters |
|---|---|---|---|
| Dense | $50\,176$ | $256$ | $50\,176 \times 256 + 256 \approx 12.8$M |
| Conv ($3 \times 3$, 64 filters) | $1 \times 224 \times 224$ | $64 \times 222 \times 222$ | $64 \times (1 \times 3 \times 3) + 64 = 640$ |

**Result:** Convolution achieves a 20 000× reduction in parameters for this example, while
producing a richer spatial output.

---

## 3. HOW — Forward and Backward Pass

### 3.1 Forward pass (cross-correlation)

The forward pass for a single output channel $f$ at position $(i, j)$:

$$Y_{f, i, j} = b_f + \sum_{c=0}^{C_{\text{in}}-1} \sum_{m=0}^{k_h-1} \sum_{n=0}^{k_w-1}
  K_{f, c, m, n} \, X_{c,\, i \cdot s + m,\, j \cdot s + n}.$$

### 3.2 Backpropagation through convolution

Let $G_Y = \frac{\partial L}{\partial Y}$ be the upstream gradient (same shape as $Y$).

**Gradient w.r.t. kernel** (for filter $f$, channel $c$):

$$\frac{\partial L}{\partial K_{f,c,m,n}} = \sum_{i} \sum_{j}
  G_{Y_{f,i,j}} \cdot X_{c,\, i \cdot s + m,\, j \cdot s + n}.$$

This is itself a cross-correlation of the input $X$ with the upstream gradient $G_Y$.

**Gradient w.r.t. bias:**

$$\frac{\partial L}{\partial b_f} = \sum_{i} \sum_{j} G_{Y_{f,i,j}}.$$

**Gradient w.r.t. input** (for stride 1, valid padding):

$$\frac{\partial L}{\partial X_{c,i,j}} = \sum_{f} \sum_{m} \sum_{n}
  K_{f,c,m,n} \cdot G_{Y_{f,\, i-m,\, j-n}},$$

where out-of-bounds indices contribute zero. This is equivalent to convolving $G_Y$ with
the *180°-rotated* (flipped) kernel — a "full" convolution with the flipped kernel. This
is the connection to true mathematical convolution.

**Result:** Backprop through convolution requires: (1) cross-correlating the input with the
upstream gradient to get $\nabla K$, and (2) convolving the upstream gradient with the
flipped kernel to get $\nabla X$.

---

## 4. Pooling

Pooling reduces spatial dimensions and introduces a degree of local translation invariance.

### 4.1 Max pooling

For a $p_h \times p_w$ pooling window with stride $s$:

$$Y_{i,j} = \max_{0 \le m < p_h,\; 0 \le n < p_w} X_{i \cdot s + m,\; j \cdot s + n}.$$

**Backprop:** The gradient passes only to the position of the maximum in each window.
All other positions receive zero gradient.

### 4.2 Average pooling

$$Y_{i,j} = \frac{1}{p_h \cdot p_w} \sum_{m=0}^{p_h-1} \sum_{n=0}^{p_w-1}
  X_{i \cdot s + m,\; j \cdot s + n}.$$

**Backprop:** The gradient is distributed equally to all positions in the window.

### 4.3 Global average pooling

Takes the mean over the entire spatial extent of each channel, reducing a
$C \times H \times W$ tensor to a $C$-dimensional vector. Used before the final
classifier layer in modern architectures (e.g., ResNet) to avoid large dense layers.

---

## 5. Receptive Field

The **receptive field** of a unit in layer $\ell$ is the region of the *original input*
that can influence its value.

### 5.1 Single layer

With kernel $k \times k$, the receptive field of each output unit covers a $k \times k$
patch of the input.

### 5.2 Two stacked layers

Two $3 \times 3$ layers (stride 1) have a combined receptive field of $5 \times 5$.
In general, $L$ layers of $k \times k$ kernels (stride 1) produce a receptive field of

$$r = L(k - 1) + 1.$$

**Result:** Deeper networks see larger portions of the input. A stack of small ($3 \times 3$)
filters is preferred over a single large filter because it achieves the same receptive
field with fewer parameters and more nonlinearities.

### 5.3 Comparison

| Configuration | Receptive field | Parameters (single channel) |
|---|---|---|
| One $7 \times 7$ layer | $7 \times 7$ | $49$ |
| Three $3 \times 3$ layers | $7 \times 7$ | $3 \times 9 = 27$ |

---

## 6. Common Architectures

### 6.1 LeNet-5 (LeCun et al., 1998)

The original CNN for digit recognition:

| Layer | Output shape | Parameters |
|---|---|---|
| Input | $1 \times 32 \times 32$ | — |
| Conv $5 \times 5$, 6 filters | $6 \times 28 \times 28$ | $6 \times 25 + 6 = 156$ |
| AvgPool $2 \times 2$ | $6 \times 14 \times 14$ | 0 |
| Conv $5 \times 5$, 16 filters | $16 \times 10 \times 10$ | $16 \times 150 + 16 = 2\,416$ |
| AvgPool $2 \times 2$ | $16 \times 5 \times 5$ | 0 |
| Flatten → Dense 120 | 120 | $400 \times 120 + 120 = 48\,120$ |
| Dense 84 | 84 | $120 \times 84 + 84 = 10\,164$ |
| Dense 10 | 10 | $84 \times 10 + 10 = 850$ |
| **Total** | | **$\approx 61\,700$** |

### 6.2 Deeper architectures (overview)

| Architecture | Year | Key idea |
|---|---|---|
| AlexNet | 2012 | Deeper, ReLU, dropout, GPU training |
| VGGNet | 2014 | Uniform $3 \times 3$ conv, very deep (16–19 layers) |
| GoogLeNet/Inception | 2014 | Parallel multi-scale filters (inception modules) |
| ResNet | 2015 | Skip (residual) connections, 100+ layers |
| EfficientNet | 2019 | Compound scaling of depth, width, resolution |

The trend: deeper + skip connections + efficient building blocks.

---

## 7. Translation Equivariance vs. Invariance

**Equivariance.** Convolution is *equivariant* to translation: shifting the input by
$(dx, dy)$ shifts every feature map by the same amount. Formally, if $T$ is a translation
operator, then $\text{Conv}(T[X]) = T[\text{Conv}(X)]$.

**Invariance.** Pooling (especially global average pooling) adds a degree of translation
*invariance*: small shifts in the input may produce the same output.

A trained CNN is approximately translation-invariant because of pooling layers and
data augmentation. However, it is **not** invariant to rotation, scaling, or other
geometric transformations unless augmented accordingly.

---

## 8. Failure Cases

1. **Rotation and scale.** CNNs are equivariant to translation only. A rotated digit
   may not be recognized without rotation augmentation or specialized architectures
   (e.g., spatial transformer networks).

2. **Destroyed spatial structure.** If pixels are randomly permuted (shuffled), the
   spatial locality assumption is violated and a CNN performs no better than (or worse
   than) a dense network. The convolution prior becomes a liability.

3. **Small datasets.** CNNs have fewer parameters than dense nets but still need
   substantial data. On tabular data or very small image datasets, simpler models
   (gradient boosted trees, SVMs) often outperform CNNs.

4. **1D sequential data.** While 1D convolutions exist and work, recurrent architectures
   (RNN/LSTM) and transformers are often more natural for variable-length sequences
   because they can model long-range dependencies without a deep stack of conv layers.

5. **Boundary effects.** Zero-padding introduces artificial zeros at the edges. Feature
   maps near the boundary are computed from a mix of real pixels and zeros, which can
   degrade edge features. Circular or reflective padding can mitigate this.

6. **Pooling discards location.** Max pooling retains *what* was detected but discards
   *where*. This is useful for classification but harmful for tasks requiring precise
   localization (segmentation, detection), which use architectures like U-Net or
   feature pyramid networks instead.

---

## 9. Connections

- [Neural Networks](../13_neural_networks/README.md) — MLP is a CNN with
  kernel size = input size (one global filter per unit)
- [RNN/LSTM](../15_rnn_lstm/README.md) — temporal analog: shared weights across time
  steps instead of spatial locations
- [Transformer](../16_transformer/README.md) — replaces local convolution with
  global self-attention; Vision Transformer (ViT) applies this to images
- [Gradient Descent](../02_gradient_descent/README.md) — optimizer used for training
- [Regularization](../03_regularization/README.md) — dropout, weight decay, data
  augmentation are standard CNN regularizers

---

## 10. References

- **LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998).** Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278–2324.
- **Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012).** ImageNet classification with deep convolutional neural networks. *Advances in Neural Information Processing Systems (NIPS)*, 25, 1097–1105.
- **He, K., Zhang, X., Ren, S., & Sun, J. (2016).** Deep residual learning for image recognition. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 770–778.
- **Goodfellow, I., Bengio, Y., & Courville, A. (2016).** *Deep Learning*. MIT Press. Chapter 9: *Convolutional Networks*.

