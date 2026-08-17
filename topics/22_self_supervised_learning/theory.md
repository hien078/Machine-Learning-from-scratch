# Topic 22: Self-Supervised Learning — Theory and Derivations

## 0. Notation Table

| Symbol | Type | Meaning |
|--------|------|---------|
| $\mathcal{D}$ | Set | Unlabeled dataset $\lbrace x_1, x_2, \dots, x_N\rbrace$ |
| $N$ | Integer | Batch size / number of anchor samples |
| $x_i$ | Vector | Raw input sample ($x_i \in \mathbb{R}^D$) |
| $\tilde{x}_i, \tilde{x}_j$ | Vector | Stochastic augmented views of $x_i$ |
| $\mathcal{T}$ | Distribution | Family of data augmentations (crop, flip, color jitter) |
| $f_\theta(\cdot)$ | Function | Encoder network $f_\theta: \mathbb{R}^D \to \mathbb{R}^d$ |
| $h_i$ | Vector | Representation vector $h_i = f_\theta(\tilde{x}_i) \in \mathbb{R}^d$ |
| $g_\phi(\cdot)$ | Function | Projection head $g_\phi: \mathbb{R}^d \to \mathbb{R}^k$ |
| $z_i$ | Vector | Normalized projection $z_i = g_\phi(h_i) / \Vert g_\phi(h_i)\Vert_2 \in S^{k-1}$ |
| $\text{sim}(u, v)$ | Scalar | Cosine similarity $u^T v / (\Vert u\Vert_2 \Vert v\Vert_2)$ |
| $\tau$ | Scalar | Temperature hyperparameter ($\tau > 0$) |
| $K$ | Integer | Number of negative samples |
| $\mathcal{L}_{\text{InfoNCE}}$ | Scalar | InfoNCE / NT-Xent loss value |

---

## 1. WHY: The Label Bottleneck

Supervised machine learning relies on human annotations. Manual labeling is expensive, unscalable, prone to annotator bias, and forces the model to discard rich structural attributes of data that are not relevant to the narrow class labels.

**Self-Supervised Learning (SSL)** circumvents the label bottleneck by constructing **pretext tasks** directly from unlabeled data. The model learns generalizable representations $h = f_\theta(x)$ that capture the intrinsic geometry and semantics of the data. These representations can then be transferred to downstream tasks (e.g., classification, detection) using linear probing or minimal fine-tuning.

Paradigms of Self-Supervised Learning:
1. **Predictive:** Predict missing or transformed features (e.g., rotation angle, colorization, relative patch position).
2. **Generative / Masked Modeling:** Reconstruct masked portions of the input (e.g., Masked Autoencoders, BERT).
3. **Contrastive:** Learn augmentation-invariant representations by pulling positive views together and pushing negative samples apart in embedding space.

---

## 2. WHAT: Representation Learning Framework

The goal is to learn an encoder $f_\theta$ that maps inputs $x \in \mathcal{X}$ to feature vectors $h \in \mathbb{R}^d$ such that semantically similar inputs map to nearby points, while distinct inputs map far apart.

### 2.1 The 4-Step SSL Pipeline
1. **Data Augmentation ($\mathcal{T}$):** For any input $x$, draw two random transformations $t_1, t_2 \sim \mathcal{T}$ to yield a positive pair $(\tilde{x}_i, \tilde{x}_j) = (t_1(x), t_2(x))$.
2. **Base Encoder ($f_\theta$):** Extracts representation vectors $h_i = f_\theta(\tilde x_i)$ and $h_j = f_\theta(\tilde x_j)$.
3. **Projection Head ($g_\phi$):** Maps $h$ to a lower-dimensional embedding space $z_i = g_\phi(h_i)$, normalized to the unit hypersphere $\Vert z_i\Vert_2 = 1$.
4. **Contrastive Loss ($\mathcal{L}$):** Maximizes $\text{sim}(z_i, z_j)$ while minimizing $\text{sim}(z_i, z_k)$ for all negative samples $z_k$.

### 2.2 Augmentation Invariance as Inductive Bias
By training the encoder to produce identical representations for $t_1(x)$ and $t_2(x)$, we explicitly force the model to be invariant to spatial cropping, color shifts, and high-frequency noise. The choice of $\mathcal{T}$ defines what semantic information is preserved versus discarded.

---

## 3. Contrastive Learning & Mathematical Derivations

### 3.1 Noise Contrastive Estimation (NCE)
NCE reformulates density estimation $p(x)$ as binary logistic regression: distinguishing true data samples $x \sim p_{\text{data}}$ from synthetic noise samples $x \sim q_{\text{noise}}$.

### 3.2 Step-by-Step InfoNCE Loss Derivation

InfoNCE generalizes NCE to a multi-class classification problem. Given an anchor sample $x$ and its positive target $y^+ \sim p(y|x)$, along with $K$ independent noise samples $\lbrace y_1, y_2, \dots, y_K\rbrace \sim p(y)$, the model must identify $y^+$ out of the set of $K+1$ candidates.

Let the score function be $f(x,y) = \frac{z_x^T z_y}{\tau}$. The probability of correctly selecting $y^+$ is given by the softmax function:

```math
P(\text{pos} \mid x, \{y\}) = \frac{\exp(f(x, y^+))}{\exp(f(x, y^+)) + \sum_{k=1}^K \exp(f(x, y_k))}
```

The InfoNCE loss is defined as the negative log-likelihood of selecting the true positive:

$$ \mathcal{L}_{\text{InfoNCE}} = -\mathbb{E} \left[ \log \frac{\exp(f(x, y^+))}{\exp(f(x, y^+)) + \sum_{k=1}^K \exp(f(x, y_k))} \right] $$

#### Deriving the Mutual Information Lower Bound (Oord et al., 2018)

We prove that minimizing $\mathcal{L}_{\text{InfoNCE}}$ maximizes a lower bound on the mutual information $I(X; Y)$ between positive views.

1. **Optimal Classifier Probability:**
   By Bayes' Rule, the true posterior probability that candidate $i$ is the positive sample given candidates $V = \lbrace y_1, \dots, y_{K+1}\rbrace$ is:

   $$P(i = \text{pos} \mid V, x) = \frac{\frac{p(y_i \mid x)}{p(y_i)}}{\sum_{j=1}^{K+1} \frac{p(y_j \mid x)}{p(y_j)}}$$

2. **Matching Density Ratio:**
   Comparing this optimal posterior to the softmax model with $f(x, y) = \log \frac{p(y \mid x)}{p(y)}$:

   $$P(i = \text{pos} \mid V, x) = \frac{\exp(f(x, y_i))}{\sum_{j=1}^{K+1} \exp(f(x, y_j))}$$

3. **Evaluating Expected Loss:**
   Substituting $f(x, y^+) = \log \frac{p(y^+ \mid x)}{p(y^+)}$ into the InfoNCE loss:

   $$\mathcal{L}_ {\text{InfoNCE}} = -\mathbb{E} \left[ \log \frac{\frac{p(y^+ \mid x)}{p(y^+)}}{\frac{p(y^+ \mid x)}{p(y^+)} + \sum_{k=1}^K \frac{p(y_k \mid x)}{p(y_k)}} \right]$$

   Take the expectation over negative samples $y_k \sim p(y)$:

   $$\mathbb{E}_ {y_k \sim p(y)} \left[ \sum_{k=1}^K \frac{p(y_k \mid x)}{p(y_k)} \right] = \sum_{k=1}^K \mathbb{E}_ {y_k \sim p(y)} \left[ \frac{p(y_k \mid x)}{p(y_k)} \right] = \sum_{k=1}^K \int p(y_k) \frac{p(y_k \mid x)}{p(y_k)} dy_k = K$$

   Using Jensen's Inequality on the convex function $-\log(\cdot)$:

   $$\mathcal{L}_ {\text{InfoNCE}} \ge -\mathbb{E}_ {x, y^+} \left[ \log \frac{\frac{p(y^+ \mid x)}{p(y^+)}}{\frac{p(y^+ \mid x)}{p(y^+)} + K} \right]$$

   For large $K$, $\frac{p(y^+ \mid x)}{p(y^+)} + K \approx K$, so:

   $$\mathcal{L}_ {\text{InfoNCE}} \ge -\mathbb{E}_ {x, y^+} \left[ \log \frac{p(y^+ \mid x)}{p(y^+)} \right] + \log(K) = -I(X; Y) + \log(K)$$

   Rearranging terms yields the lower bound:

   $$I(X; Y) \ge \log(K) - \mathcal{L}_ {\text{InfoNCE}}$$

**Result:** $\displaystyle I(X; Y) \ge \log(K) - \mathcal{L}_{\text{InfoNCE}}$

---

### 3.3 The Temperature Parameter ($\tau$)

The temperature parameter $\tau$ controls the hardness of negative mining during training.

Let $s_{i,k} = \frac{z_i^T z_k}{\tau}$. The InfoNCE loss for anchor $i$ is:

```math
\ell_i = -\log \frac{\exp(s_{i,j})}{\sum_{k \neq i} \exp(s_{i,k})}
```

The gradient with respect to similarity $s_{i,k}$ of a negative candidate $k$ is:

```math
\frac{\partial \ell_i}{\partial (z_i^T z_k)} = \frac{1}{\tau} P(k \mid i) = \frac{1}{\tau} \frac{\exp(z_i^T z_k / \tau)}{\sum_{m \neq i} \exp(z_i^T z_m / \tau)}
```

- **Small $\tau$ ($\tau \to 0$):** $P(k \mid i)$ acts as a hard `max`. The gradient is dominated entirely by the hardest negative (the sample closest to $z_i$). This forces tight separation but can cause instability if negative samples contain false negatives (same class).
- **Large $\tau$ ($\tau \to \infty$):** $P(k \mid i) \to \frac{1}{K}$. Gradients become uniform across all negative samples, leading to slow feature convergence.

**Result:** $\displaystyle \frac{\partial \ell_i}{\partial (z_i^T z_k)} = \frac{1}{\tau} \frac{\exp(z_i^T z_k / \tau)}{\sum_{m \neq i} \exp(z_i^T z_m / \tau)}$

---

### 3.4 SimCLR & The Projection Head

**SimCLR (Chen et al., 2020)** uses a batch size of $N$ to create $2N$ augmented views. For each positive pair $(i, j)$, all other $2(N-1)$ views in the batch act as negative samples (NT-Xent loss).

**Why use a projection head $g_\phi$?**
The contrastive loss explicitly removes task-irrelevant information (e.g., orientation or exact background color) to maintain augmentation invariance. If loss is applied directly to representation $h$, useful visual details are irreversibly erased. By applying loss to $z = g_\phi(h)$, $z$ discards nuisance features while $h$ retains rich downstream information.

---

### 3.5 MoCo (Momentum Contrast)

SimCLR requires huge batch sizes ($N = 4096$) to provide sufficient negative samples. **MoCo (He et al., 2020)** decouples dictionary size $K$ from batch size by maintaining a dynamic queue of negative keys.

To keep representations in the queue consistent across iterations, MoCo uses a **momentum encoder**:

```math
\theta_k \leftarrow m \theta_k + (1 - m) \theta_q
```

where $m \approx 0.999$. This momentum update ensures key representations change smoothly as the query encoder updates.

---

## 4. Non-Contrastive Methods (BYOL & Barlow Twins)

Non-contrastive SSL methods eliminate negative pairs altogether. However, learning positive pairs alone risks **representation collapse** (where $f(x) = \mathbf{c}$ for all $x$).

### 4.1 BYOL (Bootstrap Your Own Latent)
BYOL uses two neural networks:
1. **Online Network:** Encoder $f_\theta$, Projector $g_\theta$, Predictor $q_\theta$.
2. **Target Network:** Encoder $f_\xi$, Projector $g_\xi$. Parameters $\xi$ are updated via momentum $\xi \leftarrow m \xi + (1-m)\theta$.

**Loss:** Normalized Mean Squared Error between online prediction and target projection:

```math
\mathcal{L}_{\text{BYOL}} = 2 - 2 \cdot \frac{q_\theta(z_\theta)^T z_\xi}{\|q_\theta(z_\theta)\|_2 \|z_\xi\|_2}
```

**Why BYOL avoids collapse:**
The combination of a **stop-gradient** on the target network, the predictor module $q_\theta$, and the momentum update breaks architectural symmetry, preventing the trivial constant solution.

---

### 4.2 Barlow Twins
Barlow Twins applies a cross-correlation objective directly to the normalized outputs $Z^A, Z^B \in \mathbb{R}^{N \times d}$ of two augmented views:

$$ \mathcal{C}_{ij} = \frac{\sum_b z_{b,i}^A z_{b,j}^B}{\sqrt{\sum_b (z_{b,i}^A)^2} \sqrt{\sum_b (z_{b,j}^B)^2}} $$

$$ \mathcal{L}_{\text{Barlow}} = \sum_i (1 - \mathcal{C}_{ii})^2 + \lambda \sum_i \sum_{j \neq i} \mathcal{C}_{ij}^2 $$

- **Invariance Term ($\mathcal{C}_{ii} \to 1$):** Forces representations of positive views to correlate perfectly.
- **Redundancy Reduction ($\mathcal{C}_{ij} \to 0$):** Forces off-diagonal feature dimensions to be uncorrelated, ensuring each vector component captures independent information.

**Result:** $\displaystyle \mathcal L_{\text{Barlow}} = \sum_i (1 - \mathcal C_{ii})^2 + \lambda \sum_i \sum_{j \neq i} \mathcal C_{ij}^2$

---

## 5. Masked Image Modeling (MAE)

**Masked Autoencoders (MAE; He et al., 2022)** adapt generative self-supervised learning to Vision Transformers (ViT).

### 5.1 Architecture & Mechanics
1. **Patch Extraction:** An image $x \in \mathbb{R}^{H \times W \times C}$ is divided into non-overlapping patches.
2. **High Masking Ratio (75%):** 75% of patches are randomly removed.
3. **Encoder:** A heavy ViT encoder processes **only the 25% visible patches** (reducing FLOPs by ~4x).
4. **Decoder:** Learnable mask tokens `[M]` are re-inserted for missing patches. A lightweight ViT decoder reconstructs raw pixel values for the masked patches.

### 5.2 Reconstruction Objective

```math
\mathcal{L}_{\text{MAE}} = \frac{1}{|M|} \sum_{i \in M} \| x_i - \hat{x}_i \|_2^2
```

Because image pixels are highly redundant, high masking ratios (75%) prevent simple local interpolation, forcing the ViT to learn high-level semantic understanding of object structures.

---

## 6. Multimodal Contrastive Learning (CLIP)

**CLIP (Radford et al., 2021)** aligns vision and language representations over a dataset of $N$ (image, text) pairs $(x_i^I, x_i^T)$.

1. **Dual Encoders:** Image encoder $E_I(x^I) \to z^I$ and Text encoder $E_T(x^T) \to z^T$, normalized to unit length.
2. **Similarity Matrix:** $S_{ij} = (z_i^I)^T z_j^T / \tau \in \mathbb{R}^{N \times N}$.
3. **Symmetric Cross-Entropy Loss:**

   $$\mathcal{L}_ {\text{CLIP}} = \frac{1}{2} \left( \mathcal{L}_ {\text{Image} \to \text{Text}} + \mathcal{L}_ {\text{Text} \to \text{Image}} \right)$$

This cross-modal alignment enables zero-shot classification without retraining.

---

## 7. Failure Cases

1. **Representation Collapse:** If negative contrastive terms or asymmetric momentum targets are omitted, encoders collapse to constant representations $f(x) = \mathbf{c}$.
2. **Augmentation Shortcut Exploitation:** If color jitter is omitted, contrastive models exploit color histograms to match views, failing to encode object shapes.
3. **False Negative Penalty:** Random sampling treats same-class samples in a mini-batch as negative pairs, forcing the encoder to separate semantically identical objects.
4. **Batch Size Sensitivity:** SimCLR performance drops sharply at batch sizes $< 256$ due to insufficient negative sampling density.
5. **Downstream Task Mismatch:** Features engineered to be invariant to spatial orientation (e.g., random flip) perform poorly on tasks requiring spatial geometry (e.g., 3D bounding box estimation).
6. **High Compute Overhead:** Training contrastive encoders to saturation requires 2-10x more compute epochs than supervised training.

---

## 8. Connections

- **Dimensionality Reduction (PCA, Topic 10):** SSL generalizes linear subspace projections to non-linear neural manifolds.
- **Autoencoders (Topic 17):** MAE generalizes denoising autoencoders to ViT patch masking.
- **Generative Models (Topic 19):** InfoNCE shares density ratio principles with GAN discriminators and noise-contrastive estimation.

---

## 9. References

- Oord, A. v. d., Li, Y., & Vinyals, O. (2018). *Representation Learning with Contrastive Predictive Coding*. arXiv:1807.03748.
- Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). *A Simple Framework for Contrastive Learning of Visual Representations*. ICML.
- He, K., Fan, H., Wu, Y., Xie, S., & Girshick, R. (2020). *Momentum Contrast for Unsupervised Visual Representation Learning*. CVPR.
- Grill, J. B., et al. (2020). *Bootstrap Your Own Latent - A New Approach to Self-Supervised Learning*. NeurIPS.
- Zbontar, J., Jing, L., Misra, I., LeCun, Y., & Deny, S. (2021). *Barlow Twins: Self-Supervised Learning via Redundancy Reduction*. ICML.
- He, K., Chen, X., Xie, S., Li, Y., Dollár, P., & Girshick, R. (2022). *Masked Autoencoders Are Scalable Vision Learners*. CVPR.
- Radford, A., et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision*. ICML.
