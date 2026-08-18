# Autoencoder — Theory

## 1. WHY — Learning Compressed Representations

High-dimensional data often lives near a lower-dimensional manifold. An
autoencoder learns this manifold by training a neural network to reconstruct
its input through a bottleneck. Unlike PCA, which is limited to linear
subspaces, a nonlinear autoencoder can capture curved manifolds. The
bottleneck forces the network to discover the most informative features
rather than memorizing an identity map.

**Applications:** dimensionality reduction, feature learning, denoising,
anomaly detection, and (with the variational extension) generative modeling.

## 2. WHAT — Notation and Architecture

### 2.1 Notation table

| Symbol | Type | Meaning |
|---|---:|---|
| $x \in \mathbb{R}^d$ | vector | input data point |
| $z \in \mathbb{R}^k$ | vector | latent (code) representation, $k < d$ |
| $\hat{x} \in \mathbb{R}^d$ | vector | reconstruction |
| $f_\theta: \mathbb{R}^d \to \mathbb{R}^k$ | function | encoder with parameters $\theta$ |
| $g_\phi: \mathbb{R}^k \to \mathbb{R}^d$ | function | decoder with parameters $\phi$ |
| $n$ | scalar | number of training samples |
| $\lambda$ | scalar | regularization coefficient |

### 2.2 Basic autoencoder

The encoder maps input to code, the decoder maps code back to input space:

$$z = f_\theta(x), \qquad \hat{x} = g_\phi(z)$$

The objective minimizes reconstruction error over the dataset:

$$\mathcal{L}(\theta, \phi) = \frac{1}{n}\sum_{i=1}^{n} \lVert x_i - g_\phi(f_\theta(x_i)) \rVert_2^2$$

For binary/probability outputs, binary cross-entropy may replace MSE:

$$\mathcal{L}_{\text{BCE}} = -\frac{1}{n}\sum_{i=1}^{n}\sum_{j=1}^{d}\bigl[x_{ij}\log\hat{x}_{ij} + (1-x_{ij})\log(1-\hat{x}_{ij})\bigr]$$

### 2.3 Architecture choices

| Variant | Constraint | Effect |
|---|---|---|
| Undercomplete | $k < d$ | Bottleneck forces compression |
| Overcomplete | $k \geq d$ | Needs additional regularization |
| Tied weights | $W_{\text{dec}} = W_{\text{enc}}^T$ | Reduces parameters, regularizes |

## 3. HOW — Linear Autoencoder and PCA Connection

### 3.1 Linear autoencoder

Consider a single-layer encoder and decoder with no activation:

$$z = W_e x + b_e, \qquad \hat{x} = W_d z + b_d$$

where $W_e \in \mathbb{R}^{k \times d}$, $W_d \in \mathbb{R}^{d \times k}$.

The reconstruction objective becomes:

$$\min_{W_e, W_d, b_e, b_d} \frac{1}{n}\sum_{i=1}^{n}\lVert x_i - W_d(W_e x_i + b_e) - b_d\rVert_2^2$$

### 3.2 Equivalence with PCA subspace

**Theorem (Baldi & Hornik, 1989).** For mean-centered data, the optimal
linear autoencoder with $k$-dimensional bottleneck spans the same subspace as
the $\text{top-}k$ principal components.

**Derivation sketch.** With centered data ($\bar{x} = 0$), set $b_e = 0$,
$b_d = 0$. The objective is:

$$\min_{W_e, W_d} \frac{1}{n}\lVert X - XW_e^T W_d^T\rVert_F^2$$

Let $M = W_d W_e$. This is a $\text{rank-}k$ approximation problem. By the
Eckart–Young theorem, the optimal $\text{rank-}k$ approximation of $X$ in Frobenius
norm uses the $\text{top-}k$ singular vectors. Therefore $M$ projects onto the
subspace spanned by the $\text{top-}k$ right singular vectors of $X$, which equals
the $\text{top-}k$ eigenvectors of the covariance matrix $\frac{1}{n}X^T X$.

**Result:** A linear autoencoder with $k$ latent dimensions recovers the PCA
subspace. However, the individual encoder/decoder weight matrices are not
unique — any invertible $k \times k$ rotation gives the same reconstruction.

## 4. Denoising Autoencoder (DAE)

### 4.1 Idea

Instead of reconstructing a clean input from itself, corrupt the input and
reconstruct the original:

$$\tilde{x} = x + \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, \sigma^2 I)$$

$$\mathcal{L}_{\text{DAE}} = \frac{1}{n}\sum_{i=1}^{n}\lVert x_i - g_\phi(f_\theta(\tilde{x}_i))\rVert_2^2$$

### 4.2 Why it works

By forcing the network to denoise, the encoder must learn features that are
robust to noise rather than memorizing individual samples. The DAE implicitly
learns a vector field pointing toward the data manifold (Vincent et al.,
2010). This acts as a regularizer even when $k \geq d$.

## 5. Sparse Autoencoder

For an overcomplete autoencoder ($k \geq d$), add a sparsity penalty on
the hidden activations:

$$\mathcal{L}_{\text{sparse}} = \frac{1}{n}\sum_{i=1}^{n}\lVert x_i - \hat{x}_i\rVert_2^2 + \lambda \sum_{j=1}^{k} |h_j|$$

where $h_j = \frac{1}{n}\sum_{i=1}^{n} z_j^{(i)}$ is the average activation
of unit $j$ across the dataset. The L1 penalty encourages most hidden units
to be inactive for any given input, resulting in a sparse distributed code.

Alternatively, a KL-divergence penalty targets a desired activation probability
$\rho$ (typically $\rho \approx 0.05$):

```math
\Omega_{\text{KL}} = \sum_{j=1}^{k} \text{KL}(\rho \,\|\, \hat{\rho}_j) = \sum_{j=1}^{k}\left[\rho\log\frac{\rho}{\hat{\rho}_j} + (1-\rho)\log\frac{1-\rho}{1-\hat{\rho}_j}\right]
```

## 6. Variational Autoencoder (VAE)

### 6.1 Generative model

A VAE is a latent-variable generative model. The generative process is:

$$z \sim p(z) = \mathcal{N}(0, I), \qquad x \mid z \sim p_\phi(x \mid z)$$

The goal is to maximize the marginal log-likelihood:

```math
\log p_\phi(x) = \log \int p_\phi(x \mid z)\, p(z)\, dz
```

This integral is intractable because it requires integrating over all
possible latent codes.

### 6.2 Variational inference and the ELBO

Introduce an approximate posterior $q_\theta(z \mid x)$ (the encoder) and
derive a tractable lower bound.

**ELBO derivation.** Start from the log-likelihood and apply Jensen's inequality:

```math
\log p_\phi(x) = \log \int p_\phi(x \mid z)\,p(z)\,dz
```

Multiply and divide by $q_\theta(z \mid x)$:

```math
= \log \int q_\theta(z \mid x)\,\frac{p_\phi(x \mid z)\,p(z)}{q_\theta(z \mid x)}\,dz
```

```math
= \log\, \mathbb{E}_{q_\theta(z|x)}\left[\frac{p_\phi(x \mid z)\,p(z)}{q_\theta(z \mid x)}\right]
```

By Jensen's inequality ($\log$ is concave):

```math
\geq \mathbb{E}_{q_\theta(z|x)}\left[\log\frac{p_\phi(x \mid z)\,p(z)}{q_\theta(z \mid x)}\right]
```

Split the log:

```math
= \underbrace{\mathbb{E}_{q_\theta(z|x)}[\log p_\phi(x \mid z)]}_{\text{reconstruction}} - \underbrace{D_{\text{KL}}\bigl(q_\theta(z \mid x) \,\|\, p(z)\bigr)}_{\text{regularization}}
```

**Result:** The Evidence Lower Bound (ELBO) is:

```math
\text{ELBO}(x; \theta, \phi) = \mathbb{E}_{q_\theta(z|x)}[\log p_\phi(x \mid z)] - D_{\text{KL}}\bigl(q_\theta(z \mid x) \,\|\, p(z)\bigr)
```

The gap between $\log p_\phi(x)$ and the ELBO equals
$D_{\text{KL}}(q_\theta(z|x) \thinspace\Vert\thinspace p_\phi(z|x)) \geq 0$, so maximizing the ELBO
simultaneously improves the generative model and tightens the approximation.

### 6.3 Gaussian encoder

Choose the encoder to output a diagonal Gaussian:

```math
q_\theta(z \mid x) = \mathcal{N}\bigl(\mu_\theta(x),\; \text{diag}(\sigma_\theta^2(x))\bigr)
```

where $\mu_\theta(x)$ and $\log\sigma_\theta^2(x)$ are outputs of the encoder network.
We parameterize $\log\sigma^2$ instead of $\sigma$ for numerical stability.

### 6.4 Closed-form KL divergence

For $q = \mathcal{N}(\mu, \text{diag}(\sigma^2))$ and $p = \mathcal{N}(0, I)$, both $k$-dimensional:

```math
D_{\text{KL}}(q \,\|\, p) = \frac{1}{2}\sum_{j=1}^{k}\left[\mu_j^2 + \sigma_j^2 - \log\sigma_j^2 - 1\right]
```

**Derivation.** Using the general formula for KL between two Gaussians:

```math
D_{\text{KL}}(\mathcal{N}_1 \,\|\, \mathcal{N}_0) = \frac{1}{2}\left[\text{tr}(\Sigma_0^{-1}\Sigma_1) + (\mu_0 - \mu_1)^T\Sigma_0^{-1}(\mu_0 - \mu_1) - k + \log\frac{|\Sigma_0|}{|\Sigma_1|}\right]
```

Substitute $\mu_0 = 0$, $\Sigma_0 = I$, $\mu_1 = \mu$, $\Sigma_1 = \text{diag}(\sigma^2)$:

- $\text{tr}(I^{-1}\thinspace\text{diag}(\sigma^2)) = \sum_j \sigma_j^2$
- $\mu^T I^{-1} \mu = \sum_j \mu_j^2$
- $\log\frac{|I|}{|\text{diag}(\sigma^2)|} = -\sum_j \log\sigma_j^2$

**Result:** $D_{\text{KL}} = \frac{1}{2}\sum_{j=1}^{k}(\mu_j^2 + \sigma_j^2 - \log\sigma_j^2 - 1)$

### 6.5 Reparameterization trick

To backpropagate through the sampling $z \sim q_\theta(z \mid x)$, express the
random variable as a deterministic function of the parameters plus external noise:

$$z = \mu_\theta(x) + \sigma_\theta(x) \odot \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, I)$$

This moves the stochasticity to $\varepsilon$, which does not depend on $\theta$,
allowing gradients to flow through $\mu$ and $\sigma$ via standard backpropagation.

Without this trick, computing $\nabla_\theta \mathbb E_{q_\theta}[\cdot]$ would
require high-variance score function estimators (REINFORCE).

### 6.6 Full VAE loss

For a single sample, using one Monte Carlo sample of $\varepsilon$:

```math
\mathcal{L}_{\text{VAE}}(x) = \lVert x - g_\phi(\mu + \sigma \odot \varepsilon)\rVert_2^2 + D_{\text{KL}}\bigl(q_\theta(z \mid x) \,\|\, p(z)\bigr)
```

Over the dataset:

$$\mathcal{L} = \frac{1}{n}\sum_{i=1}^{n}\mathcal{L}_{\text{VAE}}(x_i)$$

## 7. Failure Cases

1. **Identity map (no bottleneck).** If $k \geq d$ with no regularization, the
   autoencoder can learn $f = g^{-1}$: perfect reconstruction but no useful
   features. Always constrain: bottleneck ($k < d$), sparsity, noise, or KL.

2. **Blurry VAE reconstructions.** MSE reconstruction loss averages over modes,
   producing blurry outputs. This is fundamental: the Gaussian decoder
   $p_\phi(x|z) = \mathcal{N}(\mu_\phi(z), \sigma^2 I)$ minimizes MSE, which
   penalizes sharpness. Alternatives: learned variance, adversarial loss.

3. **KL vanishing / posterior collapse.** In VAE training, a powerful decoder
   can reconstruct from $p(z)$ alone, ignoring the encoder. The KL term drops
   to zero ($q \approx p$) and the latent code carries no information. Mitigations:
   KL annealing (warmup $\beta$ from 0 to 1), free bits, or weaker decoders.

4. **Uninterpretable latent space.** A deterministic autoencoder's latent space
   may have gaps and irregular structure. The latent code is optimized for
   reconstruction, not for downstream tasks or smooth interpolation.

5. **Bottleneck too small.** If $k$ is much smaller than the intrinsic
   dimensionality of the data, reconstruction quality degrades sharply and
   important structure is lost.

6. **Bottleneck too large.** If $k$ is close to $d$, the model memorizes rather
   than compresses. Validation reconstruction error plateaus but generalization
   to new data does not improve.

## 8. Connections

- [PCA](../10_pca/README.md) — linear autoencoder recovers PCA subspace (§3.2)
- [Neural Networks](../13_neural_networks/README.md) — encoder/decoder are MLPs trained with backpropagation
- [Dimensionality Reduction](../12_dimensionality_reduction/README.md) — autoencoder as nonlinear alternative
- [Probabilistic View](../synthesis/probabilistic_view_of_ml.md) — VAE as variational inference
- [Information Theory](https://github.com/hien078/applied-mathematics-foundation) — KL divergence, ELBO

---

## 9. References

- **Kingma, D. P., & Welling, M. (2013).** Auto-encoding variational Bayes. *arXiv preprint arXiv:1312.6114*.
- **Baldi, P., & Hornik, K. (1989).** Neural networks and principal component analysis: Learning from examples without local minima. *Neural Networks*, 2(1), 53–58.
- **Vincent, P., Larochelle, H., Lajoie, I., Bengio, Y., & Manzagol, P. A. (2010).** Stacked denoising autoencoders: Learning useful representations in a deep network with a local denoising criterion. *Journal of Machine Learning Research*, 11, 3371–3408.
- **Goodfellow, I., Bengio, Y., & Courville, A. (2016).** *Deep Learning*. MIT Press. Chapter 14: *Autoencoders* & Chapter 20: *Generative Models*.

