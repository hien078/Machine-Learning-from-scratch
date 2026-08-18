# Generative Models Theory

## 0. Notation

| Symbol | Type | Meaning |
|---|---|---|
| $x$ | Vector | Observed data vector (e.g., an image) |
| $z$ | Vector | Latent representation |
| $p_{\text{data}}(x)$ | Probability | True underlying data distribution |
| $p_\theta(x)$ | Probability | Model distribution parameterized by $\theta$ |
| $q_\phi(z \mid x)$ | Probability | Approximate posterior (Encoder in VAE) |
| $p_\theta(x \mid z)$ | Probability | Likelihood (Decoder in VAE) |
| $D(x)$ | Function | Discriminator / Critic output |
| $G(z)$ | Function | Generator output |
| $\epsilon$ | Vector | Standard Gaussian noise |
| $\beta_t$ | Scalar | Forward process variance schedule in Diffusion |
| $\alpha_t$ | Scalar | $1 - \beta_t$ in Diffusion |
| $\bar{\alpha}_t$ | Scalar | $\prod_{s=1}^t \alpha_s$ |

## 1. WHY: The Generative Modeling Problem

The core objective of generative modeling is density estimation and sampling. 
Given a dataset $\mathcal{D} = \lbrace x^{(1)}, \ldots, x^{(N)}\rbrace$ drawn from an unknown true distribution $p_{\text{data}}(x)$, we seek to learn a model $p_\theta(x)$ that accurately approximates $p_{\text{data}}(x)$.

Generative models solve fundamental problems:
1. **Unsupervised Representation Learning:** Discovering meaningful latent factors without labels.
2. **Synthesis:** Generating novel, realistic images, text, and audio.
3. **Density Estimation:** Assigning likelihoods to data points, useful for anomaly detection.

**Taxonomy:**
* **Prescribed Models (Explicit Density):** Tractable or approximate densities. 
  * Examples: VAEs (approximate), Autoregressive Models (tractable), Normalizing Flows (tractable).
* **Implicit Models:** Can sample from the model but lack explicit density evaluation.
  * Examples: GANs.

## 2. WHAT: Latent Variable Models

Latent variable models introduce an unobserved variable $z$ to explain the observed variable $x$. 
The marginal likelihood is given by:

$$ p_\theta(x) = \int p_\theta(x \mid z) p(z) dz $$

In deep generative models, $p_\theta(x \mid z)$ is parameterized by neural networks. 
This integral is generally intractable because we must integrate over all possible values of $z$.
This intractable marginal likelihood leads to two major paths in generative modeling:
1. Variational approximations (VAEs) which maximize a lower bound.
2. Adversarial training (GANs) which bypass density estimation entirely.

## 3. HOW: Variational Autoencoders (Extensions)

Building on Topic 17, which covers the standard Evidence Lower Bound (ELBO), we focus on advanced concepts.

### $\beta$-VAE and Disentanglement

The standard ELBO is:

```math
\mathcal{L}_{\text{ELBO}} = \mathbb{E}_{q_\phi(z \mid x)}[\log p_\theta(x \mid z)] - D_{\text{KL}}(q_\phi(z \mid x) \| p(z))
```

$\beta$-VAE modifies this by explicitly weighting the KL divergence term:

```math
\mathcal{L}_{\beta} = \mathbb{E}_{q_\phi(z \mid x)}[\log p_\theta(x \mid z)] - \beta D_{\text{KL}}(q_\phi(z \mid x) \| p(z))
```

**Rule (Information Bottleneck):** $\beta > 1$ heavily penalizes the capacity of the latent channel. It forces the encoder to be extremely efficient, encouraging *disentangled representations* where individual dimensions of $z$ correspond to independent, interpretable factors of variation (like color, size, rotation).

### Posterior Collapse and KL Annealing

**Failure Mode:** If the decoder is highly flexible (e.g., an autoregressive PixelCNN decoder), it may completely ignore the latent variable $z$. The model trivially sets $q_\phi(z \mid x) = p(z)$, making the KL term 0. This is known as *posterior collapse*.

**Rule (KL Annealing):** We start training with $\beta = 0$ (only reconstruction loss) and slowly increase $\beta$ to 1 (or higher) over many epochs. This forces the decoder to rely on $z$ early in training before the KL penalty becomes too severe.

### The Information Preference Property

When the decoder is powerful, the model prefers to model local statistics with the decoder and ignore the latent code. By restricting the capacity of the decoder or using KL annealing, we force the model to encode global structure into $z$.

### Conditional VAE (CVAE)

To generate specific classes or conditions, we condition both the encoder and decoder on a label $y$:

```math
\mathcal{L}_{\text{CVAE}} = \mathbb{E}_{q_\phi(z \mid x, y)}[\log p_\theta(x \mid z, y)] - D_{\text{KL}}(q_\phi(z \mid x, y) \| p(z \mid y))
```

**Result:** This allows controlled generation. We sample $z \sim p(z)$ and generate $x \sim p_\theta(x \mid z, y)$.

## 4. HOW: Generative Adversarial Networks (GAN)

GANs avoid explicit density estimation entirely, setting up a minimax game between a Generator $G$ and Discriminator $D$.

### Minimax Formulation

$$ \min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p(z)}[\log (1 - D(G(z)))] $$

Here, $D(x)$ predicts the probability that $x$ is real. $G$ tries to minimize the chance that $D$ correctly classifies its outputs as fake.

### Optimal Discriminator Proof

For a fixed generator $G$, we find the optimal discriminator $D^\ast$.

**Step 1 (Calculus of Variations):** Write the expected value as an integral over $x$. Note that sampling $z \sim p(z)$ and applying $G(z)$ is equivalent to sampling $x \sim p_g(x)$:

$$ V(G, D) = \int \left[ p_{\text{data}}(x) \log D(x) + p_g(x) \log(1 - D(x)) \right] dx $$

**Step 2 (Pointwise Optimization):** To maximize this integral, we maximize the integrand for every $x$. Let $y = D(x)$, $a = p_{\text{data}}(x)$, $b = p_g(x)$.
We maximize $f(y) = a \log(y) + b \log(1 - y)$ with respect to $y$.

**Step 3 (Derivative):** Set derivative to zero:

```math
\frac{df}{dy} = \frac{a}{y} - \frac{b}{1 - y} = 0 \implies a(1 - y) = by \implies y = \frac{a}{a+b}
```

**Result:** The optimal discriminator is

```math
D^\ast(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}
```

Pointwise, $D^\ast(x)$ is the share of the density at $x$ that belongs to the data, so $D^\ast \equiv \frac{1}{2}$ exactly when $p_g = p_{\text{data}}$.

### Connection to Jensen-Shannon Divergence

Substitute $D^\ast$ back into $V$:

$$ V(G, D^\ast) = \mathbb{E}_{x \sim p_{\text{data}}}\left[ \log \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)} \right] + \mathbb{E}_{x \sim p_g}\left[ \log \frac{p_g(x)}{p_{\text{data}}(x) + p_g(x)} \right] $$

Multiply numerator and denominator by $\frac{1}{2}$:

$$ = \mathbb{E}_{x \sim p_{\text{data}}}\left[ \log \left( \frac{1}{2} \frac{p_{\text{data}}(x)}{\frac{p_{\text{data}}(x) + p_g(x)}{2}} \right) \right] + \mathbb{E}_{x \sim p_g}\left[ \log \left( \frac{1}{2} \frac{p_g(x)}{\frac{p_{\text{data}}(x) + p_g(x)}{2}} \right) \right] $$

Extract $\log \frac{1}{2} = -\log 2$:

```math
= -2 \log 2 + D_{\text{KL}}\left(p_{\text{data}} \Big\| \frac{p_{\text{data}} + p_g}{2}\right) + D_{\text{KL}}\left(p_g \Big\| \frac{p_{\text{data}} + p_g}{2}\right)
```

**Result:** At the optimal discriminator the objective becomes

```math
V(G, D^\ast) = -\log 4 + 2 \cdot JSD(p_{\text{data}} \Vert p_g)
```

Training $G$ against a perfect discriminator therefore minimises the Jensen-Shannon divergence between $p_g$ and $p_{\text{data}}$, whose minimum is attained exactly when the two distributions coincide.

### Training Instabilities

1. **Mode Collapse:** The generator finds a single output (or a few) that fools the discriminator and produces only that, ignoring the rest of the data distribution.
2. **Vanishing Gradients:** If $D$ becomes too good (closer to $D^\ast$), the function $-\log(1 - D(G(z)))$ saturates and gives vanishing gradients to $G$.

### Wasserstein GAN (WGAN)

**Problem:** When supports of $p_{\text{data}}$ and $p_g$ are disjoint (common in high dimensions), JSD is constant ($\log 2$), giving 0 gradients.

**Solution:** Use Earth Mover's (Wasserstein-1) Distance:

```math
W(p_{\text{data}}, p_g) = \inf_{\gamma \in \Pi(p_{\text{data}}, p_g)} \mathbb{E}_{(x,y) \sim \gamma}[\| x - y \|]
```

**Kantorovich-Rubinstein Duality:** 
This primal form is intractable. We use the dual form:

```math
W(p_{\text{data}}, p_g) = \sup_{\|f\|_L \le 1} \mathbb{E}_{x \sim p_{\text{data}}}[f(x)] - \mathbb{E}_{x \sim p_g}[f(x)]
```

**Result (WGAN Objective):** Replace $D$ with a Critic $f$ constrained to be 1-Lipschitz. 
The gradient penalty $\lambda \mathbb{E}[(\Vert\nabla_x f(x)\Vert_2 - 1)^2]$ is typically used to enforce the 1-Lipschitz constraint.

## 5. HOW: Diffusion Models

Diffusion models learn to reverse a gradual noising process over $T$ steps.

### Forward Process

A Markov chain adding Gaussian noise with variance schedule $\beta_1, \ldots, \beta_T$:

```math
q(x_t \mid x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t I)
```

**Closed-form $q(x_t \mid x_0)$ derivation:**
Let $\alpha_t = 1 - \beta_t$ and $\bar\alpha_t = \prod_{s=1}^t \alpha_s$. Using properties of sums of independent Gaussians:

**Step 1:**

```math
x_t = \sqrt{\alpha_t} x_{t-1} + \sqrt{1 - \alpha_t}\epsilon_{t-1}
```

**Step 2:**

```math
x_{t-1} = \sqrt{\alpha_{t-1}} x_{t-2} + \sqrt{1 - \alpha_{t-1}}\epsilon_{t-2}
```

**Step 3:** Substitute Step 2 into Step 1:

```math
x_t = \sqrt{\alpha_t \alpha_{t-1}} x_{t-2} + \sqrt{\alpha_t(1 - \alpha_{t-1})}\epsilon_{t-2} + \sqrt{1 - \alpha_t}\epsilon_{t-1}
```

Since $\epsilon_i \sim \mathcal{N}(0, I)$, the sum of the noise terms is distributed as

```math
\mathcal{N}(0, (\alpha_t(1 - \alpha_{t-1}) + 1 - \alpha_t)I) = \mathcal{N}(0, (1 - \alpha_t \alpha_{t-1})I)
```

**Step 4:** By induction, extending this to $t$ steps gives:

```math
x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1 - \bar{\alpha}_t}\epsilon
```

**Result:** The closed form of the forward marginal is

```math
q(x_t \mid x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) I)
```

Any $x_t$ can therefore be sampled directly from $x_0$ in a single step, without simulating the chain.

### Reverse Process

We approximate the true reverse process $q(x_{t-1} \mid x_t)$ with a neural network $p_\theta$:

```math
p_\theta(x_{t-1} \mid x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))
```

Each reverse step is again Gaussian: the network reads the mean and covariance of $x_{t-1}$ off the current noisy sample $x_t$ and the step index $t$.

### Simplified Noise Prediction Objective

The variational bound on $\log p(x_0)$ yields a loss that matches $\mu_\theta$ to the true posterior mean. 
However, Ho et al. showed it is empirically better to parameterize the network to predict the noise $\epsilon$ added at step $t$:

```math
\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]
```

**Connection to Score Matching:**
Predicting the noise $\epsilon$ is mathematically equivalent to predicting the score function $\nabla_x \log p_t(x)$. 
The generation process is effectively Langevin dynamics, iteratively moving samples along the score gradients towards high-density regions of $p_{\text{data}}$.

## 6. Failure Cases

1. **Posterior Collapse (VAE):** Powerful decoders ignore the latent code.
   - *Explanation:* The KL term dominates the optimization, causing the encoder to output the prior $p(z)$ and the decoder to drop its dependence on $z$.
2. **Mode Collapse (GAN):** Generator produces only a few modes of the true distribution.
   - *Explanation:* The generator finds a specific region that fools the discriminator and maps all $z$ to that region, failing to capture the full diversity.
3. **Training Instability (GAN):** Oscillatory behavior and vanishing gradients.
   - *Explanation:* The discriminator easily perfectly separates real/fake data, causing gradients to vanish for $G$. Addressed by WGAN.
4. **Blurry Outputs (VAE):** VAEs often generate blurry images.
   - *Explanation:* The $L_2$ pixel loss (derived from a Gaussian likelihood assumption) heavily penalizes outliers, leading the model to predict the average of possible pixels, resulting in blur.
5. **Slow Sampling (Diffusion):** Generation is computationally expensive.
   - *Explanation:* Simulating the reverse process requires hundreds or thousands of sequential network evaluations. Addressed by DDIM and ODE solvers.
6. **Evaluation Metrics Challenges:** Difficulty in evaluating implicit models.
   - *Explanation:* Inception Score (IS) and Fréchet Inception Distance (FID) have biases, rely on ImageNet features, and don't directly measure test log-likelihood.

## 7. Connections

* **Topic 13: Neural Networks:** Multilayer perceptrons and CNNs form the foundation of our generators, discriminators, and denoisers.
* **Topic 17: Autoencoders:** Understanding informational bottlenecks and latent spaces is crucial for building VAEs.
* **Probabilistic View of ML:** Generative models explicitly map neural networks to probability distributions, replacing deterministic targets with likelihood objectives.

## 8. References

* Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes.
* Goodfellow, I., et al. (2014). Generative Adversarial Nets.
* Arjovsky, M., Chintala, S., & Bottou, L. (2017). Wasserstein GAN.
* Ho, J., Jain, A., & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models.
* Song, Y., & Ermon, S. (2019). Generative Modeling by Estimating Gradients of the Data Distribution.
