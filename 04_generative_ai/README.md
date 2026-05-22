# Generative AI

Models that learn the data distribution **p(x)** (or a conditional **p(x|c)**) and generate new samples: images, text, audio. Covers VAEs, GANs, Diffusion Models, and Large Language Models.

---

## Prerequisites

### Math (from [math_for_ai_roadmap.md](../00_foundations/01_math_essentials/math_for_ai_roadmap.md))
- **(1) Linear Algebra:** all of deep_learning prerequisites + low-rank decomposition (LoRA).
- **(2) Calculus:** reparameterization trick, score function (∇ₓ log p(x)).
- **(3) Probability & Statistics:** latent variables, variational inference, ELBO, MLE, sampling (MCMC, importance).
- **(4) Optimization:** Adam/AdamW, learning rate warmup, gradient clipping.
- **(5) Information Theory:** KL divergence, cross-entropy, mutual information, ELBO decomposition.
- **(8) Differential Equations** *(for diffusion)*: SDE, ODE solvers, Fokker-Planck, Itô calculus basics.

### Code
- PyTorch (all subprojects).
- Hugging Face Transformers + PEFT (for LLM fine-tuning).
- torchvision (for image datasets).

### Recommended prior modules
- `03_deep_learning/01_multi_layer_perceptron/` (backprop fundamentals).
- `03_deep_learning/04_transformer/` (attention, for LLMs and diffusion transformers).

---

## Subprojects

### [`01_vae/`](01_vae/)
Variational Autoencoders.

| Topic | What to implement | Key math |
|---|---|---|
| Vanilla VAE | Encoder qφ(z|x), decoder pθ(x|z) | ELBO = E[log p(x|z)] − KL(q‖p) |
| Reparameterization trick | z = μ + σ ⊙ ε | Gradient flow through sampling |
| β-VAE | Tunable KL weight | Disentanglement vs reconstruction |
| Conditional VAE | Generation conditioned on label | p(x|z, c) |

### [`02_gan/`](02_gan/)
Generative Adversarial Networks.

| Topic | What to implement | Key math |
|---|---|---|
| Vanilla GAN | Generator + Discriminator | min_G max_D E[log D(x)] + E[log(1−D(G(z)))] |
| DCGAN | Convolutional GAN | Architectural guidelines (no pooling, BN) |
| WGAN / WGAN-GP | Wasserstein distance | Lipschitz constraint, gradient penalty |
| Conditional GAN | Class-conditioned generation | Concat label to input |

### [`03_diffusion_models/`](03_diffusion_models/)
Denoising Diffusion Probabilistic Models and score-based methods.

| Topic | What to implement | Key math |
|---|---|---|
| DDPM | Forward noise + reverse denoise | q(xₜ|xₜ₋₁) = N(√αₜ xₜ₋₁, βₜI) |
| Noise schedule | Linear, cosine | αₜ, ᾱₜ cumulative product |
| Score matching | ∇ₓ log p(x) estimation | Denoising score matching objective |
| Sampling | DDPM, DDIM | Deterministic (ODE) vs stochastic (SDE) |

### [`04_large_language_models/`](04_large_language_models/)
LLM architectures, fine-tuning, and alignment.

| Topic | What to implement | Key math |
|---|---|---|
| GPT-style architecture | Decoder-only transformer | Causal attention mask, KV cache |
| Tokenization | BPE from scratch | Byte-pair encoding algorithm |
| Fine-tuning (LoRA) | Low-rank adapter | ΔW = BA, rank r ≪ d |
| RLHF / DPO | Alignment from preferences | KL-regularized policy optimization |

---

## Learning Objectives

After completing this module, you should be able to:

- [ ] Implement a VAE from scratch in PyTorch; derive ELBO two ways (KL gap + Jensen).
- [ ] Train a DCGAN on MNIST/CelebA and explain mode collapse.
- [ ] Implement DDPM forward + reverse process; generate images from noise.
- [ ] Explain the connection between diffusion (discrete) and score-based SDE (continuous).
- [ ] Fine-tune a small LLM with LoRA and evaluate with perplexity.
- [ ] Implement BPE tokenization from scratch.

---

## Key References

- Kingma & Welling (2014) — *Auto-Encoding Variational Bayes*.
- Goodfellow et al. (2014) — *Generative Adversarial Nets*.
- Ho et al. (2020) — *Denoising Diffusion Probabilistic Models*.
- Song & Ermon (2019, 2021) — *Score-Based Generative Modeling*.
- Radford et al. (2018, 2019) — *GPT, GPT-2*.
- Hu et al. (2022) — *LoRA: Low-Rank Adaptation of Large Language Models*.
- Rafailov et al. (2023) — *Direct Preference Optimization*.

---

## Subproject Layout

Each subproject should follow:
```
algorithm_name/
├── data/           # Datasets (gitignored if large)
├── notebooks/      # Generation samples, training curves, latent space visualization
├── src/            # From-scratch implementation + PyTorch version
├── tests/          # Unit tests (loss computation, output shapes)
└── reports/        # Generated samples, FID scores, findings
```
