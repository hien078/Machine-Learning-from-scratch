# Generative and Self-Supervised Learning — Cross-Topic Synthesis

> The "learning without labels" map across topics 17, 19, 21, 22 — and the RL bridge (18).
> See [INDEX.md](../INDEX.md) for the full curriculum index.

---

## Overview

Topics 17, 19, 21, and 22 all answer the same question — *what can be learned from data
alone?* — with two families of answers:

- **Generative:** model $p_{\text{data}}(x)$ well enough to sample from it (VAE, GAN,
  diffusion, autoregressive LLMs).
- **Self-supervised:** invent a pretext task whose labels come from the data itself
  (reconstruction, masking, contrastive discrimination, next-token prediction) and
  keep the learned encoder.

The two overlap: every generative objective doubles as a pretext task, and several
"representation" objectives are secretly density-ratio or likelihood estimators.

---

## Three Routes to Generation

| Route | Idea | Representative |
|---|---|---|
| Explicit density | Maximize (a bound on) $\log p_\theta(x)$ | VAE (ELBO), autoregressive LLM (exact factorized likelihood) |
| Implicit | Never write down $p_\theta(x)$; train a sampler adversarially | GAN |
| Denoising / score | Learn to invert gradual noising, i.e. estimate $\nabla_x \log p_t(x)$ | Diffusion |

### The trade-off table ([19](../topics/19_generative_models/theory.md))

| Criterion | VAE | GAN | Diffusion |
|---|---|---|---|
| Likelihood | Lower bound (ELBO) | None (implicit) | Bound via variational objective |
| Sample quality | Blurry (Gaussian decoder averages modes) | Sharp | Sharp, state of the art |
| Mode coverage | Good (likelihood covers support) | Poor — mode collapse | Good |
| Training stability | High (single objective) | Low — minimax game, vanishing $D$ gradients | High (simple regression $\Vert \epsilon - \epsilon_\theta \Vert^2$) |
| Sampling cost | One decoder pass | One generator pass | Hundreds of sequential steps (DDIM helps) |
| Latent space | Explicit, smooth | Explicit but unstructured | Implicit (noise trajectory) |

No column wins every row — the field's history is a walk around this table.

---

## The ELBO Reappears

The same variational bound powers both latent-variable eras:

```math
\log p_\theta(x) \ge \mathbb{E}_{q(z \mid x)}[\log p_\theta(x \mid z)] - D_{\text{KL}}\bigl(q(z \mid x) \,\Vert\, p(z)\bigr)
```

- **VAE ([17 §6](../topics/17_autoencoder/theory.md)):** one latent $z$, encoder =
  learned $q$, reparameterization trick to backprop through sampling.
- **Diffusion ([19 §5](../topics/19_generative_models/theory.md)):** the *same* bound
  applied to a chain of latents $x_1, \dots, x_T$, where $q$ is the **fixed** forward
  noising process — no encoder to learn. The bound simplifies (Ho et al.) to noise
  regression, $\mathcal{L}_{\text{simple}} = \mathbb{E}\left[\Vert \epsilon - \epsilon_\theta(x_t, t) \Vert^2\right]$
  — why diffusion trains as stably as an autoencoder while sampling like a likelihood
  model. A diffusion model is usefully seen as a hierarchical VAE with a frozen
  encoder.

---

## Reconstruction Objectives: Autoencoder → Denoising → MAE

One objective, escalating corruption:

| Model | Corruption | Reconstructs | Purpose |
|---|---|---|---|
| [Autoencoder (17)](../topics/17_autoencoder/theory.md) | None (bottleneck $k < d$) | Whole input | Compression / manifold learning |
| Denoising AE (17 §4) | Additive Gaussian noise | Clean input | Robust features; learns a vector field toward the manifold |
| [MAE (22 §5)](../topics/22_self_supervised_learning/theory.md) | Mask 75% of ViT patches | Masked patches only | Semantic pretraining at scale |
| Diffusion (19 §5) | Full noising schedule to $\mathcal{N}(0, I)$ | The noise itself, at every level | Generation |

The through-line: without corruption, reconstruction risks the identity map; the harder
the corruption, the more *semantic* the features must be. MAE's 75% masking and
diffusion's full noise schedule are the same denoising idea pushed to opposite ends —
one keeps the encoder, the other keeps the sampler.

---

## Contrastive Objectives: InfoNCE

Instead of rebuilding pixels, classify which candidate is the true positive among
$K$ negatives ([22 §3](../topics/22_self_supervised_learning/theory.md)):

```math
\mathcal{L}_{\text{InfoNCE}} = -\mathbb{E}\left[ \log \frac{\exp(f(x, y^+))}{\exp(f(x, y^+)) + \sum_{k=1}^{K} \exp(f(x, y_k))} \right],
\qquad
I(X; Y) \ge \log K - \mathcal{L}_{\text{InfoNCE}}
```

- The optimal critic recovers the **density ratio** $p(y \mid x)/p(y)$ — the same
  quantity a GAN discriminator estimates ($D^\ast = p_{\text{data}}/(p_{\text{data}} + p_g)$):
  contrastive and adversarial training are cousins via noise-contrastive estimation.
- Augmentation choice *is* the inductive bias: it declares which variation is nuisance.
- Dropping negatives requires an anti-collapse mechanism (BYOL's stop-gradient +
  predictor, Barlow Twins' redundancy reduction); CLIP applies InfoNCE symmetrically
  to image/text pairs.

---

## Next-Token Prediction as Self-Supervision

The largest self-supervised system in existence is the LLM
([21 §1](../topics/21_llm_engineering/theory.md)). The chain rule of probability
factorizes the density exactly:

```math
\log p_\theta(x) = \sum_{i=1}^{n} \log p_\theta(t_i \mid t_1, \dots, t_{i-1})
```

so next-token prediction is simultaneously (a) an *exact* explicit-density generative
model — no ELBO gap, no adversary — and (b) a masked-prediction pretext task where the
"mask" is always the future (BERT masks random positions; MAE masks patches). It is
cross-entropy from the [Loss Functions Map](loss_functions_map.md) at web scale, with
the Transformer supplying the parallelism
(see [Sequence Models and Attention](sequence_models_and_attention.md)).

---

## The RL Bridge: RLHF and DPO

Pre-training gives a density model of text, not an assistant. Alignment reuses
[topic 18](../topics/18_reinforcement_learning/theory.md):

1. **RLHF ([18 §12](../topics/18_reinforcement_learning/theory.md)):** fit a reward
   model on human preference pairs (Bradley–Terry), then run PPO with a KL leash to
   the reference policy:

```math
\max_{\pi_\theta}\; \mathbb{E}\left[ r_\phi(x, y) \right] - \beta\, D_{\text{KL}}\bigl(\pi_\theta(\cdot \mid x) \,\Vert\, \pi_{\text{ref}}(\cdot \mid x)\bigr)
```

2. **DPO ([21 §4](../topics/21_llm_engineering/theory.md)):** the optimal policy of
   that objective has closed form, the partition function cancels in preference
   differences, and the RL loop collapses to a supervised NLL on preferences.

The KL-regularized objective is the ELBO pattern once more: data-fit plus a KL term
pulling toward a reference distribution — the VAE keeps $q$ near the prior, RLHF keeps
$\pi_\theta$ near $\pi_{\text{ref}}$.

---

## Decision Guide

```
No labels — what do you want?
├── Samples / synthesis
│   ├── Need likelihoods or a latent space → VAE (17, 19)
│   ├── Best image quality, have compute → Diffusion (19)
│   ├── One-pass sampling, accept instability → GAN (19)
│   └── Text / discrete sequences → autoregressive LLM (16, 21)
└── Representations for downstream tasks
    ├── Strong augmentations known → contrastive (SimCLR/MoCo) or BYOL (22)
    ├── ViT at scale, minimal augmentation design → MAE (22)
    ├── Paired modalities → CLIP (22)
    └── Text → pre-trained LLM features (21)
```

---

## Connections

- **Topics:** [17 Autoencoder](../topics/17_autoencoder/theory.md), [19 Generative Models](../topics/19_generative_models/theory.md), [21 LLM Engineering](../topics/21_llm_engineering/theory.md), [22 Self-Supervised Learning](../topics/22_self_supervised_learning/theory.md), [18 Reinforcement Learning](../topics/18_reinforcement_learning/theory.md), [10 PCA](../topics/10_pca/theory.md) (linear ancestor of the autoencoder)
- **Related synthesis:** [Probabilistic View of ML](probabilistic_view_of_ml.md) (MLE and variational inference foundations), [Loss Functions Map](loss_functions_map.md) (KL divergence, cross-entropy), [Sequence Models and Attention](sequence_models_and_attention.md), [Supervised vs Unsupervised](supervised_vs_unsupervised.md)
- **Maps:** [INDEX.md](../INDEX.md)
