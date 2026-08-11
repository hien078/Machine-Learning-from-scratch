# Topic 19: Generative Models

> **Phase:** 5 | **Status:** ✅ Complete | **Prerequisites:** [13 Neural Networks](../13_neural_networks/README.md), [17 Autoencoder](../17_autoencoder/README.md), Probability & Statistics

## Overview

Generative models aim to learn the underlying true data distribution $p_{\text{data}}(x)$ from empirical samples, enabling the generation of novel, realistic data. This module explores deep generative architectures, focusing on VAE extensions, Generative Adversarial Networks (GANs, WGANs), and Denoising Diffusion Probabilistic Models (DDPMs).

## Scope

- **In scope:** VAE (ELBO, reparameterization), GAN minimax objectives and WGAN intuition, diffusion forward process, and an exact-score reverse process on a toy Gaussian mixture — all in pure NumPy.
- **Out of scope:** training a learned noise-prediction network, image-scale diffusion, and production generative pipelines.

## Module Contents

| File | Type | Description |
|------|------|-------------|
| [`theory.md`](theory.md) | 📚 Theory | Formulations for $\beta$-VAE, GAN minimax and WGAN, and DDPM derivations. |
| [`first_principles.ipynb`](first_principles.ipynb) | 💻 Implementation | NumPy implementation of VAE, GAN, and the diffusion forward process plus an exact-score reverse process on a toy mixture. |
| [`exercises.ipynb`](exercises.ipynb) | 🛠️ Practice | KL divergence computation, reparameterization gradients, conceptual comparisons. |

The reusable, unit-tested reference implementations live in [`src/ml_first_principles/generative_models.py`](../../src/ml_first_principles/generative_models.py) (`VAE`, `vae_elbo_loss`, `GANGenerator`, `GANDiscriminator`, GAN losses), covered by `tests/test_phase5_models.py`.

## Connections

- **Prerequisites:**
  - [13_neural_networks](../13_neural_networks/README.md)
  - [17_autoencoder](../17_autoencoder/README.md) (Crucial for VAE basics)
- **Related Synthesis:**
  - [Probabilistic View of ML](../../synthesis/probabilistic_view_of_ml.md)
  - [Loss Functions Map](../../synthesis/loss_functions_map.md)
- **Next Topics:**
  - [21 LLM Engineering](../21_llm_engineering/README.md)
  - [22 Self-Supervised Learning](../22_self_supervised_learning/README.md)
