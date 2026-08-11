# Autoencoder

> **Phase:** 3 | **Status:** ✅ Complete | **Prerequisites:** 13 Neural Networks, 10 PCA, Information Theory

## Overview

Encoder-decoder architecture, representation learning via reconstruction,
linear autoencoder–PCA equivalence, denoising and sparse variants,
variational autoencoder (ELBO, reparameterization trick, KL regularization).

## Contents

| # | File | Type | Description |
|--:|---|---|---|
| 1 | `theory.md` | Theory | Architecture, PCA connection, denoising/sparse/variational AE, ELBO derivation, reparameterization trick, failure cases |
| 2 | `first_principles.ipynb` | Computation | Linear AE ↔ PCA, nonlinear AE, denoising, VAE from scratch, latent space visualization, PyTorch comparison |
| 3 | `exercises.ipynb` | Practice | Hand calculation (linear AE reconstruction), reparameterization trick coding, KL divergence conceptual analysis |

## Connections

- **Prereqs:** [13 Neural Networks](../13_neural_networks/README.md), [10 PCA](../10_pca/README.md), [Information Theory](https://github.com/hien078/applied-mathematics-foundation)
- **Synthesis:** [Probabilistic View of ML](../../synthesis/probabilistic_view_of_ml.md), [Geometry of ML](../../synthesis/geometry_of_ml.md)
- **Next:** Generative models (GANs, diffusion)
