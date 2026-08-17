# Topic 22: Self-Supervised Learning

> **Phase:** 5 | **Status:** ✅ Complete | **Prerequisites:** [13 Neural Networks](../13_neural_networks/README.md), [14 CNN](../14_cnn/README.md), [17 Autoencoder](../17_autoencoder/README.md)

## Overview

Self-Supervised Learning (SSL) enables representation learning without human annotations by constructing pretext tasks directly from the unlabeled data. This module explores contrastive learning (SimCLR, MoCo), non-contrastive methods (BYOL), and masked image modeling (MAE) to understand how robust, generalizable embeddings are formed.

## Scope

- **In scope:** the InfoNCE/NT-Xent objective in log-space, a contrastive training loop on toy data, temperature ablation, MAE-style random patch masking, and the representation-collapse failure case — all in pure NumPy (scikit-learn is used for toy data generation only).
- **Out of scope:** image augmentation pipelines, momentum encoders at scale, and pre-training on real vision datasets.

## Contents

| File | Type | Description |
|------|------|-------------|
| [theory.md](theory.md) | Theory | InfoNCE derivation, contrastive vs non-contrastive objectives, MAE mechanics, and representation collapse. |
| [first_principles.ipynb](first_principles.ipynb) | Code | NumPy implementation of InfoNCE, contrastive training loop, embedding visualization, and temperature ablation. |
| [exercises.ipynb](exercises.ipynb) | Practice | Hand-computing contrastive loss, NT-Xent implementation, and conceptual comparisons between SSL paradigms. |

The reusable, unit-tested reference implementations live in [`src/ml_first_principles/ssl_models.py`](https://github.com/hien078/Machine-Learning-from-scratch/blob/master/src/ml_first_principles/ssl_models.py) (`InfoNCELoss`, `PatchMasking`), covered by `tests/test_phase5_models.py`.

## Connections

- **Prerequisites:** [Topic 13: Neural Networks](../13_neural_networks/README.md), [Topic 14: CNN](../14_cnn/README.md), [Topic 17: Autoencoder](../17_autoencoder/README.md)
- **Related:** [Synthesis: Loss Functions Map](../../synthesis/loss_functions_map.md)
- **Next:** [Topic 19: Generative Models](../19_generative_models/README.md) — generative pre-training complements contrastive representation learning
