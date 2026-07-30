# Dimensionality Reduction (LDA + t-SNE)

> **Phase:** 2 | **Status:** ✅ Complete | **Prerequisites:** [10 PCA](../10_pca/README.md), [Information Theory](../../foundations/information_theory/README.md)

## Overview

Supervised and nonlinear dimensionality reduction beyond PCA. Fisher's Linear
Discriminant Analysis (LDA) maximizes between-class scatter relative to within-class
scatter for labeled data. t-SNE preserves local pairwise similarities through a
nonlinear embedding using KL divergence minimization with a heavy-tailed Student-t
kernel. Includes comparison of PCA vs LDA vs t-SNE on classification datasets.

## Contents

| # | File | Type | Description |
|--:|---|---|---|
| 1 | `theory.md` | Theory | LDA scatter matrices, Fisher criterion, t-SNE KL objective, gradient intuition, failure cases |
| 2 | `first_principles.ipynb` | Computation | WHY→WHAT→HOW→BUILD→VERIFY — LDA and t-SNE from scratch, PCA vs LDA vs t-SNE comparison |
| 3 | `exercises.ipynb` | Practice | Hand scatter-matrix calculation, LDA coding task, t-SNE conceptual questions |

## Connections

- **Prereqs:** [10 PCA](../10_pca/README.md), [Information Theory](../../foundations/information_theory/README.md)
- **Synthesis:** [Geometry of ML](../../synthesis/geometry_of_ml.md)
- **Next:** [17 Autoencoder](../17_autoencoder/README.md)
