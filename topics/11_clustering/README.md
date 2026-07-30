# Clustering

> **Phase:** 2 | **Status:** ✅ Complete | **Prerequisites:** Linear Algebra, Probability & Statistics

## Overview

Unsupervised grouping of data points via three complementary approaches: K-Means (centroid-based, minimises within-cluster sum of squares), DBSCAN (density-based, finds arbitrarily shaped clusters and noise), and Gaussian Mixture Models with EM (probabilistic soft clustering).

## Contents

| # | File | Type | Description |
|--:|---|---|---|
| 1 | `theory.md` | Theory | K-Means objective & coordinate descent derivation, DBSCAN density definitions, GMM with EM derivation, failure cases |
| 2 | `first_principles.ipynb` | Computation | From-scratch K-Means (random + K-Means++ init), DBSCAN, GMM/EM; convergence visualisation; elbow method; sklearn comparison; failure experiments |
| 3 | `exercises.ipynb` | Practice | Hand K-Means iteration, K-Means++ coding task, conceptual analysis of method tradeoffs |

## Connections

- **Prereqs:** [Linear Algebra](../../foundations/linear_algebra/README.md), [Probability & Statistics](../../foundations/probability_statistics/README.md)
- **Related:** [10 PCA](../10_pca/README.md) (dimensionality reduction before clustering), [08 KNN](../08_knn/README.md) (distance-based, supervised counterpart)
- **Synthesis:** [Supervised vs. Unsupervised](../../synthesis/supervised_vs_unsupervised.md)
- **Next:** [12 Neural Network Fundamentals](../12_neural_net_fundamentals/README.md)
