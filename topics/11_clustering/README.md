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

- **Prereqs:** [Linear Algebra](https://github.com/hien078/applied-mathematics-foundation), [Probability & Statistics](https://github.com/hien078/applied-mathematics-foundation)
- **Related:** [10 PCA](../10_pca/README.md) (dimensionality reduction before clustering), [07 KNN](../07_knn/README.md) (distance-based, supervised counterpart)
- **Synthesis:** [Supervised vs. Unsupervised](../synthesis/supervised_vs_unsupervised.md)
- **Next:** [12 Dimensionality Reduction](../12_dimensionality_reduction/README.md), [13 Neural Networks](../13_neural_networks/README.md)
