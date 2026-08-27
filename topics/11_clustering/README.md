# Clustering

> **Phase:** 2 | **Status:** ✅ Complete | **Prerequisites:** Linear Algebra, Probability & Statistics

## Overview

Unsupervised grouping of data points via three complementary approaches: K-Means (centroid-based, minimises within-cluster sum of squares), DBSCAN (density-based, finds arbitrarily shaped clusters and noise), and Gaussian Mixture Models with EM (probabilistic soft clustering).

## Scope

- **In scope:** the K-Means objective and its coordinate-descent derivation, K-Means++ initialization, DBSCAN's density definitions, GMM fitted with a fully derived EM algorithm, the elbow method, and failure cases on non-spherical data — all in pure NumPy (scikit-learn for comparison only).
- **Out of scope:** hierarchical/agglomerative clustering, spectral clustering, and cluster-validity indices beyond inertia.

## Contents

| # | File | Type | Description |
|--:|---|---|---|
| 1 | [theory.md](theory.md) | Theory | K-Means objective & coordinate descent derivation, DBSCAN density definitions, GMM with EM derivation, failure cases |
| 2 | [first_principles.ipynb](first_principles.ipynb) | Computation | From-scratch K-Means (random + K-Means++ init), DBSCAN, GMM/EM; convergence visualisation; elbow method; sklearn comparison; failure experiments |
| 3 | [exercises.ipynb](exercises.ipynb) | Practice | Hand K-Means iteration, K-Means++ coding task, conceptual analysis of method tradeoffs |

The reusable, unit-tested reference implementation lives in [`src/ml_first_principles/distance_models.py`](https://github.com/hien078/Machine-Learning-from-scratch/blob/master/src/ml_first_principles/distance_models.py) (`KMeans`), covered by `tests/test_phase2_models.py`; DBSCAN and GMM/EM are implemented in the notebook.

## Connections

- **Prereqs:** [Linear Algebra](https://github.com/hien078/applied-mathematics-foundation), [Probability & Statistics](https://github.com/hien078/applied-mathematics-foundation)
- **Related:** [10 PCA](../10_pca/README.md) (dimensionality reduction before clustering), [07 KNN](../07_knn/README.md) (distance-based, supervised counterpart)
- **Synthesis:** [Supervised vs. Unsupervised](../synthesis/supervised_vs_unsupervised.md)
- **Next:** [12 Dimensionality Reduction](../12_dimensionality_reduction/README.md), [13 Neural Networks](../13_neural_networks/README.md)
