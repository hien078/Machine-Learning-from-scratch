# Principal Component Analysis

> **Phase:** 1 | **Status:** ✅ Complete | **Prerequisites:** Linear Algebra, Probability & Statistics

## Overview

Finding the directions of maximal variance in data. The topic poses PCA as a constrained variance-maximization problem, solves it with eigendecomposition of the covariance matrix, proves the equivalence with minimal reconstruction error, connects both to the SVD, and studies when PCA misleads — unstandardized features, nonlinear structure, and identifiability of signs.

## Scope

- **In scope:** the variance-maximization derivation, the reconstruction-error view and their equivalence, the SVD connection, choosing $k$ via explained variance, whitening, standardization effects, and failure cases — all in pure NumPy.
- **Out of scope:** probabilistic PCA, kernel PCA, and nonlinear methods (t-SNE, UMAP) — those are [Topic 12](../12_dimensionality_reduction/README.md).

## Contents

| # | File | Type | Description |
|--:|---|---|---|
| 1 | [theory.md](theory.md) | Theory | WHY, covariance, variance maximization, reconstruction, SVD connection, failure cases |
| 2 | [first_principles.ipynb](first_principles.ipynb) | Computation | WHY→WHAT→HOW→BUILD→VERIFY — PCA from scratch via SVD, scree plot, sklearn comparison |
| 3 | [exercises.ipynb](exercises.ipynb) | Practice | Hand eigendecomposition, reconstruction error, standardization comparison, failure analysis, SVD connection |

## Connections

- **Prereqs:** [Linear Algebra](https://github.com/hien078/applied-mathematics-foundation), [Probability & Statistics](https://github.com/hien078/applied-mathematics-foundation)
- **Synthesis:** [Geometry of ML](../synthesis/geometry_of_ml.md)
- **Next:** [12 Dimensionality Reduction](../12_dimensionality_reduction/README.md), [17 Autoencoder](../17_autoencoder/README.md)
