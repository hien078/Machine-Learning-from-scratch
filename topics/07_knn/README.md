# K-Nearest Neighbors

> **Phase:** 2 | **Status:** ✅ Complete | **Prerequisites:** Linear Algebra (Norms/Distances)

## Overview

Non-parametric classification by majority vote of the $K$ nearest training points.
Covers distance metrics (L1, L2, Minkowski), the bias-variance tradeoff in K selection,
weighted KNN, the curse of dimensionality, computational complexity, and acceleration
structures (KD-trees, ball trees).

## Contents

| # | File | Type | Description |
|--:|---|---|---|
| 1 | `theory.md` | Theory | Distance metrics, K selection, curse of dimensionality, failure cases |
| 2 | `first_principles.ipynb` | Computation | From-scratch KNN, effect of K, library comparison, experiments |
| 3 | `exercises.ipynb` | Practice | Hand distance calculation, weighted KNN implementation, conceptual questions |

## Connections

- **Prereqs:** [Norms and Distances](https://github.com/hien078/applied-mathematics-foundation)
- **Synthesis:** [Model Selection Guide](../synthesis/model_selection_guide.md)
- **Next:** [Clustering](../11_clustering/README.md), [Dimensionality Reduction](../12_dimensionality_reduction/README.md)
