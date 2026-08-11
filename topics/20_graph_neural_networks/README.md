# Topic 20: Graph Neural Networks (GNNs)

> **Phase:** 5 | **Status:** ✅ Complete | **Prerequisites:** [10 PCA](../10_pca/README.md), [13 Neural Networks](../13_neural_networks/README.md), Linear Algebra

## Overview

Graph Neural Networks (GNNs) extend deep learning methods to graph-structured data. This module explores the foundations of graph deep learning, bridging spectral graph theory with modern spatial approaches like Graph Convolutional Networks (GCNs), Graph Attention Networks (GATs), and the general Message Passing framework.

## Scope

- **In scope:** graph Laplacian and spectral intuition, GCN normalization, single-head GAT attention, message passing, semi-supervised node classification on Karate Club, and the over-smoothing failure case — all in pure NumPy.
- **Out of scope:** multi-head attention at scale, graph pooling/readout architectures, PyTorch Geometric, and large-graph mini-batching.

## Contents

| File | Description |
|------|-------------|
| [`theory.md`](./theory.md) | First-principles derivation from Spectral Graph Theory to GCN, GAT, and MPNNs. |
| [`first_principles.ipynb`](./first_principles.ipynb) | Pure NumPy implementation of Graph Laplacian, GCN, and GAT layers with node-classification training and over-smoothing experiments. |
| [`exercises.ipynb`](./exercises.ipynb) | Pen-and-paper derivations, implementations, and conceptual exercises. |

The reusable, unit-tested reference implementations live in [`src/ml_first_principles/gnn_models.py`](../../src/ml_first_principles/gnn_models.py) (`GCNLayer`, `GATLayer`), covered by `tests/test_phase5_models.py`.

## Connections

- **Prerequisites:**
  - [Topic 10: PCA (Eigenvalues/Eigenvectors)](../10_pca/README.md)
  - [Topic 13: Neural Networks](../13_neural_networks/README.md)
- **Related Synthesis:** [Geometry of ML](../../synthesis/geometry_of_ml.md)
- **Next Topics:** [22 Self-Supervised Learning](../22_self_supervised_learning/README.md)
