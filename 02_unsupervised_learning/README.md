# Unsupervised Learning

Algorithms that discover hidden structure in **unlabeled** data — no target variable Y. The goal is to find patterns: groups, lower-dimensional representations, or frequent associations.

---

## Prerequisites

### Math (from [math_for_ai_roadmap.md](../00_foundations/01_math_essentials/math_for_ai_roadmap.md))
- **(1) Linear Algebra:** eigenvalues, eigenvectors, SVD, covariance matrix, matrix decomposition.
- **(2) Calculus:** gradient for optimization-based methods (e.g., GMM via EM).
- **(3) Probability & Statistics:** Gaussian mixture, EM algorithm, MLE, KL divergence.
- **(4) Optimization:** convergence of iterative algorithms (K-Means, EM).
- **(5) Information Theory:** entropy, mutual information (for feature selection, clustering metrics).

### Code
- Python, NumPy, Matplotlib, scipy.
- scikit-learn (for comparison after implementing from scratch).

---

## Subprojects (ordered by learning priority)

Grouped by intent: dimensionality reduction → clustering → association rules.

| # | Folder | Algorithm | Core idea | Key math |
|---|--------|-----------|-----------|----------|
| 1 | [`01_pca/`](01_pca/) | PCA | Project onto directions of max variance | Eigendecomposition of covariance / SVD |
| 2 | [`02_lda/`](02_lda/) | LDA | Maximize between-class / within-class scatter | Generalized eigenvalue problem |
| 3 | [`03_t_sne/`](03_t_sne/) | t-SNE | Preserve local neighbor structure in 2D/3D | KL divergence, Student-t kernel |
| 4 | [`04_k_means_clustering/`](04_k_means_clustering/) | K-Means | Minimize within-cluster sum of squares | Lloyd's algorithm, Voronoi partitions |
| 5 | [`05_hierarchical_clustering/`](05_hierarchical_clustering/) | Hierarchical | Agglomerative bottom-up merging | Linkage criteria (single, complete, Ward) |
| 6 | [`06_dbscan/`](06_dbscan/) | DBSCAN | Density-based: core/border/noise points | ε-neighborhood, minPts |
| 7 | [`07_gaussian_mixture_model/`](07_gaussian_mixture_model/) | GMM | Soft clustering via Gaussian mixture | EM algorithm, MLE, latent variables |
| 8 | [`08_apriori/`](08_apriori/) | Apriori | Frequent itemset mining via candidate generation | Support, confidence, lift |
| 9 | [`09_fp_growth/`](09_fp_growth/) | FP-Growth | Frequent itemset mining without candidate generation | FP-tree, conditional pattern base |

---

## Learning Objectives

After completing this module, you should be able to:

- [ ] Implement K-Means from scratch in NumPy, including the elbow method for choosing k.
- [ ] Implement PCA from scratch using eigendecomposition (not `np.linalg.svd`).
- [ ] Explain why t-SNE is non-convex and why re-runs give different embeddings.
- [ ] Articulate when DBSCAN is better than K-Means (non-convex clusters, noise).
- [ ] Implement Apriori and verify against a small market-basket dataset.
- [ ] Evaluate clustering quality using silhouette score and ARI.

---

## Key References

- Bishop — *Pattern Recognition and Machine Learning*, Chapters 9 (EM, K-Means), 12 (PCA).
- Hastie, Tibshirani, Friedman — *The Elements of Statistical Learning*, Chapter 14.
- van der Maaten & Hinton (2008) — *Visualizing Data using t-SNE*.
- Ester et al. (1996) — *A Density-Based Algorithm for Discovering Clusters* (DBSCAN).

---

## Subproject Layout

Each subproject should follow:
```
algorithm_name/
├── data/           # Datasets (gitignored if large)
├── notebooks/      # EDA, visualization of clusters/embeddings
├── src/            # From-scratch implementation
├── tests/          # Unit tests
└── reports/        # Findings, plots, comparisons
```
