# Topics — First Principles ML Stories

Each topic tells a complete reasoning story: **WHY → WHAT → HOW → BUILD → VERIFY → CONNECTIONS**

## Status Dashboard

| # | Topic | Status | Phase | Prerequisites | Main Notebook | Notes |
|---|---|---|---|---|---|---|
| 01 | Linear Regression (+ Polynomial) | 🟡 Draft | 1 | Linear Algebra, Calculus | first_principles.ipynb | Full theory, notebook, exercises |
| 02 | Gradient Descent | 🟡 Draft | 1 | Calculus, 01 LR | first_principles.ipynb | |
| 03 | Regularization (Ridge + Lasso) | 🟡 Draft | 1 | 01 LR, 02 GD | first_principles.ipynb | |
| 04 | Logistic Regression | 🟡 Draft | 1 | Probability, 02 GD | first_principles.ipynb | |
| 05 | Decision Tree | 🟡 Draft | 2 | Information Theory | first_principles.ipynb | |
| 06 | Ensemble Methods (RF + GBM) | 🟡 Draft | 2 | 05 Decision Tree | first_principles.ipynb | |
| 07 | KNN | ⚪ Planned | 2 | Norms/Distances | first_principles.ipynb | Skeleton only |
| 08 | Naive Bayes | ⚪ Planned | 2 | Probability | first_principles.ipynb | |
| 09 | SVM | ⚪ Planned | 2 | Optimization | first_principles.ipynb | |
| 10 | PCA | 🟡 Draft | 1 | Eigenvalues, SVD | first_principles.ipynb | |
| 11 | Clustering (KM+HC+DB+GMM) | 🟡 Draft | 2 | Norms/Distances | first_principles.ipynb | |
| 12 | Dimensionality Reduction (LDA+tSNE) | ⚪ Planned | 2 | PCA, Eigenvalues | first_principles.ipynb | Theory only; first_principles.ipynb & exercises.ipynb planned |
| 13 | Neural Networks | 🟡 Draft | 3 | 02 GD, 04 LogReg | first_principles.ipynb | |
| 14 | CNN | 🟡 Draft | 3 | 13 NN | first_principles.ipynb | exercises.ipynb planned |
| 15 | RNN/LSTM | 🟡 Draft | 3 | 13 NN | first_principles.ipynb | exercises.ipynb planned |
| 16 | Transformer (condensed) | ⚪ Planned | 4 | 13 NN, 15 RNN | first_principles.ipynb | Title only; exercises.ipynb planned |
| 17 | Autoencoder | ⚪ Planned | 3 | 13 NN | first_principles.ipynb | Title only; exercises.ipynb planned |

**Status labels:** `⚪ Planned` → `🟡 Draft` → `🟢 Complete` → `✅ Verified`

## Phases

- **Phase 1 (Core Classical ML):** Topics 01-04, 10
- **Phase 2 (Classical Models):** Topics 05-09, 11-12
- **Phase 3 (Deep Learning):** Topics 13-15, 17
- **Phase 4 (Modern AI):** Topic 16

## File Structure per Topic

```
topics/XX_topic_name/
├── README.md              ← Overview, prerequisites, status
├── theory.md              ← Pure theory (whiteboard content)
├── first_principles.ipynb ← Computation (computer content)
└── exercises.ipynb        ← Practice problems
```

## Navigation

See [`INDEX.md`](../INDEX.md) for prerequisite graph, math-to-algorithm mapping,
and algorithm families.
