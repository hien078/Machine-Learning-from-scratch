# Topics — First Principles ML Stories

Each topic tells a complete reasoning story: **WHY → WHAT → HOW → BUILD → VERIFY → CONNECTIONS**

## Status Dashboard

| # | Topic | Status | Phase | Prerequisites | Notes |
|---|---|---|---|---|---|
| 01 | Linear Regression (+ Polynomial) | ✅ Complete | 1 | Linear Algebra, Calculus | |
| 02 | Gradient Descent | ✅ Complete | 1 | Calculus, 01 LR | |
| 03 | Regularization (Ridge + Lasso) | ✅ Complete | 1 | 01 LR, 02 GD | |
| 04 | Logistic Regression | ✅ Complete | 1 | Probability, 02 GD | |
| 05 | Decision Tree | ✅ Complete | 2 | Information Theory | |
| 06 | Ensemble Methods (RF + GBM) | ✅ Complete | 2 | 05 Decision Tree | |
| 07 | KNN | ✅ Complete | 2 | Norms/Distances | |
| 08 | Naive Bayes | ✅ Complete | 2 | Probability | |
| 09 | SVM | ✅ Complete | 2 | Optimization | |
| 10 | PCA | ✅ Complete | 1 | Eigenvalues, SVD | |
| 11 | Clustering (KM+HC+DB+GMM) | ✅ Complete | 2 | Norms/Distances | |
| 12 | Dimensionality Reduction (LDA+tSNE) | ✅ Complete | 2 | 10 PCA, Information Theory | |
| 13 | Neural Networks | ✅ Complete | 3 | 02 GD, 04 LogReg | |
| 14 | CNN | ✅ Complete | 3 | 13 NN | Library comparison uses optional PyTorch |
| 15 | RNN/LSTM | ✅ Complete | 3 | 13 NN | Library comparison uses optional PyTorch |
| 16 | Transformer | ✅ Complete | 4 | 13 NN, 15 RNN | Library comparison uses optional PyTorch |
| 17 | Autoencoder | ✅ Complete | 3 | 13 NN, 10 PCA | Library comparison uses optional PyTorch |
| 18 | Reinforcement Learning | ✅ Complete | 5 | 02 GD, 13 NN, Probability | Six notebooks (bandits, MC/TD, policy gradients, deep RL) |
| 19 | Generative Models (VAE+GAN+Diffusion) | ✅ Complete | 5 | 13 NN, 17 AE, Probability | |
| 20 | Graph Neural Networks (GCN+GAT) | ✅ Complete | 5 | 10 PCA, 13 NN | |
| 21 | LLM Engineering (BPE+LoRA+DPO) | ✅ Complete | 5 | 13 NN, 16 Transformer | |
| 22 | Self-Supervised Learning (InfoNCE+MAE) | ✅ Complete | 5 | 13 NN, 14 CNN, 17 AE | |

**Status labels:** `⚪ Planned` → `🟡 Draft` → `✅ Complete` (all four files present and current). The
stricter *Verified* gate (clean top-to-bottom execution plus resolved links) is defined in
[NOTEBOOK_STANDARDS.md](../NOTEBOOK_STANDARDS.md) §9–§10 and checked with
`python scripts/execute_all_notebooks.py` / `python scripts/normalize_notebooks.py`.

## Phases

- **Phase 1 (Core Mathematical ML):** Topics 01–04, 10
- **Phase 2 (Classical Machine Learning):** Topics 05–09, 11–12
- **Phase 3 (Deep Learning):** Topics 13–15, 17
- **Phase 4 (Transformers):** Topic 16
- **Phase 5 (Modern AI & Advanced Architectures):** Topics 18–22

## File Structure per Topic

```
topics/XX_topic_name/
├── README.md              ← Overview, prerequisites, status
├── theory.md              ← Pure theory (whiteboard content)
├── first_principles.ipynb ← Computation (computer content)
└── exercises.ipynb        ← Practice problems
```

Large topics may add focused notebooks (topic 18 adds `bandits_and_exploration.ipynb`,
`monte_carlo_and_td.ipynb`, `policy_gradient_methods.ipynb`, and `deep_rl_advances.ipynb`).

## Navigation

See [`INDEX.md`](../INDEX.md) for prerequisite graph, math-to-algorithm mapping,
and algorithm families.
