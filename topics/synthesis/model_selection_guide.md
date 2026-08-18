# Model Selection Guide — Cross-Topic Synthesis

> Which model to try, in what order, and how to compare candidates without fooling yourself.
> See [INDEX.md](../../INDEX.md) for the full curriculum index.

---

## Overview

Model selection is three decisions in a fixed order: **evaluation design** (how you will
measure), **candidate set** (which families are plausible), and **comparison protocol**
(how you record and decide). Evaluation design comes first because it is the only step
whose mistakes are irreversible — a leaked test set cannot be un-leaked.

---

## Step 1 — Evaluation Design First

Choose the split strategy and metric before comparing models. Model selection performed
on the test set leaks information; the test set is used once, after the complete pipeline
has been selected.

| Data situation | Validation design |
|---|---|
| Independent, identically distributed observations | Random train/validation/test split or shuffled k-fold CV |
| Time-ordered observations | Forward or rolling validation; never shuffle the future into training |
| Multiple rows per person/device/group | Group-aware split (all rows of a group on one side) |
| Rare classes | Stratified split plus class-appropriate metrics (PR-AUC, per-class recall) |
| Small data and tuning | Nested CV when an unbiased comparison matters |

**Leakage pitfalls** (each produces a "too good to be true" score):

- Fitting scalers, PCA, or feature selection on the full dataset before splitting —
  preprocessing must be fit on each fold's training portion only.
- Tuning hyperparameters, or monitoring early stopping, on the test set.
- Shuffled CV on temporal data — the model "predicts" the past from the future.
- Duplicates or grouped rows landing on both sides of a split.
- Target-derived features (target encoding, imputation using $y$) computed on all rows.

---

## Step 2 — Problem Type → Candidate Families

| Problem | Trivial baseline | First real model | Main candidates | Escalate to |
|---|---|---|---|---|
| Tabular regression | Predict the mean | [Linear/Ridge](../01_linear_regression/README.md) | [Random Forest, Boosting](../06_ensemble_methods/README.md) | NN (rarely wins on tabular) |
| Tabular classification | Majority class | [Logistic Regression](../04_logistic_regression/README.md) | [Trees](../05_decision_tree/README.md), [Ensembles](../06_ensemble_methods/README.md), [SVM](../09_svm/README.md), [KNN](../07_knn/README.md) | Boosting, tuned |
| High-dim sparse (text counts) | Majority class | [Naive Bayes](../08_naive_bayes/README.md) | Linear SVM, Logistic | [Transformer](../16_transformer/README.md), pretrained |
| Images | — | Small [CNN](../14_cnn/README.md) | CNN + augmentation | Pretrained backbone |
| Sequences | Last-value / n-gram | [RNN/LSTM](../15_rnn_lstm/README.md) | [Transformer](../16_transformer/README.md) | Pretrained LLM ([21](../21_llm_engineering/README.md)) |
| Graph-structured data | Logistic on node features | [GNN](../20_graph_neural_networks/README.md) | GCN/GAT variants | — |
| Unlabeled grouping | Single cluster | [K-Means](../11_clustering/README.md) | GMM, DBSCAN, hierarchical | — |
| Dimensionality reduction | Keep raw features | [PCA](../10_pca/README.md) | [t-SNE/UMAP](../12_dimensionality_reduction/README.md) (visualization only) | [Autoencoder](../17_autoencoder/README.md) |
| Density / generation | Histogram / GMM | GMM | [VAE, GAN, diffusion](../19_generative_models/README.md) | — |
| Sequential decisions | Random / greedy policy | Tabular Q-learning | [Deep RL](../18_reinforcement_learning/README.md) | — |

---

## Step 3 — Trade-off Table

| Family | Small $n$ | Large $n$ | High $d$ | Interpretability | Train cost | Inference cost |
|---|---|---|---|---|---|---|
| Linear / Logistic | Good | Good | Good with $\ell_1/\ell_2$ | High (coefficients) | Low | Very low |
| Naive Bayes | Very good | OK | Very good (sparse) | Medium | Very low | Very low |
| KNN | Good | Memory-bound | Poor (distance concentration) | Medium (neighbors) | None | High: $O(nd)$ per query |
| Decision Tree | Overfits fast | OK | OK | High (small trees) | Low | Very low |
| Random Forest | Good | Good | Good | Low–medium (importances) | Medium | Medium |
| Gradient Boosting | Good if tuned | Very good | Good | Low–medium | Medium | Low–medium |
| Kernel SVM | Good | Poor: $O(n^2)$–$O(n^3)$ | Good (margin) | Low | High | Medium ($\propto$ support vectors) |
| Neural Networks | Poor without transfer | Very good | Good (learned features) | Low | High | Medium |

---

## Step 4 — Hyperparameter Sensitivity by Family

| Family | Critical hyperparameters | Sensitivity | Practical note |
|---|---|---|---|
| Ridge / Lasso | $\lambda$ | Moderate | Log-spaced grid; loss is smooth in $\log\lambda$ |
| Logistic Regression | Regularization strength | Low–moderate | Rarely the bottleneck |
| KNN | $k$, metric, feature scaling | High | Unscaled features silently break distances |
| Decision Tree | Depth, min samples per leaf | High | Unpruned trees interpolate noise |
| Random Forest | Number of trees, features per split | Low | More trees never hurts accuracy, only time |
| Gradient Boosting | Learning rate $\times$ rounds, depth | High | Small learning rate + early stopping |
| Kernel SVM | $C$, $\gamma$ | Very high | Joint 2-D log grid; wrong $\gamma$ means all points are support vectors |
| K-Means | $k$, initialization | High | Multiple restarts; elbow/silhouette for $k$ |
| DBSCAN | $\varepsilon$, min points | Very high | Small $\varepsilon$ changes flip cluster count |
| Neural Networks | Learning rate first; then width/depth, batch size | Very high | Learning rate dominates every other choice — see [Optimization Methods](optimization_methods_compared.md) |

---

## When From-Scratch Understanding Says "Don't Use the Fancy Model"

Implementing these models from first principles yields concrete stop signs:

- **$n \lesssim d$ or $n$ small:** variance dominates; a regularized linear model beats a
  deep net because the [bias–variance trade-off](bias_variance_tradeoff.md) is against you.
- **KNN in high dimensions:** pairwise distances concentrate, nearest and farthest
  neighbors become indistinguishable — see [Geometry of ML](geometry_of_ml.md).
- **Boosting on noisy labels:** boosting keeps reweighting the mislabeled points it can
  never fit; random forests degrade more gracefully.
- **K-Means on non-convex or unequal-density clusters:** the objective is distortion to
  centroids, so it *will* return $k$ convex cells whether or not they exist in the data.
- **Deep nets on small tabular data:** tree ensembles usually win at a fraction of the
  tuning cost.
- **The tie-break rule:** if the simple model is within one standard deviation
  (across folds/seeds) of the complex one, select the simple model — lower variance,
  cheaper inference, easier audits.

---

## Required Comparison Record

For every selected model record the baseline, split, preprocessing fitted on training
data only, hyperparameter search space, primary and secondary metrics, uncertainty across
folds/seeds, runtime, error slices, and final limitations.

---

## Connections

- **Topics:** [01 Linear Regression](../01_linear_regression/README.md), [04 Logistic Regression](../04_logistic_regression/README.md), [06 Ensemble Methods](../06_ensemble_methods/README.md), [09 SVM](../09_svm/README.md), [11 Clustering](../11_clustering/README.md), [13 Neural Networks](../13_neural_networks/README.md)
- **Related synthesis:** [Bias–Variance Trade-off](bias_variance_tradeoff.md), [Loss Functions Map](loss_functions_map.md), [Optimization Methods Compared](optimization_methods_compared.md), [Supervised vs Unsupervised](supervised_vs_unsupervised.md)
- **Maps:** [INDEX.md](../../INDEX.md)
