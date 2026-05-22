# Model Evaluation — Roadmap

> Training a model is easy. Knowing whether it's actually any good is the hard part.
> Most production ML failures trace back to evaluation, not modeling.

---

## Table of Contents

1. [Why this matters](#why-this-matters)
2. [Quick priority table](#quick-priority-table)
3. Topics in detail:
   - [1. The bias-variance frame](#1-the-bias-variance-frame)
   - [2. Train / val / test splitting](#2-train--val--test-splitting)
   - [3. Cross-validation](#3-cross-validation)
   - [4. Regression metrics](#4-regression-metrics)
   - [5. Classification metrics](#5-classification-metrics)
   - [6. Ranking & retrieval metrics](#6-ranking--retrieval-metrics)
   - [7. Calibration](#7-calibration)
   - [8. Statistical significance](#8-statistical-significance)
   - [9. Hyperparameter tuning](#9-hyperparameter-tuning)
   - [10. Diagnostics & error analysis](#10-diagnostics--error-analysis)

---

## Why this matters

- **A model is what you measure.** Optimize accuracy → get a calibration-broken classifier. Optimize AUC → get probabilities that work for ranking but not decisions.
- **The validation set is the actual product.** You'll spend more time looking at it than at the training loss.
- **"99% accuracy" lies more often than it tells.** Class imbalance, leakage, test-set overfitting, distribution shift — all hide behind a clean headline number.

---

## Quick priority table

| # | Topic | When you hit it | Priority |
|---|---|---|---|
| 1 | Bias-variance frame | Every project | 🔴 Must |
| 2 | Splitting strategy | Every project | 🔴 Must |
| 3 | Cross-validation | Every project | 🔴 Must |
| 4 | Regression metrics | Continuous targets | 🔴 Must |
| 5 | Classification metrics | Categorical targets | 🔴 Must |
| 6 | Ranking metrics | Search / recommend | 🟠 Soon |
| 7 | Calibration | When probabilities are decisions | 🟠 Soon |
| 8 | Statistical significance | "Is the new model actually better?" | 🟠 Soon |
| 9 | Hyperparameter tuning | Every non-trivial project | 🔴 Must |
| 10 | Error analysis | Before any next-iteration decision | 🔴 Must |

---

## 1. The bias-variance frame

- **Bias** — underfitting. Model can't capture the signal even on training data.
- **Variance** — overfitting. Model memorizes training noise.
- **Irreducible error** — Bayes-optimal lower bound; you can't get below it.

```
Test error  ≈  Bias²  +  Variance  +  Noise
```

Visual diagnostic: plot train loss vs val loss over epochs / model complexity.
- Train low, val low, gap small → good.
- Train low, val high, gap large → overfit → regularize / more data / simpler model.
- Train high, val high → underfit → bigger model / more features.

**Self-check:** Given two learning curves, identify which one is bias-limited vs variance-limited in under 10 seconds.

---

## 2. Train / val / test splitting

- **3-way split:** train (fit), val (tune), test (final, touched once).
- **Holdout sizes:** 60/20/20 for small data, 98/1/1 for million-scale.
- **The test set rule:** look at it once, at the very end. Every peek leaks information.
- **Distribution match:** train, val, test should be drawn from the same distribution as production. If not — that's the bug, not the model.

---

## 3. Cross-validation

| Variant | Use case |
|---|---|
| K-fold | i.i.d. tabular, modest dataset |
| Stratified K-fold | Classification with imbalance |
| Group K-fold | Multiple rows per entity (user/patient) |
| TimeSeriesSplit | Temporal data — train on past, validate on future |
| Leave-one-out (LOO) | Tiny datasets (< 100) |
| Nested CV | Unbiased estimate when tuning hyperparameters |

**Self-check:** Explain why ordinary K-fold on time series gives optimistic estimates.

---

## 4. Regression metrics

| Metric | Formula | Property |
|---|---|---|
| MSE | mean((y − ŷ)²) | Penalizes large errors heavily; scale-dependent |
| RMSE | √MSE | Same units as target |
| MAE | mean(|y − ŷ|) | Robust to outliers |
| MAPE | mean(|y − ŷ|/|y|) | Scale-free, but explodes near y=0 |
| R² | 1 − SS_res/SS_tot | Scale-free; ∈ (−∞, 1] |
| Pinball loss | quantile regression | When predicting intervals not means |

**Rule:** RMSE for "mean" prediction; MAE for robustness; quantile for "give me an interval".

---

## 5. Classification metrics

**Confusion matrix-derived:**
- Accuracy — useless on imbalanced data
- Precision = TP / (TP + FP) — "of predicted positives, how many were right"
- Recall = TP / (TP + FN) — "of actual positives, how many did we catch"
- F1 = harmonic mean of P and R
- Matthews CC — balanced, works with imbalance
- Specificity = TN / (TN + FP)

**Threshold-independent:**
- ROC-AUC — global ranking quality; can be misleading with heavy imbalance
- PR-AUC (Average Precision) — preferred when positives are rare
- Log loss / cross-entropy — penalizes confident-wrong predictions

**Multiclass extras:**
- Macro vs micro vs weighted averaging — pick deliberately, not by default

**Self-check:** Articulate why a model with 0.95 AUC can be useless in production (hint: threshold + base rate).

---

## 6. Ranking & retrieval metrics

When the model produces ordered lists (search, recsys):
- Precision@k, Recall@k
- MAP (Mean Average Precision)
- nDCG (normalized Discounted Cumulative Gain) — weights position
- MRR (Mean Reciprocal Rank) — for "first correct" tasks
- Hit@k — binary "is the answer in top k"

---

## 7. Calibration

A model is **calibrated** if `P(y=1 | model_score = 0.8) ≈ 0.8` empirically.

- **Reliability diagram** — bucketed predicted-prob vs observed-frequency.
- **Brier score** — mean((p − y)²); proper scoring rule.
- **Expected Calibration Error (ECE)** — average bucketed |p − freq|.
- **Calibrators:** Platt scaling (sigmoid fit), Isotonic regression (more flexible, needs more data).

> Tree ensembles tend to be overconfident; SVMs and NN with cross-entropy are often miscalibrated too. Always check before using `predict_proba` for decisions.

**Self-check:** Plot a reliability diagram for any classifier you've trained.

---

## 8. Statistical significance

"Model B beats A by 0.3% — is it real?"

- **Paired bootstrap** — most general; resample with replacement, recompute metric, get a confidence interval.
- **McNemar's test** — paired binary predictions.
- **5×2 CV t-test** — Dietterich's recommendation for comparing classifiers.
- **Permutation test** — randomize labels, see how often you beat the observed gap by chance.

**Rule:** Always report a confidence interval on the metric, not just a point estimate.

---

## 9. Hyperparameter tuning

| Method | When |
|---|---|
| Grid search | < 4 hyperparameters, cheap models |
| Random search | > 4 hyperparameters (Bergstra & Bengio) |
| Bayesian (Optuna, Hyperopt) | Expensive models, many trials |
| Hyperband / ASHA | Early-stopping bad trials |
| Population-Based Training | RL, deep learning at scale |

**Tuning loop:** train → val metric → search step → re-train. **Test set never enters this loop.**

**Self-check:** Spot the leakage when someone tunes on the test set "just this once".

---

## 10. Diagnostics & error analysis

After metrics, look at *which* examples the model gets wrong:
- Confusion matrix per class — which classes confuse with which.
- Slice-based metrics — accuracy by subgroup (gender, region, device). Find the underperforming slice before users do.
- Manual review of worst-N predictions — almost always reveals labeling errors, leakage, or systematic bias.
- Residual plots for regression — heteroscedasticity, missing features.
- Learning curves vs dataset size — answers "would more data help?"

**Self-check:** For your last project, can you name the 3 hardest slices for the model and why?

---

## Anti-patterns to flag

- Reporting accuracy on imbalanced data.
- Tuning on the test set.
- One number, no confidence interval.
- Using AUC when you care about a specific operating point.
- Treating `predict_proba` outputs as probabilities without calibration check.
- "Model B is better" with no significance test.
- Skipping error analysis to "save time".

---

## Recommended next move

Loop back to [02_data_preprocessing/](../02_data_preprocessing/) and [03_feature_engineering/](../03_feature_engineering/) — error analysis usually surfaces a preprocessing fix or a missing feature, not a modeling fix.
