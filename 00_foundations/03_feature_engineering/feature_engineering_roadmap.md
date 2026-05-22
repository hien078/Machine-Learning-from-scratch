# Feature Engineering — Roadmap

> Clean data → informative features. In tabular ML this is where most of the lift comes from.
> In deep learning this layer shrinks (the network learns features) but never disappears.

---

## Table of Contents

1. [Why this matters](#why-this-matters)
2. [Quick priority table](#quick-priority-table)
3. Topics in detail:
   - [1. Numeric transformations](#1-numeric-transformations)
   - [2. Categorical interactions](#2-categorical-interactions)
   - [3. Date & time features](#3-date--time-features)
   - [4. Text features](#4-text-features)
   - [5. Image features (classical)](#5-image-features-classical)
   - [6. Aggregation & lag features](#6-aggregation--lag-features)
   - [7. Feature selection](#7-feature-selection)
   - [8. Learned vs hand-crafted](#8-learned-vs-hand-crafted)

---

## Why this matters

- **Kaggle Grandmasters spend ~80% of their time here.** Models tend to converge across teams; features differentiate.
- **Domain knowledge encodes as features.** "Days since last purchase" beats raw timestamps for churn.
- **The right feature can collapse the problem.** A polynomial regression with `x²` added is just linear regression — feature engineering ≈ chosen inductive bias.

---

## Quick priority table

| # | Topic | Where it shines | Priority |
|---|---|---|---|
| 1 | Numeric transformations (log, binning, ratios) | Tabular | 🔴 Must |
| 2 | Categorical interactions | Tabular | 🟠 Soon |
| 3 | Date/time decomposition | Anything with timestamps | 🔴 Must |
| 4 | Text features (TF-IDF, embeddings) | NLP without DL | 🟠 Soon |
| 5 | Classical image features | Vision without DL | 🟡 Niche |
| 6 | Aggregation / lag features | Time series, sessions | 🔴 Must |
| 7 | Feature selection | When p > n or sparse | 🟠 Soon |
| 8 | Learned vs hand-crafted | DL projects | 🔴 Must understand the tradeoff |

---

## 1. Numeric transformations

- **Log / sqrt** for right-skewed distributions (income, file size, view count).
- **Binning / discretization** — turn continuous into ordinal categories (age groups). Loses info but adds robustness to outliers.
- **Polynomial features** — `x → [x, x², x³]`. Cheap nonlinearity for linear models.
- **Ratios & differences** — `price / area`, `current_balance − previous_balance`. Often the single most predictive feature.
- **Clipping / winsorizing** — cap at percentiles to tame outliers without dropping rows.

**Self-check:** Given a target with skew = 5, choose a transformation that brings residuals to ~normal.

---

## 2. Categorical interactions

- **Cross features** — `city × product_category` → new high-cardinality feature.
- **Frequency / count encoding** — replace category with its count in train set.
- **Target encoding with smoothing** — mean target per category, blended toward global mean for rare categories. **Always inside CV folds** to avoid leakage.

**Self-check:** Implement target encoding with leave-one-out + smoothing in <30 lines.

---

## 3. Date & time features

A timestamp is *never* a feature on its own. Decompose:
- Year / month / day / hour / minute / second
- Day-of-week, day-of-year, week-of-year
- is_weekend, is_holiday (use `holidays` package)
- Cyclic encoding for hour (sin / cos of `2π · hour / 24`) — preserves "23h and 0h are close"
- Time since reference event (days since signup, hours since last login)

**Self-check:** Predict store traffic; the most predictive features should be cyclic hour + day-of-week + is_holiday.

---

## 4. Text features

| Method | Output | When |
|---|---|---|
| Length / word count | Scalar | Cheap baseline |
| Bag-of-Words | Sparse matrix | Small corpus |
| TF-IDF | Sparse matrix | Classical NLP baseline |
| Char n-grams | Sparse matrix | Robust to typos, multilingual |
| Word embeddings (Word2Vec, GloVe) | Dense vector | Pre-LLM era, still fast |
| Sentence embeddings (SBERT, OpenAI) | Dense vector | Modern default |

**Self-check:** Build a sentiment classifier with TF-IDF + LogReg in 50 lines, hit > 80% on IMDB.

---

## 5. Image features (classical)

Mostly historical now (CNNs ate this), but useful for explainability or low-data regimes:
- Color histograms (RGB, HSV)
- HOG (Histogram of Oriented Gradients)
- SIFT / SURF / ORB keypoints
- Texture: GLCM, LBP

**Self-check:** Classify cats vs dogs with HOG + SVM and beat random chance — see why CNNs replaced this.

---

## 6. Aggregation & lag features

For grouped / sequential data:
- **Group statistics** — mean, std, min, max, percentiles per user/session.
- **Lag features** — value at t-1, t-7, t-30 (price yesterday, last week, last month).
- **Rolling windows** — moving average, rolling std, exponential moving average.
- **Cumulative** — running sum, time-since-event.
- **Counts** — number of past purchases / errors / clicks.

**Self-check:** Predict next-day energy consumption; the lag-7 (same day last week) feature should be top-3.

---

## 7. Feature selection

When *p* (features) is comparable to *n* (samples), or features are noisy:

| Method | Idea |
|---|---|
| Filter: variance threshold | Drop near-constant features |
| Filter: correlation with target | Pearson / Spearman / mutual info |
| Filter: pairwise correlation | Drop redundant features (|ρ| > 0.95) |
| Wrapper: forward / backward | Iterative subset search (expensive) |
| Embedded: L1 (Lasso) | Sparsity-inducing regularizer |
| Embedded: tree importance | XGBoost/RF feature_importances_ |
| SHAP / permutation importance | Model-agnostic, post-hoc |

**Self-check:** Reduce 1000 features to 50 without losing > 1% AUC on a held-out set.

---

## 8. Learned vs hand-crafted

The deep learning bargain: trade hand-crafted features for compute + data.

| Regime | Win |
|---|---|
| n < 10k, tabular | Hand-crafted + tree boosting (XGBoost still SOTA on Kaggle) |
| n > 1M, images/text/audio | Learned (CNN, Transformer) |
| Hybrid: tabular + text + images | Hand-craft numeric, embed text/images, concat |

**Even with DL, you still hand-craft:**
- Time features (calendar info)
- Group-level aggregates (user history)
- Domain-specific signals (medical lab ratios, financial indicators)

**Self-check:** Articulate why "let the neural net figure it out" still loses to lag features + LightGBM on most tabular contests.

---

## Anti-patterns to flag

- Throwing 5000 polynomial features at a linear model "to be safe".
- Target encoding without CV → leakage.
- One-hot encoding raw datetime strings.
- Building features on the full dataset before splitting.
- "Tree importance says feature X matters" — without permutation importance / SHAP for confirmation, can be misleading.

---

## Recommended next module

→ [04_model_evaluation/](../04_model_evaluation/) — features are ready, models are trained, but *how do you know it's actually good?*
