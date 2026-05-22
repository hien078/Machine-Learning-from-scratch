# Data Preprocessing — Roadmap

> Raw data → clean tensors. The unglamorous 60% of any real ML project.
> Skipping this layer = "garbage in, garbage out" — no model on top can rescue it.

---

## Table of Contents

1. [Why this matters](#why-this-matters)
2. [Quick priority table](#quick-priority-table)
3. Topics in detail:
   - [1. Loading & I/O](#1-loading--io)
   - [2. Cleaning](#2-cleaning)
   - [3. Encoding categorical](#3-encoding-categorical)
   - [4. Scaling & normalization](#4-scaling--normalization)
   - [5. Imbalanced data](#5-imbalanced-data)
   - [6. Splitting strategy](#6-splitting-strategy)
   - [7. Leakage prevention](#7-leakage-prevention)
   - [8. Reproducibility](#8-reproducibility)

---

## Why this matters

- **Most "model improvements" in industry come from better data, not better architectures.** Andrew Ng's "data-centric AI" frame.
- **Leakage is the #1 silent killer.** A 99% accuracy that drops to 60% in production is almost always leakage.
- **Pre-processing pipeline = part of the model.** What you do to training data, you must do identically (and *only after fit*) on val/test.

---

## Quick priority table

| # | Topic | When you hit it | Priority |
|---|---|---|---|
| 1 | Loading & I/O | Day 1 of every project | 🔴 Must |
| 2 | Cleaning (missing/dup/outlier) | Day 1 of every project | 🔴 Must |
| 3 | Categorical encoding | Tabular data | 🔴 Must |
| 4 | Scaling & normalization | Any distance/gradient method | 🔴 Must |
| 5 | Imbalanced data | Classification on real data | 🟠 Soon |
| 6 | Splitting strategy | Every project (don't shuffle naively) | 🔴 Must |
| 7 | Leakage prevention | Production / competition | 🔴 Must |
| 8 | Reproducibility | Every project (seed everything) | 🔴 Must |

---

## 1. Loading & I/O

- File formats: CSV / Parquet / JSON / NPZ / HDF5 / TFRecord — when to pick which.
- `pandas` for tabular, `polars` when CSV > 1 GB, `pyarrow` underneath.
- Lazy loading vs in-memory; chunked reads for files larger than RAM.
- Image: `PIL`, `cv2`, `torchvision.io`. Audio: `librosa`, `torchaudio`. Text: just `open()`.

**Self-check:** Load a 5GB CSV without crashing the kernel.

---

## 2. Cleaning

- **Missing values:** drop vs impute (mean / median / mode / KNN / model-based). When *missingness itself* is a signal → add an indicator column.
- **Duplicates:** exact vs near-duplicate (`fuzzywuzzy`, MinHash for text).
- **Outliers:** IQR rule, z-score, Isolation Forest. Decide: remove vs cap vs leave.
- **Type coercion:** strings that should be numbers, dates parsed as strings, currencies with `$,`.

**Self-check:** Given a CSV with mixed types, produce a clean DataFrame in ≤ 20 lines, no surprises.

---

## 3. Encoding categorical

| Method | Use when | Risk |
|---|---|---|
| One-hot | Low cardinality (< 50) | Curse of dimensionality |
| Ordinal / label | Ordered categories (S/M/L) | Implies false order if used wrongly |
| Target / mean encoding | High cardinality + supervised | **Leakage** if not done in CV folds |
| Frequency encoding | Cardinality > 1000 | Loses information |
| Embedding (learned) | Deep learning + huge cardinality | Needs training data |
| Hashing trick | Streaming / unknown vocabulary | Collisions |

**Self-check:** Encode 10k unique zip codes without exploding memory.

---

## 4. Scaling & normalization

- **StandardScaler** (μ=0, σ=1) — default for linear models, SVM, KNN, PCA.
- **MinMaxScaler** ([0,1]) — bounded outputs, image pixels.
- **RobustScaler** (median, IQR) — when outliers exist.
- **Log / Box-Cox / Yeo-Johnson** — for skewed distributions.
- **Per-sample L2 normalization** — text TF-IDF, retrieval embeddings.

> Tree-based models (RF, XGBoost, LightGBM) do **not** need scaling. Linear models, NN, distance-based methods do.

**Self-check:** Know why scaling MNIST pixels to [0,1] vs standardizing changes training dynamics.

---

## 5. Imbalanced data

- **Resampling:** oversample minority (SMOTE / ADASYN), undersample majority, hybrid.
- **Class weights:** `class_weight='balanced'` in sklearn, `pos_weight` in `BCEWithLogitsLoss`.
- **Threshold moving:** train as-is, pick threshold on validation set (don't default to 0.5).
- **Metric choice:** stop using accuracy. Use F1 / AUPRC / matthews_corrcoef.

**Self-check:** Explain why a 99% accuracy classifier on a 99:1 dataset is useless.

---

## 6. Splitting strategy

| Strategy | When |
|---|---|
| Random split | i.i.d. tabular |
| Stratified split | Imbalanced classification |
| Group split | Multiple rows per user/patient (prevent leakage) |
| Time-based split | Any temporal data — **never shuffle time series** |
| Nested CV | Hyperparameter tuning + unbiased estimate |

**Self-check:** Articulate why `train_test_split(shuffle=True)` on time-series data invalidates your results.

---

## 7. Leakage prevention

The cardinal sin. Sources:
- **Pre-processing on the whole dataset before splitting** — `StandardScaler.fit(X)` then split = leakage.
- **Target leakage** — feature derived from the target (e.g., "days until churn" when predicting churn).
- **Temporal leakage** — using future info to predict past.
- **Group leakage** — same user/patient in train and test.

**Rule of thumb:** Wrap every transformation in `sklearn.pipeline.Pipeline` so `fit` is only called on the train split.

**Self-check:** Spot the leakage in someone else's Kaggle notebook within 5 minutes.

---

## 8. Reproducibility

- Seed: `random`, `numpy`, `torch` (+ CUDA), `tensorflow` if used.
- `PYTHONHASHSEED` for ordering of `set` / `dict` iteration.
- `torch.use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG=:4096:8` for full determinism on GPU.
- Dataset hash: SHA256 of the raw file; log it.
- Pin every library in `requirements.txt`.

**Self-check:** Two runs on the same machine should produce bit-identical loss curves.

---

## Anti-patterns to flag

- "I'll handle missing values later" → you won't.
- One-hot encoding 50k+ unique categories.
- `StandardScaler.fit_transform(X)` then `train_test_split` — leakage.
- "Just SMOTE it" without thinking about the validation strategy.
- `accuracy_score` on imbalanced classification.
- Re-shuffling time-series.

---

## Recommended next module

→ [03_feature_engineering/](../03_feature_engineering/) — once data is clean, ask: *what features should I build from it?*
