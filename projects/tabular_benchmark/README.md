# Tabular benchmark: from-scratch vs scikit-learn

Capstone project: an end-to-end benchmark of this repo's `ml_first_principles`
from-scratch models against their scikit-learn equivalents on two sklearn
built-in datasets (nothing is downloaded):

- **Regression** — `load_diabetes`: LinearRegression, RidgeRegression,
  LassoRegression (metrics: $R^2$, RMSE).
- **Classification** — `load_breast_cancer`: LogisticRegression,
  DecisionTreeClassifier, RandomForestClassifier, KNeighborsClassifier,
  GaussianNB, LinearSVC (metrics: accuracy, F1).

Both sides get the identical 80/20 split (seed 42) with features standardized
using train-fold statistics, and both are scored with the same
`ml_first_principles.metrics` functions. Fit and predict wall times are
recorded per model.

## How to run

From the **repo root**:

```bash
source .venv/bin/activate
python projects/tabular_benchmark/src/tb_benchmark.py
```

Expected output: a console summary of the headline metric per model pair, and
the full report written to [`reports/benchmark.md`](reports/benchmark.md)
(deterministic under the fixed seed — the file is committed).

Tests:

```bash
pytest projects/tabular_benchmark -q
```

## Findings summary

- Closed-form / same-objective linear models (linear, ridge, lasso) match
  sklearn to 4 decimal places.
- Scratch logistic regression *beats* sklearn's unregularized lbfgs on this
  nearly separable dataset — but only because 1000 plain GD steps act as
  accidental early stopping; the other classifiers land within 0-2 test
  samples of sklearn (solver/RNG differences, same algorithms).
- Accuracy parity does not imply speed parity: the pure-Python tree grower
  makes scratch trees/forests roughly 40-80x slower than sklearn's Cython
  trees, while closed-form linear models and vectorized KNN keep pace.

See the **Findings** section of [`reports/benchmark.md`](reports/benchmark.md)
for the full discussion.

## Layout

| Path | Contents |
|---|---|
| `src/tb_benchmark.py` | The whole benchmark: data loading, model pairs, evaluation, report writer, `main()`. |
| `tests/` | Fast unit tests on small pieces (single pairs, report formatting, determinism). |
| `reports/benchmark.md` | Generated benchmark report (committed). |
| `data/` | Empty by design — datasets are sklearn built-ins. |
| `notebooks/` | Placeholder for future exploration notebooks. |
