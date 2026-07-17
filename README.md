# Machine Learning from First Principles

An educational repository for deriving, implementing, and testing machine-learning
methods from mathematical assumptions rather than treating libraries as black boxes.

> **WHY → WHAT → HOW → BUILD → VERIFY → CONNECTIONS**

## Repository Layout

| Path | Role |
|---|---|
| [`foundations/`](foundations/) | Reusable mathematics and numerical-computing tools |
| [`topics/`](topics/) | Seventeen First Principles algorithm stories |
| [`synthesis/`](synthesis/) | Cross-model comparisons and decision guides |
| [`src/ml_first_principles/`](src/ml_first_principles/) | Installable from-scratch implementations |
| [`tests/`](tests/) | Numerical and behavioral regression tests |

## Topic Sequence

| Phase | Topics |
|---|---|
| Core mathematical ML | Linear Regression, Gradient Descent, Regularization, Logistic Regression, PCA |
| Classical models | Decision Tree, Ensemble Methods, KNN, Naive Bayes, SVM, Clustering, Dimensionality Reduction |
| Deep learning | Neural Networks, CNN, RNN/LSTM, Transformer, Autoencoder |

See [`INDEX.md`](INDEX.md) for the full curriculum index with prerequisite graph,
math-to-algorithm mapping, and cross-topic navigation.

## File Roles

Each topic uses the smallest set of files needed for its reasoning story:

| File | Responsibility |
|---|---|
| `README.md` | Scope, prerequisites, maturity, and navigation |
| `theory.md` | WHY, formal WHAT, assumptions, notation, and derivations |
| `first_principles.ipynb` | HOW, BUILD, experiments, failures, and VERIFY |
| `exercises.ipynb` | Hand calculations, coding tasks, and conceptual checks |

The boundary between Markdown, notebooks, and reusable Python is defined in
[`NOTEBOOK_STANDARDS.md`](NOTEBOOK_STANDARDS.md).

## Installation

Python 3.12 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps --no-build-isolation -e .
```

For deep-learning comparisons, also install PyTorch (`pip install torch`).

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

## Verification

```bash
python -m pytest -p no:cacheprovider
ruff check src tests scripts
python scripts/normalize_notebooks.py
```

## License

Released under the [MIT License](LICENSE).
