# Machine Learning from First Principles

> **Derive, implement, and verify machine learning algorithms from mathematical principles without black-box dependencies.**

[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg?style=flat-square&logo=python)](https://www.python.org/)
[![Mathematics](https://img.shields.io/badge/math-first__principles-crimson.svg?style=flat-square)](https://en.wikipedia.org/wiki/First_principle)
[![Pytest](https://img.shields.io/badge/tests-passing-brightgreen.svg?style=flat-square&logo=pytest)](tests/)
[![License](https://img.shields.io/badge/license-MIT-green.svg?style=flat-square)](LICENSE)

---

## 🎯 Core Philosophy

Machine Learning is applied mathematics and numerical computation. This repository strictly follows a **first-principles methodology**:

```text
Phenomenon & Motivation
→ Mathematical Formulation
→ Analytical Derivation
→ From-Scratch NumPy/PyTorch Implementation
→ Numerical Verification & Behavioral Tests
→ ML/AI Connections & Trade-offs
```

Every algorithm is built step-by-step from raw matrix operations and calculus before comparing with production libraries.

---

## 📂 Repository Structure

```text
Machine-Learning-from-scratch/
├── foundations/               # Mathematical prerequisites (calculus, linalg, probability, information theory)
├── topics/                    # 17 algorithm modules (theory, implementation, exercises)
├── synthesis/                 # Cross-model comparisons & decision guides
├── src/ml_first_principles/   # Clean, installable Python library written from scratch
├── tests/                     # Unit tests & numerical regression suites
├── INDEX.md                   # Full curriculum index & prerequisite DAG
├── NOTEBOOK_STANDARDS.md      # Writing & coding standards
└── README.md
```

---

## 🗺️ Topics & Curriculum

The repository covers 17 distinct algorithmic modules organized into three main phases:

### Phase 1: Core Mathematical ML
| Topic | Module | Mathematical Core | Status |
|---|---|---|---|
| `01` | Linear Regression | OLS, Normal Equations, QR Decomposition | ✅ Complete |
| `02` | Gradient Descent | Convexity, Step Size, Momentum, Adaptive Rates | ✅ Complete |
| `03` | Regularization | L1/L2 Norms, Lasso, Ridge, ElasticNet, KKT | ✅ Complete |
| `04` | Logistic Regression | MLE, Sigmoid, Cross-Entropy, Newton-Raphson | ✅ Complete |
| `10` | PCA | SVD, Eigendecomposition, Covariance Manifolds | ✅ Complete |

### Phase 2: Classical Machine Learning
| Topic | Module | Mathematical Core | Status |
|---|---|---|---|
| `05` | Decision Trees | Gini Impurity, Information Gain, Entropy | ✅ Complete |
| `06` | Ensemble Methods | Bagging, Random Forest, AdaBoost, Gradient Boosting | ✅ Complete |
| `07` | K-Nearest Neighbors | Metric Spaces, KD-Trees, Distance Measures | ✅ Complete |
| `08` | Naive Bayes | Bayes Theorem, MAP, Gaussian/Multinomial Priors | ✅ Complete |
| `09` | Support Vector Machines | Dual Formulation, Convex Quadratic Program, Kernels | ✅ Complete |
| `11` | Clustering | K-Means, EM Algorithm, Gaussian Mixture Models | ✅ Complete |
| `12` | Dimensionality Reduction | t-SNE, UMAP, Spectral Embeddings | ✅ Complete |

### Phase 3: Deep Learning & Neural Architectures
| Topic | Module | Mathematical Core | Status |
|---|---|---|---|
| `13` | Neural Networks | Computational Graphs, Chain Rule, Backprop | ✅ Complete |
| `14` | Convolutional Networks | Cross-Correlation, Receptive Fields, Pooling | ✅ Complete |
| `15` | Recurrent Networks | RNNs, BPTT, Vanishing Gradients, LSTM, GRU | ✅ Complete |
| `16` | Transformers | Scaled Dot-Product, Multi-Head Self-Attention, Positional Encoding | ✅ Complete |
| `17` | Autoencoders | Bottleneck Representations, Variational Inference (VAE) | ✅ Complete |

---

## ⚡ Quick Start & Installation

### Prerequisites
- **Python 3.12+**
- Virtual environment (`venv` or `conda`)

### 1. Clone & Setup Environment

```bash
git clone https://github.com/hien078/Machine-Learning-from-scratch.git
cd Machine-Learning-from-scratch

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### 3. Run Verification Tests

Ensure all algorithm implementations pass the unit test suite:

```bash
pytest
```

---

## 🔬 Software Engineering & Testing

All algorithm implementations inside `src/ml_first_principles/` are paired with automated regression tests in `tests/`:

- Linear Models, Optimizers, Tree Models, Ensembles
- Distance Metrics, Probabilistic Models, Neural Core
- Gradient checks and numerical stability checks

---

## 📄 License

This repository is released under the [MIT License](LICENSE).
