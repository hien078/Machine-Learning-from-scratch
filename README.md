# Machine Learning from First Principles

> **Derive, implement, and verify machine learning algorithms from mathematical principles without black-box dependencies.**

[![Python Version](https://img.shields.io/badge/python-3.12+-blue.svg?style=flat-square&logo=python)](https://www.python.org/)
[![Mathematics](https://img.shields.io/badge/math-first__principles-crimson.svg?style=flat-square)](https://en.wikipedia.org/wiki/First_principle)
[![CI](https://github.com/hien078/Machine-Learning-from-scratch/actions/workflows/ci.yml/badge.svg)](https://github.com/hien078/Machine-Learning-from-scratch/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT-green.svg?style=flat-square)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-online-blue.svg?style=flat-square&logo=materialformkdocs)](https://hien078.github.io/Machine-Learning-from-scratch/)

> 📖 **Browse this curriculum as a website:** [hien078.github.io/Machine-Learning-from-scratch](https://hien078.github.io/Machine-Learning-from-scratch/) — rendered theory, executed notebooks, and cross-topic maps.

---

> 🧮 **Sister Repository:** For standalone, deep-dive mathematical prerequisites (Linear Algebra, Calculus & Optimization, Probability & Statistics, Information Theory, Numerical Computing), check out [applied-mathematics-foundation](https://github.com/hien078/applied-mathematics-foundation).

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
├── topics/                    # 22 algorithm modules (theory, implementation, exercises)
├── synthesis/                 # Cross-model comparisons & decision guides
├── src/ml_first_principles/   # Clean, installable Python library written from scratch
├── tests/                     # Unit tests & numerical regression suites
├── scripts/                   # Notebook validation & normalization tooling
├── INDEX.md                   # Full curriculum index & prerequisite DAG
├── ROADMAP.md                 # Ecosystem phases, milestones & decisions log
├── NOTEBOOK_STANDARDS.md      # Writing & coding standards
├── pyproject.toml             # Package metadata, dev extras, lint & test config
└── README.md
```

> 📐 **Mathematical Prerequisites** (Linear Algebra, Calculus, Probability, Information Theory, etc.) are maintained in the dedicated **[applied-mathematics-foundation](https://github.com/hien078/applied-mathematics-foundation)** repository.

---

## 🗺️ Topics & Curriculum

The repository covers 22 distinct algorithmic modules organized into five phases:

### Phase 1: Core Mathematical ML
| Topic | Module | Mathematical Core | Status |
|---|---|---|---|
| `01` | [Linear Regression](topics/01_linear_regression/README.md) | OLS, Normal Equations, Projection Geometry | ✅ Complete |
| `02` | [Gradient Descent](topics/02_gradient_descent/README.md) | Convexity, Step Size, Momentum, Adaptive Rates | ✅ Complete |
| `03` | [Regularization](topics/03_regularization/README.md) | L1/L2 Norms, Lasso, Ridge, ElasticNet, KKT | ✅ Complete |
| `04` | [Logistic Regression](topics/04_logistic_regression/README.md) | MLE, Sigmoid, Cross-Entropy, Newton-Raphson | ✅ Complete |
| `10` | [PCA](topics/10_pca/README.md) | SVD, Eigendecomposition, Covariance Manifolds | ✅ Complete |

### Phase 2: Classical Machine Learning
| Topic | Module | Mathematical Core | Status |
|---|---|---|---|
| `05` | [Decision Trees](topics/05_decision_tree/README.md) | Gini Impurity, Information Gain, Entropy | ✅ Complete |
| `06` | [Ensemble Methods](topics/06_ensemble_methods/README.md) | Bagging, Random Forest, AdaBoost, Gradient Boosting | ✅ Complete |
| `07` | [K-Nearest Neighbors](topics/07_knn/README.md) | Metric Spaces, KD-Trees, Distance Measures | ✅ Complete |
| `08` | [Naive Bayes](topics/08_naive_bayes/README.md) | Bayes Theorem, MAP, Gaussian/Multinomial Priors | ✅ Complete |
| `09` | [Support Vector Machines](topics/09_svm/README.md) | Dual Formulation, Convex Quadratic Program, Kernels | ✅ Complete |
| `11` | [Clustering](topics/11_clustering/README.md) | K-Means, EM Algorithm, Gaussian Mixture Models | ✅ Complete |
| `12` | [Dimensionality Reduction](topics/12_dimensionality_reduction/README.md) | LDA, t-SNE, KL-Divergence Embeddings | ✅ Complete |

### Phase 3: Deep Learning & Neural Architectures
| Topic | Module | Mathematical Core | Status |
|---|---|---|---|
| `13` | [Neural Networks](topics/13_neural_networks/README.md) | Computational Graphs, Chain Rule, Backprop | ✅ Complete |
| `14` | [Convolutional Networks](topics/14_cnn/README.md) | Cross-Correlation, Receptive Fields, Pooling | ✅ Complete |
| `15` | [Recurrent Networks](topics/15_rnn_lstm/README.md) | RNNs, BPTT, Vanishing Gradients, LSTM, GRU | ✅ Complete |
| `17` | [Autoencoders](topics/17_autoencoder/README.md) | Bottleneck Representations, Reconstruction Loss | ✅ Complete |

### Phase 4: Transformers
| Topic | Module | Mathematical Core | Status |
|---|---|---|---|
| `16` | [Transformers](topics/16_transformer/README.md) | Scaled Dot-Product, Multi-Head Self-Attention, Positional Encoding | ✅ Complete |

### Phase 5: Modern AI & Advanced Architectures
| Topic | Module | Mathematical Core | Status |
|---|---|---|---|
| `18` | [Reinforcement Learning](topics/18_reinforcement_learning/README.md) | MDPs, Bellman Equations, Q-Learning, DQN, Policy Gradients | ✅ Complete |
| `19` | [Generative Models](topics/19_generative_models/README.md) | VAE (ELBO), GAN (Minimax/Wasserstein), Diffusion (DDPM) | ✅ Complete |
| `20` | [Graph Neural Networks](topics/20_graph_neural_networks/README.md) | Spectral Graph Convolutions, Graph Laplacian, Message Passing (GCN, GAT) | ✅ Complete |
| `21` | [LLM Engineering](topics/21_llm_engineering/README.md) | BPE Tokenizer, LoRA Rank Factorization, DPO Alignment | ✅ Complete |
| `22` | [Self-Supervised Learning](topics/22_self_supervised_learning/README.md) | InfoNCE Contrastive Loss, SimCLR, Masked Autoencoders | ✅ Complete |

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
pip install -e ".[dev]"   # library + pytest test tooling
```

> **Optional — PyTorch:** the library-comparison sections of topics 13–17 additionally
> import `torch`. Install it separately (`pip install torch`) to execute those notebooks;
> everything else runs on the pinned dependencies above.

### 3. Run Verification Tests

Ensure all algorithm implementations pass the unit test suite:

```bash
pytest
```

---

## 🔬 Software Engineering & Testing

All algorithm implementations inside `src/ml_first_principles/` are paired with automated regression tests in `tests/`:

- Linear Models, Optimizers, Tree Models, Ensembles
- Distance Metrics, Probabilistic Models, Neural Core, Visualization
- Phase 5 modules: RL (GridWorld, Q-Learning), Generative (VAE/GAN), GNN (GCN/GAT), LLM (BPE/LoRA/DPO), SSL (InfoNCE/MAE)
- Gradient checks, numerical stability checks, and package-export consistency

Notebook health is validated separately: `python scripts/execute_all_notebooks.py` executes
every notebook top-to-bottom on a fresh kernel (add `--write` to refresh the committed
outputs — the only sanctioned way to produce them), and
`python scripts/normalize_notebooks.py` verifies canonical format and metadata.

---

## 📄 License

This repository is released under the [MIT License](LICENSE).
