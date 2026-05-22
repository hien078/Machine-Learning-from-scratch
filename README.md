# Machine Learning Learning Workspace

A structured, hands-on educational repository designed to master Machine Learning, Deep Learning, and Generative AI from first principles (implementing algorithms from scratch using NumPy) to advanced library applications (PyTorch, Hugging Face).

---

## 📂 Project Structure

Each algorithm folder follows the same 5-notebook pedagogical pattern:
`01_intuition.ipynb` → `02_mathematics.ipynb` → `03_optimization.ipynb` → `04_statistics.ipynb` → `05_hands_on_programming.ipynb`.

* **[`00_foundations/`](00_foundations/)** — Prerequisites that every later module assumes.
    * [`01_math_essentials/`](00_foundations/01_math_essentials/) — math roadmap for AI/ML.
    * [`02_data_preprocessing/`](00_foundations/02_data_preprocessing/) — loading, cleaning, encoding, splitting, leakage prevention.
    * [`03_feature_engineering/`](00_foundations/03_feature_engineering/) — numeric / categorical / time / text features.
    * [`04_model_evaluation/`](00_foundations/04_model_evaluation/) — metrics, CV, calibration, significance testing.
* **[`01_supervised_learning/`](01_supervised_learning/)** — Learning **f: X → Y** from labeled data.
    * `01_linear_regression` → `02_polynomial_regression` → `03_ridge_regression` → `04_lasso_regression`
    * `05_logistic_regression` → `06_decision_tree` → `07_random_forest` → `08_gradient_boosting`
    * `09_k_nearest_neighbors` → `10_naive_bayes` → `11_support_vector_machine`
* **[`02_unsupervised_learning/`](02_unsupervised_learning/)** — Finding structure in unlabeled data.
    * `01_pca` → `02_lda` → `03_t_sne` (dimensionality reduction)
    * `04_k_means_clustering` → `05_hierarchical_clustering` → `06_dbscan` → `07_gaussian_mixture_model` (clustering)
    * `08_apriori` → `09_fp_growth` (association rules)
* **[`03_deep_learning/`](03_deep_learning/)** — Neural networks and advanced architectures.
    * `01_multi_layer_perceptron` → `02_cnn` → `03_rnn_lstm_gru` → `04_transformer` → `05_autoencoder`
* **[`04_generative_ai/`](04_generative_ai/)** — Models that learn p(x) and generate new samples.
    * `01_vae` → `02_gan` → `03_diffusion_models` → `04_large_language_models`
* **[`05_reinforcement_learning/`](05_reinforcement_learning/)** — Learning through interaction and reward.
    * `00_markov_decision_process` (framework)
    * `01_q_learning` → `02_dqn` → `03_double_dqn` (value-based)
    * `04_policy_gradient` → `05_a2c` → `06_ddpg` → `07_ppo` → `08_sac` (policy-based / actor-critic)
* **[`99_capstone/`](99_capstone/)** — End-to-end projects combining multiple modules.
* **[`applications/`](applications/)** — Project-style code (not concept learning): agent SDKs, integration examples.
    * `mini_agent_sdk/` — lightweight agent SDK powered by Google Gemini.
    * `openai_agent/` — OpenAI Agents SDK examples.

> **Cross-reference matrix:** [INDEX.md](INDEX.md) maps every algorithm to which math pillar (linear algebra / calculus / probability / optimization / information theory) it exercises.

---

## 🗺️ Learning Roadmap & Guidelines

Before diving in, make sure to read:
*   [00_foundations/01_math_essentials/math_for_ai_roadmap.md](00_foundations/01_math_essentials/math_for_ai_roadmap.md) — practical math curriculum for modern AI/ML (Vietnamese).
*   [ML.md](ML.md) — the 20+ foundational ML models, organized by paradigm (Vietnamese).
*   [CLAUDE.md](CLAUDE.md) — strict guidelines on implementation, reproducibility (seeding), simplicity, and code cleanliness.

---

## 🛠️ Environment Setup

This workspace uses a Python virtual environment (`.venv`). Follow these steps to set it up and install the required dependencies:

### 1. Activate the Virtual Environment

**On Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

**On Windows (Command Prompt - cmd):**
```cmd
.venv\Scripts\activate.bat
```

**On macOS/Linux:**
```bash
source .venv/bin/activate
```

### 2. Install Dependencies
Once activated, install the required packages:
```bash
pip install -r requirements.txt
```

---

## 🚀 Getting Started

To begin learning:
1. Skim [ML.md](ML.md) to see the 20+ algorithms at a glance.
2. Open [INDEX.md](INDEX.md) to find a specific algorithm or math pillar.
3. Inside any algorithm folder, walk through the 5 notebooks in order: intuition → mathematics → optimization → statistics → hands-on.
4. Set your random seeds for reproducibility (as detailed in [CLAUDE.md](CLAUDE.md)).
