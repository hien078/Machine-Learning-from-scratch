# Capstone Projects

End-to-end projects that combine multiple modules. The purpose is integration: data pipeline + modeling + evaluation + (optionally) deployment.

---

## Why a capstone module

After grinding through 20+ algorithms in isolation, the gap to "I can build a real ML system" is still large. A capstone forces:

- Choosing the right model for an actual problem (not the one you just learned).
- A realistic data pipeline — messy CSV → features → model input.
- A real evaluation strategy — not just `score = model.score(X_test, y_test)`.
- Reporting and reproducibility — someone else can run your code.

---

## Suggested capstone ideas

Pick ideas that **combine ≥ 2 paradigms** so the integration is real:

| Project | Combines | Why it's a good capstone |
|---|---|---|
| **House-price prediction** | Preprocessing + feature engineering + gradient boosting + calibration | Classic tabular pipeline, end-to-end |
| **Image classifier with explainability** | CNN + SHAP / Grad-CAM | Shows you understand *what* the model learned |
| **Recommender from scratch** | PCA / matrix factorization + ranking metrics | Bridges unsupervised + evaluation |
| **Text classifier with LLM baseline** | TF-IDF + LogReg vs zero-shot LLM | Honest comparison: do you need the big model? |
| **RL agent for a custom env** | MDP design + DQN / PPO | Forces you to think about reward shaping |
| **Generate-then-classify pipeline** | Diffusion / GAN for augmentation + downstream classifier | Probes whether generative ≠ useful |

---

## Capstone layout (per project)

```
99_capstone/
└── project_name/
    ├── README.md         # Problem statement, dataset, results
    ├── data/             # Raw data (gitignored if > 100MB)
    ├── notebooks/        # EDA, exploratory modeling
    ├── src/              # Reusable code (preprocessing.py, model.py, train.py)
    ├── tests/            # At minimum: data integrity + smoke test for train
    ├── reports/          # Plots, metrics, comparison tables
    └── requirements.txt  # Pinned dependencies for this project
```

Project README should answer:
1. **Problem** — what are you predicting / generating?
2. **Data** — source, size, license, splits.
3. **Approach** — which modules from `01_*` through `05_*` you used and why.
4. **Results** — metric on held-out test, with a confidence interval.
5. **What you'd do next** — honest list of known limitations.
