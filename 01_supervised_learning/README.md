# Supervised Learning

Algorithms that learn a mapping **f: X → Y** from labeled training data {(xᵢ, yᵢ)}. The goal is to generalize — predict correctly on unseen inputs.

---

## Prerequisites

### Math (from [math_for_ai_roadmap.md](../00_foundations/01_math_essentials/math_for_ai_roadmap.md))
- **(1) Linear Algebra:** matrix multiplication, transpose, inverse, norms (L1/L2), eigenvalues.
- **(2) Calculus:** partial derivatives, gradient, chain rule, Taylor expansion.
- **(3) Probability & Statistics:** Bayes' theorem, MLE, MAP, Gaussian distribution, bias-variance tradeoff.
- **(4) Optimization:** gradient descent, SGD, convexity, Lagrange multipliers (for SVM).

### Code
- Python, NumPy, Matplotlib.
- scikit-learn (for comparison after implementing from scratch).

---

## Subprojects (ordered by learning priority)

| # | Folder | Algorithm | Core idea | Key math |
|---|--------|-----------|-----------|----------|
| 1 | [`01_linear_regression/`](01_linear_regression/) | Linear Regression | Minimize ‖y − Xw‖² | Normal equation, gradient descent |
| 2 | [`02_polynomial_regression/`](02_polynomial_regression/) | Polynomial Regression | Feature expansion x → [x, x², …, xᵈ] | Bias-variance tradeoff |
| 3 | [`03_ridge_regression/`](03_ridge_regression/) | Ridge (L2) | + λ‖w‖² penalty | L2 regularization, closed-form |
| 4 | [`04_lasso_regression/`](04_lasso_regression/) | Lasso (L1) | + λ‖w‖₁ penalty | L1 regularization, sparsity |
| 5 | [`05_logistic_regression/`](05_logistic_regression/) | Logistic Regression | σ(wᵀx) → probability | Cross-entropy loss, sigmoid |
| 6 | [`06_decision_tree/`](06_decision_tree/) | Decision Tree | Recursive splits maximizing info gain | Entropy, Gini impurity |
| 7 | [`07_random_forest/`](07_random_forest/) | Random Forest | Bagging + feature subsampling | Variance reduction |
| 8 | [`08_gradient_boosting/`](08_gradient_boosting/) | Gradient Boosting | Sequential residual fitting | Functional gradient descent, 2nd-order Taylor |
| 9 | [`09_k_nearest_neighbors/`](09_k_nearest_neighbors/) | KNN | Majority vote of k nearest points | Distance metrics, no training |
| 10 | [`10_naive_bayes/`](10_naive_bayes/) | Naive Bayes | P(y\|x) ∝ P(x\|y)P(y) with feature independence | Bayes' theorem, MLE |
| 11 | [`11_support_vector_machine/`](11_support_vector_machine/) | SVM | Maximum margin hyperplane | Hinge loss, KKT, kernel trick |

---

## Learning Objectives

After completing this module, you should be able to:

- [ ] Implement linear regression from scratch (normal equation + gradient descent) in NumPy.
- [ ] Derive the gradient of cross-entropy loss through softmax.
- [ ] Explain bias-variance tradeoff with a concrete example.
- [ ] Implement SVM dual form for a linearly separable 2D dataset.
- [ ] Train a Random Forest and explain why it reduces variance but not bias.
- [ ] Compare your from-scratch implementation against scikit-learn on the same dataset.

---

## Key References

- Andrew Ng — CS229 Lecture Notes (Stanford, free).
- Bishop — *Pattern Recognition and Machine Learning*, Chapters 3-4, 7, 14.
- Hastie, Tibshirani, Friedman — *The Elements of Statistical Learning* (free PDF), Chapters 3-4, 9-10.
- scikit-learn documentation — [scikit-learn.org](https://scikit-learn.org/).

---

## Subproject Layout

Each subproject should follow:
```
algorithm_name/
├── data/           # Datasets (gitignored if large)
├── notebooks/      # EDA, training visualization
├── src/            # From-scratch implementation
├── tests/          # Unit tests
└── reports/        # Findings, plots, comparisons
```
