# Naive Bayes

> **Phase:** 2 | **Status:** ✅ Complete | **Prerequisites:** Probability & Statistics, Information Theory

## Overview

Generative classification via Bayes' theorem and the naive conditional independence
assumption. Covers Gaussian, Multinomial, and Bernoulli variants, MAP parameter
estimation in closed form, log-space computation for numerical stability, Laplace
smoothing for zero-frequency problems, and the generative–discriminative connection
to logistic regression.

## Contents

| # | File | Type | Description |
|--:|---|---|---|
| 1 | `theory.md` | Theory | Bayes' theorem, naive independence, Gaussian/Multinomial/Bernoulli NB, log-space computation, Laplace smoothing, failure cases |
| 2 | `first_principles.ipynb` | Computation | WHY→BUILD→VERIFY — from-scratch GaussianNB, Iris dataset, sklearn comparison, parameter matching, correlated features experiment |
| 3 | `exercises.ipynb` | Practice | Hand posterior calculation, Laplace-smoothed MultinomialNB coding task, conceptual question on independence assumption |

## Connections

- **Prereqs:** [Probability & Statistics](https://github.com/hien078/applied-mathematics-foundation), [Information Theory](https://github.com/hien078/applied-mathematics-foundation)
- **Generative–Discriminative pair:** [04 Logistic Regression](../04_logistic_regression/README.md) (discriminative counterpart)
- **Synthesis:** [Probabilistic View of ML](../synthesis/probabilistic_view_of_ml.md)
- **Next:** [09 SVM](../09_svm/README.md) (geometric separation), [11 Clustering](../11_clustering/README.md) (unsupervised Gaussian mixtures)
