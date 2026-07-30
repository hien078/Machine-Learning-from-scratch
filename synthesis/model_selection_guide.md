# Model Selection Guide

## Start with the Evaluation Design

Choose the split strategy and metric before comparing models. Model selection performed
on the test set leaks information; the test set is used once after the complete pipeline
has been selected.

| Data situation | Validation design |
|---|---|
| Independent, identically distributed observations | Random train/validation/test split or shuffled CV |
| Time-ordered observations | Forward or rolling validation; never shuffle the future into training |
| Multiple rows per person/device/group | Group-aware split |
| Rare classes | Stratified split plus class-appropriate metrics |
| Small data and tuning | Nested CV when an unbiased comparison matters |

## Practical Decision Table

| Requirement | Useful starting point | Escalate when |
|---|---|---|
| Interpretable continuous prediction | Linear/Ridge Regression | Residual structure remains nonlinear |
| Calibrated linear classification baseline | Logistic Regression | Boundaries remain nonlinear |
| Mixed nonlinear tabular interactions | Decision Tree or Random Forest | Boosting is justified by validation |
| Local similarity is meaningful | KNN | Dimension or inference cost becomes excessive |
| Very small data with plausible independence | Naive Bayes | Calibration or feature dependence dominates |
| High-dimensional margin | Linear SVM | A justified kernel or representation is available |
| Unlabeled grouping | K-Means baseline | Shape/noise assumptions fail; compare DBSCAN/GMM |

## Required Comparison Record

For every selected model record the baseline, split, preprocessing fitted on training
data only, hyperparameter search space, primary and secondary metrics, uncertainty across
folds/seeds, runtime, error slices, and final limitations.

## Connections

- [Bias–Variance](bias_variance_tradeoff.md)
- [Loss Functions](loss_functions_map.md)
- [Supervised vs Unsupervised](supervised_vs_unsupervised.md)
