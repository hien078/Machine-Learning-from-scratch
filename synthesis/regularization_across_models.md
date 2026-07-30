# Regularization Across Models

Regularization controls effective model capacity; it is broader than adding a penalty to
an objective.

| Mechanism | Primary effect | Models | Failure mode |
|---|---|---|---|
| $\ell_2$ penalty | Shrinks parameters smoothly | Linear models, SVM, neural networks | Can retain many weak irrelevant features |
| $\ell_1$ penalty | Encourages exact zeros | Linear and generalized linear models | Unstable selection among correlated features |
| Elastic Net | Combines shrinkage and sparsity | Linear models | Adds another tuning dimension |
| Tree depth/leaf constraints | Limits partition complexity | Trees and forests | Excessive limits create high bias |
| Early stopping | Restricts optimization trajectory | Boosting, neural networks | Requires a leakage-free validation rule |
| Data augmentation | Encodes invariances through examples | CNNs and other neural models | Invalid transformations inject label noise |
| Dropout | Stochastic activation masking | Neural networks | Not equivalent to simple $\ell_2$ in general |

## Penalty and Constraint Views

For suitable regularity conditions, a penalized problem

$$
\min_\theta L(\theta)+\lambda R(\theta)
$$

corresponds to a constrained problem $\min_\theta L(\theta)$ subject to
$R(\theta)\le t$. The mapping between $\lambda$ and $t$ need not be one-to-one in
degenerate or inactive cases.

## Validation Rule

Regularization strength is selected inside the training/validation procedure. Feature
scaling, imputation, and selection must be fitted separately inside every fold; fitting
them once on all data leaks validation information.

## Connections

- [Regularization Topic](../topics/03_regularization/README.md)
- [Bias–Variance](bias_variance_tradeoff.md)
- [Geometry of ML](geometry_of_ml.md)
- [Model Selection](model_selection_guide.md)
