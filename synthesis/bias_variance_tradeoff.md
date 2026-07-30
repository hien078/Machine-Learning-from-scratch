# Bias–Variance Trade-off

## Purpose

Explain how model flexibility, data size, noise, and regularization change expected
prediction error across model families.

## Decomposition

Assume observations satisfy $Y=f(X)+\varepsilon$, where
$\mathbb{E}[\varepsilon\mid X]=0$ and
$\operatorname{Var}(\varepsilon\mid X)=\sigma^2$. For a fitted predictor
$\hat{f}_{\mathcal{D}}(x)$ that varies with training set $\mathcal{D}$,

$$
\mathbb{E}_{\mathcal{D},\varepsilon}
\left[(Y-\hat{f}_{\mathcal{D}}(x))^2\mid X=x\right]
=\sigma^2+
\left(\mathbb{E}_{\mathcal{D}}[\hat{f}_{\mathcal{D}}(x)]-f(x)\right)^2+
\operatorname{Var}_{\mathcal{D}}(\hat{f}_{\mathcal{D}}(x)).
$$

The terms are irreducible noise, squared bias, and estimator variance. This identity is
pointwise and specific to squared error; classification needs an analogous stability and
calibration analysis rather than blindly reusing the formula.

## Cross-Model View

| Change | Typical bias effect | Typical variance effect | Important caveat |
|---|---|---|---|
| Increase polynomial degree | Down | Up | Depends on data and regularization |
| Increase KNN $k$ | Up | Down | Scaling and dimension can dominate |
| Increase tree depth | Down | Up | Pruning and minimum leaf size matter |
| Bag independent trees | Similar | Down | Correlated trees limit the reduction |
| Increase regularization | Up | Down | Excessive regularization underfits |
| Add training data | Similar | Down | Only if train and deployment distributions agree |

## Verification Questions

1. Does test error form a U-shaped curve as capacity changes?
2. Across repeated training samples, how variable are predictions at the same $x$?
3. Does a lower-variance model introduce systematic residual structure?
4. Does cross-validation choose a capacity near the held-out optimum?

## Connections

- [Linear Regression](../topics/01_linear_regression/README.md)
- [Regularization](../topics/03_regularization/README.md)
- [Decision Trees](../topics/05_decision_tree/README.md)
- [Ensemble Methods](../topics/06_ensemble_methods/README.md)
- [Model Selection Guide](model_selection_guide.md)
