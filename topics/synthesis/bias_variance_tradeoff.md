# Bias–Variance Trade-off — Cross-Topic Synthesis

> How model flexibility, data size, noise, and regularization change expected
> prediction error across model families.
> See [INDEX.md](../../INDEX.md) for the full curriculum index.

---

## Overview

Every model family in this curriculum makes the same bargain in a different currency:
a rigid model is systematically wrong (bias), a flexible model is unstable across
training sets (variance). The decomposition below makes the bargain exact for squared
error; the per-family sections show which knob each model uses to move along the curve.

---

## Decomposition

Assume observations satisfy $Y=f(X)+\varepsilon$, where
$\mathbb{E}[\varepsilon\mid X]=0$ and
$\operatorname{Var}(\varepsilon\mid X)=\sigma^2$, with $\varepsilon$ independent of the
training set $\mathcal{D}$. Write $\hat{f}_{\mathcal{D}}(x)$ for the fitted predictor and
$\bar{f}(x)=\mathbb{E}_{\mathcal{D}}[\hat{f}_{\mathcal{D}}(x)]$ for its average over
training sets.

**Step 1 — split off the noise.** At a fixed test point $x$,

```math
\mathbb{E}\left[(Y-\hat{f}_{\mathcal{D}}(x))^2\right]
=\mathbb{E}\left[(f(x)+\varepsilon-\hat{f}_{\mathcal{D}}(x))^2\right]
=\sigma^2+\mathbb{E}_{\mathcal{D}}\left[(f(x)-\hat{f}_{\mathcal{D}}(x))^2\right],
```

because the cross-term
$2\,\mathbb{E}\left[\varepsilon\,(f(x)-\hat{f}_{\mathcal{D}}(x))\right]=0$: the fresh
noise $\varepsilon$ is independent of $\hat{f}_{\mathcal{D}}$ and has mean zero.

**Step 2 — split bias from variance.** Add and subtract $\bar{f}(x)$:

```math
\mathbb{E}_{\mathcal{D}}\left[(f(x)-\hat{f}_{\mathcal{D}}(x))^2\right]
=\underbrace{(f(x)-\bar{f}(x))^2}_{\text{bias}^2}
+\underbrace{\mathbb{E}_{\mathcal{D}}\left[(\bar{f}(x)-\hat{f}_{\mathcal{D}}(x))^2\right]}_{\text{variance}}
+\;2\,(f(x)-\bar{f}(x))\,\mathbb{E}_{\mathcal{D}}\left[\bar{f}(x)-\hat{f}_{\mathcal{D}}(x)\right].
```

The cross-term vanishes because
$\mathbb{E}_{\mathcal{D}}\left[\bar{f}(x)-\hat{f}_{\mathcal{D}}(x)\right]
=\bar{f}(x)-\bar{f}(x)=0$ — the first factor $(f(x)-\bar{f}(x))$ is a constant with
respect to $\mathcal{D}$, so it slides outside the expectation. **Result:**

```math
\text{Expected error at }x=\sigma^2+\text{Bias}^2(x)+\text{Variance}(x).
```

This identity is pointwise and specific to squared error. For 0–1 classification loss
there is no clean additive decomposition; the analogue is a stability and calibration
analysis, not a blind reuse of the formula.

---

## How Each Model Family Trades Bias for Variance

| Model family | Capacity knob | More capacity → | Notes |
|---|---|---|---|
| Linear / polynomial regression | Degree, feature count | Bias ↓, variance ↑ | Variance grows with collinearity |
| Ridge / Lasso | $\lambda$ (smaller = more capacity) | Bias ↓, variance ↑ | Explicit, continuous dial |
| Decision tree | Depth, min samples per leaf | Bias ↓, variance ↑ | Deep trees are near-zero bias, huge variance |
| Random forest | Number of trees $B$, feature subsampling | Variance ↓, bias $\approx$ same | Averaging attacks variance only |
| Boosting | Number of rounds, learning rate | Bias ↓ first, then variance ↑ | Sequential bias reduction |
| KNN | $k$ (smaller = more capacity) | Bias ↓, variance ↑ | $k=1$: zero training error |
| SVM | $C$ (larger = more capacity), kernel width | Bias ↓, variance ↑ | Margin softness is the dial |
| Neural network | Width, depth, epochs | Nonmonotone (double descent) | See below |

### Linear models and regularization

Ridge shrinks coefficients toward zero: predictions become systematically biased
(shrunk) but far less sensitive to which sample was drawn — the exact trade the
decomposition prices. There exists $\lambda>0$ whose ridge estimator beats OLS in
expected error whenever variance dominates. Lasso makes the same trade with a
selection twist: zeroed coefficients are maximal bias on those features, bought for
variance reduction elsewhere. Full derivations:
[Regularization theory](../03_regularization/theory.md).

### Trees, forests, boosting

A single deep tree interpolates training data: bias near zero, variance dominated by
which samples landed in each leaf. Averaging $B$ trees with pairwise correlation
$\rho$ gives variance

```math
\rho\sigma_t^2+\frac{1-\rho}{B}\sigma_t^2,
```

so bagging shrinks the second term while $\rho$ floors the first — random forests
decorrelate trees (feature subsampling) precisely to lower $\rho$. Boosting moves in
the opposite direction: each round reduces the bias of the current ensemble, so
overfitting arrives through too many rounds rather than too deep a learner. See
[Ensemble Methods](../06_ensemble_methods/README.md).

### KNN's $k$

The two terms are visible directly: prediction at $x$ averages $k$ neighbor labels, so
variance scales like $\sigma^2/k$, while bias grows as the neighborhood widens to
include points where $f$ differs from $f(x)$. Small $k$ = low bias, high variance;
large $k$ = smooth, biased. In high dimensions the "neighborhood" widens catastrophically
regardless of $k$ — see [Geometry of ML](geometry_of_ml.md) on distance concentration.

### SVM's $C$

Large $C$ punishes margin violations hard: the boundary bends to fit individual points
(low bias, high variance). Small $C$ tolerates violations for a wider margin (higher
bias, lower variance). With RBF kernels, kernel width $\gamma$ is a second capacity
dial acting the same way. See [SVM](../09_svm/README.md).

### Neural networks and double descent

Classical theory predicts a U-shaped test error in capacity. Modern overparameterized
networks show a second descent: past the interpolation threshold (capacity just enough
to fit training data exactly — the variance peak), test error can *decrease* again as
width grows, because among the many interpolating solutions, gradient descent finds
low-norm, smooth ones. The classical decomposition still holds pointwise; what changes
is that variance stops growing monotonically with parameter count. See
[Neural Networks](../13_neural_networks/README.md).

---

## Diagnosing via Learning Curves

Plot training and validation error against training-set size $n$:

| Symptom | High bias (underfit) | High variance (overfit) |
|---|---|---|
| Training error | High, near validation error | Low, far below validation error |
| Gap (val $-$ train) | Small | Large |
| Effect of more data | Curves plateau early, little help | Gap narrows — more data helps |
| Effective fix | More capacity, better features, less $\lambda$ | More data, regularization, ensembling |

Two practical rules:

1. If training error already exceeds the target, no amount of data fixes it — the
   model family is too biased.
2. If the gap is large but shrinking with $n$, collecting data competes with
   regularization; extrapolate the validation curve before buying either.

Model choice from these diagnostics is covered in the
[Model Selection Guide](model_selection_guide.md).

---

## Verification Questions

1. Does test error form a U-shaped curve as capacity changes (or a double descent for
   overparameterized networks)?
2. Across repeated training samples, how variable are predictions at the same $x$?
3. Does a lower-variance model introduce systematic residual structure?
4. Does cross-validation choose a capacity near the held-out optimum?

---

## Connections

- **Topics:** [01 Linear Regression](../01_linear_regression/README.md), [03 Regularization](../03_regularization/README.md), [05 Decision Tree](../05_decision_tree/README.md), [06 Ensemble Methods](../06_ensemble_methods/README.md), [07 KNN](../07_knn/README.md), [09 SVM](../09_svm/README.md), [13 Neural Networks](../13_neural_networks/README.md)
- **Related synthesis:** [Model Selection Guide](model_selection_guide.md), [Regularization Across Models](regularization_across_models.md), [Geometry of ML](geometry_of_ml.md)
- **Maps:** [INDEX.md](../../INDEX.md)
