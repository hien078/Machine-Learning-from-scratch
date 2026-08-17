# Regularization Across Models — Cross-Topic Synthesis

> Every regularizer is the same idea wearing a different costume: control effective
> capacity so the model fits signal, not sample.
> See [INDEX.md](../INDEX.md) for the full curriculum index.

---

## Overview

Regularization controls effective model capacity; it is broader than adding a penalty
to an objective. Penalties, constraints, noise injection, averaging, and stopping early
all restrict *which functions the training procedure can actually reach* — and that is
the common definition this page uses. Two equivalent lenses organize everything below.

## One Frame, Two Lenses

**Lens 1 — capacity control.** For suitable regularity conditions, a penalized problem

```math
\min_\theta L(\theta)+\lambda R(\theta)
```

corresponds to a constrained problem $\min_\theta L(\theta)$ subject to
$R(\theta)\le t$: shrinking $t$ shrinks the hypothesis space, trading variance for bias
(see [Bias–Variance](bias_variance_tradeoff.md); the region-shape geometry is in
[Geometry of ML](geometry_of_ml.md)). The mapping between $\lambda$ and $t$ need not be
one-to-one in degenerate or inactive cases.

**Lens 2 — MAP priors.** If the loss is a negative log-likelihood, the penalty is a
negative log-prior:

```math
\hat{\theta}_{\text{MAP}}
=\arg\max_\theta\;\log p(\mathcal{D}\mid\theta)+\log p(\theta)
=\arg\min_\theta\;L(\theta)-\log p(\theta).
```

| Penalty | Prior on $\theta_j$ | Correspondence |
|---|---|---|
| $\lambda\Vert\theta\Vert_2^2$ | Gaussian $\mathcal{N}(0,\tau^2)$ | $\lambda=\sigma^2/(2\tau^2)$ for Gaussian noise variance $\sigma^2$ |
| $\lambda\Vert\theta\Vert_1$ | Laplace$(0,b)$ | $\lambda=\sigma^2/b$; sharp peak at 0 → exact zeros |

Tighter prior (smaller $\tau$ or $b$) = stronger penalty = smaller reachable region:
the two lenses are one statement. Full derivations:
[Regularization theory](../topics/03_regularization/theory.md); the MLE/MAP framing is
in [Probabilistic View of ML](probabilistic_view_of_ml.md).

---

## Mechanism Map

| Mechanism | Primary effect | Models | Failure mode |
|---|---|---|---|
| $\ell_2$ penalty | Shrinks parameters smoothly | Linear models, SVM, neural networks | Can retain many weak irrelevant features |
| $\ell_1$ penalty | Encourages exact zeros | Linear and generalized linear models | Unstable selection among correlated features |
| Elastic Net | Combines shrinkage and sparsity | Linear models | Adds another tuning dimension |
| Tree depth / leaf constraints | Limits partition complexity | Trees, forests, boosting | Excessive limits create high bias |
| Cost-complexity pruning | Removes weak subtrees post hoc | Decision trees | Pruning parameter needs validation |
| Bagging | Averages away variance | Forests, any high-variance learner | Cannot reduce bias; correlated members limit gains |
| Early stopping | Restricts optimization trajectory | Boosting, neural networks | Requires a leakage-free validation rule |
| Weight decay | $\ell_2$-style shrinkage during updates | Deep networks | Not identical to $\ell_2$ loss penalty under Adam |
| Dropout | Stochastic activation masking | Neural networks | Not equivalent to simple $\ell_2$ in general |
| Data augmentation | Encodes invariances through examples | CNNs and other neural models | Invalid transformations inject label noise |
| Label smoothing | Softens targets, caps confidence | Classifiers with cross-entropy | Distorts calibration and distillation targets |

---

## Per-Family Instances

### Linear models: explicit penalties

Ridge and Lasso are the canonical, fully analyzable cases — closed-form shrinkage for
ridge, soft-thresholding and sparsity for Lasso, with the ball-vs-diamond geometry in
[Geometry of ML](geometry_of_ml.md) and the priors above. Everything else on this page
is a generalization of this picture to models where the penalty cannot be written down
so cleanly.

### Gradient descent: early stopping as implicit $\ell_2$

Running GD from $\theta_0=0$ on least squares fits fast directions (large singular
values of $X$) first and slow directions later. Stopping at step $t$ therefore leaves
low-signal directions near zero — the same directional shrinkage pattern as ridge, with
$\lambda\sim 1/(\eta t)$ as the rough correspondence. Early stopping is a regularizer
you get for free from the optimizer; its "hyperparameter" is the validation-based
stopping time. See [Gradient Descent](../topics/02_gradient_descent/README.md).

### Trees: structural constraints

Trees have no weights to penalize, so capacity is constrained structurally: maximum
depth, minimum samples per leaf/split, and cost-complexity pruning

```math
R_\alpha(T)=R(T)+\alpha\,\vert T\vert,
```

which is the penalized-objective template with $R(\theta)$ replaced by leaf count
$\vert T\vert$. Depth limits act *before* fitting (pre-pruning), cost-complexity acts
*after* (post-pruning). See [Decision Tree](../topics/05_decision_tree/README.md).

### Bagging: regularization by averaging

Bagging changes no objective and no single model — it reduces variance by averaging $B$
resampled fits, leaving bias untouched (the $\rho$-limited variance formula is in
[Bias–Variance](bias_variance_tradeoff.md)). It is the regularizer of choice when the
base learner is deliberately overfit (deep trees), which is exactly the random-forest
recipe. Boosting's counterpart knob is shrinkage: scaling each round by a learning rate
$\nu<1$ plus early stopping on rounds. See
[Ensemble Methods](../topics/06_ensemble_methods/README.md).

### Deep networks: decay, dropout, augmentation, smoothing

- **Weight decay** subtracts $\eta\lambda\theta$ in each update. With plain SGD this
  equals an $\ell_2$ loss penalty; with adaptive optimizers it does not, which is why
  AdamW *decouples* decay from the gradient step (see
  [Optimization Methods](optimization_methods_compared.md)).
- **Dropout** masks activations with probability $p$ during training and rescales at
  test time. Two readings: an implicit ensemble of exponentially many subnetworks, and
  noise injection that penalizes co-adapted features. Not equivalent to $\ell_2$ except
  in special linear cases.
- **Data augmentation** is a prior over invariances expressed as data: if labels are
  invariant to a transform group (flips, crops, shifts), sampling that group constrains
  the learned function to respect it — capacity control in function space rather than
  parameter space. See [CNN](../topics/14_cnn/README.md).
- **Label smoothing** replaces one-hot targets with
  $(1-\epsilon)\,y+\epsilon/K$, capping achievable confidence and shrinking logit
  magnitudes — an output-space analogue of weight shrinkage.

See [Neural Networks](../topics/13_neural_networks/README.md).

---

## When to Prefer Which

| Situation | Preferred regularizer | Why |
|---|---|---|
| Many features, few believed relevant | $\ell_1$ / Elastic Net | Sparsity doubles as selection |
| Correlated features, all mildly useful | $\ell_2$ | Stable shrinkage, no arbitrary dropping |
| Single tree overfits | Depth / min-samples limits, then bagging | Structural first, averaging second |
| Any high-variance learner, compute available | Bagging | Variance reduction without retuning the learner |
| Deep net, limited data | Augmentation first, then decay + dropout | Invariance priors are the cheapest capacity cut |
| Long training runs | Early stopping | Free, needs only a clean validation split |
| Overconfident classifier | Label smoothing | Directly targets the confidence pathology |

Rule of thumb: prefer the regularizer whose *assumption* matches your prior knowledge —
sparsity, smoothness, invariance, or "the model is simply too flexible" — rather than
stacking all of them by default.

---

## Validation Rule

Regularization strength is selected inside the training/validation procedure. Feature
scaling, imputation, and selection must be fitted separately inside every fold; fitting
them once on all data leaks validation information. Selection workflow:
[Model Selection Guide](model_selection_guide.md).

---

## Connections

- **Topics:** [02 Gradient Descent](../topics/02_gradient_descent/README.md), [03 Regularization](../topics/03_regularization/README.md), [05 Decision Tree](../topics/05_decision_tree/README.md), [06 Ensemble Methods](../topics/06_ensemble_methods/README.md), [13 Neural Networks](../topics/13_neural_networks/README.md), [14 CNN](../topics/14_cnn/README.md)
- **Related synthesis:** [Bias–Variance Trade-off](bias_variance_tradeoff.md), [Geometry of ML](geometry_of_ml.md), [Probabilistic View of ML](probabilistic_view_of_ml.md), [Optimization Methods Compared](optimization_methods_compared.md), [Model Selection Guide](model_selection_guide.md)
- **Maps:** [INDEX.md](../INDEX.md)
