# Ensemble Methods — Theory

## 0. Notation

All symbols used below — defined once.

| Symbol | Type | Meaning |
|---|---|---|
| $n$ | scalar | number of training examples |
| $d$ | scalar | number of features |
| $M$ | scalar | number of base learners (trees) in the ensemble |
| $B$ | scalar | number of bootstrap samples (often $B = M$) |
| $h_m$ | function | $m$-th base learner (decision tree) |
| $H(x)$ | function | ensemble prediction: vote or average of $h_1, \dots, h_M$ |
| $\sigma^2$ | scalar | variance of a single base learner's prediction |
| $\rho$ | scalar in $[0, 1]$ | average pairwise correlation between base learners |
| $m_{\text{try}}$ | scalar | number of candidate features evaluated at each split |
| $w_i$ | scalar $\ge 0$ | sample weight for the $i$-th example (boosting) |
| $\alpha_m$ | scalar | weight of the $m$-th weak learner (AdaBoost) |
| $\epsilon_m$ | scalar in $(0, 1)$ | weighted error rate of the $m$-th weak learner |
| $\eta$ | scalar in $(0, 1]$ | learning rate (shrinkage) in gradient boosting |
| $L(y, F)$ | scalar | loss function: measures cost of predicting $F$ when truth is $y$ |
| $F_m(x)$ | function | ensemble prediction after $m$ boosting rounds |
| $r_{im}$ | scalar | pseudo-residual: $-\partial L(y_i, F) / \partial F$ evaluated at $F = F_{m-1}(x_i)$ |

**Convention.** Classification labels are $y_i \in \{-1, +1\}$ for AdaBoost and
$y_i \in \{0, 1, \dots, K-1\}$ for random forest. Regression targets are real-valued.

---

## 1. WHY — The Instability Problem

A single decision tree is a **high-variance** model: small changes in the training set can
change the root split and cascade through the entire structure (see topic 05, §8.2). Two
bootstrap samples of the same data may produce trees that disagree on 15–30% of test
points.

**Key insight.** If we could generate many independent datasets from the true distribution,
train a tree on each, and average their predictions, the variance of the average would
shrink. We cannot access the true distribution, but we can simulate it with **bootstrap
sampling** from the training set.

This is the fundamental idea behind ensembles: **combine many imperfect predictors whose
errors partially cancel**.

---

## 2. WHAT — Bagging (Bootstrap Aggregating)

### 2.1 Procedure

Given training data $\{(x_i, y_i)\}_{i=1}^n$:

1. For $m = 1, \dots, M$:
   a. Draw a **bootstrap sample** $S_m$: sample $n$ examples from the training set
      **with replacement**.
   b. Train a base learner $h_m$ on $S_m$.
2. Aggregate:
   - **Regression:** $H(x) = \frac{1}{M} \sum_{m=1}^M h_m(x)$ (average).
   - **Classification:** $H(x) = \text{mode}\{h_1(x), \dots, h_M(x)\}$ (majority vote).

### 2.2 Why averaging reduces variance

Suppose we have $M$ predictors with identical variance $\sigma^2$ and pairwise
correlation $\rho$. The variance of their average is:

$$\text{Var}\!\left(\frac{1}{M}\sum_{m=1}^M h_m\right) = \frac{1}{M^2} \left[\sum_{m=1}^M \text{Var}(h_m) + \sum_{m \ne m'} \text{Cov}(h_m, h_{m'})\right].$$

Since $\text{Var}(h_m) = \sigma^2$ and $\text{Cov}(h_m, h_{m'}) = \rho\sigma^2$:

$$= \frac{1}{M^2}\left[M\sigma^2 + M(M-1)\rho\sigma^2\right] = \rho\sigma^2 + \frac{1-\rho}{M}\sigma^2.$$

**Result:** The variance of the bagged ensemble is

$$\text{Var}(H) = \rho\sigma^2 + \frac{1-\rho}{M}\sigma^2. \qquad (2.1)$$

- **Special case $\rho = 0$ (uncorrelated predictors):** $\text{Var}(H) = \sigma^2 / M$.
  Variance shrinks as $1/M$ — the ideal case.
- **Special case $\rho = 1$ (identical predictors):** $\text{Var}(H) = \sigma^2$. No
  reduction at all — averaging copies of the same model is useless.
- **General case $0 < \rho < 1$:** increasing $M$ drives the second term to zero, but
  the first term $\rho\sigma^2$ remains as an irreducible floor.

**Implication.** To reduce variance maximally, we need many base learners with **low
pairwise correlation**.

---

## 3. HOW — Random Forest

### 3.1 From bagging to random forest

Bagged trees are still correlated because they see the same features and tend to pick the
same strong features at the root. **Random forests** (Breiman, 2001) add one modification:

> At each split, restrict the candidate features to a **random subset** of size $m_{\text{try}}$.

This decorrelates the trees: if feature $j$ is the globally strongest, not every tree
gets to see it at every split. Different trees build different structures, reducing $\rho$.

### 3.2 Typical values of $m_{\text{try}}$

| Task | Default $m_{\text{try}}$ |
|---|---|
| Classification | $\lfloor\sqrt{d}\rfloor$ |
| Regression | $\lfloor d/3 \rfloor$ |

Smaller $m_{\text{try}}$ → more decorrelation (lower $\rho$) but weaker individual trees
(higher single-tree error). The optimal value balances this trade-off.

### 3.3 Algorithm summary

1. For $m = 1, \dots, M$:
   a. Draw bootstrap sample $S_m$ of size $n$.
   b. Grow a decision tree on $S_m$, but at each node:
      - Select $m_{\text{try}}$ features at random (without replacement).
      - Find the best split among only those features.
   c. Grow the tree fully (no pruning).
2. Predict by majority vote (classification) or averaging (regression).

---

## 4. Out-of-Bag (OOB) Error Estimation

### 4.1 The OOB idea

Each bootstrap sample $S_m$ contains about $1 - (1 - 1/n)^n \approx 1 - 1/e \approx 63.2\%$
of the original training examples. The remaining ~36.8% are **out-of-bag** for tree $m$.

For each training example $x_i$, let $\text{OOB}(i) = \{m : x_i \notin S_m\}$ be the
set of trees that did not train on $x_i$. The OOB prediction is:

$$\hat{y}_i^{\text{OOB}} = \text{aggregate}\{h_m(x_i) : m \in \text{OOB}(i)\}.$$

### 4.2 Why it works

The OOB prediction for $x_i$ uses only trees that never saw $x_i$ during training — it is
a legitimate held-out prediction. The OOB error aggregated over all examples approximates
the leave-one-out cross-validation error, without the cost of retraining.

**Result:** The OOB error is an unbiased estimate of the generalization error, computed
for free during training.

---

## 5. Boosting

### 5.1 Core idea

While bagging reduces **variance** by averaging independent models, boosting reduces
**bias** by building models **sequentially**, each one focusing on the mistakes of the
previous ensemble.

### 5.2 AdaBoost (Adaptive Boosting)

For binary classification with $y_i \in \{-1, +1\}$:

1. Initialize equal weights: $w_i^{(1)} = 1/n$ for all $i$.
2. For $m = 1, \dots, M$:
   a. Train weak learner $h_m$ on the weighted training set.
   b. Compute weighted error:
      $$\epsilon_m = \frac{\sum_{i=1}^n w_i^{(m)} \cdot \mathbb{1}[h_m(x_i) \ne y_i]}{\sum_{i=1}^n w_i^{(m)}}. \qquad (5.1)$$
   c. Compute learner weight:
      $$\alpha_m = \frac{1}{2} \ln\frac{1 - \epsilon_m}{\epsilon_m}. \qquad (5.2)$$
   d. Update sample weights:
      $$w_i^{(m+1)} = w_i^{(m)} \cdot \exp(-\alpha_m \cdot y_i \cdot h_m(x_i)). \qquad (5.3)$$
   e. Normalize: $w_i^{(m+1)} \leftarrow w_i^{(m+1)} / \sum_j w_j^{(m+1)}$.
3. Final prediction: $H(x) = \text{sign}\!\left(\sum_{m=1}^M \alpha_m h_m(x)\right)$.

**Key properties of $\alpha_m$:**
- If $\epsilon_m < 0.5$ (better than random), then $\alpha_m > 0$: the learner contributes positively.
- If $\epsilon_m = 0$, then $\alpha_m \to \infty$: a perfect learner gets infinite weight.
- If $\epsilon_m = 0.5$ (random), then $\alpha_m = 0$: the learner is ignored.

**Weight update intuition.** When $y_i \cdot h_m(x_i) = -1$ (misclassified), the exponent
is $+\alpha_m > 0$, so $w_i$ increases. When $y_i \cdot h_m(x_i) = +1$ (correct), $w_i$
decreases. The next learner focuses on hard examples.

### 5.3 Gradient Boosting

Gradient boosting generalises boosting to arbitrary differentiable loss functions by
interpreting each step as **functional gradient descent**.

Given a differentiable loss $L(y, F)$ and current ensemble $F_{m-1}$:

1. Initialize: $F_0(x) = \arg\min_\gamma \sum_{i=1}^n L(y_i, \gamma)$.
2. For $m = 1, \dots, M$:
   a. Compute **pseudo-residuals** (negative gradient of loss):
      $$r_{im} = -\frac{\partial L(y_i, F)}{\partial F}\bigg|_{F = F_{m-1}(x_i)}. \qquad (5.4)$$
   b. Fit a regression tree $h_m$ to the pseudo-residuals $\{(x_i, r_{im})\}$.
   c. Update: $F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$.

For **squared loss** $L(y, F) = \frac{1}{2}(y - F)^2$:

$$r_{im} = -\frac{\partial}{\partial F}\frac{1}{2}(y_i - F)^2\bigg|_{F=F_{m-1}(x_i)} = y_i - F_{m-1}(x_i).$$

**Result:** Under squared loss, the pseudo-residuals are simply the residuals
$r_{im} = y_i - F_{m-1}(x_i)$, and each new tree fits the errors of the current ensemble.

The learning rate $\eta$ (shrinkage) controls the step size. Smaller $\eta$ requires more
trees but typically produces better generalization — analogous to step-size tuning in
gradient descent.

---

## 6. Bias-Variance Decomposition of Ensembles

| Method | Primary effect | Mechanism |
|---|---|---|
| Bagging / RF | Reduces **variance** | Averaging decorrelated models; bias stays approximately unchanged |
| Boosting | Reduces **bias** | Each stage corrects residual errors; variance may increase with too many stages |

**Bagging:** Each bagged tree has roughly the same bias as a single tree (bootstrap samples
approximate the full training set). The averaging reduces variance per equation (2.1).
Bias is unchanged or slightly increased because bootstrap samples are slightly smaller
effective samples.

**Boosting:** The initial model has high bias (e.g. a shallow stump). Each boosting round
reduces the training error, lowering bias. However, with too many rounds or noisy data,
boosting can overfit — increasing variance.

---

## 7. Feature Importance

### 7.1 Impurity-based importance (MDI)

For each feature $j$, sum the weighted impurity decrease over all splits that use $j$,
across all trees:

$$\text{Importance}(j) = \frac{1}{M}\sum_{m=1}^M \sum_{\text{node } t \text{ in tree } m \atop \text{splits on } j} \frac{|t|}{n} \cdot \Delta I(t). \qquad (7.1)$$

where $|t|$ is the number of samples at node $t$ and $\Delta I(t)$ is the impurity decrease.

### 7.2 Permutation importance

1. Compute baseline score on validation data.
2. For each feature $j$: randomly permute column $j$, recompute score.
3. Importance = drop in score.

Permutation importance is model-agnostic and avoids the bias of MDI toward high-cardinality
features.

---

## 8. Failure Cases

### 8.1 Boosting overfits noisy data

Boosting aggressively upweights misclassified examples. If some examples are mislabelled
or intrinsically noisy, boosting assigns them enormous weight, forcing subsequent learners
to memorise noise. Symptoms:

- Training error → 0 while test error increases.
- A few examples dominate the weight distribution.

**Mitigation:** Early stopping, subsampling (stochastic gradient boosting), or limiting
tree depth.

### 8.2 Random forest: diminishing returns and computational cost

- **Diminishing returns:** After $M \approx 100$–$500$ trees, additional trees reduce
  variance negligibly (the $\rho\sigma^2$ floor in equation 2.1).
- **Computational cost:** Each tree costs $O(m_{\text{try}} \cdot n \log n)$ to build.
  Total training cost is $O(M \cdot m_{\text{try}} \cdot n \log n)$.
- **Memory:** Storing $M$ unpruned trees can be expensive for large $M$.

### 8.3 Extrapolation failure (tree-based ensembles)

All tree-based methods predict piecewise-constant functions. They **cannot extrapolate**
beyond the range of the training targets. For regression on trending data (e.g.
time-series), the ensemble's predictions are bounded by the training target range.

### 8.4 Loss of interpretability

A single tree can be read as a set of if-then rules. An ensemble of hundreds of trees
sacrifices this interpretability. Feature importance and partial dependence plots provide
partial compensation but do not recover full transparency.

---

## 9. Connections

- **Decision Tree** (topic 05). The base learner. Single trees are high-variance, low-bias
  models — the perfect building block for bagging/random forest.
- **Gradient Descent** (topic 02). Gradient boosting is gradient descent in function space.
  The learning rate $\eta$ plays the same role as step size in parameter-space optimisation.
- **Bias–Variance Tradeoff** (synthesis). Bagging and boosting attack opposite ends of the
  decomposition. Understanding this tradeoff is key to choosing between them.
- **Regularization** (topic 03). Shrinkage ($\eta < 1$) in gradient boosting is a form of
  regularization. Early stopping is analogous to limiting model complexity.
- **Neural Networks** (topic 13). Dropout can be interpreted as an ensemble method
  (averaging over exponentially many sub-networks).

---

## 10. References

- **Breiman, L. (2001).** Random forests. *Machine Learning*, 45(1), 5–32.
- **Freund, Y., & Schapire, R. E. (1997).** A decision-theoretic generalization of on-line learning and an application to boosting. *Journal of Computer and System Sciences*, 55(1), 119–139.
- **Friedman, J. H. (2001).** Greedy function approximation: A gradient boosting machine. *Annals of Statistics*, 29(5), 1189–1232.
- **Chen, T., & Guestrin, C. (2016).** XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785–794.
- **Hastie, T., Tibshirani, R., & Friedman, J. (2009).** *The Elements of Statistical Learning: Data Mining, Inference, and Prediction* (2nd ed.). Springer. Chapter 10 (*Boosting and Additive Trees*) & Chapter 15 (*Random Forests*).

