# Decision Trees — Theory

## 0. Notation

All symbols used below — defined once.

| Symbol | Type | Meaning |
|---|---|---|
| $n$ | scalar | number of training examples |
| $d$ | scalar | number of features |
| $x_i$ | vector of length $d$ | feature vector of the $i$-th example |
| $y_i$ | scalar | label of the $i$-th example (class label or continuous value) |
| $K$ | scalar | number of distinct classes (classification) |
| $\mathcal{R}$ | set | set of examples reaching a given node |
| $\vert \mathcal{R}\vert $ | scalar | number of examples in a node |
| $p_k$ | scalar in $[0, 1]$ | proportion of class $k$ in a node: $p_k = \vert \lbrace i \in \mathcal{R} : y_i = k\rbrace\vert  / \vert \mathcal{R}\vert $ |
| $G(\mathcal{R})$ | scalar in $[0, 1]$ | Gini impurity of node $\mathcal{R}$ |
| $H(\mathcal{R})$ | scalar $\ge 0$ | entropy of node $\mathcal{R}$ |
| $j$ | scalar | feature index used for splitting |
| $t$ | scalar | threshold for the split: left child gets $\lbrace x : x_j < t\rbrace$ |
| $\mathcal{R}_L, \mathcal{R}_R$ | sets | left and right child partitions after a split |
| $D$ | scalar | maximum allowed tree depth |
| $n_{\min}$ | scalar | minimum samples required to split a node |
| $\alpha$ | scalar $\ge 0$ | cost-complexity pruning parameter |
| $\vert T\vert $ | scalar | number of leaf nodes in tree $T$ |

**Convention.** All logarithms are base-2 ($\log_2$) unless stated otherwise. For entropy
in nats, use $\ln$; the formulas are identical up to a constant factor. We define
$0 \log 0 := 0$ by continuity.

---

## 1. WHY — From Linear Boundaries to Recursive Partitioning

Linear models (logistic regression, SVMs) carve feature space with a single hyperplane.
This fails when the decision depends on **threshold interactions**:

- "Approve loan if age $\ge$ 25 **and** income $\ge$ 50k." No single hyperplane captures
  this rectangular region.
- "Flag fraud if transaction $> 5{,}000$ **or** country is high-risk." The decision
  boundary is a union of axis-aligned rectangles.

A decision tree builds an **interpretable, piecewise-constant predictor** by recursively
splitting the feature space along one feature at a time. Each split creates two children;
each leaf returns a constant prediction (majority class or mean target value). The result
is a set of axis-aligned rectangles, each with its own prediction.

**Key properties.**

1. **No feature scaling needed** — splits depend only on sort order.
2. **Handles mixed types** — continuous and categorical features coexist naturally.
3. **Interpretable** — the tree can be read as a sequence of if/then rules.
4. **Non-parametric** — no assumption about the functional form of the boundary.

---

## 2. WHAT — Impurity Measures and Objective

### 2.1 Classification: Gini impurity

For a node with class proportions $p_1, \dots, p_K$:

$$G = 1 - \sum_{k=1}^{K} p_k^2. \qquad (2.1)$$

**Interpretation.** $G$ is the probability that two randomly drawn examples from the node
(with replacement) belong to different classes. $G = 0$ when the node is pure (all one
class); $G$ is maximised when all classes are equally represented.

### 2.2 Classification: Entropy

$$H = -\sum_{k=1}^{K} p_k \log_2 p_k. \qquad (2.2)$$

**Interpretation.** $H$ measures the average number of bits needed to encode the class
label. $H = 0$ for a pure node; $H = \log_2 K$ when all classes are equally likely.

### 2.3 Regression: Mean Squared Error

For a regression tree, each leaf predicts the mean of its targets:
$\hat y_{\mathcal{R}} = \frac{1}{|\mathcal{R}|} \sum_{i \in \mathcal{R}} y_i$. The
impurity of a node is the variance of the targets:

$$\text{MSE}(\mathcal{R}) = \frac{1}{|\mathcal{R}|} \sum_{i \in \mathcal{R}} (y_i - \hat{y}_{\mathcal{R}})^2. \qquad (2.3)$$

### 2.4 Information Gain

Given a parent node $\mathcal{R}$ split into children $\mathcal{R}_L$ and $\mathcal{R}_R$:

$$\Delta I = I(\mathcal{R}) - \frac{|\mathcal{R}_L|}{|\mathcal{R}|} I(\mathcal{R}_L) - \frac{|\mathcal{R}_R|}{|\mathcal{R}|} I(\mathcal{R}_R), \qquad (2.4)$$

where $I$ is any impurity measure (Gini, entropy, or MSE). The greedy algorithm picks
the split $(j, t)$ that **maximises** $\Delta I$.

---

## 3. HOW — Greedy Recursive Splitting

### 3.1 The split-selection algorithm

At each node $\mathcal{R}$:

1. **For each feature** $j \in \lbrace1, \dots, d\rbrace$:
   a. Sort the distinct values of $x_j$ in $\mathcal{R}$.
   b. For each candidate threshold $t$ (midpoint between consecutive sorted values):
      - Partition $\mathcal{R}$ into $\mathcal R_L = \lbrace i : x_{ij} < t\rbrace$ and
        $\mathcal R_R = \lbrace i : x_{ij} \ge t\rbrace$.
      - Compute the weighted impurity
        $\frac{|\mathcal{R}_L|}{|\mathcal{R}|} I(\mathcal{R}_L) + \frac{|\mathcal{R}_R|}{|\mathcal{R}|} I(\mathcal{R}_R)$.
2. **Pick** the $(j^\ast, t^\ast)$ that minimises the weighted child impurity (equivalently,
   maximises information gain).
3. **Recurse** on $\mathcal{R}_L$ and $\mathcal{R}_R$.

**The algorithm is greedy:** it optimises each split independently without look-ahead.
Finding the globally optimal tree (minimising total leaves or error) is NP-hard.

### 3.2 Derivation: why Gini and entropy prefer balanced splits

> **Claim.** Both $G$ and $H$ are strictly concave functions of the class-proportion
> vector $(p_1, \dots, p_K)$. Concavity implies that the weighted average of children's
> impurity is always $\le$ the parent's impurity (Jensen's inequality), and the gap is
> largest when the split creates the most "different" children.

**Proof for the binary case** ($K = 2$, so $p_1 = p$, $p_2 = 1 - p$):

**Gini:** $G(p) = 2p(1-p)$. Second derivative: $G''(p) = -4 < 0$. So $G$ is strictly
concave on $[0, 1]$.

**Entropy:** $H(p) = -p \log p - (1-p) \log(1-p)$. Second derivative:
$H''(p) = -\frac{1}{p(1-p) \ln 2} < 0$ for $p \in (0, 1)$. Strictly concave. $\blacksquare$

**Consequence.** By Jensen's inequality applied to the concave function $I$:

```math
I\!\left(\frac{|\mathcal{R}_L|}{|\mathcal{R}|} \mathbf{p}_L + \frac{|\mathcal{R}_R|}{|\mathcal{R}|} \mathbf{p}_R\right) \ge \frac{|\mathcal{R}_L|}{|\mathcal{R}|} I(\mathbf{p}_L) + \frac{|\mathcal{R}_R|}{|\mathcal{R}|} I(\mathbf{p}_R),
```

where $\mathbf{p}_L, \mathbf{p}_R$ are the class-proportion vectors of the children.
The left side is the parent's impurity (since proportions mix linearly). Equality holds
only when $\mathbf{p}_L = \mathbf{p}_R$ — a useless split. The largest gap (maximum
information gain) occurs when the children are most "separated" in class composition.

**Result:** Both Gini and entropy are concave functions of class proportions; the
information gain is always non-negative and is maximised by splits that separate classes.

### 3.3 Comparison of Gini and Entropy

For binary classification ($K = 2$):

| $p$ | Gini $2p(1-p)$ | Entropy $-p\log_2 p - (1-p)\log_2(1-p)$ |
|---:|---:|---:|
| 0.0 | 0.000 | 0.000 |
| 0.1 | 0.180 | 0.469 |
| 0.3 | 0.420 | 0.881 |
| 0.5 | 0.500 | 1.000 |

Both are symmetric about $p = 0.5$, zero at $p = 0$ and $p = 1$, and maximal at
$p = 0.5$. In practice they produce very similar trees; Gini is slightly cheaper to
compute (no logarithm).

---

## 4. Stopping Criteria and Leaf Prediction

The recursion terminates at a node when any of these conditions hold:

1. **Pure node.** All examples have the same class ($G = 0$, $H = 0$).
2. **Maximum depth.** The node's depth equals $D$.
3. **Minimum samples.** $|\mathcal{R}| < n_{\min}$ (too few examples to split).
4. **No gain.** The best split yields $\Delta I \le 0$ (or below a threshold).

**Leaf prediction:**

- **Classification:** predict the majority class
  $\hat{y} = \arg\max_k p_k$.
- **Regression:** predict the mean
  $\hat{y} = \frac{1}{|\mathcal{R}|} \sum_{i \in \mathcal{R}} y_i$.

---

## 5. Pruning — Cost-Complexity (Minimal Cost-Complexity)

### 5.1 Motivation

Deep trees overfit: they memorise training noise. Stopping early (via $D$ or $n_{\min}$)
is a blunt tool — a tree may need depth 10 in one branch but only 3 in another.
**Pruning** grows a full tree first, then removes subtrees that don't justify their
complexity.

### 5.2 Cost-complexity criterion

For a subtree $T$ with leaves $l_1, \dots, l_{|T|}$, define:

$$R_\alpha(T) = \sum_{l=1}^{|T|} \frac{|\mathcal{R}_l|}{n} \cdot I(\mathcal{R}_l) + \alpha |T|. \qquad (5.1)$$

The first term is the **total weighted impurity** (resubstitution error). The second is a
**complexity penalty** proportional to the number of leaves. The parameter $\alpha \ge 0$
controls the trade-off:

- $\alpha = 0$: the full tree is optimal (no penalty for leaves).
- $\alpha \to \infty$: the root node alone is optimal (any split costs more than it saves).

### 5.3 Pruning algorithm (Breiman et al., 1984)

1. Grow the full tree $T_{\max}$.
2. For increasing $\alpha$, find the weakest link — the internal node whose subtree
   contributes the least per-leaf reduction in impurity:

   $$\alpha_{\text{eff}}(t) = \frac{R(t) - R(T_t)}{|T_t| - 1},$$

   where $R(t)$ is the impurity of node $t$ treated as a leaf and $R(T_t)$ is the
   total impurity of the subtree rooted at $t$.
3. Collapse the weakest link to a leaf. Repeat to generate a nested sequence
   $T_{\max} \supset T_1 \supset T_2 \supset \cdots \supset \lbrace root\rbrace$.
4. Select the best $\alpha$ (and corresponding tree) by cross-validation.

**Result:** Cost-complexity pruning produces a nested sequence of optimally pruned
subtrees. Cross-validation selects the right complexity.

---

## 6. Computational Complexity

### 6.1 Building one node

At a single node with $m$ samples:

1. **Sort each feature:** $O(d \cdot m \log m)$.
2. **Scan sorted values for best threshold:** $O(d \cdot m)$ (linear scan, updating
   left/right counts incrementally).
3. **Total per node:** $O(d \cdot m \log m)$.

### 6.2 Full tree

- **Balanced tree** of depth $D$: at each level, the total work across all nodes is
  $O(d \cdot n \log n)$ (each sample is scanned once per level). With $D$ levels:
  $O(D \cdot d \cdot n \log n)$.
- A fully grown tree has $D \le n$, giving worst case $O(d \cdot n^2 \log n)$.
- In practice, $D = O(\log n)$ (balanced data), so training is $O(d \cdot n \log^2 n)$.

### 6.3 Prediction

Prediction follows a root-to-leaf path of length $\le D$: $O(D)$ per sample.

---

## 7. Classification vs Regression Trees (CART)

CART (Classification and Regression Trees) unifies both tasks:

| Aspect | Classification | Regression |
|---|---|---|
| Impurity | Gini or Entropy | MSE (variance) |
| Leaf prediction | Majority class | Mean of targets |
| Loss | Misclassification rate | MSE |
| Split objective | Maximise class separation | Minimise within-child variance |

The algorithm structure — greedy binary splits, recursive build, pruning — is identical.
Only the impurity measure and leaf prediction rule change.

---

## 8. Failure Cases

### 8.1 Axis-aligned boundaries only

Every split is perpendicular to one feature axis. A diagonal boundary like
$x_1 + x_2 > 0$ requires many staircase-like splits to approximate. Linear models
handle this in one step.

### 8.2 High variance (instability)

Small changes in training data can change the root split, cascading through the entire
tree. Two bootstrap samples of the same dataset may produce very different trees. This is
the fundamental motivation for **ensemble methods** (bagging, random forests).

**Quantifying instability.** Train $B$ trees on bootstrap samples and measure the
disagreement rate: the fraction of test points where at least two trees predict different
classes. Deep, unpruned trees typically show 15–30% disagreement.

### 8.3 Overfitting

A fully grown tree can memorise the training data (zero training error, high test error).
Cures:

1. Limit depth ($D$) or minimum leaf samples ($n_{\min}$).
2. Post-hoc pruning (§5).
3. Ensemble averaging (random forests, boosting).

### 8.4 Biased feature selection

Features with many distinct values (e.g. continuous vs binary) tend to be selected more
often because they offer more candidate thresholds. Entropy-based gain ratio
(Quinlan, 1993) normalises by the split's own entropy to counteract this bias.

---

## 9. Connections

- **Information Theory** (foundations). Entropy and information gain are the core
  quantities from Shannon's theory applied to supervised learning.
- **Ensemble Methods** (topic 06). Bagging (Random Forests) reduces tree variance by
  averaging many high-variance trees. Boosting (AdaBoost, Gradient Boosting) sequentially
  fits trees to residuals.
- **Bias–Variance Tradeoff** (synthesis). A single deep tree = low bias, high variance.
  Shallow trees or pruned trees increase bias to reduce variance.
- **Linear Models** (topics 01, 04). Trees are complementary: linear models excel at
  smooth boundaries; trees excel at axis-aligned threshold interactions. Combining both
  (e.g. model trees) is an active area.
- **SVM** (topic 09). Kernel SVMs handle non-linear boundaries via feature mapping;
  trees handle them via recursive partitioning. Neither requires the other's machinery.

---

## 10. References

- **Breiman, L., Friedman, J., Olshen, R., & Stone, C. (1984).** *Classification and Regression Trees*. Wadsworth & Brooks/Cole.
- **Quinlan, J. R. (1986).** Induction of decision trees. *Machine Learning*, 1(1), 81–106.
- **Quinlan, J. R. (1993).** *C4.5: Programs for Machine Learning*. Morgan Kaufmann.
- **Hastie, T., Tibshirani, R., & Friedman, J. (2009).** *The Elements of Statistical Learning: Data Mining, Inference, and Prediction* (2nd ed.). Springer. Chapter 9.2: *Tree-Based Methods*.

