# K-Nearest Neighbors — Theory

## 0. Notation

All symbols used below — defined once.

| Symbol | Type | Meaning |
|---|---|---|
| $n$ | scalar | number of training examples |
| $p$ | scalar | number of features (dimensionality) |
| $x_i$ | vector of length $p$ | feature vector of the $i$-th training example |
| $X$ | matrix of size $n \times p$ | training feature matrix; row $i$ is $x_i^T$ |
| $y_i$ | scalar | class label of the $i$-th example |
| $x_q$ | vector of length $p$ | query (test) point |
| $K$ | scalar | number of neighbors to consult |
| $d(x, z)$ | scalar $\ge 0$ | distance between vectors $x$ and $z$ |
| $\mathcal{N}_K(x_q)$ | index set of size $K$ | indices of the $K$ nearest training points to $x_q$ |
| $w_i$ | scalar $\ge 0$ | weight assigned to neighbor $i$ |
| $C$ | scalar | number of distinct classes |
| $V_p(r)$ | scalar | volume of a $p$-dimensional ball of radius $r$ |

**Vector convention.** All vectors are column vectors. Lowercase Latin = vector;
uppercase = matrix.

Norms carry explicit subscripts: $\Vert x\Vert_1$, $\Vert x\Vert_2$.

---

## 1. WHY — Instance-Based Learning

Most classifiers (logistic regression, SVM, neural networks) learn a **parametric model**
during training: they compress the data into a fixed set of parameters $\theta$, then
discard the training set. KNN takes the opposite approach:

- **No training phase.** Store the entire training set $\lbrace(x_i, y_i)\rbrace_{i=1}^n$.
- **Prediction = local lookup.** For a query $x_q$, find the $K$ training points closest
  to $x_q$ and let them vote.
- **Non-parametric.** The model complexity grows with $n$, not with a fixed $p$-dimensional
  parameter vector.

### 1.1 When does this make sense?

1. Decision boundaries are irregular and hard to express as a parametric function.
2. The dataset is small enough that storing and scanning it is feasible.
3. Local structure matters more than global patterns.

---

## 2. WHAT — The KNN Rule

### 2.1 Classification rule

Given a query $x_q$:

1. Compute $d(x_q, x_i)$ for every training point $i = 1, \dots, n$.
2. Select the $K$ indices with the smallest distances: $\mathcal{N}_K(x_q)$.
3. Predict by **majority vote**:

```math
\hat{y}(x_q) = \arg\max_{c \in \{1, \dots, C\}} \sum_{i \in \mathcal{N}_K(x_q)} \mathbb{1}[y_i = c].
```

**Result:** KNN predicts the most frequent class among the $K$ nearest neighbors.

### 2.2 Distance metrics

The choice of distance function defines "nearest."

**Euclidean distance (L2):**

```math
d_2(x, z) = \|x - z\|_2 = \left(\sum_{j=1}^p (x_j - z_j)^2 \right)^{1/2}.
```

**Manhattan distance (L1):**

```math
d_1(x, z) = \|x - z\|_1 = \sum_{j=1}^p |x_j - z_j|.
```

**Minkowski distance (Lp):**

```math
d_q(x, z) = \|x - z\|_q = \left(\sum_{j=1}^p |x_j - z_j|^q \right)^{1/q}, \qquad q \ge 1.
```

Euclidean is $q = 2$, Manhattan is $q = 1$. As $q \to \infty$, $d_q \to \max_j |x_j - z_j|$
(Chebyshev distance).

### 2.3 No closed-form training

KNN has **no training phase** — no objective function to minimize, no parameters to
optimize. The "model" *is* the training data. All computation happens at prediction time.

---

## 3. HOW — Choosing K (Bias-Variance Tradeoff)

### 3.1 K = 1: minimum bias, maximum variance

With $K = 1$, the decision boundary passes between every pair of differently-labeled
neighbors. This produces:

- **Zero training error** (each point is its own nearest neighbor).
- A jagged, complex boundary that overfits noise.

### 3.2 K = n: maximum bias, minimum variance

With $K = n$, every query consults all training points. The prediction is always the
overall majority class — a constant classifier that ignores the input.

### 3.3 Intermediate K

**Result:** Increasing $K$ smooths the decision boundary. The optimal $K$ balances:

| Small K | Large K |
|---|---|
| Low bias | High bias |
| High variance | Low variance |
| Captures fine structure | Misses local patterns |
| Sensitive to noise | Robust to noise |

A common heuristic: $K \approx \sqrt{n}$. In practice, use cross-validation.

---

## 4. Weighted KNN

Uniform voting treats all $K$ neighbors equally. A distant neighbor has the same
influence as an adjacent one. **Weighted KNN** assigns more weight to closer neighbors.

### 4.1 Inverse-distance weighting

$$w_i = \frac{1}{d(x_q, x_i) + \epsilon}, \qquad \epsilon > 0 \text{ (small constant to avoid division by zero)}.$$

The prediction becomes:

$$\hat{y}(x_q) = \arg\max_{c} \sum_{i \in \mathcal{N}_K(x_q)} w_i \cdot \mathbb{1}[y_i = c].$$

**Result:** Weighted KNN is less sensitive to the exact choice of $K$ because distant
neighbors contribute less. It also reduces the impact of ties.

---

## 5. Curse of Dimensionality

KNN relies on the assumption that nearby points have similar labels. In high dimensions,
this assumption breaks down.

### 5.1 Volume of a hypersphere

The volume of a $p$-dimensional ball of radius $r$ is:

$$V_p(r) = \frac{\pi^{p/2}}{\Gamma(p/2 + 1)} \cdot r^p.$$

**Key fact:** As $p$ grows, the ratio $V_p(r) / V_p(R)$ for $r < R$ shrinks exponentially.
Almost all the volume of a high-dimensional ball lies in a thin shell near the surface.

### 5.2 Concentration of distances

For points drawn uniformly in $[0, 1]^p$, the expected nearest and farthest distances
satisfy:

$$\frac{d_{\max} - d_{\min}}{d_{\min}} \to 0 \qquad \text{as } p \to \infty.$$

**Result:** In high dimensions, all pairwise distances become nearly equal. The concept
of "nearest neighbor" loses meaning — the closest point is barely closer than the
farthest. KNN degrades to random guessing.

### 5.3 Practical consequence

With $p$ features, the fraction of training data needed to maintain a fixed neighborhood
density grows as $n \propto r^p$. For $p = 20$, covering even 1% of the feature space
requires an astronomical number of samples.

---

## 6. Computational Complexity

### 6.1 Brute force

| Phase | Cost |
|---|---|
| Training (fit) | $O(np)$ — just store the data |
| Single query (predict) | $O(np)$ — compute $n$ distances, each costing $O(p)$ |
| Sorting/selection | $O(n \log K)$ — partial sort for $\text{top-}K$ |
| $m$ queries | $O(mnp)$ — linear in dataset size |

### 6.2 Acceleration structures

**KD-tree.** A binary tree that recursively partitions the feature space along coordinate
axes.

Average query time: $O(p \log n)$ for low $p$. Degrades to $O(np)$ for $p \gtrsim 20$.

**Ball tree.** Partitions data into nested hyperspheres. Works better than KD-trees in
moderate dimensions ($p \lesssim 50$), but still suffers from the curse of dimensionality.

**Approximate nearest neighbors (ANN).** Methods like locality-sensitive hashing (LSH)
trade exactness for speed: $O(p)$ per query with sublinear index lookups.

**Result:** For low-dimensional data ($p < 20$), tree structures give $O(\log n)$ query
time. For high-dimensional data, brute force or approximate methods are necessary.

---

## 7. Feature Scaling

KNN computes distances, so features with larger scales dominate. If feature 1 is in
$[0, 1000]$ and feature 2 is in $[0, 1]$, the distance is effectively determined by
feature 1 alone.

**Cure:** Standardize features to zero mean and unit variance before computing distances:

$$x_j' = \frac{x_j - \mu_j}{\sigma_j}.$$

This ensures each feature contributes proportionally to the distance.

---

## 8. Failure Cases

1. **High dimensionality.** Distances concentrate (§5) — KNN degrades to random guessing.
   **Cure:** Dimensionality reduction (PCA) or feature selection before KNN.

2. **Imbalanced classes.** If class A has 95% of training points, majority vote with any
   reasonable $K$ will almost always predict A.
   **Cure:** Weighted voting by inverse class frequency, or resampling.

3. **Feature scaling sensitivity.** Unscaled features bias distances toward high-magnitude
   features (§7). **Cure:** Standardize features.

4. **Computational cost.** Prediction is $O(np)$ per query — prohibitive for large $n$ or
   real-time applications. **Cure:** KD-trees, ball trees, or approximate methods (§6.2).

5. **Irrelevant features.** Every feature contributes equally to distance. Irrelevant
   features add noise to distance computations and degrade accuracy.
   **Cure:** Feature selection or weighted distance metrics.

---

## 9. Connections

- **Clustering (K-Means).** Both are distance-based. K-Means assigns points to the nearest
  *centroid*; KNN assigns to the majority class of nearest *data points*. K-Means is
  unsupervised; KNN is supervised.
- **Voronoi tessellation.** The 1-NN decision boundary partitions the feature space into
  Voronoi cells — each cell contains the points closest to one training point.
- **Kernel methods / SVM.** A kernel $K(x, z)$ can be seen as a similarity measure, just
  as $-d(x, z)$ is. Kernel density estimation is a continuous analogue of KNN counting.
- **Decision trees.** Both are non-parametric and create piecewise-constant predictions.
  Decision trees partition axis-aligned; KNN partitions by distance spheres.
- **Bias-variance tradeoff.** $K$ plays the same role as regularization strength $\lambda$
  in parametric models: controls the smoothness of the decision boundary.
- [Norms and Distances](https://github.com/hien078/applied-mathematics-foundation)
- [Clustering](../11_clustering/README.md)
- [Dimensionality Reduction](../12_dimensionality_reduction/README.md)
- [Model Selection](../synthesis/model_selection_guide.md)

---

## 10. References

- **Cover, T., & Hart, P. (1967).** Nearest neighbor pattern classification. *IEEE Transactions on Information Theory*, 13(1), 21–27. (Cover-Hart theorem: $R^\ast \le R_{1\text{-NN}} \le 2 R^\ast$).
- **Hastie, T., Tibshirani, R., & Friedman, J. (2009).** *The Elements of Statistical Learning: Data Mining, Inference, and Prediction* (2nd ed.). Springer. Chapter 13.3: $k$*-Nearest-Neighbor Classifiers*.
- **Bishop, C. M. (2006).** *Pattern Recognition and Machine Learning*. Springer. Chapter 2.5: *Non-parametric Methods*.
