# Clustering — Theory

> **Purpose:** Pure theory, definitions, and derivations for unsupervised clustering.
> Read this first, then open `first_principles.ipynb` for computation and experiments.

## Prerequisites

- [Foundations: Linear Algebra](https://github.com/hien078/applied-mathematics-foundation)
- [Foundations: Probability & Statistics](https://github.com/hien078/applied-mathematics-foundation)

---

## 0. Notation

| Symbol | Type | Meaning |
|---|---|---|
| $n$ | scalar $\in \mathbb{N}$ | number of data points |
| $d$ | scalar $\in \mathbb{N}$ | number of features (dimensionality) |
| $K$ | scalar $\in \mathbb{N}$ | number of clusters |
| $x_i$ | vector $\in \mathbb{R}^d$ | feature vector of the $i$-th point |
| $X$ | matrix $\in \mathbb{R}^{n \times d}$ | data matrix; row $i$ is $x_i^\top$ |
| $c_i$ | scalar $\in \{1, \dots, K\}$ | cluster assignment of point $i$ |
| $C_k$ | set | set of indices assigned to cluster $k$: $C_k = \{i : c_i = k\}$ |
| $\mu_k$ | vector $\in \mathbb{R}^d$ | centroid (mean) of cluster $k$ |
| $\pi_k$ | scalar $\in (0, 1)$ | mixture weight for component $k$; $\sum_k \pi_k = 1$ |
| $\Sigma_k$ | matrix $\in \mathbb{R}^{d \times d}$ | covariance matrix of component $k$ |
| $r_{ik}$ | scalar $\in [0, 1]$ | responsibility: posterior probability that point $i$ belongs to component $k$ |
| $\varepsilon$ | scalar $> 0$ | DBSCAN neighbourhood radius |
| $N_\varepsilon(x_i)$ | set | $\varepsilon$-neighbourhood: $\{x_j : \Vert x_j - x_i\Vert  \le \varepsilon\}$ |
| $\text{minPts}$ | scalar $\in \mathbb{N}$ | DBSCAN minimum density threshold |
| $J$ | scalar | K-Means objective (inertia) |
| $\mathcal{L}$ | scalar | log-likelihood of the GMM |

**Conventions.**

- All vectors are column vectors. $\|\cdot\|$ denotes the Euclidean ($\ell^2$) norm.
- $\mathcal{N}(x; \mu, \Sigma)$ is the multivariate Gaussian density evaluated at $x$.

---

## 1. The Problem — WHY

Given $n$ data points $\{x_1, \dots, x_n\}$ with **no labels**, we want to discover natural groups (clusters) in the data. This arises whenever we need to:

- **Segment** customers, images, or documents into meaningful categories.
- **Compress** data by replacing each point with its cluster representative.
- **Pre-process** for downstream supervised tasks (e.g., feature engineering).

The challenge: "cluster" has no single mathematical definition. Different algorithms encode different inductive biases — centroid distance, density connectivity, or probabilistic membership.

---

## 2. K-Means — WHAT

### 2.1 Objective function

K-Means seeks assignments $c = (c_1, \dots, c_n)$ and centroids $\mu = (\mu_1, \dots, \mu_K)$ that minimise the **within-cluster sum of squares (WCSS)**, also called **inertia**:

$$J(c, \mu) = \sum_{i=1}^n \|x_i - \mu_{c_i}\|^2 = \sum_{k=1}^K \sum_{i \in C_k} \|x_i - \mu_k\|^2$$

This is equivalent to minimising the total squared distance from each point to its assigned centroid.

### 2.2 K-Means as coordinate descent

The joint optimisation over $(c, \mu)$ is NP-hard in general. Lloyd's algorithm solves it by **coordinate descent** — alternating between optimising one variable while holding the other fixed.

**Step 1 — Assignment (fix $\mu$, optimise $c$).** For each point $i$, choose the nearest centroid:

$$c_i = \arg\min_{k \in \{1, \dots, K\}} \|x_i - \mu_k\|^2$$

This minimises $J$ with respect to $c$ because each term depends only on its own $c_i$.

**Step 2 — Update (fix $c$, optimise $\mu$).** For each cluster $k$, find the centroid that minimises the sum of squared distances to its members.

> **Derivation.** Fix $k$ and consider the sub-problem:
>
> $$\min_{\mu_k} \sum_{i \in C_k} \|x_i - \mu_k\|^2$$
>
> Expand: $\sum_{i \in C_k} (x_i - \mu_k)^\top (x_i - \mu_k)$. Take the gradient with respect to $\mu_k$:
>
> $$\nabla_{\mu_k} \sum_{i \in C_k} \|x_i - \mu_k\|^2 = -2 \sum_{i \in C_k} (x_i - \mu_k) = 0$$
>
> Solving: $|C_k| \cdot \mu_k = \sum_{i \in C_k} x_i$

**Result:** $\mu_k = \frac{1}{|C_k|} \sum_{i \in C_k} x_i$ — the arithmetic mean of the assigned points.

### 2.3 Lloyd's algorithm

1. **Initialise** centroids $\mu_1, \dots, \mu_K$ (e.g., random selection from data).
2. **Assign** each point to its nearest centroid (Step 1).
3. **Update** each centroid to the mean of its cluster (Step 2).
4. **Repeat** steps 2–3 until assignments no longer change or a maximum iteration count is reached.

### 2.4 Convergence guarantee

> **Theorem.** Lloyd's algorithm converges in a finite number of steps.

**Proof sketch.** Each step (assign or update) does not increase $J$. Since $J \ge 0$ and there are finitely many possible assignment vectors $c \in \{1, \dots, K\}^n$, the algorithm must terminate. $\blacksquare$

**Caveat:** Convergence is to a **local** minimum, not necessarily the global one. The final result depends on initialisation.

### 2.5 K-Means++ initialisation

Random initialisation often leads to poor local minima. K-Means++ selects initial centroids that are spread apart:

1. Choose $\mu_1$ uniformly at random from $\{x_1, \dots, x_n\}$.
2. For $k = 2, \dots, K$:
   - Compute $D(x_i) = \min_{j < k} \|x_i - \mu_j\|^2$ for each point.
   - Select $\mu_k = x_i$ with probability proportional to $D(x_i)$.

> **Theorem (Arthur & Vassilvitskii, 2007).** K-Means++ initialisation guarantees $\mathbb{E}[J] \le 8(\ln K + 2) \cdot J^\ast$, where $J^\ast$ is the optimal objective.

**Result:** K-Means++ provides an $O(\log K)$-competitive initialisation in expectation.

### 2.6 Choosing K — the Elbow method

The inertia $J$ always decreases as $K$ increases (trivially, $J = 0$ when $K = n$). The **elbow method** plots $J$ vs. $K$ and looks for a "bend" where the marginal decrease sharply slows — indicating that additional clusters add little explanatory power.

---

## 3. DBSCAN — Density-Based Clustering

### 3.1 Core definitions

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) defines clusters as dense regions separated by sparse areas. Two parameters control the density threshold:

- $\varepsilon > 0$: neighbourhood radius.
- $\text{minPts} \in \mathbb{N}$: minimum number of points in the $\varepsilon$-ball.

**Point classification:**

| Type | Condition | Role |
|---|---|---|
| **Core point** | $\vert N_\varepsilon(x_i)\vert  \ge \text{minPts}$ | Forms the dense backbone of a cluster |
| **Border point** | Not core, but $\exists$ core point $x_j$ with $\Vert x_i - x_j\Vert  \le \varepsilon$ | Belongs to a cluster but does not expand it |
| **Noise point** | Neither core nor border | Not assigned to any cluster |

### 3.2 Algorithm

1. For each unvisited point $x_i$:
   a. Compute $N_\varepsilon(x_i)$.
   b. If $|N_\varepsilon(x_i)| < \text{minPts}$, label $x_i$ as noise (may be re-labelled later as border).
   c. Otherwise, $x_i$ is a core point — start a new cluster and **expand**:
      - Add all points in $N_\varepsilon(x_i)$ to the cluster.
      - For each new core point found, recursively add its $\varepsilon$-neighbours.
2. Continue until all points are visited.

### 3.3 Key properties

- **No need to specify $K$**: the number of clusters is determined by the data and parameters.
- **Arbitrary shapes**: clusters are defined by density connectivity, not by distance to a centroid.
- **Noise detection**: points in low-density regions are explicitly labelled as noise.

---

## 4. Gaussian Mixture Models (GMM) and EM

### 4.1 The generative model

A GMM assumes the data is generated from a mixture of $K$ Gaussian components:

$$p(x) = \sum_{k=1}^K \pi_k \, \mathcal{N}(x; \mu_k, \Sigma_k)$$

where each component density is:

$$\mathcal{N}(x; \mu_k, \Sigma_k) = \frac{1}{(2\pi)^{d/2} |\Sigma_k|^{1/2}} \exp\!\left( -\frac{1}{2} (x - \mu_k)^\top \Sigma_k^{-1} (x - \mu_k) \right)$$

The parameters are $\theta = \{\pi_k, \mu_k, \Sigma_k\}_{k=1}^K$.

### 4.2 The log-likelihood

Given observed data $X = \{x_1, \dots, x_n\}$, the log-likelihood is:

$$\mathcal{L}(\theta) = \sum_{i=1}^n \log \left( \sum_{k=1}^K \pi_k \, \mathcal{N}(x_i; \mu_k, \Sigma_k) \right)$$

Direct maximisation is intractable because the log is outside the sum. The **Expectation-Maximisation (EM)** algorithm circumvents this.

### 4.3 EM algorithm

**E-step — compute responsibilities.** Using Bayes' rule, compute the posterior probability that point $i$ belongs to component $k$:

$$r_{ik} = \frac{\pi_k \, \mathcal{N}(x_i; \mu_k, \Sigma_k)}{\sum_{j=1}^K \pi_j \, \mathcal{N}(x_i; \mu_j, \Sigma_j)}$$

**M-step — update parameters.** Let $N_k = \sum_{i=1}^n r_{ik}$ be the effective number of points in component $k$.

> **Derivation of M-step updates.** Setting the derivative of the expected complete-data log-likelihood with respect to each parameter to zero:
>
> For $\mu_k$: differentiate $\sum_i r_{ik} \log \mathcal{N}(x_i; \mu_k, \Sigma_k)$ w.r.t. $\mu_k$ and set to zero.
> The key term is $\sum_i r_{ik} \, \Sigma_k^{-1}(x_i - \mu_k) = 0$, giving $\mu_k = \frac{1}{N_k} \sum_i r_{ik} \, x_i$.
>
> For $\Sigma_k$: similarly, $\Sigma_k = \frac{1}{N_k} \sum_i r_{ik} (x_i - \mu_k)(x_i - \mu_k)^\top$.
>
> For $\pi_k$: maximise subject to $\sum_k \pi_k = 1$ using a Lagrange multiplier, giving $\pi_k = N_k / n$.

**Result:** The M-step updates are:

$$\mu_k = \frac{\sum_{i=1}^n r_{ik} \, x_i}{N_k}, \qquad \Sigma_k = \frac{\sum_{i=1}^n r_{ik} (x_i - \mu_k)(x_i - \mu_k)^\top}{N_k}, \qquad \pi_k = \frac{N_k}{n}$$

### 4.4 Convergence

> **Theorem.** The EM algorithm monotonically increases the log-likelihood: $\mathcal{L}(\theta^{(t+1)}) \ge \mathcal{L}(\theta^{(t)})$.

**Proof sketch.** EM maximises a lower bound (the ELBO) on $\mathcal{L}$. The E-step tightens the bound; the M-step increases it. Since the bound touches $\mathcal{L}$ at the current parameters, neither step can decrease $\mathcal{L}$. $\blacksquare$

**Caveat:** Like K-Means, EM converges to a local maximum — multiple restarts are recommended.

### 4.5 K-Means as a special case of EM

K-Means can be viewed as a degenerate GMM where:
- All covariances are $\Sigma_k = \sigma^2 I$ with $\sigma \to 0$.
- Responsibilities become hard assignments: $r_{ik} \in \{0, 1\}$.
- Mixture weights are uniform.

---

## 5. Comparison of Clustering Methods

| Property | K-Means | DBSCAN | GMM (EM) |
|---|---|---|---|
| Cluster shape | Spherical (Voronoi cells) | Arbitrary | Ellipsoidal |
| Must specify $K$? | Yes | No ($\varepsilon$, minPts instead) | Yes |
| Soft assignments? | No (hard) | No (hard + noise) | Yes (responsibilities $r_{ik}$) |
| Handles noise? | No | Yes (explicit noise label) | Not directly |
| Scalability | $O(nKd)$ per iteration | $O(n^2)$ naive, $O(n \log n)$ with spatial index | $O(nKd^2)$ per iteration |
| Convergence | Local minimum of $J$ | Deterministic (given $\varepsilon$, minPts) | Local maximum of $\mathcal{L}$ |

---

## 6. Failure Cases

### 6.1 K-Means failures

| Failure | Cause | Diagnostic |
|---|---|---|
| Non-spherical clusters | Objective assumes equal-variance spherical clusters | Visualise; try GMM or DBSCAN |
| Unequal cluster sizes | Larger clusters dominate the objective | Inspect cluster populations |
| Wrong $K$ | Elbow is ambiguous for some data | Try silhouette score, gap statistic |
| Sensitivity to initialisation | Converges to local minima | Use K-Means++, run multiple restarts |

### 6.2 DBSCAN failures

| Failure | Cause | Diagnostic |
|---|---|---|
| Varying density | Single $\varepsilon$ cannot capture both dense and sparse regions | Try OPTICS or HDBSCAN |
| High dimensions | Distances become less meaningful (curse of dimensionality) | Reduce dimensionality first (PCA) |
| Parameter sensitivity | Results change drastically with $\varepsilon$ and minPts | Use k-distance plot to guide $\varepsilon$ selection |

### 6.3 GMM failures

| Failure | Cause | Diagnostic |
|---|---|---|
| Covariance collapse | A component collapses onto a single point ($\vert \Sigma_k\vert  \to 0$) | Add regularisation: $\Sigma_k \leftarrow \Sigma_k + \epsilon I$ |
| Non-Gaussian data | Model assumes Gaussian components | Visualise; consider kernel methods |
| Too many components | Overfits; some components become redundant | Use BIC/AIC for model selection |

---

## 7. Connections

- **Distance foundation:** [Norms and Distances](https://github.com/hien078/applied-mathematics-foundation) — K-Means and DBSCAN both rely on Euclidean distance.
- **Probability foundation:** [Probability & Statistics](https://github.com/hien078/applied-mathematics-foundation) — GMM builds on multivariate Gaussians and Bayes' rule.
- **Dimensionality reduction:** [10 PCA](../10_pca/README.md) — often used as a pre-processing step before clustering; PCA + K-Means is a common pipeline.
- **Distance-based supervised:** [07 KNN](../07_knn/README.md) — shares the Euclidean distance primitive; KNN is supervised while K-Means is unsupervised.
- **Synthesis:** [Supervised vs. Unsupervised](../../synthesis/supervised_vs_unsupervised.md) — clustering is the prototypical unsupervised task.
- **Graph Map:** See [INDEX.md](../../INDEX.md)

---

## 8. References

- **MacQueen, J. (1967).** Some methods for classification and analysis of multivariate observations. *Proceedings of the 5th Berkeley Symposium on Mathematical Statistics and Probability*, 1, 281–297.
- **Arthur, D., & Vassilvitskii, S. (2007).** k-means++: The advantages of careful seeding. *Proceedings of the 18th Annual ACM-SIAM Symposium on Discrete Algorithms*, 1027–1035.
- **Ester, M., Kriegel, H. P., Sander, J., & Xu, X. (1996).** A density-based algorithm for discovering clusters in large spatial databases with noise. *KDD-96*, 96(34), 226–231.
- **Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977).** Maximum likelihood from incomplete data via the EM algorithm. *Journal of the Royal Statistical Society: Series B (Methodological)*, 39(1), 1–22.
- **Bishop, C. M. (2006).** *Pattern Recognition and Machine Learning*. Springer. Chapter 9: *Mixture Models and EM*.

