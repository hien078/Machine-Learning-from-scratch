# Naive Bayes — Theory

## 0. Notation

| Symbol | Type | Meaning |
|---|---|---|
| $n$ | scalar | number of training examples |
| $d$ | scalar | number of features |
| $x_i$ | vector of length $d$ | feature vector of example $i$ |
| $x_{ij}$ | scalar | $j$-th feature of example $i$ |
| $y_i$ | scalar | class label of example $i$, $y_i \in \lbrace1, \dots, K\rbrace$ |
| $K$ | scalar | number of classes |
| $n_k$ | scalar | number of training examples in class $k$ |
| $\pi_k$ | scalar in $(0, 1)$ | prior probability $P(y = k)$ |
| $\mu_{kj}$ | scalar | mean of feature $j$ in class $k$ (Gaussian NB) |
| $\sigma^2_{kj}$ | scalar $> 0$ | variance of feature $j$ in class $k$ (Gaussian NB) |
| $\theta_{kj}$ | scalar in $[0, 1]$ | probability parameter for feature $j$ in class $k$ (Multinomial/Bernoulli NB) |
| $\alpha$ | scalar $\ge 0$ | Laplace smoothing parameter |

**Vector convention.** Lowercase = vector, uppercase = matrix. All log operations
are natural logarithm.

---

## 1. WHY — When Strong Assumptions Beat Flexible Models

Consider classifying emails as spam vs. not-spam. Each email is described by
thousands of word features. With limited training data relative to the feature
dimensionality, flexible models (logistic regression, neural networks) risk
overfitting. A model with a *strong structural assumption* can trade bias for
dramatically lower variance.

Naive Bayes takes the strongest possible assumption: **every feature is
conditionally independent of every other feature, given the class label**.
This assumption is almost always wrong — word co-occurrences are clearly
dependent — yet Naive Bayes works surprisingly well because:

1. **Classification only needs the ranking of $P(y \mid x)$ across classes**, not
   the actual probability values. Even with biased probability estimates, the
   correct class often still gets the highest score.
2. **Parameter estimation is trivially parallelizable** — each feature's
   distribution is estimated independently, requiring only one pass through the data.
3. **No iterative optimization** — parameters are computed in closed form from
   sufficient statistics (counts, means, variances).

---

## 2. WHAT — Bayes' Theorem and the Naive Assumption

### 2.1 Bayes' Theorem

For a class label $y$ and feature vector $x = (x_1, \dots, x_d)$:

```math
P(y = k \mid x) = \frac{P(x \mid y = k) \, P(y = k)}{P(x)}. \qquad (2.1)
```

- **Posterior** $P(y = k \mid x)$: what we want — the probability of class $k$ given the observed features.
- **Likelihood** $P(x \mid y = k)$: how likely these features are if the example belongs to class $k$.
- **Prior** $P(y = k)$: how common class $k$ is overall.
- **Evidence** $P(x) = \sum_{j=1}^{K} P(x \mid y = j) P(y = j)$: a normalizing constant, identical across classes.

Since $P(x)$ is the same for all classes, the classification rule only needs:

```math
\hat{y} = \arg\max_k \; P(x \mid y = k) \, P(y = k). \qquad (2.2)
```

### 2.2 The Naive Conditional Independence Assumption

The likelihood $P(x \mid y = k)$ is a $d$-dimensional joint density. Estimating it
directly requires exponential amounts of data. The **naive** assumption factorizes it:

$$P(x \mid y = k) = \prod_{j=1}^{d} P(x_j \mid y = k). \qquad (2.3)$$

Each feature is modeled by a **univariate** distribution conditioned on the class,
reducing the number of parameters from exponential to $O(K \cdot d)$.

### 2.3 The Decision Rule

Substituting (2.3) into (2.2):

$$\hat{y} = \arg\max_k \left[ P(y = k) \prod_{j=1}^{d} P(x_j \mid y = k) \right]. \qquad (2.4)$$

**Result:** The Naive Bayes classifier selects the class that maximizes the product
of the prior and the per-feature likelihoods.

---

## 3. HOW — Log-Space Computation

### 3.1 The Numerical Problem

Multiplying many probabilities (each $< 1$) causes underflow. With $d = 1000$
features, a product like $0.1^{1000} = 10^{-1000}$ is far below the smallest
representable float64 ($\approx 10^{-308}$).

### 3.2 Log-Posterior

Take the logarithm of the decision criterion (2.4):

$$\log P(y = k \mid x) + \text{const} = \log P(y = k) + \sum_{j=1}^{d} \log P(x_j \mid y = k). \qquad (3.1)$$

The argmax is preserved because $\log$ is monotonically increasing:

$$\hat{y} = \arg\max_k \left[ \log P(y = k) + \sum_{j=1}^{d} \log P(x_j \mid y = k) \right]. \qquad (3.2)$$

**Result:** Working in log-space converts products to sums, avoiding underflow.
This is mandatory for any practical implementation.

### 3.3 Log-Sum-Exp for Posterior Probabilities

When actual posterior probabilities (not just the argmax) are needed, compute the
normalizing constant via the **log-sum-exp** trick. Let $a_k = \log P(y = k) + \sum_j \log P(x_j \mid y = k)$:

$$\log P(y = k \mid x) = a_k - \log \sum_{j=1}^{K} e^{a_j} = a_k - \left[ a_{\max} + \log \sum_{j=1}^{K} e^{a_j - a_{\max}} \right],$$

where $a_{\max} = \max_j a_j$. Subtracting $a_{\max}$ before exponentiation prevents overflow.

---

## 4. Gaussian Naive Bayes

### 4.1 Model

For continuous features, assume each $P(x_j \mid y = k)$ is a **univariate Gaussian**
with class-specific mean $\mu_{kj}$ and variance $\sigma^2_{kj}$:

```math
P(x_j \mid y = k) = \frac{1}{\sqrt{2\pi \sigma^2_{kj}}} \exp\!\left( -\frac{(x_j - \mu_{kj})^2}{2\sigma^2_{kj}} \right). \qquad (4.1)
```

### 4.2 Log-Likelihood Contribution

$$\log P(x_j \mid y = k) = -\frac{1}{2} \log(2\pi \sigma^2_{kj}) - \frac{(x_j - \mu_{kj})^2}{2\sigma^2_{kj}}. \qquad (4.2)$$

### 4.3 Parameter Estimation (MAP)

Given training data $\lbrace(x_i, y_i)\rbrace_{i=1}^{n}$, estimate parameters by maximum
likelihood (which coincides with MAP under a flat prior):

$$\hat{\pi}_k = \frac{n_k}{n}, \qquad \hat{\mu}_{kj} = \frac{1}{n_k} \sum_{i: y_i = k} x_{ij}, \qquad \hat{\sigma}^2_{kj} = \frac{1}{n_k} \sum_{i: y_i = k} (x_{ij} - \hat{\mu}_{kj})^2. \qquad (4.3)$$

**Variance smoothing.** To avoid division by zero when a feature is constant within
a class, add a small positive value $\epsilon$ to all variances:
$\hat{\sigma}^2_{kj} \leftarrow \hat{\sigma}^2_{kj} + \epsilon$.

**Result:** Gaussian NB requires one pass through the data to compute class counts,
per-class means, and per-class variances. No iterative optimization needed.

---

## 5. Multinomial Naive Bayes

### 5.1 Model

For count/frequency data (e.g., word counts in text), model each class as a
**multinomial** distribution. Let $x_j$ be the count of feature $j$:

$$P(x \mid y = k) \propto \prod_{j=1}^{d} \theta_{kj}^{x_j}, \qquad (5.1)$$

where $\theta_{kj} = P(\text{feature } j \mid y = k)$ and $\sum_j \theta_{kj} = 1$.

### 5.2 Parameter Estimation with Laplace Smoothing

The MLE is $\hat\theta_{kj} = N_{kj} / N_k$ where $N_{kj} = \sum_{i: y_i = k} x_{ij}$
is the total count of feature $j$ in class $k$, and $N_k = \sum_j N_{kj}$.

**The zero-frequency problem.** If feature $j$ never appears in class $k$, then
$\hat{\theta}_{kj} = 0$, and any test example with $x_j > 0$ gets
$P(x \mid y = k) = 0$ — one missing word kills the entire class probability.

**Laplace ($\text{add-}\alpha$) smoothing:**

$$\hat{\theta}_{kj} = \frac{N_{kj} + \alpha}{N_k + \alpha \cdot d}, \qquad \alpha > 0. \qquad (5.2)$$

**Why $\alpha \cdot d$ in the denominator?** Adding $\alpha$ pseudo-counts to each of the $d$ feature categories increases the total count in class $k$ by $\sum_{j=1}^d \alpha = \alpha \cdot d$. This exact normalization guarantees that the parameters form a valid probability distribution summing to 1: $\sum_{j=1}^d \hat\theta_{kj} = \frac{\sum_{j=1}^d (N_{kj} + \alpha)}{N_k + \alpha d} = \frac{N_k + \alpha d}{N_k + \alpha d} = 1$.

With $\alpha = 1$ (Laplace smoothing), this is equivalent to placing a symmetric Dirichlet prior $\text{Dir}(\alpha, \dots, \alpha)$ on $\theta_k$ and computing the MAP estimate.

**Result:** Laplace smoothing ensures every feature has nonzero probability in every class, preventing a single unseen feature from zeroing out a class.

---

## 6. Bernoulli Naive Bayes

### 6.1 Model

For binary features $x_j \in \lbrace0, 1\rbrace$:

$$P(x_j \mid y = k) = \theta_{kj}^{x_j} (1 - \theta_{kj})^{1 - x_j}. \qquad (6.1)$$

Unlike Multinomial NB, Bernoulli NB explicitly models the *absence* of a feature (the $(1 - \theta_{kj})$ term). This makes it more suitable when feature absence carries information (e.g., a spam word being absent is evidence against spam).

### 6.2 Parameter Estimation

$$\hat{\theta}_{kj} = \frac{(\text{count of examples in class } k \text{ with } x_j = 1) + \alpha}{n_k + 2\alpha}. \qquad (6.2)$$

The denominator uses $2\alpha$ because each binary feature has exactly 2 possible states ($x_j = 1$ and $x_j = 0$), so adding $\alpha$ to each state adds $2\alpha$ to the total count.

---

## 7. Variant Comparison

| Variant | Feature type | Likelihood model | Smoothing | Use case |
|---|---|---|---|---|
| **Gaussian** | Continuous | Univariate Normal | Variance floor $\epsilon$ | Sensor readings, measurements |
| **Multinomial** | Counts / frequencies | Multinomial | $\text{Add-}\alpha$ | Text classification (bag of words) |
| **Bernoulli** | Binary (0/1) | Bernoulli | $\text{Add-}\alpha$ | Binary text features, presence/absence |

---

## 8. Generative vs. Discriminative

Naive Bayes is a **generative** classifier: it models the joint distribution $P(x, y) = P(x \mid y) P(y)$ and uses Bayes' rule to infer $P(y \mid x)$.

Logistic regression is a **discriminative** classifier: it models $P(y \mid x)$ directly as $\sigma(\theta^T x)$ without modeling $P(x \mid y)$.

**Theorem (Ng & Jordan, 2001).** Under the naive Bayes model assumptions (class-conditional feature independence with exponential-family likelihoods), the posterior $P(y \mid x)$ has the logistic (sigmoid/softmax) form. Naive Bayes and logistic regression are therefore a **generative-discriminative pair** — they share the same model family but differ in parameter estimation:

| Aspect | Naive Bayes | Logistic Regression |
|---|---|---|
| Estimates | $P(x \mid y)$ and $P(y)$ | $P(y \mid x)$ directly |
| Parameters | Closed-form (counts/means) | Iterative (gradient descent) |
| Asymptotic | Lower asymptotic accuracy | Higher asymptotic accuracy |
| Sample efficiency | Converges faster with few samples | Needs more data |
| Independence assumption | Required | Not required |

---

## 9. Failure Cases

### 9.1 Correlated Features Violate Independence

When features are highly correlated, the naive assumption double-counts evidence. Example: if $x_1$ and $x_2$ are copies of the same feature, Naive Bayes treats them as two independent pieces of evidence, making the posterior over-confident.

**Effect:** Probability estimates are badly calibrated (too close to 0 or 1), though classification accuracy may still be reasonable.

### 9.2 Continuous Features with Non-Gaussian Distribution

Gaussian NB assumes each feature is normally distributed within each class. If the true distribution is multimodal, heavy-tailed, or skewed, the Gaussian likelihood assigns wrong density values.

**Cure:** Transform features (log, Box-Cox), discretize into bins, or use kernel density estimation.

### 9.3 Zero Variance Features

If a feature is constant within a class, the Gaussian variance is zero and the density is undefined ($1 / \sqrt{0}$). Variance smoothing ($\epsilon > 0$) is required.

### 9.4 Unseen Feature Values (Discrete NB)

Without smoothing, a single unseen feature-value pair produces $P(x_j \mid y = k) = 0$, which zeros out the entire class posterior regardless of all other features. Laplace smoothing (§5.2) is the standard fix.

---

## 10. Connections

- **[Probability & Statistics](https://github.com/hien078/applied-mathematics-foundation):** Bayes' theorem, MLE, MAP estimation — the mathematical foundation.
- **[Logistic Regression](../04_logistic_regression/README.md):** The discriminative counterpart. Same posterior form (sigmoid/softmax), different parameter estimation (iterative vs. closed-form).
- **[Information Theory](https://github.com/hien078/applied-mathematics-foundation):** Cross-entropy loss in logistic regression connects to KL divergence between the empirical and model distributions.
- **[Probabilistic View](../synthesis/probabilistic_view_of_ml.md):** Naive Bayes illustrates the generative modeling paradigm — model $P(x, y)$, then derive $P(y \mid x)$ via Bayes' rule.

---

## 11. References

- **Ng, A. Y., & Jordan, M. I. (2001).** On discriminative vs. generative classifiers: A comparison of logistic regression and naive bayes. *Advances in Neural Information Processing Systems (NIPS)*, 14, 841–848.
- **Bishop, C. M. (2006).** *Pattern Recognition and Machine Learning*. Springer. Chapter 4.2: *Probabilistic Generative Models*.
- **McCallum, A., & Nigam, K. (1998).** A comparison of event models for Naive Bayes text classification. *AAAI-98 Workshop on Learning for Text Categorization*, 752, 41–48.

