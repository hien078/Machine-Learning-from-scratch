# Foundational Mathematical Models in Machine Learning

> A "model" in Machine Learning is in fact a combination of several pieces: the **hypothesis equation**, the **loss function**, and the **optimization algorithm** (optimizer). Just by swapping a function, adding a parameter, or ensembling the outputs of multiple models, you create a brand-new model. Every day researchers around the world keep publishing dozens more variants.

> Yet that vast Machine Learning "universe" is built on roughly **20+ foundational mathematical models**. From these core building blocks, all the more complex algorithms are developed.

---

## 1. Supervised Learning

*Learn a mapping f: X → Y from input data that already has labels (answers).*

| # | Model | Core idea | Underlying math |
|---|-------|-----------|-----------------|
| 1 | **Linear Regression** | Use the equation of a line / hyperplane (ŷ = Wx + b) to predict continuous values (house prices, temperature). | Normal equation, gradient descent |
| 2 | **Logistic Regression** | Use the Sigmoid function σ(z) = 1/(1+e⁻ᶻ) to squash values into [0, 1] → classification probability (spam or not). | Cross-entropy loss, MLE |
| 3 | **Support Vector Machine (SVM)** | Find a hyperplane in high-dimensional space with the **maximum margin** separating classes. | Hinge loss, KKT conditions, kernel trick |
| 4 | **Decision Tree** | Ask successive Yes/No questions, splitting the data by criteria that reduce information impurity. | Entropy, Gini impurity |
| 5 | **Naive Bayes** | Predict P(y\|x) via Bayes' theorem, assuming features are conditionally independent. | Bayes' theorem, MLE |
| 6 | **K-Nearest Neighbors (KNN)** | Predict via "majority vote" of the K closest data points — a lazy learner with no training phase. | Euclidean / Manhattan distance |

### Ensemble techniques (still Supervised Learning)

*Combine several base models into one that is stronger than the sum of its parts. This is an **ensembling technique**, not a separate learning paradigm.*

| # | Model | Core idea | Underlying math |
|---|-------|-----------|-----------------|
| 7 | **Random Forest** | Build hundreds of different Decision Trees (bagging + random feature sampling); the final output is the mean / vote of all trees. | Variance reduction, bootstrap sampling |
| 8 | **Gradient Boosting** (XGBoost, LightGBM) | Build models **sequentially**; each new model uses gradients to correct the errors of the previous one. | Functional gradient descent, 2nd-order Taylor expansion |

---

## 2. Unsupervised Learning

*Automatically discover hidden structure in the data without any labels.*

| # | Model | Core idea | Underlying math |
|---|-------|-----------|-----------------|
| 9 | **K-Means Clustering** | Group data into K clusters by iterating: assign points → update centroids. K must be chosen in advance. | Lloyd's algorithm, Voronoi partitions |
| 10 | **DBSCAN** | **Density-based** clustering — auto-detects the number of clusters, handles arbitrary cluster shapes, and removes noise. | ε-neighborhood, minPts, core/border/noise points |
| 11 | **Hierarchical Clustering** | Build a hierarchical tree (dendrogram) by repeatedly merging the closest cluster pair — no need to pre-pick K. | Linkage criteria (single, complete, Ward) |
| 12 | **GMM** (Gaussian Mixture Model) | Soft clustering — each point belongs to multiple clusters with different probabilities; each cluster is a Gaussian distribution. | EM algorithm, MLE, latent variables |
| 13 | **PCA** (Principal Component Analysis) | "Compress" data from a high-dimensional space into fewer dimensions while preserving maximum variance. | Eigendecomposition, SVD, covariance matrix |

---

## 3. Neural Networks & Deep Learning

*Stack linear transformations + non-linear activations into many layers. The number of possible architectures is essentially unlimited because they can be freely composed.*

| # | Model | Core idea | Underlying math |
|---|-------|-----------|-----------------|
| 14 | **Multi-Layer Perceptron (MLP)** | The most basic neural network: a chain of matrix multiplications interleaved with non-linear activations (ReLU, tanh). | Chain rule (backpropagation) |
| 15 | **CNN** (Convolutional Neural Network) | Use the **convolution** operation to extract spatial features — outstanding on images. | Cross-correlation, pooling, stride/padding |
| 16 | **RNN / LSTM / GRU** | Contain a mathematical loop giving the model "memory" — specialized for sequential data (text, time series). | BPTT, gating mechanisms (forget/input/output gates) |
| 17 | **Transformer** | The foundational architecture of ChatGPT, Gemini, and Claude. Uses the **Self-Attention** mechanism — computing how relevant every element in a sequence is to every other. | Attention = softmax(QKᵀ/√dₖ)V, positional encoding |

---

## 4. Generative Models

*Learn the data distribution p(x) in order to **generate new samples** (images, text, audio) that never existed in the training set.*

| # | Model | Core idea | Underlying math |
|---|-------|-----------|-----------------|
| 18 | **VAE** (Variational Autoencoder) | The encoder compresses data into a latent space; the decoder reconstructs it. Learns a continuous latent distribution → new samples are produced via sampling. | ELBO, KL divergence, reparameterization trick |
| 19 | **GAN** (Generative Adversarial Network) | An adversarial game: the Generator produces fake data, the Discriminator tries to tell real from fake — they compete until the Generator's output is indistinguishable. | Minimax game: min_G max_D, Nash equilibrium |
| 20 | **Diffusion Models** (DDPM) | Gradually add Gaussian noise to an image until it becomes pure noise (forward process), then learn how to **reverse** that process to generate images from noise. The foundation behind DALL·E and Stable Diffusion. | Markov chain, variational lower bound, score matching |

---

## 5. Reinforcement Learning

*An agent learns through the loop: observe state → choose action → receive reward → update strategy. The goal: maximize long-term cumulative reward.*

| # | Model | Core idea | Underlying math |
|---|-------|-----------|-----------------|
| 21 | **Q-Learning** (Value-based) | Learn a value function Q(s, a) — the expected reward of taking action a in state s. The agent picks the action with the highest Q. | MDP (Markov Decision Process), Bellman equation, TD learning |
| 22 | **Policy Gradient** (REINFORCE, PPO) | Optimize the policy π(a\|s) **directly** via gradient ascent on the expected-reward objective — no Q-value estimation needed. A fundamentally different approach from Q-Learning. | ∇J = E[∇ log π(a\|s) · Gₜ], advantage estimation |

> **Note:** MDP (Markov Decision Process) is the **mathematical framework** describing an RL problem (states, actions, rewards, transition probabilities) — it is not a model itself. Q-Learning and Policy Gradient are the two main algorithm families used to solve MDPs.

---

## Summary: The 5 Mathematical Pillars of Machine Learning

| Pillar | Role in ML | Examples |
|--------|-----------|----------|
| **Linear Algebra** | Represent data as matrices and vectors; the native language of every neural network. | Matrix multiplication in MLPs, SVD in PCA, attention QKᵀ |
| **Calculus** | Compute derivatives to determine the direction that reduces error — the foundation of backpropagation. | Chain rule, Jacobian, gradient descent |
| **Probability & Statistics** | Reason under uncertainty; estimate parameters from data. | Bayes' theorem, MLE/MAP, cross-entropy |
| **Optimization** | Find optimal model parameters — SGD, Adam, convex vs non-convex, constrained optimization. A distinct field, broader than just "applied Calculus". | SGD, Adam, KKT (SVM), EM algorithm |
| **Information Theory** | Measure information content and the divergence between distributions — essential for loss functions and generative models. | Entropy, KL divergence, mutual information, ELBO |

---

> **Advice:** You don't need to learn thousands of models. Mastering the mathematical essence of about **10–15 foundational algorithms** (Linear / Logistic Regression, Decision Tree, Random Forest, K-Means, PCA, MLP, basic CNN, …) is already enough to tackle **more than 80% of real-world problems**.
