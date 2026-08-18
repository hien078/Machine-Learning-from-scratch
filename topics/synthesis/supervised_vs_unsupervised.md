# Supervised vs Unsupervised — Cross-Topic Synthesis

> A taxonomy of all 22 topics by supervision signal, and how the same math reappears across paradigms.
> See [INDEX.md](../../INDEX.md) for the full curriculum index.

---

## The Four Paradigms

| Paradigm | Observed training signal | Typical objective | Examples in this repo |
|---|---|---|---|
| Supervised | Inputs and targets $(x,y)$ | Predict $y$ from $x$ | Regression, classification, most of topics 01–09 |
| Unsupervised | Inputs $x$ only | Describe structure in $p(x)$ or geometry | PCA, clustering, density models |
| Self-supervised | Targets constructed from $x$ itself | Predict masked, future, or transformed parts | Language-model pretraining, contrastive learning |
| Reinforcement | Scalar reward from interaction | Maximize expected return | Q-learning, policy gradients |

---

## What "Label" Means in Each Paradigm

The paradigms differ less in their math than in *where the target comes from*:

- **Supervised:** the target is *external* — annotated by a human or upstream process.
  Expensive, but the objective directly measures the task.
- **Unsupervised:** no target; the objective is *self-referential* — fit $p(x)$,
  minimize distortion, preserve variance or neighborhoods.
- **Self-supervised:** the target is *manufactured from the input* — mask a token,
  corrupt an image, decide which augmented view shares an origin. Labels are free; the
  bet is that the pretext task forces representations useful elsewhere.
- **Reinforcement:** the signal is a *reward* — scalar, often delayed, and evaluative
  rather than instructive: it says how good the action was, never what the correct
  action would have been. Credit assignment replaces label lookup.

---

## All 22 Topics Classified

| Topic | Paradigm | Training signal | Core objective |
|---|---|---|---|
| [01 Linear Regression](../01_linear_regression/README.md) | Supervised | Real-valued $y$ | MSE |
| [02 Gradient Descent](../02_gradient_descent/README.md) | — (optimizer, paradigm-agnostic) | Any differentiable loss | Loss minimization |
| [03 Regularization](../03_regularization/README.md) | — (objective modifier) | Any loss + penalty | Loss $+\lambda\,\Omega(w)$ |
| [04 Logistic Regression](../04_logistic_regression/README.md) | Supervised | Class label | Binary cross-entropy |
| [05 Decision Tree](../05_decision_tree/README.md) | Supervised | Class label / real $y$ | Impurity reduction |
| [06 Ensemble Methods](../06_ensemble_methods/README.md) | Supervised | Class label / real $y$ | Aggregated base-learner loss |
| [07 KNN](../07_knn/README.md) | Supervised | Class label / real $y$ | None (memorize; vote at query) |
| [08 Naive Bayes](../08_naive_bayes/README.md) | Supervised (generative) | Class label | Joint likelihood $p(x,y)$ |
| [09 SVM](../09_svm/README.md) | Supervised | Class label $y\in\lbrace-1,+1\rbrace$ | Hinge loss + margin |
| [10 PCA](../10_pca/README.md) | Unsupervised | None | Variance / reconstruction error |
| [11 Clustering](../11_clustering/README.md) | Unsupervised | None | Distortion (K-Means), likelihood (GMM) |
| [12 Dimensionality Reduction](../12_dimensionality_reduction/README.md) | Unsupervised | None | Neighborhood preservation |
| [13 Neural Networks](../13_neural_networks/README.md) | Supervised (architecture is paradigm-agnostic) | Class label / real $y$ | Cross-entropy / MSE |
| [14 CNN](../14_cnn/README.md) | Supervised (architecture) | Class label | Cross-entropy |
| [15 RNN/LSTM](../15_rnn_lstm/README.md) | Supervised / self-supervised | Next element of the sequence | Sequence cross-entropy |
| [16 Transformer](../16_transformer/README.md) | Self-supervised (pretrain) + supervised (fine-tune) | Next/masked token | Cross-entropy |
| [17 Autoencoder](../17_autoencoder/README.md) | Self-supervised (input is its own target) | $x$ itself | Reconstruction; $-\mathrm{ELBO}$ (VAE) |
| [18 Reinforcement Learning](../18_reinforcement_learning/README.md) | Reinforcement | Reward $r_t$ | Expected return |
| [19 Generative Models](../19_generative_models/README.md) | Unsupervised / self-supervised | None ($x$ only) | Likelihood, ELBO, adversarial, denoising |
| [20 Graph Neural Networks](../20_graph_neural_networks/README.md) | Supervised (also SSL pretraining) | Node/graph labels | Cross-entropy over graph structure |
| [21 LLM Engineering](../21_llm_engineering/README.md) | All four in one pipeline | Tokens → preferences → rewards | CE pretrain, supervised fine-tune, RLHF |
| [22 Self-Supervised Learning](../22_self_supervised_learning/README.md) | Self-supervised | Constructed pretext targets | InfoNCE, masked prediction |

Autoencoders sit on the unsupervised/self-supervised boundary: no external label exists,
yet training is a standard supervised regression with target $y:=x$. The taxonomy
classifies the *signal source*, not the loss shape — which is exactly why the same math
keeps reappearing.

---

## Same Math, Different Paradigm

| Loss shape | Supervised home | Reappears as |
|---|---|---|
| MSE $\Vert y-\hat y\Vert_2^2$ | Linear regression | Autoencoder reconstruction ($y:=x$); PCA (linear AE); value-function regression in RL |
| Categorical cross-entropy | Softmax classifier | Next-token prediction (class = vocabulary entry); masked-token prediction; InfoNCE |
| Log-likelihood $\log p_\theta(\cdot)$ | Naive Bayes joint fit | GMM density fit; VAE ELBO; policy gradient's $\log\pi_\theta(a\mid s)$ weighted by return |
| Distortion to prototypes | KNN/nearest-centroid classification | K-Means; vector quantization in discrete autoencoders |

Two identities are worth internalizing:

- **Reconstruction is prediction with $y:=x$.** The autoencoder objective
  $\Vert x-g(f(x))\Vert_2^2$ is [Topic 01](../01_linear_regression/theory.md)'s
  loss with the input as its own target; restricting $f,g$ to linear maps recovers PCA.
- **InfoNCE is classification in disguise.** For anchor $z_i$, positive $z_i^{+}$, and
  negatives $\lbrace z_j\rbrace$,

```math
\mathcal{L}_{\mathrm{InfoNCE}}=-\log\frac{\exp\left(\mathrm{sim}(z_i,z_i^{+})/\tau\right)}{\sum_{j}\exp\left(\mathrm{sim}(z_i,z_j)/\tau\right)},
```

  which is exactly softmax cross-entropy for an $N$-way classification problem whose
  "class" is *which candidate is the positive pair* — a label manufactured by data
  augmentation ([22 Self-Supervised Learning](../22_self_supervised_learning/theory.md)).
  Likewise, a language model is "just" a classifier over the vocabulary whose labels
  come free from the corpus ([16 Transformer](../16_transformer/theory.md)).

---

## The Semi-Supervised Bridge

Real datasets are usually a few labeled examples plus a large unlabeled pool. The
bridges between paradigms:

- **Pretrain, then fine-tune** — the dominant modern form: self-supervised pretraining
  on unlabeled data, supervised fine-tuning on the small labeled set
  ([21 LLM Engineering](../21_llm_engineering/README.md)).
- **Pseudo-labeling / self-training:** a supervised model labels the unlabeled pool and
  retrains on its own confident predictions.
- **Consistency regularization:** predictions must agree across augmentations of the
  same input — supervision without labels.
- **Generative assistance:** modeling $p(x)$ helps $p(y\mid x)$ under the cluster
  assumption — decision boundaries should pass through low-density regions of $p(x)$.

---

## Evaluation Changes Without Labels

There is no task target to determine a unique notion of success. Cluster compactness,
reconstruction error, likelihood, and neighborhood preservation answer different
questions. An unsupervised score can improve while downstream usefulness worsens.

Common evaluation errors:

- interpreting clusters as real categories without external evidence;
- tuning an unsupervised representation on test labels;
- judging t-SNE by apparent visual separation alone;
- reporting only training accuracy for supervised models;
- calling a reconstruction objective proof that the latent representation is meaningful.

---

## Connections

- **Topics:** [10 PCA](../10_pca/README.md), [11 Clustering](../11_clustering/README.md), [12 Dimensionality Reduction](../12_dimensionality_reduction/README.md), [17 Autoencoder](../17_autoencoder/README.md), [18 Reinforcement Learning](../18_reinforcement_learning/README.md), [22 Self-Supervised Learning](../22_self_supervised_learning/README.md)
- **Related synthesis:** [Probabilistic View of ML](probabilistic_view_of_ml.md), [Loss Functions Map](loss_functions_map.md), [Model Selection Guide](model_selection_guide.md)
- **Maps:** [INDEX.md](../../INDEX.md)
