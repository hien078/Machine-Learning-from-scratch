# Math for AI/ML — A Roadmap by Real-World Necessity

> Ranked by **how often you actually use it** in modern AI/ML (2026), not by mathematical "difficulty" or "beauty".
> All math notation uses Unicode (∑ ∫ ∂ ∇ → ≈), no LaTeX.

---

## Table of Contents

1. [How to use this document](#how-to-use-this-document)
2. [Quick ranking table](#quick-ranking-table)
3. [Grouping by priority](#grouping-by-priority)
4. Detail per area:
   - [1. Linear Algebra](#1-linear-algebra)
   - [2. Multivariable Calculus](#2-multivariable-calculus)
   - [3. Probability & Statistics](#3-probability--statistics)
   - [4. Optimization](#4-optimization)
   - [5. Information Theory](#5-information-theory)
   - [6. Numerical Methods](#6-numerical-methods)
   - [7. Discrete Math & Graph Theory](#7-discrete-math--graph-theory)
   - [8. Differential Equations](#8-differential-equations)
   - [9. Functional Analysis](#9-functional-analysis)
   - [10. Differential Geometry & Topology](#10-differential-geometry--topology)
   - [11. Abstract Algebra](#11-abstract-algebra)
   - [12. Measure Theory](#12-measure-theory)
   - [13. Math for the LLM Era (2024-2026)](#13-math-for-the-llm-era-2024-2026)
5. [Applied math "beyond the table" — worth learning once your foundation is solid](#applied-math-beyond-the-table)
6. [Meta pitfalls when learning math for AI](#meta-pitfalls-when-learning-math-for-ai)
7. [Recommended 12-month roadmap (3 paths)](#recommended-12-month-roadmap)
8. [Glossary of key terms](#glossary-of-key-terms)

---

## How to use this document

- **Do not read sequentially.** Identify your goal (engineer / research / a specific subfield), then study only the depth you need in each area.
- **Learn through projects, not through books.** Each area has a "self-check" — if you can't do it, you're not there yet.
- **Use it or lose it.** Studying areas (10)(11)(12) without a project that uses them = forgotten within 6 months.
- **"Enough" is relative.** Enough for an engineer ≠ enough for a research scientist ≠ enough for a theory researcher.

---

## Quick ranking table

| # | Area | Usage frequency | Priority | Best learned through | Where you hit it most |
|---|---|---|---|---|---|
| 1 | Linear Algebra | Daily | 🔴 Must-have | Code + textbook | Every model |
| 2 | Multivariable Calculus | Daily | 🔴 Must-have | Backprop from scratch | Training |
| 3 | Probability & Statistics | Weekly | 🔴 Must-have | Implementing distributions | Loss, generative, eval |
| 4 | Optimization | Weekly | 🔴 Must-have | Implementing an optimizer | Every training loop |
| 5 | Information Theory | Often | 🟠 Sooner or later | Cross-entropy, KL from scratch | Loss, VAE, contrastive |
| 6 | Numerical Methods | When debugging | 🟠 Sooner or later | Stability tricks | FP16, softmax, log-space |
| 7 | Discrete Math & Graphs | Project-dependent | 🟠 Sooner or later | Algorithms | GNN, attention, search |
| 8 | Differential Equations | Subfield-dependent | 🟡 As needed | Neural ODE/SDE | Diffusion, flow |
| 9 | Functional Analysis | Subfield-dependent | 🟡 As needed | Kernel, GP | SVM, GP, theory |
| 10 | Differential Geometry & Topology | Subfield-dependent | 🟡 As needed | Manifold papers | Geometric DL, hyperbolic embedding |
| 11 | Abstract Algebra | Niche | ⚪ Optional | Group-equivariant nets | Sciences ML, physics |
| 12 | Measure Theory | Very rare | ⚪ Optional | Reading rigorous papers | Pure theory |
| 13 | Math for the LLM era | Weekly (LLM era) | 🟠 Combo of 1-7 | FlashAttention/Mamba/SSM | LLM train, serve, align |

> Section #13 is not new math — it is the combination of (1)(2)(3)(4)(6) applied to LLM architectures of 2024-2026. It is separated out because engineers/researchers in 2026 use it daily.

---

## Grouping by priority

```
🔴 NEED RIGHT NOW (1-4): without these you cannot read papers, cannot train models.
🟠 NEEDED SOONER OR LATER (5-7): you hit them constantly while debugging + reading modern papers.
🟡 AS NEEDED (8-10): only learn when your subfield requires it.
⚪ OPTIONAL (11-12): pure research / specific niches.
🟠 LLM ERA (13): combo of 🔴 + 🟠 applied to the LLM stack.
```

**Principle:** don't reach for a higher area while a lower one is still shaky. Studying (10) Differential Geometry before mastering (2) Calculus = wasted time.

---

## 1. Linear Algebra

### Why #1
All AI data = tensors (generalization of matrices). Every neural-network layer = a linear transformation followed by a non-linearity. Understanding linear algebra = understanding the spine of deep learning.

### Core topics (must-know)

| Topic | Why you need it |
|---|---|
| Vectors, vector spaces | Embeddings, feature spaces |
| Dot product, outer product | Similarity, attention |
| Matrices, matmul, transpose | The forward pass = a matmul chain |
| Rank, kernel, image, subspaces | Low-rank, LoRA, bottlenecks |
| Determinant, trace | Jacobian, determinants in normalizing flows |
| Eigenvalues, eigenvectors | PCA, spectral methods, Hessian stability |
| Singular Value Decomposition (SVD) | PCA, low-rank approximation, recommendation |
| Norms: L1, L2, L∞, Frobenius, nuclear, spectral | Regularization, weight decay |
| Symmetric matrices, PSD/PD | Covariance, Hessian, kernels |
| Projection, orthogonality | Gram-Schmidt, QR, orthogonal init |
| Kronecker product, Hadamard product | Efficient computation |
| LU, QR, Cholesky decomposition | Solving linear systems, sampling |

### Concrete uses in AI

- **Forward pass:** y = σ(Wx + b) — just matmul + bias + non-linearity.
- **Attention:** softmax(QKᵀ/√d)V — all linear operations.
- **PCA:** diagonalizing the covariance matrix = SVD.
- **LoRA:** ΔW ≈ BA with rank(BA) ≪ rank(W) — low-rank decomposition.
- **Batch Normalization:** whitening = projection onto a subspace.
- **Word/Sentence Embeddings:** points in R^d.
- **Graph Neural Networks:** Laplacian L = D − A is a linear-algebra object.

### Recommended resources

- **Foundational textbook:** Gilbert Strang — *Introduction to Linear Algebra* (very readable, paired with MIT 18.06).
- **Rigorous textbook:** Sheldon Axler — *Linear Algebra Done Right* (no use of determinants).
- **For ML:** Chapter 2 of *Mathematics for Machine Learning* (Deisenroth, Faisal, Ong) — free PDF.
- **Advanced reference:** Trefethen & Bau — *Numerical Linear Algebra*.

### Self-check

- [ ] Compute the SVD of a 2×2 matrix by hand.
- [ ] Explain why attention divides by √d (hint: variance of a dot product).
- [ ] Prove that a covariance matrix is always PSD.
- [ ] Implement PCA in NumPy without using `np.linalg.svd` (use power iteration).
- [ ] Why does LoRA work when it only trains rank r ≪ d?

### Common pitfalls

- Confusing rank with dimension.
- Forgetting that AᵀA ≠ AAᵀ.
- Assuming eigenvalues are always real — only true for symmetric/Hermitian matrices.
- Confusing "orthogonal" with "orthonormal".

---

## 2. Multivariable Calculus

### Why #2
All training = optimizing a loss over millions/billions of parameters. Backpropagation = the chain rule applied to a computation graph. No calculus = no understanding of why the model learns.

### Core topics

| Topic | Why you need it |
|---|---|
| Partial derivatives | Every gradient component |
| Gradient ∇f | Direction of steepest ascent of the loss |
| Jacobian J | Vector → vector derivative, backprop through layers |
| Hessian H | Curvature, second-order methods, sharpness |
| Chain rule (1-var, multi-var) | Backpropagation |
| Matrix/vector calculus | Computing ∂loss/∂W |
| Taylor expansion | Linearization, second-order approximations |
| Multivariable integration | Marginalization in probability |
| Implicit function theorem | Implicit layers, equilibrium models |
| Leibniz rule | Differentiation under the integral, reparameterization |

### Concrete uses

- **Backpropagation:** the chain rule generalized over function composition.
- **Reverse-mode autodiff:** why PyTorch/JAX are fast — they compute Jacobian-vector products efficiently.
- **Reparameterization trick (VAE):** z = μ + σ ⊙ ε, lets gradients flow through samples.
- **Sharpness-aware minimization (SAM):** uses Hessian/curvature.
- **Natural gradient:** F⁻¹ ∇L with F the Fisher information matrix.
- **Score matching:** ∇_x log p(x) — gradient of a log-density.

### Resources

- **Foundational textbook:** Stewart — *Multivariable Calculus* (mainstream, accessible).
- **For ML:** Chapter 5 of *Mathematics for Machine Learning*.
- **Matrix calculus:** *The Matrix Cookbook* (Petersen & Pedersen) — formula reference.
- **Autodiff:** *Automatic Differentiation in Machine Learning: a Survey* (Baydin et al.).

### Self-check

- [ ] Compute ∂(xᵀAx)/∂x by hand.
- [ ] Derive the gradient of the cross-entropy loss through softmax.
- [ ] Implement backprop from scratch for a 2-layer MLP in NumPy.
- [ ] Explain the reparameterization trick: why is it necessary for VAEs?
- [ ] Draw a computation graph and run reverse-mode autodiff by hand.

### Common pitfalls

- Thinking the gradient is a scalar — the gradient is always a vector.
- Forgetting transposes when differentiating w.r.t. a matrix.
- Assuming the Hessian is always computable — for large models it doesn't fit in memory.
- Confusing Jacobian-vector product (forward-mode) with vector-Jacobian product (reverse-mode).

---

## 3. Probability & Statistics

### Why #3
ML = learning from data = learning distributions. Every loss has a probabilistic interpretation. Generative models = learn p(x). Bayesian DL, uncertainty, A/B tests — all require this.

### Core topics

| Topic | Why you need it |
|---|---|
| Discrete and continuous random variables | All stochastic models |
| Distributions: Bernoulli, Categorical, Gaussian, Poisson, Exponential, Beta, Dirichlet | Likelihoods, priors |
| Expectation, variance, covariance | Loss, regularization |
| Joint, marginal, conditional | Bayes, graphical models |
| Bayes' theorem | MAP, Bayesian inference |
| MLE, MAP, EM | Parameter learning |
| Central Limit Theorem (CLT) | Why Gaussians are everywhere |
| Law of large numbers | Convergence of SGD |
| Sampling: rejection, importance, MCMC, Gibbs | Bayesian inference, diffusion |
| Variational inference, ELBO | VAE, variational methods |
| Hypothesis testing, p-values, confidence intervals | Model evaluation, A/B |
| Concentration inequalities (Markov, Chebyshev, Hoeffding) | Generalization bounds |
| Exponential family of distributions | Unified framework |

### Concrete uses

- **Cross-entropy loss:** −E[log p(y\|x)] = negative log-likelihood.
- **VAE:** maximize ELBO = log p(x) − KL(q(z\|x) ‖ p(z)).
- **Diffusion model:** learn the gradient of log-density (score) at many noise levels.
- **Bayesian Neural Network:** place a prior on weights, posterior via VI or MCMC.
- **Uncertainty estimation:** epistemic vs aleatoric uncertainty.
- **Active learning:** pick the sample with the highest entropy.
- **GAN:** not directly probabilistic but ultimately optimizes a form of divergence.
- **Reinforcement learning:** the policy = a distribution over actions.

### Resources

- **Foundational textbook:** Wasserman — *All of Statistics* (compact, sufficient for ML).
- **ML textbook:** Bishop — *Pattern Recognition and Machine Learning* (PRML, classic).
- **Bayesian:** Gelman — *Bayesian Data Analysis*.
- **Deeper:** Murphy — *Probabilistic Machine Learning: Advanced Topics*.
- **For beginners:** Blitzstein — *Introduction to Probability*.

### Self-check

- [ ] Derive that cross-entropy = negative log-likelihood for Bernoulli/Categorical.
- [ ] Derive the VAE ELBO two ways: (a) from KL(q ‖ posterior) ≥ 0, (b) from Jensen's inequality. Approach (a) shows the exact gap = KL.
- [ ] Explain why log-likelihood is preferable to likelihood (numerical reasons).
- [ ] For a Gaussian model, compute the MLE of μ and σ² by hand.
- [ ] Implement Gibbs sampling for a simple model.
- [ ] Distinguish epistemic uncertainty vs aleatoric uncertainty.

### Common pitfalls

- Confusing p(A\|B) with p(B\|A) — base-rate fallacy.
- Forgetting the Jacobian when changing variables.
- Confusing posterior with likelihood.
- Assuming MLE and MAP always give the same answer (only when the prior is uniform).
- Forgetting that KL is asymmetric: KL(p‖q) ≠ KL(q‖p).

---

## 4. Optimization

### Why #4
Training = an optimization problem. Understanding optimizers = understanding why Adam is usually better than SGD, why learning rate matters, why models get stuck at saddle points.

### Core topics

| Topic | Why you need it |
|---|---|
| Convex vs non-convex | DL is non-convex, you need to understand the difference |
| Gradient descent, SGD | Basic optimizer |
| Momentum, Nesterov | Acceleration, escaping local minima |
| Adaptive methods: AdaGrad, RMSProp, Adam, AdamW | Modern optimizers |
| Learning rate schedules | Warmup, cosine decay — vital for LLMs |
| Second-order: Newton, quasi-Newton (L-BFGS) | When they apply |
| Constrained optimization, Lagrange, KKT | SVM, RLHF |
| Duality | Deeper understanding of SVMs, convex problems |
| Stochastic optimization, variance reduction | SGD analysis |
| Convergence analysis (rate, bounds) | Reading optimization papers |
| Sharpness, flatness, generalization | Generalization theory |
| Trust region, proximal methods | TRPO, RL |

### Concrete uses

- **SGD with momentum:** v ← βv + ∇L; θ ← θ − ηv. Still used by every large model.
- **Adam:** combines momentum + per-parameter adaptive lr. Default for NLP/CV.
- **AdamW:** Adam + decoupled weight decay. Default for LLM training.
- **Learning rate warmup:** linear ramp-up of lr for the first N steps, then decay. Vital for transformers.
- **Gradient clipping:** clip the norm to avoid exploding gradients.
- **TRPO/PPO:** trust region in RL — the policy can't change too much per step.
- **DPO (Direct Preference Optimization):** optimize preferences directly through an implicit reward, avoiding a separate reward model + PPO. It doesn't always beat RLHF (PPO/GRPO) — still under debate, each has its own sweet spot.
- **GRPO (Group Relative Policy Optimization):** RLHF variant without a value network, advantage estimated from group rewards. Used by the DeepSeek/R1 line of reasoning models.
- **Muon, SOAP, Shampoo:** optimizers that use matrix-valued curvature (matrix-valued second moments). 2024-2026 results show they beat AdamW on some LLM setups, papers still rolling out.

### Resources

- **Convex:** Boyd & Vandenberghe — *Convex Optimization* (free PDF, classic).
- **Non-convex / ML:** Bottou, Curtis, Nocedal — *Optimization Methods for Large-Scale ML* (survey).
- **Textbook:** Nocedal & Wright — *Numerical Optimization*.
- **DL-specific:** Goodfellow et al. — *Deep Learning*, Chapter 8.

### Self-check

- [ ] Implement SGD, momentum, and Adam from scratch.
- [ ] Derive Adam's update rule from its definition.
- [ ] Explain bias correction in Adam: why divide by (1 − β^t).
- [ ] Why does decoupled weight decay differ from L2 regularization in Adam?
- [ ] Sketch the loss landscape of a convex vs non-convex problem.
- [ ] Solve an SVM by hand using KKT conditions (2-D case).

### Common pitfalls

- Believing Adam is always better than SGD — no, SGD+momentum still wins on many CV setups.
- Forgetting to zero the gradient (`optimizer.zero_grad()`).
- Setting the learning rate too high → loss spike / NaN.
- Skipping warmup when training a transformer → loss divergence.
- Believing a larger batch size = faster training — you actually need to scale lr with √batch or linearly.

---

## 5. Information Theory

### Why it matters
Most classification and generative losses originate in information theory. A deep understanding of cross-entropy, KL, and MI = a much more fluent reading of modern papers.

### Core topics

| Topic | Why you need it |
|---|---|
| Entropy H(X) | Measure of uncertainty |
| Joint, conditional, mutual entropy | Measuring dependence |
| Cross-entropy H(p,q) | Classification loss |
| KL divergence D_KL(p‖q) | VAE, variational inference, distillation |
| Mutual information I(X;Y) | Contrastive learning, InfoMax |
| Jensen-Shannon divergence | GAN |
| Rate-distortion theory | VAE bottleneck, compression |
| Channel capacity | Information bottleneck principle |
| MDL, Kolmogorov complexity | The Occam's-razor theory |
| f-divergence (generalization) | f-GAN, advanced |

### Concrete uses

- **Cross-entropy loss:** every classification loss = H(p_true, p_model).
- **KL in VAE:** ELBO = E[log p(x\|z)] − KL(q(z\|x) ‖ p(z)).
- **Knowledge distillation:** student learns from teacher via KL on softmax outputs.
- **InfoNCE (Contrastive learning):** a lower bound on mutual information → SimCLR, CLIP.
- **Information Bottleneck:** learn a representation Z such that I(X;Z) is minimized and I(Y;Z) is maximized.
- **Perplexity:** exp(cross-entropy) — the standard language-model metric.
- **β-VAE:** add a β coefficient in front of the KL term to control disentanglement.

### Resources

- **Classic:** David MacKay — *Information Theory, Inference, and Learning Algorithms* (free PDF, unbeatable).
- **Textbook:** Cover & Thomas — *Elements of Information Theory*.
- **ML-focused:** Tishby's papers on the Information Bottleneck.
- **Modern:** Poole et al. (2019) *On Variational Bounds of Mutual Information*.

### Self-check

- [ ] Derive cross-entropy from the definition of entropy.
- [ ] Compute the KL between two Gaussians by hand.
- [ ] Explain what InfoNCE is actually estimating.
- [ ] Why is the asymmetry of KL important in VAEs? (forward vs reverse KL)
- [ ] Implement perplexity for a language model.
- [ ] Distinguish H(X) vs H(X\|Y) vs I(X;Y).

### Common pitfalls

- Forgetting that KL ≥ 0 with equality iff p = q a.e.
- Confusing forward KL (mode-covering) with reverse KL (mode-seeking).
- Believing MI is easy to estimate — it's extremely hard, you need lower bounds (MINE, InfoNCE).
- Forgetting log base: log_2 (bits) vs log_e (nats).

---

## 6. Numerical Methods

### Why it matters
All ML computation runs on hardware with finite-precision floats. Understanding floating-point = understanding why your loss goes NaN, why FP16 training breaks, why softmax overflows.

### Core topics

| Topic | Why you need it |
|---|---|
| Floating-point formats (FP32, TF32, FP16, BF16, FP8 E4M3/E5M2) | Mixed-precision training |
| Round-off error | Numerical stability |
| Numerical stability | Avoiding NaN/Inf |
| Condition number | When a matrix is hard to invert |
| Log-space trick: logsumexp | Numerically stable softmax |
| Cancellation, catastrophic cancellation | Error when subtracting nearby values |
| Iterative methods: Jacobi, Gauss-Seidel, CG | Solving large linear systems |
| Numerical integration, quadrature | Sampling, normalization |
| Power iteration, Lanczos | Largest eigenvalues |

### Concrete uses

- **Stable softmax:** softmax(x) = softmax(x − max(x)) to avoid overflow.
- **logsumexp:** log(Σ exp(x_i)) = max + log(Σ exp(x_i − max)).
- **Mixed precision (FP16/BF16/FP8):** faster training + less memory. FP16 needs loss scaling because its range is narrow; BF16 is the default for LLMs (same exponent as FP32); FP8 (H100/Blackwell) requires per-tensor/per-block scales + care with softmax and layernorm, which are precision-sensitive.
- **Gradient clipping:** clip by norm to prevent explosion.
- **Layer normalization:** divide by √(σ² + ε), the ε avoids division by zero.
- **Stable cross-entropy:** use logsoftmax + NLLLoss instead of softmax + log + NLL.

### Resources

- **Classic textbook:** Trefethen & Bau — *Numerical Linear Algebra*.
- **Deeper:** Higham — *Accuracy and Stability of Numerical Algorithms*.
- **Practical:** *What Every Computer Scientist Should Know About Floating-Point Arithmetic* (Goldberg).
- **DL-specific:** the NVIDIA Mixed Precision Training guide.

### Self-check

- [ ] Why does softmax([1000, 1001, 1002]) overflow? How to fix it?
- [ ] What is 1 + 1e-20 in FP32?
- [ ] Distinguish FP16 vs BF16 vs FP8: range vs precision tradeoff; which is good for training/inference and why?
- [ ] Implement logsumexp.
- [ ] When does a high-condition-number matrix cause problems?
- [ ] Why does the loss spike when training with FP16 without loss scaling?

### Common pitfalls

- Believing 0.1 + 0.2 = 0.3 (no, because of binary floats).
- Forgetting ε in LayerNorm, BatchNorm, divisions.
- Training in FP16 without loss scaling → underflowing gradients.
- Subtracting nearby values (catastrophic cancellation) → loss of precision.
- Trusting 4-digit accuracy — FP32 only gives about 7 significant digits across many operations.

---

## 7. Discrete Math & Graph Theory

### Why it matters
The data structure behind many modern models is a graph: GNNs, attention (a fully connected graph), tokenization (BPE), search in RL/planning, retrieval.

### Core topics

| Topic | Why you need it |
|---|---|
| Sets, mappings, relations | Foundations |
| Combinatorics, counting | Combinatorial search |
| Graphs: undirected, directed, weighted | GNN, computation graph |
| Paths, cycles, connectivity | Reachability, search |
| Trees, spanning trees | Decision trees, parsing |
| Graph Laplacian L = D − A | Spectral GNN, manifold |
| Graph algorithms: BFS, DFS, Dijkstra, A* | Search, planning |
| Dynamic programming | RL value iteration, Viterbi |
| Complexity (Big-O) | Efficient algorithms |
| Hashing, locality-sensitive hashing | ANN search, retrieval |

### Concrete uses

- **Graph Neural Network (GNN):** message passing over a graph.
- **Graph Attention (GAT):** attention weights between neighbouring nodes.
- **Transformer = GNN on a fully-connected graph:** every token "connects" to every other token.
- **Beam search, top-k sampling:** decoding in NLP.
- **BPE tokenization:** greedy merging of frequent pairs in a corpus.
- **Retrieval (ANN, HNSW, FAISS):** kNN on the embedding space, hashing.
- **MCTS (Monte Carlo Tree Search):** AlphaGo, AlphaZero.

### Resources

- **Foundational textbook:** Rosen — *Discrete Mathematics and Its Applications*.
- **Algorithms:** CLRS — *Introduction to Algorithms*.
- **Graph-specific:** CLRS graph chapter, or *Graph Representation Learning* (Hamilton).
- **GNN:** *A Comprehensive Survey on Graph Neural Networks* (Wu et al., 2020).

### Self-check

- [ ] Implement a GCN layer from scratch in NumPy.
- [ ] Code Dijkstra/A* yourself.
- [ ] Explain the Viterbi algorithm (DP for HMM decoding).
- [ ] Draw the computation graph of a transformer block.
- [ ] Analyse the complexity of attention (O(n²)) and why linear attention is desirable.

### Common pitfalls

- Confusing a tree with a graph that has cycles.
- Implementing a GNN without normalizing the Laplacian → explodes.
- Beam search is not optimal, only a heuristic.
- Mixing up directed vs undirected when building the adjacency matrix.

---

## 8. Differential Equations

### Why it matters (subfield-dependent)
Diffusion models, neural ODEs, flow matching — hot in generative AI in 2026. If you aim for generative models, this math is mandatory.

### Core topics

| Topic | Why you need it |
|---|---|
| 1st- and 2nd-order ODEs | Neural ODE |
| Linear ODE systems | Linear dynamics |
| Euler method, Runge-Kutta | Numerical ODE solvers |
| Stochastic differential equations (SDE) | Diffusion (score-based) |
| Basic Itô calculus | SDE manipulation |
| Fokker-Planck equation | Diffusion density evolution |
| Continuous-time flow | Flow matching, rectified flow |
| Probability flow ODE | Deterministic diffusion sampling |

### Concrete uses

- **Neural ODE:** dy/dt = f(y, t, θ), trained with the adjoint method.
- **Continuous Normalizing Flow:** uses ODEs to transform a distribution.
- **DDPM (Denoising Diffusion):** learn the reverse of a (discrete) noise chain.
- **Score-based generative model:** learn the score ∇_x log p_t(x) at many noise levels; sample via SDE/ODE.
- **Flow Matching / Rectified Flow:** instead of noise → data through many steps, learn a straight vector field.
- **Consistency models:** diffusion distillation down to 1 step.

### Resources

- **ODE foundations:** Strogatz — *Nonlinear Dynamics and Chaos*.
- **SDE foundations:** Øksendal — *Stochastic Differential Equations*.
- **Diffusion models:** Yang Song's blog *Generative Modeling by Estimating Gradients of the Data Distribution*.
- **Foundational papers:**
  - Neural ODE (Chen et al., 2018).
  - DDPM (Ho et al., 2020).
  - Score-based (Song & Ermon, 2019, 2021).
  - Flow Matching (Lipman et al., 2023).

### Self-check

- [ ] Solve dy/dt = −ky by hand.
- [ ] Implement an Euler solver for a Neural ODE.
- [ ] Derive the DDPM ELBO.
- [ ] Explain the relationship between (discrete) diffusion and (continuous) score-based SDEs.
- [ ] Implement DDPM sampling from a given checkpoint.

### Common pitfalls

- Mixing up the forward and reverse process in diffusion.
- Forgetting that SDEs use the Itô integral, which differs from Riemann.
- Believing diffusion = VAE — the math is very different.
- Numerical solvers becoming unstable when the step size is large.

---

## 9. Functional Analysis

### Why it matters (subfield-dependent)
The root of kernel methods (SVM, GP). Less and less important in deep learning, but still foundational for theory.

### Core topics

| Topic | Why you need it |
|---|---|
| Banach, Hilbert spaces | Foundations |
| Inner product, norm, completeness | RKHS |
| Linear operators between function spaces | Kernel operators |
| Reproducing Kernel Hilbert Space (RKHS) | Kernel methods |
| Spectral theorem for compact operators | Mercer's theorem |
| Basic Fourier analysis | Signals, Fourier features |

### Concrete uses

- **Kernel SVM:** maximize the margin in an RKHS.
- **Gaussian Process:** a distribution over functions, with the kernel as covariance.
- **Random Fourier Features:** approximate a kernel by a random feature map.
- **Neural Tangent Kernel (NTK):** a wide NN is equivalent to a kernel method.
- **Sinusoidal positional encoding:** a Fourier basis in a transformer.
- **Implicit Neural Representations (NeRF, SIREN):** Fourier features at the input.

### Resources

- **Textbook:** Kreyszig — *Introductory Functional Analysis with Applications*.
- **Kernel methods:** Schölkopf & Smola — *Learning with Kernels*.
- **GP:** Rasmussen & Williams — *Gaussian Processes for Machine Learning* (free PDF).
- **NTK:** Jacot et al. (2018) — *Neural Tangent Kernel*.

### Self-check

- [ ] Define an RKHS via the reproducing property.
- [ ] Derive the dual form of the kernel SVM.
- [ ] Implement Gaussian Process regression in NumPy.
- [ ] Compute the NTK of a 1-layer infinite-width network.

### Common pitfalls

- Confusing "kernel" in ML (a similarity function) with "kernel" in linear algebra (null space).
- Assuming every Hilbert space is R^d — it can be an infinite-dimensional function space.
- GPs scale as O(n³) in the number of samples — they don't scale to large datasets.

---

## 10. Differential Geometry & Topology

### Why it matters (subfield-dependent)
Manifold learning, hyperbolic embedding, diffusion theory (the score = a gradient on a manifold), interpretability (loss landscape).

### Core topics

| Topic | Why you need it |
|---|---|
| Manifolds, tangent spaces | The manifold hypothesis for data |
| Riemannian metric | Distances on manifolds |
| Geodesics | Shortest paths |
| Curvature, Ricci | Loss-landscape analysis |
| Exponential map, logarithm map | Optimization on manifolds |
| Fiber bundles | Equivariant networks |
| Basic cohomology | Topology of data |
| Persistent homology | Topological Data Analysis (TDA) |

### Concrete uses

- **Manifold hypothesis:** real-world data lies on a low-dimensional manifold inside a high-dimensional ambient space.
- **Hyperbolic embedding:** used for tree/hierarchy data (Poincaré embedding).
- **Loss-landscape geometry:** sharp vs flat minima → generalization.
- **Geometric Deep Learning:** Bronstein et al. unify GNNs, CNNs, transformers through geometry.
- **Manifold optimization:** Adam on the Stiefel/Grassmann manifold for orthogonal weights.
- **TDA for ML:** persistence diagrams as features.

### Resources

- **Foundational textbook:** do Carmo — *Riemannian Geometry*.
- **For ML:** Bronstein et al. — *Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges* (free book).
- **TDA:** Edelsbrunner & Harer — *Computational Topology*.
- **Optimization on manifolds:** Absil, Mahony, Sepulchre — *Optimization Algorithms on Matrix Manifolds*.

### Self-check

- [ ] Compute the tangent space of the sphere S² at a point.
- [ ] Implement a Poincaré embedding for a simple tree.
- [ ] Explain the flat-minima hypothesis from a geometric viewpoint.
- [ ] Derive the exponential map on a sphere.

### Common pitfalls

- Manifold learning ≠ Riemannian DL — close concepts, but different.
- Assuming all data lies on a clean manifold — in reality there is noise and irregularity.
- Hyperbolic embeddings are noticeably harder to train than Euclidean.

---

## 11. Abstract Algebra

### Why it matters (niche)
Group-equivariant networks: AlphaFold, physics ML, computer vision with symmetries. Mathematically elegant, application-niche.

### Core topics

| Topic | Why you need it |
|---|---|
| Group, subgroup, normal subgroup | Symmetry |
| Group action | Symmetry in data |
| Representation theory | Equivariant layers |
| Lie group, Lie algebra | Continuous symmetry (rotation, translation) |
| Irreducible representation | Tensor field networks, e3nn |
| Quotient group | Pose estimation |

### Concrete uses

- **Equivariant CNN:** group convolution in place of standard convolution.
- **E(3)-equivariant network:** AlphaFold, used for 3-D molecules.
- **Tensor Field Network, e3nn:** equivariant under 3-D rotation.
- **Spherical CNN:** for data on the sphere (climate, omnidirectional images).
- **SE(3)-Transformer:** a transformer equivariant under rigid transformations.

### Resources

- **Foundational textbook:** Dummit & Foote — *Abstract Algebra*.
- **Representation theory:** Fulton & Harris — *Representation Theory*.
- **For ML:** Bronstein et al. *Geometric Deep Learning*, group chapter.
- **Paper:** Cohen & Welling (2016) *Group Equivariant CNNs*.

### Self-check

- [ ] List the symmetries of a square (the D₄ group).
- [ ] Explain equivariant vs invariant.
- [ ] Implement a simple equivariant layer for rotation.

### Common pitfalls

- Confusing invariant and equivariant.
- Believing "more equivariance" is always better — it heavily constrains the network and can reduce capacity.

---

## 12. Measure Theory

### Why it matters (very rarely)
Only needed when reading rigorous probability papers (e.g., convergence proofs, optimal-transport theory). Not needed to do practical AI/ML.

### Core topics

| Topic | Why you need it |
|---|---|
| σ-algebra | Rigorous definition of probability |
| Measure | Generalization of probability |
| Lebesgue integral | Integrating non-Riemann-integrable functions |
| Radon-Nikodym derivative | Density between two measures |
| Dominated convergence | Swapping limit and integral |
| Optimal transport, Wasserstein distance | WGAN, distribution matching |

### Concrete uses

- **Wasserstein distance:** used in WGAN, optimal transport.
- **Radon-Nikodym:** exactly the density p(x). Important when changing measures.
- **Convergence theorems:** used in theory-paper proofs.

### Resources

- **Textbook:** Folland — *Real Analysis*.
- **Rigorous probability:** Billingsley — *Probability and Measure*.
- **Optimal transport:** Villani — *Optimal Transport: Old and New* (very hard).
- **For ML:** Peyré & Cuturi — *Computational Optimal Transport* (easier).

### Self-check

- [ ] Define a σ-algebra.
- [ ] Distinguish Riemann and Lebesgue integration.
- [ ] Understand Wasserstein-1 in its dual form (Kantorovich-Rubinstein).

### Common pitfalls

- Studying measure theory before you need it → forgotten entirely.
- Getting overwhelmed by formalism.

---

## 13. Math for the LLM Era (2024-2026)

### Why split this out
Not a new branch of math — it's a combo of (1)(2)(3)(4)(6) applied to LLM-era architectures. Split out because it is what engineers/researchers in 2026 actually hit daily, and the required math differs noticeably from "classical 2018 ML".

### Topics & required math

| Topic | Underlying math | Why |
|---|---|---|
| FlashAttention (v1/v2/v3) | (6) IO model, tiling, online softmax | Not new math — softmax re-engineered to be HBM-aware |
| RoPE (Rotary Position Embedding) | (1) 2-D rotations / complex numbers | Per-head rotation by position |
| ALiBi, YaRN, NTK-aware scaling | (1)(2) extrapolation theory | Long context without retraining |
| Grouped/Multi-Query Attention | (1) tradeoff rank vs KV cache | Inference economics |
| Linear / sub-quadratic attention | (1)(9) kernel approximation | Performer, Hyena, RWKV |
| **State Space Models (S4, Mamba)** | (1)(8) discretized linear ODE: ḣ = Ah + Bx | Selective recurrence with parallel scan; HiPPO theory |
| Mixture of Experts (MoE) | (3)(4) routing softmax + load-balance loss | Sparse compute scale-up |
| Quantization (GPTQ, AWQ, GGUF) | (1)(6) min ‖Wx − W_q x‖² calibration | Inference cost |
| Speculative / parallel decoding | (3) rejection sampling, distribution matching | 2–4× faster decode |
| Test-time compute (search, BoN, MCTS) | (3)(7) DP + Bayesian decision | Reasoning models (o1, R1) |
| Process Reward Model (PRM) | (3)(4) per-step credit assignment | RL on reasoning traces |
| Mechanistic interpretability | (1) SVD/superposition + (3) feature distributions | Circuits, SAE (sparse autoencoder) |
| RLHF / GRPO / DPO | (3)(4) KL-regularized policy optimization | Alignment |

### Resources
- **FlashAttention:** Dao et al. (2022, 2023, 2024) — read v1→v3 in order.
- **Mamba/SSM:** Gu & Dao (2023) *Mamba*; Gu (2023) thesis on S4 for depth.
- **Speculative decoding:** Leviathan et al. (2023); Chen et al. (2023) *Accelerating LLM Inference*.
- **MoE:** Switch Transformer (Fedus et al., 2021); the Mixtral paper.
- **Quantization:** GPTQ (Frantar et al., 2022); AWQ (Lin et al., 2023).
- **Interpretability:** Anthropic *A Mathematical Framework for Transformer Circuits* (Elhage et al., 2021).
- **RL alignment:** PPO (Schulman 2017); DPO (Rafailov 2023); GRPO (DeepSeekMath 2024).

### Self-check
- [ ] Derive online softmax (running max + denominator) as used in FlashAttention.
- [ ] Implement RoPE by hand on an attention head — verify the rotation is correct.
- [ ] Discretize ḣ = Ah + Bx with ZOH; derive the recurrence form of Mamba's selective scan.
- [ ] Explain rejection sampling in speculative decoding: why is the output distribution unchanged?
- [ ] Contrast DPO loss and PPO loss — which one uses an explicit reward model?
- [ ] Read one SAE paper and explain what loss = reconstruction + L1 sparsity means.

### Common pitfalls
- Assuming FlashAttention is "exact" like a standard softmax — value-equivalent, yes, but the order of operations differs (numerically equivalent, not bit-equal).
- Implementing RoPE on the wrong axis → attention becomes noise.
- Wrong SSM discretization (Euler vs ZOH vs bilinear) → unstable.
- Quantization PTQ that benchmarks well but breaks on long-tail inputs.
- Misconfigured speculative decoding → output distribution drifts from the target model.
- DPO training on only 1 epoch is insufficient; too many epochs → preference overfitting.

---

## Applied math "beyond the table"

Once your foundation (1-7) is solid, the areas below **often have higher ROI** than (10)(11)(12):

### A. Statistical Learning Theory
- **Topics:** PAC learning, VC dimension, Rademacher complexity, generalization bounds, uniform convergence.
- **Why:** answers "why does the model generalize?". Note: classical bounds are vacuous on DL; 2020-2026 PAC-Bayes and NTK-based bounds are tighter.
- **Resources:** Shalev-Shwartz & Ben-David — *Understanding Machine Learning*.
- **Self-check:** derive Hoeffding's bound; compute the VC-dim of a linear classifier; explain why classical DL bounds are vacuous.

### B. Reinforcement Learning Math
- **Topics:** MDP, Bellman equation, policy gradient theorem, TD learning, exploration-exploitation, GAE, importance sampling.
- **Why:** RLHF, GRPO, reasoning models (o1/R1), agents, robotics. Extremely hot in 2026.
- **Resources:** Sutton & Barto — *Reinforcement Learning: An Introduction* (free PDF). Supplements: Spinning Up (OpenAI), the CleanRL repo.
- **Self-check:** derive the policy gradient theorem; implement REINFORCE + baseline; explain the GAE λ bias/variance tradeoff; derive the PPO clipped objective.

### C. Advanced Convex Analysis
- **Topics:** subdifferentials, proximal operators, ADMM, mirror descent, Bregman divergence.
- **Why:** Optimization research, sparse methods, some LLM optimizers (Muon) connect here.
- **Resources:** Boyd & Vandenberghe *Convex Optimization* (the deeper parts).
- **Self-check:** derive the proximal operator of L1 (soft-thresholding); explain mirror descent = GD on the dual space.

### D. Causal Inference
- **Topics:** do-calculus, counterfactuals, instrumental variables, DAGs, front/back-door criterion.
- **Why:** A rising niche, few AI practitioners know it. Important for rigorous evaluation + debiasing LLMs.
- **Resources:** Pearl — *Causality* or *The Book of Why*. Supplement: *Causal Inference: The Mixtape* (Cunningham, free).
- **Self-check:** distinguish p(y\|x) and p(y\|do(x)); explain Simpson's paradox via a DAG.

### E. Game Theory
- **Topics:** Nash equilibrium, minimax, mechanism design, no-regret learning.
- **Why:** GAN, multi-agent RL, alignment, self-play (AlphaZero, debate).
- **Resources:** Osborne — *An Introduction to Game Theory*.
- **Self-check:** compute the Nash equilibrium of rock-paper-scissors; explain minimax = saddle-point optimization.

### F. Computational Complexity
- **Topics:** P, NP, approximation, hardness, randomized algorithms.
- **Why:** Understand when a problem doesn't scale and why heuristics are needed.
- **Self-check:** is attention's O(n²) tight? What does linear attention trade off?

---

## Meta pitfalls when learning math for AI

Per-area pitfalls live inside each area's section. This section lists **cross-area** pitfalls — about *how* you learn, not *what*.

1. **"Learn all the math first, then start"** → you'll never feel "enough". Learn in parallel, triggered by need.
2. **Reading pure-theory books in isolation** → instantly forgotten. Implementing is mandatory; intuition comes from your fingers.
3. **Reaching for a higher area before mastering a lower one.** Topology before calculus = waste. Follow the priority order in the table.
4. **Believing "you must have a math PhD"** → wrong. Most top engineers at frontier labs learn math on demand, not systematically from scratch.
5. **Skipping (6) Numerical** → a lifetime of debugging NaN without understanding the root cause.
6. **Learning each area in isolation, without connections** → forgotten. One project that uses 3-4 areas at once cements them deeper than 6 months of reading.
7. **Putting research on a pedestal** → unfounded self-doubt. Good engineers need just as much math as researchers, only with a different depth-vs-breadth tradeoff.
8. **Not re-evaluating the path every 3 months** → walking the old road by sunk-cost inertia even after the goal has shifted.
9. **"Read 50 papers, then reproduce"** → the reading-without-doing trap. Reproducing 1 paper > reading 20.

---

## Recommended 12-month roadmap

**Baseline assumption:** you already have (1)(2)(3)(4) at the "can use" level — you can implement an MLP, you understand backprop, you know Gaussians, you know SGD/Adam. You are not yet solid on (5) Info Theory, (6) Numerical, (7) Discrete & Graph; you haven't touched 8-13.

The roadmap is split by **end goal**, not one-size-fits-all. Pick one path, don't jump between paths.

### Path 1 — Engineer (production, infra, applied)
*Target: train/serve LLMs at scale, debug performance, ship features.*

| Month | Math | Skill | Deliverable |
|---|---|---|---|
| 1-3 | Fill in (5) Info Theory + (6) Numerical, deeply | Mixed precision (BF16/FP8), gradient clipping, profiling (PyTorch Profiler, Nsight) | Blog/repo: "Numerical pitfalls in LLM training" |
| 4-6 | Deeper (1): low-rank, Kronecker; (4) optim: LR schedule, optimizer state | Distributed (DDP/FSDP/ZeRO), basic CUDA (1 kernel: fused softmax or layernorm) | Train ≥1B params on multi-GPU, log the scaling curve |
| 7-9 | (13) LLM math: FlashAttention v3, RoPE, GQA, KV cache | Inference stack (vLLM/SGLang), quantization (GPTQ/AWQ) | Serve a model in production with measured latency/throughput SLA |
| 10-12 | More of (13): speculative decoding, MoE | Eval harness, LLM regression tests | One end-to-end feature: data → train → eval → serve |

### Path 2 — Research (papers, novel methods)
*Target: read papers fluently, reproduce + extend, contribute a novel idea.*

| Month | Math | Skill | Deliverable |
|---|---|---|---|
| 1-3 | Fill in (5) Info Theory; deeper (4) optim (convergence, sharpness) | Re-implement transformer/VAE from scratch, read 1 paper/week | Blog: "Information Theory in modern DL" + 12 paper summaries |
| 4-6 | Subfield-specific math: generative → (8) ODE/SDE; theory → (A) SLT; geometric → (10)(11); alignment/RL → (B) | Reproduce 1 "tractable" paper in your subfield, without copying code | Reproduction repo + writeup of obstacles encountered |
| 7-9 | Deeper into the secondary area of your subfield (e.g. diffusion: master (8); RLHF: master (B)) | Read 1 paper/week + reproduce 1 baseline per month | Identify 2-3 limitations of the reproduced paper |
| 10-12 | As required by the paper you are writing | Design 1 small experiment testing 1 specific hypothesis | Workshop-quality writeup, submit if the results hold |

### Path 3 — Applied / Domain ML (science, finance, health, etc.)
*Target: use ML to solve a specific domain problem, not necessarily training from scratch.*

| Month | Math | Skill | Deliverable |
|---|---|---|---|
| 1-3 | Deeper (3) Probability (Bayesian, hypothesis testing); basic (5) Info Theory | Fine-tuning workflow (LoRA, QLoRA), rigorous prompt engineering | 1 fine-tuned model on a domain dataset, with full eval |
| 4-6 | (D) Causal Inference; (3) uncertainty quantification | Eval design, A/B test, calibration | Report: domain task with uncertainty + a defensible causal claim |
| 7-9 | Domain-specific math (e.g. health: survival analysis; finance: time-series, stochastic calculus) | Domain integration (data pipeline, MLOps) | End-to-end domain pipeline with stakeholder feedback |
| 10-12 | As needed | Productionization, drift monitoring | A system running in production with domain-relevant metrics |

### Cross-cutting principles (all 3 paths)

- **Don't study (9)(10)(11)(12) in the first 12 months** unless your chosen path explicitly requires it (Path 2 + a geometric/theory subfield).
- **Re-evaluate every 3 months:** is the path still right? If the goal changed, switch paths — don't finish the old one out of sunk cost.
- **Learn math on demand:** hit a problem that needs X → learn X. Don't learn X "just in case".
- **Deliverable before reading:** finish a deliverable before reading more papers. Avoid the perpetual-learning trap.

---

## Glossary of key terms

| Term | Symbol / abbreviation |
|---|---|
| Linear Algebra | LA |
| Calculus | — |
| Derivative | — |
| Gradient | ∇ |
| Partial Derivative | ∂ |
| Jacobian matrix | J |
| Hessian matrix | H |
| Chain Rule | — |
| Backpropagation | backprop |
| Eigenvalue | λ |
| Eigenvector | v |
| Singular Value Decomposition | SVD |
| Positive Definite | PD |
| Positive Semi-Definite | PSD |
| Probability distribution | — |
| Gaussian distribution | 𝒩 |
| Expectation | E |
| Variance | Var |
| Covariance | Cov |
| Maximum Likelihood Estimation | MLE |
| Maximum A Posteriori | MAP |
| Central Limit Theorem | CLT |
| Variational Inference | VI |
| Evidence Lower Bound | ELBO |
| Entropy | H |
| Cross-Entropy | H(p,q) |
| Kullback-Leibler Divergence | D_KL |
| Mutual Information | I |
| Optimization | — |
| Convex | — |
| Non-convex | — |
| Learning Rate | lr |
| Generative Model | — |
| Diffusion Model | — |
| Graph Neural Network | GNN |
| Manifold | — |
| Tangent Space | — |
| Translation / Rotation (group) | — |
| Equivariant / Invariant | — |
| State Space Model | SSM |
| Mixture of Experts | MoE |
| Quantization (PTQ/QAT) | — |
| Speculative Decoding | — |
| Test-Time Compute | TTC |
| Process Reward Model | PRM |
| Reinforcement Learning from Human Feedback | RLHF |
| Direct Preference Optimization | DPO |

---

## Closing

This document is a **map**, not a **mandatory route**. Everyone has a different background and different goals. Three principles to keep:

1. **Learn driven by real need.** A project needs math X → you learn X.
2. **Implement, don't just read.** Math intuition comes from your fingers, not your eyes.
3. **Pick one path, re-evaluate every 3 months.** Path 1 (Engineer) / Path 2 (Research) / Path 3 (Applied) — don't switch paths out of FOMO.

If you already have (1)(2)(3)(4) at the "can use" level and you're heading into the LLM era: the focus of the next 6 months should be **filling in (5) Info Theory + (6) Numerical + (13) LLM math** — not chasing (10)(11)(12) or measure theory.

> *"In theory, theory and practice are the same. In practice, they are not."*
