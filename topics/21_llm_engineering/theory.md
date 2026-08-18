# LLM Engineering — Theory

## 0. Notation

| Symbol | Type | Meaning |
|---|---|---|
| $V$ | set | Vocabulary of tokens |
| $t_i$ | integer | Token ID at position $i$ |
| $C$ | scalar | Target vocabulary size in BPE |
| $W_0 \in \mathbb{R}^{d \times k}$ | matrix | Pre-trained frozen weight matrix |
| $\Delta W \in \mathbb{R}^{d \times k}$ | matrix | Weight update matrix |
| $B \in \mathbb{R}^{d \times r}$ | matrix | LoRA down-projection matrix |
| $A \in \mathbb{R}^{r \times k}$ | matrix | LoRA up-projection matrix |
| $r$ | scalar | LoRA rank |
| $\alpha$ | scalar | LoRA scaling factor |
| $\pi_\theta(y \mid x)$ | distribution | Policy (LLM) parameterized by $\theta$ |
| $\pi_{\text{ref}}(y \mid x)$ | distribution | Reference policy (SFT model) |
| $y_w$ | sequence | Winning (preferred) completion |
| $y_l$ | sequence | Losing (rejected) completion |
| $r(x, y)$ | scalar function | Reward model evaluating completion $y$ for prompt $x$ |
| $\beta$ | scalar | KL penalty coefficient in RLHF/DPO |
| $\sigma$ | function | Sigmoid function: $\sigma(z) = 1 / (1 + \exp(-z))$ |
| $Z(x)$ | scalar function | Partition function for prompt $x$ |

---

## 1. WHY — The LLM Pipeline

Training a Large Language Model (LLM) is not a single process, but a sequence of stages, each serving a specific purpose. The modern LLM pipeline consists of three core stages:

### 1.1 Pre-training

The model is trained on a massive corpus of text using self-supervised learning (typically next-token prediction).

*   **Goal:** Learn world knowledge, grammar, and reasoning capabilities.
*   **Challenge:** Computationally expensive, requires thousands of GPUs, resulting in a "base model" that is a powerful autocomplete engine but cannot follow instructions.

### 1.2 Supervised Fine-Tuning (SFT)

The base model is fine-tuned on a smaller, high-quality dataset of instruction-response pairs.

*   **Goal:** Teach the model to act as an assistant and follow user instructions.
*   **Challenge:** Full fine-tuning updates all parameters, which is memory-intensive and prone to catastrophic forgetting. Parameter-Efficient Fine-Tuning (PEFT) methods like LoRA mitigate this.

### 1.3 Alignment (RLHF / DPO)

The SFT model is further optimized to align with human preferences, maximizing helpfulness and minimizing toxicity or hallucinations.

*   **Goal:** Instill human values and shape the model's tone and safety boundaries.
*   **Challenge:** Defining "good" behavior is hard. Reward models and preference learning are used to operationalize human feedback.

This module details the critical engineering components that make this pipeline viable at scale: subword tokenization (BPE), efficient fine-tuning (LoRA), and direct preference alignment (DPO).

---

## 2. Subword Tokenization (BPE) — WHAT & HOW

Before a text sequence can be fed into an LLM, it must be converted into a sequence of discrete integers (tokens).

### Motivation
*   **Character-level:** Small vocabulary ($\sim 256$), no Out-Of-Vocabulary (OOV) tokens, but results in extremely long sequences, making transformer self-attention computationally infeasible, $O(N^2)$.
*   **Word-level:** Short sequences, but huge vocabulary (millions of words), leading to a massive embedding matrix and OOV issues for rare words, misspellings, or morphologically rich languages.
*   **Subword-level:** The optimal middle ground. Frequent words remain single tokens, while rare words are broken into subword pieces. This balances vocabulary size and sequence length.

### Byte-Pair Encoding (BPE) Algorithm

BPE is a data compression algorithm adapted for tokenization. It builds a vocabulary iteratively from the bottom up.

**Step 1: Initialization**
Start with a vocabulary of base characters (or bytes) present in the training corpus. Represent each word in the corpus as a sequence of these base characters, followed by a special end-of-word symbol (e.g., `</w>`).

**Step 2: Frequency Counting**
Count the frequency of all adjacent token pairs in the corpus.

```math
\text{count}(t_a, t_b) = \sum_{\text{words}} \text{freq}(w) \cdot \mathbb{I}[(t_a, t_b) \in w]
```

**Step 3: Greedy Merge**
Find the most frequent adjacent pair $(t_a, t_b)$. Create a new token $t_{new} = t_a t_b$.
Add $t_{new}$ to the vocabulary $V = V \cup \lbrace t_{new}\rbrace$.

**Step 4: Update Corpus**
Replace all occurrences of the pair $(t_a, t_b)$ in the corpus with the new token $t_{new}$.

**Step 5: Iteration**
Repeat Steps 2-4 until the vocabulary reaches the target size $C$.

**Result:** A vocabulary of size $C$ and an ordered list of merge rules. To tokenize new text, apply the merge rules in the exact order they were learned.

### Worked Example
Corpus: "low": 5, "lower": 2, "newest": 6, "widest": 3
Initial vocab: `l, o, w, e, r, n, s, t, i, d, </w>`

*   **Iteration 1:** `e` and `s` appear together 9 times (6 in newest, 3 in widest). Merge `e, s` $\rightarrow$ `es`.
*   **Iteration 2:** `es` and `t` appear together 9 times. Merge `es, t` $\rightarrow$ `est`.
*   **Iteration 3:** `est` and `</w>` appear together 9 times. Merge `est, </w>` $\rightarrow$ `est</w>`.
*   **Iteration 4:** `l` and `o` appear 7 times. Merge `l, o` $\rightarrow$ `lo`.

### Alternative Tokenizers
*   **WordPiece:** Used in BERT. Instead of merging the most frequent pair, it merges the pair that maximizes the likelihood of the training data.
*   **SentencePiece:** Treats the input as a raw stream of characters (including spaces as a special character `_`). BPE is often applied *on top* of SentencePiece, ensuring spaces are modeled directly without requiring pre-tokenization rules.

---

## 3. Parameter-Efficient Fine-Tuning (PEFT) — HOW

During SFT, updating all weights of an LLM is extremely costly. For LLaMA-7B, full fine-tuning requires tracking 7B parameters, their gradients (7B), and optimizer states (e.g., Adam uses momentum and variance: 14B), resulting in massive VRAM requirements.

### Low-Rank Adaptation (LoRA)

LoRA hypothesize that the change in weights during fine-tuning has a low intrinsic dimension. Instead of updating $W_0 \in \mathbb{R}^{d \times k}$, LoRA injects trainable rank decomposition matrices.

**Formulation:**

```math
W = W_0 + \Delta W
```

```math
W = W_0 + \frac{\alpha}{r} B A
```

Where $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$, and the rank $r \ll \min(d, k)$. The scalar $\alpha$ is a constant scaling factor.

**Forward Pass for input $x \in \mathbb{R}^{1 \times d}$:**

```math
h = x W = x W_0 + \frac{\alpha}{r} x B A
```

By linearity (distributive property), we compute $x W_0$ and $x B A$ separately and sum them. $W_0$ is frozen.

**Parameter Reduction Proof:**
*   Full update parameters: $d \times k$
*   LoRA parameters: $d \times r + r \times k = r(d + k)$
*   Savings Ratio: $\frac{r(d + k)}{dk} \approx \frac{r}{d}$ (assuming $d \approx k$)
*   For $d=4096, k=4096, r=8$: LoRA uses $8(8192) = 65,536$ parameters vs $16,777,216$ for full fine-tuning. A 99.6% reduction.

**Result:** Massive reduction in trainable parameters, allowing fine-tuning of 7B models on a single GPU.

### Initialization and Gradient Flow
*   $A$ is initialized randomly (e.g., Kaiming Gaussian).
*   $B$ is initialized to zero matrices.
*   **Why?** This ensures that at initialization, $\Delta W = B A = 0$, so the model acts exactly like the pre-trained base model.

Applying the Chain Rule for loss $L$:

```math
\frac{\partial L}{\partial A} = B^T \frac{\partial L}{\partial \Delta W}
```

```math
\frac{\partial L}{\partial B} = \frac{\partial L}{\partial \Delta W} A^T
```

Both updates factor through $\partial L / \partial \Delta W$, so the frozen $W_0$ never needs a gradient of its own. Because $B = 0$ at initialization, the first step moves $B$ alone, and $A$ only starts to move once $B$ is non-zero.

### Application and QLoRA
The original LoRA paper found that applying LoRA to the Attention $W^Q$ and $W^V$ projection matrices yields the best performance-to-parameter tradeoff.

**QLoRA (Quantized LoRA)** extends this by freezing $W_0$ in a highly compressed 4-bit NormalFloat (NF4) format, while training $A$ and $B$ in 16-bit precision, further slashing VRAM usage. It uses double quantization (quantizing the quantization constants) to save more memory.

---

## 4. LLM Alignment (RLHF vs DPO) — WHAT & HOW

Base models hallucinate, generate toxic text, and fail to follow instructions perfectly. Alignment forces the model output distribution to align with human preferences.

### RLHF Pipeline (Reinforcement Learning from Human Feedback)
1.  **Reward Model (RM) Training:** Train a classifier $r_\phi(x, y)$ on human preference data to predict human scoring.
2.  **PPO Optimization:** Use Proximal Policy Optimization (RL) to update the policy $\pi_\theta$ to maximize the reward.

```math
\max_{\pi_\theta} \mathbb{E}_ {x \sim D, y \sim \pi_\theta} [r_\phi(x, y)] - \beta \mathbb{KL}[\pi_\theta(y \mid x) \Vert \pi_{\text{ref}}(y \mid x)]
```

The KL penalty prevents the policy from drifting too far from the reference model (preventing "reward hacking").

### Direct Preference Optimization (DPO)

RLHF is complex, requiring a distinct reward model and unstable RL loops. DPO mathematically demonstrates that the reward model can be bypassed entirely.

**Step 1: The Optimal Policy**
Starting from the RLHF objective, the optimal policy $\pi^\ast$ that maximizes the regularized reward has a closed-form solution:

```math
\pi^\ast(y \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp\left( \frac{1}{\beta} r(x, y) \right)
```

Where the partition function is

```math
Z(x) = \sum_y \pi_{\text{ref}}(y \mid x) \exp\left( \frac{1}{\beta} r(x, y) \right)
```

**Step 2: Rewriting the Reward**
By taking the log and rearranging (algebraic manipulation), we can express the reward function $r(x,y)$ in terms of the optimal policy and reference policy:

```math
r(x, y) = \beta \log \frac{\pi^\ast(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)
```

**Step 3: The Bradley-Terry Preference Model**
The probability that human prefers $y_w$ over $y_l$ under the Bradley-Terry model is:

```math
P(y_w \succ y_l \| x) = \sigma(r(x, y_w) - r(x, y_l))
```

**Step 4: The DPO Derivation**
Substitute the reward formulation from Step 2 into the Bradley-Terry model. The partition function $Z(x)$ cancels out because it only depends on $x$!

```math
r(x, y_w) - r(x, y_l) = \beta \log \frac{\pi_\theta(y_w\|x)}{\pi_{\text{ref}}(y_w\|x)} - \beta \log \frac{\pi_\theta(y_l\|x)}{\pi_{\text{ref}}(y_l\|x)}
```

Let the implicit reward for a completion under policy $\pi_\theta$ be

```math
\hat r_\theta(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}
```

The probability of preference becomes:

```math
P(y_w \succ y_l \| x) = \sigma\left( \hat{r}_\theta(x, y_w) - \hat{r}_\theta(x, y_l) \right)
```

**Result: DPO Loss Function**
We optimize $\pi_\theta$ directly using Negative Log-Likelihood on the preference data:

```math
\mathcal{L}_{\text{DPO}}(\theta; \pi_{\text{ref}}) = - \mathbb{E}_{(x, y_w, y_l) \sim D} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w\|x)}{\pi_{\text{ref}}(y_w\|x)} - \beta \log \frac{\pi_\theta(y_l\|x)}{\pi_{\text{ref}}(y_l\|x)} \right) \right]
```

| Feature | RLHF | DPO |
|---|---|---|
| Reward Model | Explicit separate network | Implicitly defined by the policy |
| Optimizer | PPO (Complex RL) | NLL / Cross Entropy (Standard Gradient Descent) |
| Stability | Low (RL hyperparameter sensitive) | High (Supervised learning paradigm) |

---

## 5. Failure Cases

1.  **BPE Tokenization Artifacts:** Rare words or numbers might be split into counterintuitive subwords (e.g., "SolidGoldMagikarp" glitch tokens, or arithmetic failures due to misaligned digit tokens), causing poor LLM reasoning.
2.  **LoRA Rank Too Low:** If $r$ is too small, the adaptation matrix cannot capture the complexity of the fine-tuning task, leading to underfitting.
3.  **LoRA Rank Too High:** If $r$ is too large, it approaches full fine-tuning, increasing memory usage and susceptibility to overfitting/catastrophic forgetting without adding significant expressivity.
4.  **Reward Hacking (Goodhart's Law):** In RLHF, the model might find ways to exploit the reward model (e.g., writing overly long, sycophantic responses that humans rate highly but are factually void) rather than actually improving.
5.  **Alignment Tax:** Heavy alignment (RLHF/DPO) often reduces the base model's zero-shot performance on standard benchmarks (e.g., code generation or math), trading capability for safety.
6.  **Distribution Shift in DPO:** DPO relies on off-policy preference data. If the model's generated distribution shifts too far from the data distribution where preferences were collected, the implicit reward estimates become inaccurate.

---

## 6. Connections

*   **Prereqs:** [13 Neural Networks](../13_neural_networks/README.md) for gradient descent; [16 Transformer](../16_transformer/README.md) for the base architecture on which PEFT operates.
*   **Builds on:** Cross-entropy loss (used in SFT), softmax/sigmoid functions.
*   **Next:** Scaling Laws (how these models scale with compute), Inference Optimization (KV Cache, quantization for deployment).

---

## 7. References

*   Sennrich, R., Haddow, B., & Birch, A. (2016). Neural Machine Translation of Rare Words with Subword Units. (BPE Tokenization)
*   Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models.
*   Dettmers, T., et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs.
*   Ouyang, L., et al. (2022). Training language models to follow instructions with human feedback. (InstructGPT/RLHF)
*   Rafailov, R., et al. (2023). Direct Preference Optimization: Your Language Model is Secretly a Reward Model. (DPO)
