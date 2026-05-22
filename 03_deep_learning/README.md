# Deep Learning

Neural networks: compositions of linear transformations and non-linear activations, trained end-to-end via backpropagation. This module progresses from MLPs to CNNs, RNNs, and Transformers.

---

## Prerequisites

### Math (from [math_for_ai_roadmap.md](../00_foundations/01_math_essentials/math_for_ai_roadmap.md))
- **(1) Linear Algebra:** matrix multiplication, tensor operations, rank, norms, SVD.
- **(2) Calculus:** chain rule (backpropagation), Jacobian, Hessian, computation graphs.
- **(3) Probability & Statistics:** cross-entropy, softmax, likelihood, Bayesian interpretation of regularization.
- **(4) Optimization:** SGD, Adam, learning rate schedules, gradient clipping.
- **(6) Numerical Methods:** floating-point stability, logsumexp, mixed precision (FP16/BF16).

### Code
- Python, NumPy (for from-scratch implementations).
- PyTorch (for library implementations and GPU training).
- Matplotlib (for training curves and visualization).

---

## Subprojects (ordered by learning priority)

### [`01_multi_layer_perceptron/`](01_multi_layer_perceptron/)
Multilayer Perceptrons (MLPs) and manual backpropagation.

| Topic | What to implement | Key math |
|---|---|---|
| Single neuron | Forward pass, activation functions | σ(wᵀx + b), ReLU, tanh |
| MLP | 2+ hidden layers from scratch | Matrix chain: y = σ(W₂ · σ(W₁x + b₁) + b₂) |
| Backprop | Manual gradient computation | Chain rule on computation graph |
| Regularization | Dropout, weight decay, batch norm | Variance analysis, L2 penalty |

### [`02_cnn/`](02_cnn/)
CNNs for computer vision (image classification, feature extraction).

| Topic | What to implement | Key math |
|---|---|---|
| Conv2D layer | Convolution from scratch | Cross-correlation, filter/kernel, stride, padding |
| Pooling | Max pool, average pool | Spatial downsampling |
| Architectures | LeNet → VGG → ResNet | Skip connections, depth vs width |
| Transfer learning | Fine-tune pretrained model | Feature reuse |

### [`03_rnn_lstm_gru/`](03_rnn_lstm_gru/)
RNNs, LSTMs, GRUs for sequential data (text, time series).

| Topic | What to implement | Key math |
|---|---|---|
| Vanilla RNN | Forward + BPTT from scratch | hₜ = tanh(Wₕhₜ₋₁ + Wₓxₜ + b) |
| LSTM | Gated recurrence | Forget/input/output gates, cell state |
| GRU | Simplified gating | Update/reset gates |
| Seq2Seq | Encoder-decoder | Teacher forcing, beam search |

### [`04_transformer/`](04_transformer/)
Attention mechanisms and self-attention architectures.

| Topic | What to implement | Key math |
|---|---|---|
| Scaled dot-product attention | Attention from scratch | softmax(QKᵀ/√dₖ)V |
| Multi-head attention | Parallel attention heads | Concat + linear projection |
| Positional encoding | Sinusoidal / RoPE | Fourier basis, rotation matrix |
| Transformer block | Full encoder/decoder block | LayerNorm, residual connection, FFN |

### [`05_autoencoder/`](05_autoencoder/)
Self-supervised representation learning by reconstruction.

| Topic | What to implement | Key math |
|---|---|---|
| Vanilla autoencoder | Encoder → bottleneck → decoder | Reconstruction loss ‖x − x̂‖² |
| Denoising AE | Reconstruct from corrupted input | Robust representations |
| Sparse AE | L1 / KL sparsity penalty on activations | Sparse coding |
| Contractive AE | Penalize Jacobian of encoder | ‖∂h/∂x‖_F² |

---

## Learning Objectives

After completing this module, you should be able to:

- [ ] Implement a 2-layer MLP with backpropagation from scratch in NumPy (no autograd).
- [ ] Implement a Conv2D forward pass from scratch and verify against `torch.nn.Conv2d`.
- [ ] Train a ResNet on CIFAR-10 using PyTorch and achieve >90% accuracy.
- [ ] Implement vanilla RNN + BPTT from scratch; explain vanishing gradient with a concrete example.
- [ ] Implement scaled dot-product attention from scratch; explain why dividing by √dₖ.
- [ ] Build a single Transformer encoder block from scratch in PyTorch.

---

## Key References

- Goodfellow, Bengio, Courville — *Deep Learning* (free at [deeplearningbook.org](https://www.deeplearningbook.org/)), Chapters 6-10.
- Karpathy — *Neural Networks: Zero to Hero* (YouTube series).
- He et al. (2016) — *Deep Residual Learning for Image Recognition*.
- Vaswani et al. (2017) — *Attention Is All You Need*.
- Hochreiter & Schmidhuber (1997) — *Long Short-Term Memory*.

---

## Subproject Layout

Each subproject should follow:
```
algorithm_name/
├── data/           # Datasets (gitignored if large)
├── notebooks/      # Training visualization, architecture diagrams
├── src/            # From-scratch implementation + PyTorch version
├── tests/          # Unit tests (gradient checking, output shape)
└── reports/        # Training curves, comparisons, findings
```
