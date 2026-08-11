# Topic 20: Graph Neural Networks (GNNs)

## 0. Notation Table

| Symbol | Type | Meaning |
|--------|------|---------|
| $G = (V, E)$ | Tuple | Graph with vertices $V$ and edges $E$ |
| $N$ | Scalar | Number of nodes, $|V|$ |
| $F$ | Scalar | Dimensionality of node features |
| $A$ | Matrix | Adjacency matrix of size $N \times N$ |
| $D$ | Matrix | Degree matrix (diagonal) of size $N \times N$ |
| $L$ | Matrix | Unnormalized Graph Laplacian, $L = D - A$ |
| $L_{sym}$ | Matrix | Symmetric normalized Graph Laplacian |
| $H^{(l)}$ | Matrix | Node representations at layer $l$, size $N \times F_l$ |
| $W^{(l)}$ | Matrix | Learnable weight matrix at layer $l$ |
| $h_i$ | Vector | Feature vector of node $i$ |
| $\alpha_{ij}$ | Scalar | Attention coefficient from node $j$ to node $i$ |
| $\mathcal{N}(i)$ | Set | Set of neighbors of node $i$ (sometimes including $i$) |

## 1. WHY: Motivation and Problem Statement

Standard deep learning architectures (like CNNs and RNNs) are designed for data with a regular grid-like structure, such as images (2D grids) or text sequences (1D grids). However, many real-world datasets naturally reside in non-Euclidean spaces and are best represented as graphs. Examples include:
- **Social Networks:** Users and their connections.
- **Molecules:** Atoms as nodes, chemical bonds as edges.
- **Citation Networks:** Papers as nodes, citations as edges.

### Why Standard Neural Networks Fail
Applying standard feed-forward networks to graphs poses several challenges:
1. **Arbitrary Size and Topology:** Graph nodes can have varying numbers of neighbors, unlike image pixels which have a fixed neighborhood.
2. **Permutation Invariance/Equivariance:** If we reorder the nodes in the adjacency matrix and feature matrix, the graph structure remains the same. A neural network must process this data such that the output node representations are equivariant to the node ordering, and graph-level predictions are invariant.
3. **Complex Relationships:** Standard MLPs treat data instances independently, failing to capture the explicit topological relationships encoded by the edges.

Graph Neural Networks solve this by leveraging the graph structure explicitly during the feature transformation process.

## 2. WHAT: Graph Representations and Model Formulation

Before formulating operations on graphs, we establish how graphs are mathematically represented.

**Assumptions:** We primarily deal with simple graphs that are undirected, unweighted, and without self-loops, although the concepts naturally extend to directed and weighted variants.

**Adjacency Matrix ($A$):**
An $N \times N$ matrix where $A_{ij} = 1$ if there is an edge between node $i$ and node $j$, and $0$ otherwise. For undirected graphs, $A$ is symmetric.

**Degree Matrix ($D$):**
A diagonal $N \times N$ matrix where $D_{ii} = \sum_j A_{ij}$ is the degree of node $i$.

**Node Features ($X$):**
An $N \times F$ matrix where the $i$-th row corresponds to the $F$-dimensional feature vector of node $i$. (We often denote the initial features as $H^{(0)} = X$).

**Objective Function:**
Depending on the task, GNNs optimize standard objectives:
- **Node Classification:** Cross-entropy loss on node representations.
- **Graph Classification:** Cross-entropy loss on a globally pooled graph representation.
- **Link Prediction:** Binary cross-entropy on dot products of node pairs.

## 3. HOW: Spectral Graph Theory

Early GNNs were inspired by signal processing on graphs, operating in the spectral domain.

### Graph Laplacian
The fundamental operator in spectral graph theory is the unnormalized Graph Laplacian:
$$L = D - A$$
**Rule Used:** Graph representation definition.

**Properties of $L$:**
1. **Symmetric:** Since $D$ and $A$ are symmetric for undirected graphs.
2. **Positive Semi-Definite (PSD):** For any vector $x \in \mathbb{R}^N$, $x^T L x = \frac{1}{2} \sum_{i,j} A_{ij} (x_i - x_j)^2 \geq 0$.
3. The smallest eigenvalue is always $0$, with the constant vector $\mathbf{1}$ as its eigenvector ($L\mathbf{1} = 0$).

### Normalized Laplacian
To maintain numerical stability, we often use the symmetric normalized Laplacian:
$$L_{sym} = I - D^{-1/2} A D^{-1/2}$$
**Rule Used:** Normalization by node degrees.

### Eigendecomposition and Graph Fourier Transform
Since $L$ (or $L_{sym}$) is symmetric and real, it admits an eigendecomposition:
$$L = U \Lambda U^T$$
where $U \in \mathbb{R}^{N \times N}$ is the matrix of orthonormal eigenvectors and $\Lambda$ is the diagonal matrix of eigenvalues ($\lambda_1 \leq \lambda_2 \leq \dots \leq \lambda_N$).

The eigenvectors $U$ form a basis for graph signals. We define the **Graph Fourier Transform** of a signal $x \in \mathbb{R}^N$ as:
$$\hat{x} = U^T x$$
**Rule Used:** Change of basis to Laplacian eigenbasis.

### Spectral Convolution
Given a filter $g_\theta$ in the spectral domain, the convolution of $x$ with $g_\theta$ is:
$$g_\theta \star x = U g_\theta(\Lambda) U^T x$$
where $g_\theta(\Lambda)$ is a diagonal matrix of learnable filter coefficients.

### Chebyshev Approximation (ChebNet)
Computing $U$ is $\mathcal{O}(N^3)$, which is intractable for large graphs. Defferrard et al. approximated the filter using a truncated Chebyshev polynomial of order $K$:
$$g_{\theta}(\Lambda) \approx \sum_{k=0}^{K} \theta_k T_k(\tilde{\Lambda})$$
where $\tilde{\Lambda} = \frac{2}{\lambda_{max}} \Lambda - I$ scales eigenvalues to $[-1, 1]$.
By applying this to the Laplacian, the convolution avoids the eigendecomposition:
$$g_\theta \star x \approx \sum_{k=0}^{K} \theta_k T_k(\tilde{L}) x$$
**Result:** A localized, scalable graph convolution where $K$ controls the neighborhood radius.

## 4. HOW: Graph Convolutional Networks (GCN)

Kipf and Welling simplified ChebNet by setting $K=1$ and making specific approximations to yield an efficient, spatial-like convolution layer.

### Step-by-Step Derivation
**Step 1: First-order Approximation**
Set $K=1$ and assume $\lambda_{max} \approx 2$ (true for many normalized graphs).
$$g_\theta \star x \approx \theta_0 x + \theta_1 (L_{sym} - I) x$$
**Rule Used:** Truncated Chebyshev polynomial.

**Step 2: Laplacian Substitution**
Substitute $L_{sym} = I - D^{-1/2} A D^{-1/2}$:
$$g_\theta \star x \approx \theta_0 x - \theta_1 (D^{-1/2} A D^{-1/2}) x$$
**Rule Used:** Definition of normalized Laplacian.

**Step 3: Single Parameter Restriction**
To constrain the number of parameters and avoid overfitting, set $\theta = \theta_0 = -\theta_1$:
$$g_\theta \star x \approx \theta (I + D^{-1/2} A D^{-1/2}) x$$
**Rule Used:** Parameter sharing/regularization.

**Step 4: Renormalization Trick**
The operator $I + D^{-1/2} A D^{-1/2}$ has eigenvalues in $[0, 2]$, leading to numerical instability (exploding/vanishing gradients) in deep models. Kipf & Welling proposed replacing it with:
$$\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}$$
where $\tilde{A} = A + I$ (adding self-loops) and $\tilde{D}_{ii} = \sum_j \tilde{A}_{ij}$.
**Rule Used:** Spectral norm bounding via renormalization.

**Step 5: Layer-wise Propagation**
Generalizing to multi-dimensional features $H^{(l)}$ and multiple filters $W^{(l)}$, we get the final GCN layer:
$$H^{(l+1)} = \sigma(\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(l)} W^{(l)})$$
**Result:** The standard GCN forward pass, which elegantly bridges spectral theory and spatial message passing.

### The Over-Smoothing Problem
A major limitation of GCNs is **over-smoothing**. As the number of layers increases, the repeated multiplication by the normalized adjacency matrix acts as a low-pass filter. Eventually, the node representations become indistinguishable and converge to a stationary distribution proportional to node degrees. This limits GCNs to shallow architectures (typically 2-4 layers).

## 5. HOW: Graph Attention Networks (GAT)

GCN uses fixed aggregation weights ($1/\sqrt{D_{ii}D_{jj}}$). Graph Attention Networks (Veličković et al.) replace this with learned attention, allowing the model to weigh neighbors differently based on their features.

**Motivation:** Not all neighbors are equally important. Learned attention provides expressivity and handles noisy edges better.

### Attention Coefficient Computation
For nodes $i$ and $j$, the attention mechanism computes a raw score:
$$e_{ij} = \text{LeakyReLU}(a^T [W h_i || W h_j])$$
where $a$ is a learnable weight vector and $||$ denotes concatenation.

This score is normalized across all neighbors $j \in \mathcal{N}(i)$ using a softmax function:
$$\alpha_{ij} = \text{softmax}_j(e_{ij}) = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}(i)} \exp(e_{ik})}$$
**Rule Used:** Softmax normalization.

### Node Update
The updated node representation is the weighted sum of transformed neighbor features:
$$h_i^{(l+1)} = \sigma \left( \sum_{j \in \mathcal{N}(i)} \alpha_{ij} W h_j^{(l)} \right)$$

### Multi-Head Attention
To stabilize learning, GAT employs multi-head attention. For $K$ independent attention heads, the outputs are typically concatenated for hidden layers:
$$h_i^{(l+1)} = \Big\|_{k=1}^K \sigma \left( \sum_{j \in \mathcal{N}(i)} \alpha_{ij}^k W^k h_j^{(l)} \right)$$
For the final output layer (e.g., classification), averaging is preferred over concatenation:
$$h_i^{(out)} = \sigma \left( \frac{1}{K} \sum_{k=1}^K \sum_{j \in \mathcal{N}(i)} \alpha_{ij}^k W^k h_j^{(L-1)} \right)$$
**Result:** A fully spatial, attention-driven node update mechanism.

## 6. HOW: Message Passing Neural Networks (MPNN)

Gilmer et al. formalized GNNs into a unified spatial framework called **Message Passing Neural Networks**. The forward pass consists of two phases:

1. **Aggregate Phase (Message Generation & Aggregation):**
$$m_i^{(l+1)} = \sum_{j \in \mathcal{N}(i)} M_l(h_i^{(l)}, h_j^{(l)}, e_{ij})$$
where $M_l$ is a message function (e.g., MLPs) and $e_{ij}$ represents edge features. The aggregation operator (sum, max, mean) must be permutation invariant.

2. **Combine Phase (Node Update):**
$$h_i^{(l+1)} = U_l(h_i^{(l)}, m_i^{(l+1)})$$
where $U_l$ is an update function (e.g., GRU, dense layer).

GCN and GAT are special cases of MPNN with specific message and update functions.

## 7. Failure Cases

1. **Over-smoothing:** As depth increases, node features collapse to indistinguishable vectors, preventing deep GNN architectures.
2. **Over-squashing:** Information from exponentially growing neighborhoods is compressed into a fixed-size vector, causing a loss of long-range dependencies in graphs with high degrees or small diameters (bottlenecks).
3. **Scalability ($O(N^2)$ Memory):** Standard GNN implementations require storing the dense or sparse $N \times N$ adjacency matrix and intermediate activations, failing on very large graphs without specialized mini-batching (e.g., GraphSAGE, ClusterGCN).
4. **Heterophily Assumption:** GCNs implicitly assume homophily (connected nodes share similar labels). On heterophilic graphs (e.g., fraud networks where fraudsters connect to normal users), standard smoothing degrades performance.
5. **Disconnected Components:** Without global context or virtual nodes, message passing cannot route information between disconnected subgraphs.
6. **Feature-less Nodes:** If nodes lack informative features, topological structure alone may be insufficient for high accuracy unless structural encodings are injected.

## 8. Connections

- **Dimensionality Reduction:** Similar to PCA finding the principal axes of data variance, the eigenvectors of the Graph Laplacian find the principal axes of variation on the graph topology.
- **Attention Mechanisms:** GAT relies on the same core principles as Transformers, adapted for sparse graph topologies instead of dense sequence interactions.

## 9. References

- Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. ICLR.
- Veličković, P., et al. (2018). Graph attention networks. ICLR.
- Defferrard, M., et al. (2016). Convolutional neural networks on graphs with fast localized spectral filtering. NeurIPS.
- Gilmer, J., et al. (2017). Neural message passing for quantum chemistry. ICML.
