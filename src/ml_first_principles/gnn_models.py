"""First-principles pure NumPy implementations of graph neural networks (GCN, GAT)."""

from __future__ import annotations

import numpy as np

LEAKY_RELU_ALPHA = 0.2


def _glorot_init(rng: np.random.Generator, fan_in: int, fan_out: int) -> np.ndarray:
    return rng.standard_normal((fan_in, fan_out)) * np.sqrt(2.0 / (fan_in + fan_out))


def _validated_adjacency(x: np.ndarray, adj: np.ndarray) -> np.ndarray:
    """Return the adjacency with self-loops, validating shapes and signs.

    Self-loops are added only where the diagonal is zero, so an adjacency that
    already contains them is not double-counted.
    """
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError("adj must be a square matrix")
    if x.ndim != 2 or x.shape[0] != adj.shape[0]:
        raise ValueError("x must have one row per adjacency node")
    if np.any(adj < 0):
        raise ValueError("adj must be non-negative")
    adj_tilde = np.array(adj, dtype=float)
    np.fill_diagonal(adj_tilde, np.maximum(np.diag(adj_tilde), 1.0))
    return adj_tilde


class GCNLayer:
    """Graph Convolutional Network layer in pure NumPy."""

    def __init__(
        self, in_features: int, out_features: int, random_state: int | None = None
    ) -> None:
        """Initialize the layer weight with Glorot-scaled noise.

        Args:
            in_features: Input feature dimensionality per node.
            out_features: Output feature dimensionality per node.
            random_state: Seed for the isolated random generator.

        Raises:
            ValueError: If a dimension is not positive.
        """
        if in_features < 1 or out_features < 1:
            raise ValueError("in_features and out_features must be positive")
        self.in_features = in_features
        self.out_features = out_features
        self.weight = _glorot_init(np.random.default_rng(random_state), in_features, out_features)

    def forward(self, x: np.ndarray, adj: np.ndarray) -> np.ndarray:
        r"""Apply :math:`H' = D^{-1/2}\tilde{A}D^{-1/2}XW`.

        Args:
            x: Node features of shape ``(n_nodes, in_features)``.
            adj: Non-negative adjacency of shape ``(n_nodes, n_nodes)``;
                self-loops are added where missing.

        Returns:
            Convolved node features of shape ``(n_nodes, out_features)``.

        Raises:
            ValueError: If shapes are incompatible or ``adj`` is negative.
        """
        adj_tilde = _validated_adjacency(x, adj)
        deg = np.sum(adj_tilde, axis=1)
        deg_inv_sqrt = 1.0 / np.sqrt(deg)

        norm_adj = adj_tilde * deg_inv_sqrt[:, None] * deg_inv_sqrt[None, :]
        support = x @ self.weight
        return norm_adj @ support


class GATLayer:
    """Single-head Graph Attention Network layer in pure NumPy."""

    def __init__(
        self, in_features: int, out_features: int, random_state: int | None = None
    ) -> None:
        """Initialize the projection and attention weights.

        Args:
            in_features: Input feature dimensionality per node.
            out_features: Output feature dimensionality per node.
            random_state: Seed for the isolated random generator.

        Raises:
            ValueError: If a dimension is not positive.
        """
        if in_features < 1 or out_features < 1:
            raise ValueError("in_features and out_features must be positive")
        self.in_features = in_features
        self.out_features = out_features
        rng = np.random.default_rng(random_state)
        self.weight = _glorot_init(rng, in_features, out_features)
        self.a = _glorot_init(rng, 2 * out_features, 1)

    @staticmethod
    def _leaky_relu(x: np.ndarray, alpha: float = LEAKY_RELU_ALPHA) -> np.ndarray:
        return np.where(x > 0, x, alpha * x)

    def forward(self, x: np.ndarray, adj: np.ndarray) -> np.ndarray:
        """Aggregate neighbor features with learned attention weights.

        Attention scores of non-neighbors are masked to ``-inf`` before the
        softmax, so they receive exactly zero weight; every node attends at
        least to itself through the added self-loop.

        Args:
            x: Node features of shape ``(n_nodes, in_features)``.
            adj: Non-negative adjacency of shape ``(n_nodes, n_nodes)``.

        Returns:
            Attention-aggregated features of shape ``(n_nodes, out_features)``.

        Raises:
            ValueError: If shapes are incompatible or ``adj`` is negative.
        """
        adj_tilde = _validated_adjacency(x, adj)
        num_nodes = x.shape[0]
        h = x @ self.weight  # [N, out_features]

        # Compute pairwise attention inputs.
        h_i = np.repeat(h, num_nodes, axis=0).reshape(num_nodes, num_nodes, self.out_features)
        h_j = np.tile(h, (num_nodes, 1)).reshape(num_nodes, num_nodes, self.out_features)
        a_input = np.concatenate([h_i, h_j], axis=-1)  # [N, N, 2*out_features]

        e = self._leaky_relu(a_input @ self.a).squeeze(-1)  # [N, N]
        e = np.where(adj_tilde > 0, e, -np.inf)

        # Row-wise softmax; every row has at least the self-loop entry finite.
        e_exp = np.exp(e - np.max(e, axis=-1, keepdims=True))
        alpha = e_exp / np.sum(e_exp, axis=-1, keepdims=True)

        return alpha @ h
