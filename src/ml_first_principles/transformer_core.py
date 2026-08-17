"""Sequence-model building blocks with manual backprop: embedding, norm, attention.

Every layer operates on batched sequences and follows one uniform interface designed
to pair with the step-based :class:`ml_first_principles.optimizers.Adam`:

- ``params``: dict mapping parameter name to array (empty for parameter-free layers).
- ``grads``: dict with the same keys, overwritten by each ``backward`` call.
- ``forward(x)``: compute outputs and cache whatever the backward pass needs.
- ``backward(dout)``: fill ``grads`` and return the gradient w.r.t. the input.

Composite layers expose nested parameters through prefixed keys (e.g. ``"attn.w_q"``)
so a single flat dict drives one optimizer ``step`` call. The math is ported from the
hand-derived reference implementation in ``projects/char_transformer_tiny`` (verified
there by an exhaustive finite-difference gradient check, repeated in this repo's tests).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
IntArray = NDArray[np.integer]
ParamDict = dict[str, NDArray[np.float64]]

EMBEDDING_INIT_STD = 0.02


def _causal_softmax(scores: Array) -> Array:
    """Row-wise softmax over the last axis with a causal mask, max-shifted for stability."""
    length = scores.shape[-1]
    mask = np.tril(np.ones((length, length), dtype=bool))
    masked = np.where(mask, scores, -np.inf)
    shifted = masked - masked.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


class Embedding:
    """Lookup table mapping integer token ids ``(B, T)`` to vectors ``(B, T, d_model)``.

    The weight is initialized from a scaled Gaussian ``N(0, 0.02^2)``. The backward
    pass scatter-adds the upstream gradient into the rows selected by the cached ids
    (``np.add.at``), so repeated ids accumulate correctly.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        """Initialize the embedding table.

        Args:
            vocab_size: Number of rows (distinct token ids), positive.
            d_model: Embedding dimension, positive.
            random_state: Seed or generator passed to ``np.random.default_rng``.

        Raises:
            ValueError: If ``vocab_size`` or ``d_model`` is not positive.
        """
        if vocab_size < 1 or d_model < 1:
            raise ValueError("vocab_size and d_model must be positive")
        rng = np.random.default_rng(random_state)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.params: ParamDict = {
            "weight": EMBEDDING_INIT_STD * rng.standard_normal((vocab_size, d_model))
        }
        self.grads: ParamDict = {"weight": np.zeros((vocab_size, d_model))}
        self._ids: IntArray | None = None

    def forward(self, ids: IntArray) -> Array:
        """Look up embedding rows for a batch of token-id sequences.

        Args:
            ids: Integer array of shape ``(B, T)`` with values in ``[0, vocab_size)``.

        Returns:
            Array of shape ``(B, T, d_model)``.

        Raises:
            ValueError: If ``ids`` is not a non-empty 2-D integer array in range.
        """
        tokens = np.asarray(ids)
        if tokens.ndim != 2 or tokens.size == 0 or not np.issubdtype(tokens.dtype, np.integer):
            raise ValueError("ids must be a non-empty two-dimensional integer array")
        if tokens.min() < 0 or tokens.max() >= self.vocab_size:
            raise ValueError("ids must lie in [0, vocab_size)")
        self._ids = tokens
        return self.params["weight"][tokens]

    def backward(self, dout: Array) -> None:
        """Scatter-add ``dout`` into the weight gradient.

        Token ids are not differentiable, so no input gradient is returned.

        Args:
            dout: Upstream gradient of shape ``(B, T, d_model)``.

        Raises:
            RuntimeError: If ``forward`` has not been called yet.
            ValueError: If ``dout`` does not match the cached forward shape.
        """
        if self._ids is None:
            raise RuntimeError("forward must be called before backward")
        upstream = np.asarray(dout, dtype=float)
        if upstream.shape != (*self._ids.shape, self.d_model):
            raise ValueError("dout has an incompatible shape")
        grad = np.zeros_like(self.params["weight"])
        np.add.at(grad, self._ids, upstream)
        self.grads["weight"] = grad


class LayerNorm:
    """Layer normalization over the last axis with learnable scale and shift.

    Normalizes each position to zero mean and unit variance across ``d_model``,
    then applies ``gamma * x_hat + beta``. Backward uses the standard closed form
    ``dx = inv_std * (dx_hat - mean(dx_hat) - x_hat * mean(dx_hat * x_hat))`` with
    means taken over the normalized axis.
    """

    def __init__(self, d_model: int, epsilon: float = 1e-5) -> None:
        """Initialize with ``gamma = 1`` and ``beta = 0``.

        Args:
            d_model: Size of the normalized (last) axis, positive.
            epsilon: Positive constant added to the variance for stability.

        Raises:
            ValueError: If ``d_model`` is not positive or ``epsilon`` is not positive.
        """
        if d_model < 1:
            raise ValueError("d_model must be positive")
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        self.d_model = d_model
        self.epsilon = epsilon
        self.params: ParamDict = {"gamma": np.ones(d_model), "beta": np.zeros(d_model)}
        self.grads: ParamDict = {"gamma": np.zeros(d_model), "beta": np.zeros(d_model)}
        self._x_hat: Array | None = None
        self._inv_std: Array | None = None

    def forward(self, x: Array) -> Array:
        """Normalize over the last axis and apply the affine transform.

        Args:
            x: Array of shape ``(..., d_model)`` with at least two dimensions.

        Returns:
            Array with the same shape as ``x``.

        Raises:
            ValueError: If the last axis of ``x`` is not ``d_model``.
        """
        values = np.asarray(x, dtype=float)
        if values.ndim < 2 or values.shape[-1] != self.d_model:
            raise ValueError("x must have shape (..., d_model)")
        mean = values.mean(axis=-1, keepdims=True)
        centered = values - mean
        variance = (centered**2).mean(axis=-1, keepdims=True)
        inv_std = 1.0 / np.sqrt(variance + self.epsilon)
        x_hat = centered * inv_std
        self._x_hat = x_hat
        self._inv_std = inv_std
        return self.params["gamma"] * x_hat + self.params["beta"]

    def backward(self, dout: Array) -> Array:
        """Fill ``grads`` for gamma/beta and return the input gradient.

        Args:
            dout: Upstream gradient with the forward output shape.

        Returns:
            Gradient w.r.t. the forward input, same shape as ``dout``.

        Raises:
            RuntimeError: If ``forward`` has not been called yet.
            ValueError: If ``dout`` does not match the cached forward shape.
        """
        if self._x_hat is None or self._inv_std is None:
            raise RuntimeError("forward must be called before backward")
        upstream = np.asarray(dout, dtype=float)
        if upstream.shape != self._x_hat.shape:
            raise ValueError("dout has an incompatible shape")
        batch_axes = tuple(range(upstream.ndim - 1))
        self.grads["gamma"] = (upstream * self._x_hat).sum(axis=batch_axes)
        self.grads["beta"] = upstream.sum(axis=batch_axes)
        dx_hat = upstream * self.params["gamma"]
        mean_dx_hat = dx_hat.mean(axis=-1, keepdims=True)
        mean_dx_hat_x = (dx_hat * self._x_hat).mean(axis=-1, keepdims=True)
        return self._inv_std * (dx_hat - mean_dx_hat - self._x_hat * mean_dx_hat_x)


class CausalSelfAttention:
    """Single-head causal self-attention with output projection.

    Computes ``softmax(mask(Q K^T / sqrt(d_model))) V W_o`` where the mask is lower
    triangular (each position attends only to itself and earlier positions) and the
    softmax is max-shifted for numerical stability. Masked entries have attention
    weight exactly zero, so they contribute zero gradient in the softmax backward
    ``dS = A * (dA - sum(dA * A))``. The residual connection is the caller's job
    (see :class:`TransformerBlock`).
    """

    def __init__(self, d_model: int, random_state: int | np.random.Generator | None = None) -> None:
        """Initialize the four projection matrices with std ``1 / sqrt(d_model)``.

        Args:
            d_model: Model width, positive. Queries, keys, values, and outputs all
                have this dimension (single head).
            random_state: Seed or generator passed to ``np.random.default_rng``.

        Raises:
            ValueError: If ``d_model`` is not positive.
        """
        if d_model < 1:
            raise ValueError("d_model must be positive")
        rng = np.random.default_rng(random_state)
        self.d_model = d_model
        scale = 1.0 / np.sqrt(d_model)
        self.params: ParamDict = {
            name: scale * rng.standard_normal((d_model, d_model))
            for name in ("w_q", "w_k", "w_v", "w_o")
        }
        self.grads: ParamDict = {name: np.zeros((d_model, d_model)) for name in self.params}
        self._cache: dict[str, Array] | None = None

    def forward(self, x: Array) -> Array:
        """Apply causal self-attention to a batch of sequences.

        Args:
            x: Array of shape ``(B, T, d_model)``.

        Returns:
            Array of shape ``(B, T, d_model)``.

        Raises:
            ValueError: If ``x`` is not 3-D with last axis ``d_model``.
        """
        values = np.asarray(x, dtype=float)
        if values.ndim != 3 or values.shape[-1] != self.d_model:
            raise ValueError("x must have shape (B, T, d_model)")
        scale = 1.0 / np.sqrt(self.d_model)
        q = values @ self.params["w_q"]
        k = values @ self.params["w_k"]
        v = values @ self.params["w_v"]
        attn = _causal_softmax(q @ k.transpose(0, 2, 1) * scale)  # (B, T, T)
        ctx = attn @ v  # (B, T, d_model)
        self._cache = {"x": values, "q": q, "k": k, "v": v, "attn": attn, "ctx": ctx}
        return ctx @ self.params["w_o"]

    def backward(self, dout: Array) -> Array:
        """Fill ``grads`` for the four projections and return the input gradient.

        Args:
            dout: Upstream gradient of shape ``(B, T, d_model)``.

        Returns:
            Gradient w.r.t. the forward input, shape ``(B, T, d_model)``.

        Raises:
            RuntimeError: If ``forward`` has not been called yet.
            ValueError: If ``dout`` does not match the cached forward shape.
        """
        if self._cache is None:
            raise RuntimeError("forward must be called before backward")
        cache = self._cache
        upstream = np.asarray(dout, dtype=float)
        if upstream.shape != cache["x"].shape:
            raise ValueError("dout has an incompatible shape")
        self.grads["w_o"] = np.einsum("btd,bte->de", cache["ctx"], upstream)
        dctx = upstream @ self.params["w_o"].T
        dattn = dctx @ cache["v"].transpose(0, 2, 1)
        dv = cache["attn"].transpose(0, 2, 1) @ dctx
        # Softmax backward; masked entries have attn == 0, so their gradients vanish.
        dscores = cache["attn"] * (dattn - (dattn * cache["attn"]).sum(axis=-1, keepdims=True))
        dscores *= 1.0 / np.sqrt(self.d_model)
        dq = dscores @ cache["k"]
        dk = dscores.transpose(0, 2, 1) @ cache["q"]
        self.grads["w_q"] = np.einsum("btd,bte->de", cache["x"], dq)
        self.grads["w_k"] = np.einsum("btd,bte->de", cache["x"], dk)
        self.grads["w_v"] = np.einsum("btd,bte->de", cache["x"], dv)
        return dq @ self.params["w_q"].T + dk @ self.params["w_k"].T + dv @ self.params["w_v"].T


class TransformerBlock:
    """Causal self-attention and a two-layer ReLU MLP, each wrapped in a residual.

    ``out = a + relu(a W1 + b1) W2 + b2`` with ``a = x + attn(x)``. No LayerNorm —
    this matches the hand-derived reference model in ``projects/char_transformer_tiny``
    (a deliberate simplification that trains fine at small scale); compose with
    :class:`LayerNorm` explicitly if normalization is wanted.

    ``params``/``grads`` are flat dicts exposing the nested attention layer through
    prefixed keys (``"attn.w_q"``, ...) next to the block-owned MLP parameters
    (``"w1"``, ``"b1"``, ``"w2"``, ``"b2"``), so one dict drives ``Adam.step``.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        """Initialize the attention sublayer and MLP weights (std ``1/sqrt(fan_in)``).

        Args:
            d_model: Model width, positive.
            d_ff: Hidden width of the MLP, positive.
            random_state: Seed or generator passed to ``np.random.default_rng``.
                Draw order: attention ``w_q, w_k, w_v, w_o``, then ``w1``, ``w2``.

        Raises:
            ValueError: If ``d_model`` or ``d_ff`` is not positive.
        """
        if d_model < 1 or d_ff < 1:
            raise ValueError("d_model and d_ff must be positive")
        rng = np.random.default_rng(random_state)
        self.d_model = d_model
        self.d_ff = d_ff
        self.attn = CausalSelfAttention(d_model, random_state=rng)
        self._mlp_params: ParamDict = {
            "w1": rng.standard_normal((d_model, d_ff)) / np.sqrt(d_model),
            "b1": np.zeros(d_ff),
            "w2": rng.standard_normal((d_ff, d_model)) / np.sqrt(d_ff),
            "b2": np.zeros(d_model),
        }
        self._mlp_grads: ParamDict = {
            name: np.zeros_like(value) for name, value in self._mlp_params.items()
        }
        self._cache: dict[str, Array] | None = None

    @property
    def params(self) -> ParamDict:
        """Flat parameter dict: ``attn.``-prefixed sublayer entries plus MLP weights."""
        merged = {f"attn.{name}": value for name, value in self.attn.params.items()}
        merged.update(self._mlp_params)
        return merged

    @property
    def grads(self) -> ParamDict:
        """Flat gradient dict with the same keys as :attr:`params`."""
        merged = {f"attn.{name}": value for name, value in self.attn.grads.items()}
        merged.update(self._mlp_grads)
        return merged

    def forward(self, x: Array) -> Array:
        """Apply the block to a batch of sequences.

        Args:
            x: Array of shape ``(B, T, d_model)``.

        Returns:
            Array of shape ``(B, T, d_model)``.

        Raises:
            ValueError: If ``x`` is not 3-D with last axis ``d_model``.
        """
        values = np.asarray(x, dtype=float)
        if values.ndim != 3 or values.shape[-1] != self.d_model:
            raise ValueError("x must have shape (B, T, d_model)")
        attended = values + self.attn.forward(values)  # residual around attention
        pre = attended @ self._mlp_params["w1"] + self._mlp_params["b1"]
        hidden = np.maximum(pre, 0.0)
        self._cache = {"attended": attended, "pre": pre, "hidden": hidden}
        return attended + hidden @ self._mlp_params["w2"] + self._mlp_params["b2"]

    def backward(self, dout: Array) -> Array:
        """Fill all gradients (attention and MLP) and return the input gradient.

        Args:
            dout: Upstream gradient of shape ``(B, T, d_model)``.

        Returns:
            Gradient w.r.t. the forward input, shape ``(B, T, d_model)``.

        Raises:
            RuntimeError: If ``forward`` has not been called yet.
            ValueError: If ``dout`` does not match the cached forward shape.
        """
        if self._cache is None:
            raise RuntimeError("forward must be called before backward")
        cache = self._cache
        upstream = np.asarray(dout, dtype=float)
        if upstream.shape != cache["attended"].shape:
            raise ValueError("dout has an incompatible shape")
        self._mlp_grads["w2"] = np.einsum("btf,btd->fd", cache["hidden"], upstream)
        self._mlp_grads["b2"] = upstream.sum(axis=(0, 1))
        dhidden = upstream @ self._mlp_params["w2"].T
        dpre = dhidden * (cache["pre"] > 0.0)
        self._mlp_grads["w1"] = np.einsum("btd,btf->df", cache["attended"], dpre)
        self._mlp_grads["b1"] = dpre.sum(axis=(0, 1))
        dattended = upstream + dpre @ self._mlp_params["w1"].T  # residual around MLP
        return dattended + self.attn.backward(dattended)  # residual around attention


def softmax_cross_entropy(logits: Array, targets: IntArray) -> tuple[float, Array]:
    """Mean softmax cross-entropy over all positions and its gradient w.r.t. logits.

    Computed in log-space with a max shift: ``log p = z - max(z) - log sum exp(z - max(z))``,
    so no probability is ever exponentiated before normalization. The gradient is the
    standard ``(softmax(z) - onehot(y)) / N`` with ``N = B * T``.

    Args:
        logits: Array of shape ``(B, T, V)``.
        targets: Integer array of shape ``(B, T)`` with values in ``[0, V)``.

    Returns:
        Tuple of the scalar mean loss and ``dL/dlogits`` with the shape of ``logits``.

    Raises:
        ValueError: If shapes are incompatible or targets are out of range.
    """
    values = np.asarray(logits, dtype=float)
    labels = np.asarray(targets)
    if values.ndim != 3:
        raise ValueError("logits must have shape (B, T, V)")
    if labels.shape != values.shape[:2] or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("targets must be an integer array of shape (B, T)")
    n_batch, n_steps, n_classes = values.shape
    if labels.size == 0 or labels.min() < 0 or labels.max() >= n_classes:
        raise ValueError("targets must be non-empty with values in [0, V)")
    shifted = values - values.max(axis=-1, keepdims=True)
    log_z = np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    log_probs = shifted - log_z  # (B, T, V)
    n_positions = n_batch * n_steps
    flat_labels = labels.reshape(n_positions)
    loss = -log_probs.reshape(n_positions, n_classes)[np.arange(n_positions), flat_labels].mean()
    dlogits = np.exp(log_probs)  # softmax probabilities
    dlogits.reshape(n_positions, n_classes)[np.arange(n_positions), flat_labels] -= 1.0
    dlogits /= n_positions
    return float(loss), dlogits
