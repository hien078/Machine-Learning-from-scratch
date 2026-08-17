"""Tests for transformer_core: gradient checks, causality, norm stats, determinism."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from ml_first_principles.optimizers import Adam
from ml_first_principles.transformer_core import (
    CausalSelfAttention,
    Embedding,
    LayerNorm,
    TransformerBlock,
    softmax_cross_entropy,
)

ATOL = 1e-7
RTOL = 1e-4
STEP = 1e-5


def numeric_gradient(loss_fn: Callable[[], float], param: np.ndarray) -> np.ndarray:
    """Central finite differences of ``loss_fn`` w.r.t. every entry of ``param`` (in place)."""
    numeric = np.zeros_like(param)
    flat = param.reshape(-1)
    for i in range(flat.size):
        orig = flat[i]
        flat[i] = orig + STEP
        loss_plus = loss_fn()
        flat[i] = orig - STEP
        loss_minus = loss_fn()
        flat[i] = orig
        numeric.reshape(-1)[i] = (loss_plus - loss_minus) / (2.0 * STEP)
    return numeric


def check_all_params(
    loss_fn: Callable[[], float], params: dict[str, np.ndarray], grads: dict[str, np.ndarray]
) -> None:
    """Compare analytic grads against finite differences for every parameter entry."""
    assert set(grads) == set(params)
    for name, param in params.items():
        numeric = numeric_gradient(loss_fn, param)
        np.testing.assert_allclose(
            grads[name], numeric, atol=ATOL, rtol=RTOL, err_msg=f"gradient mismatch in {name}"
        )


# ---------------------------------------------------------------------------
# Gradient checks: every parameter of every layer, plus input gradients.
# ---------------------------------------------------------------------------


def test_embedding_gradient_check() -> None:
    rng = np.random.default_rng(0)
    layer = Embedding(7, 4, random_state=0)
    ids = rng.integers(0, 7, size=(2, 3))
    weighting = rng.standard_normal((2, 3, 4))

    layer.forward(ids)
    layer.backward(weighting)

    check_all_params(
        lambda: float(np.sum(layer.forward(ids) * weighting)), layer.params, layer.grads
    )


def test_layernorm_gradient_check() -> None:
    rng = np.random.default_rng(1)
    layer = LayerNorm(5)
    layer.params["gamma"] = rng.standard_normal(5)  # non-trivial affine params
    layer.params["beta"] = rng.standard_normal(5)
    x = rng.standard_normal((2, 3, 5))
    weighting = rng.standard_normal((2, 3, 5))

    layer.forward(x)
    dx = layer.backward(weighting)

    def loss_fn() -> float:
        return float(np.sum(layer.forward(x) * weighting))

    check_all_params(loss_fn, layer.params, layer.grads)
    np.testing.assert_allclose(dx, numeric_gradient(loss_fn, x), atol=ATOL, rtol=RTOL)


def test_causal_self_attention_gradient_check() -> None:
    rng = np.random.default_rng(2)
    layer = CausalSelfAttention(4, random_state=2)
    x = rng.standard_normal((2, 3, 4))
    weighting = rng.standard_normal((2, 3, 4))

    layer.forward(x)
    dx = layer.backward(weighting)

    def loss_fn() -> float:
        return float(np.sum(layer.forward(x) * weighting))

    check_all_params(loss_fn, layer.params, layer.grads)
    np.testing.assert_allclose(dx, numeric_gradient(loss_fn, x), atol=ATOL, rtol=RTOL)


def test_transformer_block_gradient_check() -> None:
    rng = np.random.default_rng(3)
    block = TransformerBlock(4, 6, random_state=3)
    x = rng.standard_normal((2, 3, 4))
    weighting = rng.standard_normal((2, 3, 4))

    block.forward(x)
    dx = block.backward(weighting)

    def loss_fn() -> float:
        return float(np.sum(block.forward(x) * weighting))

    check_all_params(loss_fn, block.params, block.grads)
    np.testing.assert_allclose(dx, numeric_gradient(loss_fn, x), atol=ATOL, rtol=RTOL)


def test_composed_stack_gradient_check() -> None:
    """Embedding -> TransformerBlock -> softmax_cross_entropy, checked end to end."""
    rng = np.random.default_rng(4)
    vocab_size = d_model = 6  # equal so block outputs serve directly as logits
    embedding = Embedding(vocab_size, d_model, random_state=4)
    block = TransformerBlock(d_model, 9, random_state=5)
    ids = rng.integers(0, vocab_size, size=(2, 4))
    targets = rng.integers(0, vocab_size, size=(2, 4))

    logits = block.forward(embedding.forward(ids))
    _, dlogits = softmax_cross_entropy(logits, targets)
    dembedded = block.backward(dlogits)
    embedding.backward(dembedded)

    def loss_fn() -> float:
        loss, _ = softmax_cross_entropy(block.forward(embedding.forward(ids)), targets)
        return loss

    all_params = {"embedding.weight": embedding.params["weight"]}
    all_params |= {f"block.{name}": p for name, p in block.params.items()}
    all_grads = {"embedding.weight": embedding.grads["weight"]}
    all_grads |= {f"block.{name}": g for name, g in block.grads.items()}
    check_all_params(loss_fn, all_params, all_grads)


# ---------------------------------------------------------------------------
# Causality, layer semantics, determinism.
# ---------------------------------------------------------------------------


def test_causal_attention_no_future_leak() -> None:
    """Perturbing a future position must leave all earlier outputs exactly unchanged."""
    rng = np.random.default_rng(5)
    layer = CausalSelfAttention(8, random_state=5)
    x = rng.standard_normal((2, 5, 8))
    t_perturb = 3

    out = layer.forward(x)
    perturbed = x.copy()
    perturbed[:, t_perturb, :] += 1.0
    out_perturbed = layer.forward(perturbed)

    np.testing.assert_allclose(out[:, :t_perturb], out_perturbed[:, :t_perturb], atol=0.0, rtol=0.0)
    assert not np.allclose(out[:, t_perturb:], out_perturbed[:, t_perturb:], atol=1e-6), (
        "perturbation should change outputs at and after the perturbed position"
    )


def test_layernorm_output_stats() -> None:
    """With default gamma/beta, each position is normalized to mean 0 and variance ~1."""
    rng = np.random.default_rng(6)
    layer = LayerNorm(32)
    x = 3.0 + 5.0 * rng.standard_normal((4, 7, 32))

    out = layer.forward(x)

    assert out.shape == x.shape
    np.testing.assert_allclose(out.mean(axis=-1), 0.0, atol=1e-12)
    np.testing.assert_allclose(out.var(axis=-1), 1.0, atol=1e-3)


def test_embedding_backward_matches_explicit_loop() -> None:
    """Scatter-add over repeated ids must equal a per-position accumulation loop."""
    rng = np.random.default_rng(7)
    layer = Embedding(5, 3, random_state=7)
    ids = np.array([[0, 1, 1], [4, 0, 4]])
    dout = rng.standard_normal((2, 3, 3))

    layer.forward(ids)
    layer.backward(dout)

    expected = np.zeros((5, 3))
    for b in range(ids.shape[0]):
        for t in range(ids.shape[1]):
            expected[ids[b, t]] += dout[b, t]
    np.testing.assert_allclose(layer.grads["weight"], expected, atol=0.0, rtol=0.0)


def test_seeded_init_is_deterministic() -> None:
    for layer_a, layer_b, layer_c in [
        (Embedding(9, 4, random_state=11), Embedding(9, 4, random_state=11), Embedding(9, 4, 12)),
        (
            CausalSelfAttention(4, random_state=11),
            CausalSelfAttention(4, random_state=11),
            CausalSelfAttention(4, random_state=12),
        ),
        (
            TransformerBlock(4, 6, random_state=11),
            TransformerBlock(4, 6, random_state=11),
            TransformerBlock(4, 6, random_state=12),
        ),
    ]:
        for name in layer_a.params:
            np.testing.assert_array_equal(layer_a.params[name], layer_b.params[name])
        assert any(
            not np.array_equal(layer_a.params[name], layer_c.params[name])
            for name in layer_a.params
        )


def test_softmax_cross_entropy_matches_naive_and_sums_to_zero() -> None:
    rng = np.random.default_rng(8)
    logits = rng.standard_normal((2, 3, 5))
    targets = rng.integers(0, 5, size=(2, 3))

    loss, dlogits = softmax_cross_entropy(logits, targets)

    probs = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
    naive = -np.mean([np.log(probs[b, t, targets[b, t]]) for b in range(2) for t in range(3)])
    assert loss > 0.0
    np.testing.assert_allclose(loss, naive, atol=1e-12)
    # dL/dlogits rows sum to zero: softmax minus one-hot, scaled by 1/N.
    np.testing.assert_allclose(dlogits.sum(axis=-1), 0.0, atol=1e-12)
    np.testing.assert_allclose(dlogits * 6.0, probs - np.eye(5)[targets], atol=1e-12)


def test_transformer_block_pairs_with_step_adam() -> None:
    """The flat params/grads dicts must drive optimizers.Adam and reduce the loss."""
    rng = np.random.default_rng(9)
    vocab_size = d_model = 6
    embedding = Embedding(vocab_size, d_model, random_state=9)
    block = TransformerBlock(d_model, 8, random_state=10)
    ids = rng.integers(0, vocab_size, size=(4, 5))
    targets = rng.integers(0, vocab_size, size=(4, 5))
    optimizer = Adam(learning_rate=1e-2)

    def loss_and_step() -> float:
        loss, dlogits = softmax_cross_entropy(block.forward(embedding.forward(ids)), targets)
        embedding.backward(block.backward(dlogits))
        params = {"emb.weight": embedding.params["weight"]}
        params |= {f"block.{name}": p for name, p in block.params.items()}
        grads = {"emb.weight": embedding.grads["weight"]}
        grads |= {f"block.{name}": g for name, g in block.grads.items()}
        optimizer.step(params, grads)
        return loss

    initial = loss_and_step()
    final = initial
    for _ in range(30):
        final = loss_and_step()
    assert final < 0.8 * initial, f"loss did not decrease: {initial:.4f} -> {final:.4f}"


# ---------------------------------------------------------------------------
# Validation.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "build",
    [
        lambda: Embedding(0, 4),
        lambda: Embedding(4, 0),
        lambda: LayerNorm(0),
        lambda: LayerNorm(4, epsilon=0.0),
        lambda: CausalSelfAttention(0),
        lambda: TransformerBlock(0, 4),
        lambda: TransformerBlock(4, 0),
    ],
)
def test_constructors_reject_invalid_sizes(build) -> None:
    with pytest.raises(ValueError):
        build()


def test_forward_and_backward_reject_invalid_inputs() -> None:
    embedding = Embedding(4, 3)
    with pytest.raises(ValueError, match="integer"):
        embedding.forward(np.zeros((2, 3)))  # float ids
    with pytest.raises(ValueError, match="vocab_size"):
        embedding.forward(np.array([[0, 4]]))  # id out of range
    with pytest.raises(RuntimeError, match="forward"):
        embedding.backward(np.zeros((2, 3, 3)))
    embedding.forward(np.array([[0, 1]]))
    with pytest.raises(ValueError, match="shape"):
        embedding.backward(np.zeros((2, 3, 3)))

    attention = CausalSelfAttention(4)
    with pytest.raises(ValueError, match="shape"):
        attention.forward(np.zeros((2, 3)))
    with pytest.raises(RuntimeError, match="forward"):
        attention.backward(np.zeros((2, 3, 4)))

    block = TransformerBlock(4, 6)
    with pytest.raises(ValueError, match="shape"):
        block.forward(np.zeros((2, 3, 5)))
    with pytest.raises(RuntimeError, match="forward"):
        block.backward(np.zeros((2, 3, 4)))

    norm = LayerNorm(4)
    with pytest.raises(ValueError, match="shape"):
        norm.forward(np.zeros(4))
    with pytest.raises(RuntimeError, match="forward"):
        norm.backward(np.zeros((2, 4)))

    with pytest.raises(ValueError, match="shape"):
        softmax_cross_entropy(np.zeros((2, 3)), np.zeros((2,), dtype=np.int64))
    with pytest.raises(ValueError, match="targets"):
        softmax_cross_entropy(np.zeros((2, 3, 5)), np.full((2, 3), 5, dtype=np.int64))
