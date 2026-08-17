"""Tests for the tiny character transformer: gradients, causality, learning, shapes."""

from __future__ import annotations

import numpy as np
import pytest
from ct_model import Adam, CTConfig, cross_entropy, forward, generate, init_params, loss_and_grads

TINY = CTConfig(vocab_size=11, block_size=5, d_model=8, d_ff=10)
ATOL = 1e-7
RTOL = 1e-4


def make_batch(config: CTConfig, rng: np.random.Generator, batch_size: int = 2):
    ids = rng.integers(0, config.vocab_size, size=(batch_size, config.block_size))
    targets = rng.integers(0, config.vocab_size, size=(batch_size, config.block_size))
    return ids, targets


def test_gradient_check_full_model() -> None:
    """Central finite differences vs manual backprop, every entry of every parameter."""
    rng = np.random.default_rng(0)
    params = init_params(TINY, rng)
    ids, targets = make_batch(TINY, rng)
    _, grads = loss_and_grads(params, ids, targets, TINY)

    h = 1e-5
    for name, p in params.items():
        numeric = np.zeros_like(p)
        flat = p.reshape(-1)
        for i in range(flat.size):
            orig = flat[i]
            flat[i] = orig + h
            loss_plus, _ = loss_and_grads(params, ids, targets, TINY)
            flat[i] = orig - h
            loss_minus, _ = loss_and_grads(params, ids, targets, TINY)
            flat[i] = orig
            numeric.reshape(-1)[i] = (loss_plus - loss_minus) / (2.0 * h)
        np.testing.assert_allclose(
            grads[name], numeric, atol=ATOL, rtol=RTOL, err_msg=f"gradient mismatch in {name}"
        )


def test_causal_mask_no_future_leak() -> None:
    """Perturbing a future token must leave all earlier logits exactly unchanged."""
    rng = np.random.default_rng(1)
    params = init_params(TINY, rng)
    ids, _ = make_batch(TINY, rng)
    logits, _ = forward(params, ids, TINY)

    t_perturb = 3
    perturbed = ids.copy()
    perturbed[:, t_perturb] = (perturbed[:, t_perturb] + 1) % TINY.vocab_size
    logits_perturbed, _ = forward(params, perturbed, TINY)

    np.testing.assert_allclose(
        logits[:, :t_perturb], logits_perturbed[:, :t_perturb], atol=1e-12, rtol=0.0
    )
    assert not np.allclose(logits[:, t_perturb:], logits_perturbed[:, t_perturb:], atol=1e-6), (
        "perturbation should change logits at and after the perturbed position"
    )


def test_loss_decreases_over_steps() -> None:
    """A few Adam steps on a fixed batch must reduce the loss well below its start."""
    rng = np.random.default_rng(2)
    params = init_params(TINY, rng)
    ids, targets = make_batch(TINY, rng, batch_size=4)
    optimizer = Adam(params, lr=1e-2)

    initial_loss, grads = loss_and_grads(params, ids, targets, TINY)
    for _ in range(30):
        optimizer.step(params, grads)
        loss, grads = loss_and_grads(params, ids, targets, TINY)
    assert loss < 0.8 * initial_loss, f"loss did not decrease: {initial_loss:.4f} -> {loss:.4f}"


def test_shapes_and_api() -> None:
    rng = np.random.default_rng(3)
    params = init_params(TINY, rng)
    ids, targets = make_batch(TINY, rng, batch_size=3)

    logits, _ = forward(params, ids, TINY)
    assert logits.shape == (3, TINY.block_size, TINY.vocab_size)

    loss, dlogits = cross_entropy(logits, targets)
    assert np.isfinite(loss) and loss > 0.0
    assert dlogits.shape == logits.shape
    # dL/dlogits rows sum to zero: softmax minus one-hot.
    np.testing.assert_allclose(dlogits.sum(axis=-1), 0.0, atol=1e-12)

    _, grads = loss_and_grads(params, ids, targets, TINY)
    assert set(grads) == set(params)
    for name in params:
        assert grads[name].shape == params[name].shape


def test_forward_rejects_too_long_sequence() -> None:
    rng = np.random.default_rng(4)
    params = init_params(TINY, rng)
    ids = rng.integers(0, TINY.vocab_size, size=(1, TINY.block_size + 1))
    with pytest.raises(ValueError, match="block_size"):
        forward(params, ids, TINY)


def test_generation_deterministic_and_greedy_matches() -> None:
    rng = np.random.default_rng(5)
    params = init_params(TINY, rng)
    prompt = [1, 2, 3]

    greedy_a = generate(params, TINY, prompt, 10, np.random.default_rng(7))
    greedy_b = generate(params, TINY, prompt, 10, np.random.default_rng(99))
    assert greedy_a == greedy_b, "greedy decoding must not depend on the rng"

    temp_a = generate(params, TINY, prompt, 10, np.random.default_rng(7), temperature=0.8)
    temp_b = generate(params, TINY, prompt, 10, np.random.default_rng(7), temperature=0.8)
    assert temp_a == temp_b, "sampling must be deterministic under a fixed seed"
    assert len(temp_a) == len(prompt) + 10
    assert all(0 <= i < TINY.vocab_size for i in temp_a)
