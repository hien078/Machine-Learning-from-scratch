"""Tests for the tiny character transformer: gradients, causality, learning, shapes."""

from __future__ import annotations

import numpy as np
import pytest
from ct_model import CharTransformer, CTConfig, generate

from ml_first_principles.optimizers import Adam
from ml_first_principles.transformer_core import softmax_cross_entropy

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
    model = CharTransformer(TINY, rng)
    ids, targets = make_batch(TINY, rng)
    _, grads = model.loss_and_grads(ids, targets)

    h = 1e-5
    for name, p in model.params.items():
        numeric = np.zeros_like(p)
        flat = p.reshape(-1)
        for i in range(flat.size):
            orig = flat[i]
            flat[i] = orig + h
            loss_plus, _ = model.loss_and_grads(ids, targets)
            flat[i] = orig - h
            loss_minus, _ = model.loss_and_grads(ids, targets)
            flat[i] = orig
            numeric.reshape(-1)[i] = (loss_plus - loss_minus) / (2.0 * h)
        np.testing.assert_allclose(
            grads[name], numeric, atol=ATOL, rtol=RTOL, err_msg=f"gradient mismatch in {name}"
        )


def test_causal_mask_no_future_leak() -> None:
    """Perturbing a future token must leave all earlier logits exactly unchanged."""
    rng = np.random.default_rng(1)
    model = CharTransformer(TINY, rng)
    ids, _ = make_batch(TINY, rng)
    logits = model.forward(ids)

    t_perturb = 3
    perturbed = ids.copy()
    perturbed[:, t_perturb] = (perturbed[:, t_perturb] + 1) % TINY.vocab_size
    logits_perturbed = model.forward(perturbed)

    np.testing.assert_allclose(
        logits[:, :t_perturb], logits_perturbed[:, :t_perturb], atol=1e-12, rtol=0.0
    )
    assert not np.allclose(logits[:, t_perturb:], logits_perturbed[:, t_perturb:], atol=1e-6), (
        "perturbation should change logits at and after the perturbed position"
    )


def test_loss_decreases_over_steps() -> None:
    """A few library-Adam steps on a fixed batch must reduce the loss well below its start."""
    rng = np.random.default_rng(2)
    model = CharTransformer(TINY, rng)
    ids, targets = make_batch(TINY, rng, batch_size=4)
    optimizer = Adam(learning_rate=1e-2)

    initial_loss, grads = model.loss_and_grads(ids, targets)
    for _ in range(30):
        optimizer.step(model.params, grads)
        loss, grads = model.loss_and_grads(ids, targets)
    assert loss < 0.8 * initial_loss, f"loss did not decrease: {initial_loss:.4f} -> {loss:.4f}"


def test_shapes_and_api() -> None:
    rng = np.random.default_rng(3)
    model = CharTransformer(TINY, rng)
    ids, targets = make_batch(TINY, rng, batch_size=3)

    logits = model.forward(ids)
    assert logits.shape == (3, TINY.block_size, TINY.vocab_size)

    loss, dlogits = softmax_cross_entropy(logits, targets)
    assert np.isfinite(loss) and loss > 0.0
    assert dlogits.shape == logits.shape
    # dL/dlogits rows sum to zero: softmax minus one-hot.
    np.testing.assert_allclose(dlogits.sum(axis=-1), 0.0, atol=1e-12)

    _, grads = model.loss_and_grads(ids, targets)
    assert set(grads) == set(model.params)
    for name in model.params:
        assert grads[name].shape == model.params[name].shape


def test_forward_rejects_too_long_sequence() -> None:
    rng = np.random.default_rng(4)
    model = CharTransformer(TINY, rng)
    ids = rng.integers(0, TINY.vocab_size, size=(1, TINY.block_size + 1))
    with pytest.raises(ValueError, match="block_size"):
        model.forward(ids)


def test_generation_deterministic_and_greedy_matches() -> None:
    rng = np.random.default_rng(5)
    model = CharTransformer(TINY, rng)
    prompt = [1, 2, 3]

    greedy_a = generate(model, prompt, 10, np.random.default_rng(7))
    greedy_b = generate(model, prompt, 10, np.random.default_rng(99))
    assert greedy_a == greedy_b, "greedy decoding must not depend on the rng"

    temp_a = generate(model, prompt, 10, np.random.default_rng(7), temperature=0.8)
    temp_b = generate(model, prompt, 10, np.random.default_rng(7), temperature=0.8)
    assert temp_a == temp_b, "sampling must be deterministic under a fixed seed"
    assert len(temp_a) == len(prompt) + 10
    assert all(0 <= i < TINY.vocab_size for i in temp_a)
