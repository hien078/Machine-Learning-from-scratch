"""Tiny character-level causal transformer assembled from `ml_first_principles` layers.

Architecture (weights tied between token embedding and output projection):

    ids -> Embedding(vocab) + Embedding(positions)     # library Embedding layers
        -> TransformerBlock                            # library: causal attention + ReLU MLP,
                                                       # residuals, no LayerNorm (see README)
        -> logits = x @ Wte.T                          # tied output projection (local)
        -> softmax_cross_entropy on next-token targets # library loss

Since v0.3.0 the layer math (forward and manual backward) lives in
`ml_first_principles.transformer_core`; this module keeps only project-specific glue:
weight tying (not expressible with the library `Embedding`, so the tied projection and
its extra gradient term stay local), model assembly, and autoregressive generation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ml_first_principles.transformer_core import Embedding, TransformerBlock, softmax_cross_entropy

Array = NDArray[np.float64]
Params = dict[str, Array]


@dataclass(frozen=True)
class CTConfig:
    """Model hyperparameters (sizes only; training knobs live in ct_train)."""

    vocab_size: int
    block_size: int = 32
    d_model: int = 48
    d_ff: int = 96


class CharTransformer:
    """Token + positional embedding, one transformer block, tied output projection.

    `params` (and the grads returned by `loss_and_grads`) is a flat dict with
    prefixed keys ("tok_emb.weight", "block.attn.w_q", ...) so it plugs directly
    into `ml_first_principles.optimizers.Adam.step`.
    """

    def __init__(self, config: CTConfig, rng: np.random.Generator) -> None:
        self.config = config
        self.tok_emb = Embedding(config.vocab_size, config.d_model, random_state=rng)
        self.pos_emb = Embedding(config.block_size, config.d_model, random_state=rng)
        self.block = TransformerBlock(config.d_model, config.d_ff, random_state=rng)
        self._x2: Array | None = None

    @property
    def params(self) -> Params:
        return {
            "tok_emb.weight": self.tok_emb.params["weight"],
            "pos_emb.weight": self.pos_emb.params["weight"],
            **{f"block.{name}": p for name, p in self.block.params.items()},
        }

    def forward(self, ids: NDArray[np.integer]) -> Array:
        """Compute logits (B, T, V); caches the pre-projection activations for backward."""
        t = ids.shape[1]
        if t > self.config.block_size:
            raise ValueError(f"sequence length {t} exceeds block_size {self.config.block_size}")
        positions = np.arange(t, dtype=np.int64)[None, :]  # (1, T), broadcast over batch
        x0 = self.tok_emb.forward(ids) + self.pos_emb.forward(positions)  # (B, T, D)
        x2 = self.block.forward(x0)
        self._x2 = x2
        return x2 @ self.tok_emb.params["weight"].T  # tied projection (B, T, V)

    def loss_and_grads(
        self, ids: NDArray[np.integer], targets: NDArray[np.integer]
    ) -> tuple[float, Params]:
        """Forward + backward pass; returns scalar loss and a flat gradient dict."""
        logits = self.forward(ids)
        loss, dlogits = softmax_cross_entropy(logits, targets)
        assert self._x2 is not None

        # Tied output projection: logits = x2 @ Wte.T contributes a second Wte gradient.
        wte = self.tok_emb.params["weight"]
        tied_grad = np.einsum("btv,btd->vd", dlogits, self._x2)
        dx2 = dlogits @ wte

        dx0 = self.block.backward(dx2)
        self.tok_emb.backward(dx0)  # embedding scatter-add
        self.pos_emb.backward(dx0.sum(axis=0, keepdims=True))  # undo the (1, T, D) broadcast

        grads = {
            "tok_emb.weight": self.tok_emb.grads["weight"] + tied_grad,
            "pos_emb.weight": self.pos_emb.grads["weight"],
            **{f"block.{name}": g for name, g in self.block.grads.items()},
        }
        return loss, grads


def generate(
    model: CharTransformer,
    prompt_ids: list[int],
    n_new_tokens: int,
    rng: np.random.Generator,
    temperature: float | None = None,
) -> list[int]:
    """Autoregressive sampling. temperature=None -> greedy; else softmax(logits/temperature)."""
    ids = list(prompt_ids)
    for _ in range(n_new_tokens):
        window = np.array([ids[-model.config.block_size :]])
        logits = model.forward(window)
        last = logits[0, -1]
        if temperature is None:
            ids.append(int(last.argmax()))
        else:
            scaled = last / temperature
            scaled -= scaled.max()  # log-space shift for stability
            probs = np.exp(scaled) / np.exp(scaled).sum()
            ids.append(int(rng.choice(len(probs), p=probs)))
    return ids
