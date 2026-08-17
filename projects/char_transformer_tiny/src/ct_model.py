"""Single-block, single-head causal transformer in pure NumPy with manual backprop.

Architecture (weights tied between token embedding and output projection):

    ids -> Wte[ids] + Wpe[:T]                      # token + learned positional embedding
        -> x + CausalSelfAttention(x)              # residual, single head, scale 1/sqrt(D)
        -> x + MLP(x)                              # residual, Dense -> ReLU -> Dense
        -> logits = x @ Wte.T                      # tied output projection
        -> softmax cross-entropy on next-token targets

No LayerNorm (deliberate simplification for a tiny model; noted in README).
All gradients are derived and implemented by hand; see `loss_and_grads`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
Params = dict[str, Array]


@dataclass(frozen=True)
class CTConfig:
    """Model hyperparameters (sizes only; training knobs live in ct_train)."""

    vocab_size: int
    block_size: int = 32
    d_model: int = 48
    d_ff: int = 96


def init_params(config: CTConfig, rng: np.random.Generator) -> Params:
    """Initialize all weights with scaled Gaussians (std 0.02 embeddings, 1/sqrt(fan_in) mats)."""
    v, t, d, f = config.vocab_size, config.block_size, config.d_model, config.d_ff
    return {
        "Wte": 0.02 * rng.standard_normal((v, d)),
        "Wpe": 0.02 * rng.standard_normal((t, d)),
        "Wq": rng.standard_normal((d, d)) / np.sqrt(d),
        "Wk": rng.standard_normal((d, d)) / np.sqrt(d),
        "Wv": rng.standard_normal((d, d)) / np.sqrt(d),
        "Wo": rng.standard_normal((d, d)) / np.sqrt(d),
        "W1": rng.standard_normal((d, f)) / np.sqrt(d),
        "b1": np.zeros(f),
        "W2": rng.standard_normal((f, d)) / np.sqrt(f),
        "b2": np.zeros(d),
    }


def _masked_softmax(scores: Array) -> Array:
    """Row-wise softmax over the last axis with causal mask, shifted by max for stability."""
    t = scores.shape[-1]
    mask = np.tril(np.ones((t, t), dtype=bool))
    scores = np.where(mask, scores, -np.inf)
    scores = scores - scores.max(axis=-1, keepdims=True)
    exp = np.exp(scores)
    return exp / exp.sum(axis=-1, keepdims=True)


def forward(params: Params, ids: NDArray[np.integer], config: CTConfig) -> tuple[Array, dict]:
    """Compute logits (B, T, V) and a cache of intermediates for the backward pass."""
    _, t = ids.shape
    if t > config.block_size:
        raise ValueError(f"sequence length {t} exceeds block_size {config.block_size}")
    scale = 1.0 / np.sqrt(config.d_model)

    x0 = params["Wte"][ids] + params["Wpe"][:t]  # (B, T, D)
    q = x0 @ params["Wq"]
    k = x0 @ params["Wk"]
    v = x0 @ params["Wv"]
    attn = _masked_softmax(q @ k.transpose(0, 2, 1) * scale)  # (B, T, T)
    ctx = attn @ v  # (B, T, D)
    x1 = x0 + ctx @ params["Wo"]  # residual around attention

    pre = x1 @ params["W1"] + params["b1"]
    h = np.maximum(pre, 0.0)
    x2 = x1 + h @ params["W2"] + params["b2"]  # residual around MLP

    logits = x2 @ params["Wte"].T  # tied projection (B, T, V)
    cache = {
        "ids": ids,
        "x0": x0,
        "q": q,
        "k": k,
        "v": v,
        "attn": attn,
        "ctx": ctx,
        "x1": x1,
        "pre": pre,
        "h": h,
        "x2": x2,
        "scale": scale,
    }
    return logits, cache


def cross_entropy(logits: Array, targets: NDArray[np.integer]) -> tuple[float, Array]:
    """Mean next-token cross-entropy and dL/dlogits, computed in log-space (max-shifted)."""
    b, t, v = logits.shape
    shifted = logits - logits.max(axis=-1, keepdims=True)
    log_z = np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    log_probs = shifted - log_z  # (B, T, V)
    n = b * t
    loss = -log_probs.reshape(n, v)[np.arange(n), targets.reshape(n)].mean()

    dlogits = np.exp(log_probs)  # softmax probabilities
    dlogits.reshape(n, v)[np.arange(n), targets.reshape(n)] -= 1.0
    dlogits /= n
    return float(loss), dlogits


def loss_and_grads(
    params: Params, ids: NDArray[np.integer], targets: NDArray[np.integer], config: CTConfig
) -> tuple[float, Params]:
    """Forward + manual backward pass; returns scalar loss and gradients for every parameter."""
    logits, c = forward(params, ids, config)
    loss, dlogits = cross_entropy(logits, targets)
    g = {name: np.zeros_like(p) for name, p in params.items()}

    # Tied output projection: logits = x2 @ Wte.T
    g["Wte"] += np.einsum("btv,btd->vd", dlogits, c["x2"])
    dx2 = dlogits @ params["Wte"]

    # MLP with residual: x2 = x1 + relu(x1 @ W1 + b1) @ W2 + b2
    dx1 = dx2.copy()
    g["W2"] += np.einsum("btf,btd->fd", c["h"], dx2)
    g["b2"] += dx2.sum(axis=(0, 1))
    dh = dx2 @ params["W2"].T
    dpre = dh * (c["pre"] > 0.0)
    g["W1"] += np.einsum("btd,btf->df", c["x1"], dpre)
    g["b1"] += dpre.sum(axis=(0, 1))
    dx1 += dpre @ params["W1"].T

    # Attention with residual: x1 = x0 + (attn @ v) @ Wo
    dx0 = dx1.copy()
    g["Wo"] += np.einsum("btd,bte->de", c["ctx"], dx1)
    dctx = dx1 @ params["Wo"].T
    dattn = dctx @ c["v"].transpose(0, 2, 1)
    dv = c["attn"].transpose(0, 2, 1) @ dctx
    # Softmax backward (masked entries have attn == 0, so their grads vanish).
    dscores = c["attn"] * (dattn - (dattn * c["attn"]).sum(axis=-1, keepdims=True))
    dscores *= c["scale"]
    dq = dscores @ c["k"]
    dk = dscores.transpose(0, 2, 1) @ c["q"]
    g["Wq"] += np.einsum("btd,bte->de", c["x0"], dq)
    g["Wk"] += np.einsum("btd,bte->de", c["x0"], dk)
    g["Wv"] += np.einsum("btd,bte->de", c["x0"], dv)
    dx0 += dq @ params["Wq"].T + dk @ params["Wk"].T + dv @ params["Wv"].T

    # Embeddings: x0 = Wte[ids] + Wpe[:T]
    np.add.at(g["Wte"], c["ids"], dx0)
    g["Wpe"][: ids.shape[1]] += dx0.sum(axis=0)
    return loss, g


class Adam:
    """Minimal per-tensor Adam with bias correction (dict-of-arrays interface).

    Local implementation: `ml_first_principles.optimizers.adam` drives its own full
    optimization loop over a single flat vector, which does not fit minibatch training.
    """

    def __init__(
        self,
        params: Params,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.t = 0
        self.m = {name: np.zeros_like(p) for name, p in params.items()}
        self.v = {name: np.zeros_like(p) for name, p in params.items()}

    def step(self, params: Params, grads: Params) -> None:
        self.t += 1
        for name, p in params.items():
            grad = grads[name]
            self.m[name] = self.beta1 * self.m[name] + (1.0 - self.beta1) * grad
            self.v[name] = self.beta2 * self.v[name] + (1.0 - self.beta2) * grad**2
            m_hat = self.m[name] / (1.0 - self.beta1**self.t)
            v_hat = self.v[name] / (1.0 - self.beta2**self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def generate(
    params: Params,
    config: CTConfig,
    prompt_ids: list[int],
    n_new_tokens: int,
    rng: np.random.Generator,
    temperature: float | None = None,
) -> list[int]:
    """Autoregressive sampling. temperature=None -> greedy; else softmax(logits/temperature)."""
    ids = list(prompt_ids)
    for _ in range(n_new_tokens):
        window = np.array([ids[-config.block_size :]])
        logits, _ = forward(params, window, config)
        last = logits[0, -1]
        if temperature is None:
            ids.append(int(last.argmax()))
        else:
            scaled = last / temperature
            scaled -= scaled.max()  # log-space shift for stability
            probs = np.exp(scaled) / np.exp(scaled).sum()
            ids.append(int(rng.choice(len(probs), p=probs)))
    return ids
