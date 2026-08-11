"""First-principles pure NumPy implementations of LLM engineering blocks (BPE, LoRA, DPO)."""

from __future__ import annotations

from collections import Counter

import numpy as np

END_OF_WORD = "</w>"
WEIGHT_INIT_SCALE = 0.01


def _apply_merge(
    symbols: list[str], pair: tuple[str, str], replacement: str
) -> list[str]:
    merged = []
    index = 0
    while index < len(symbols):
        if index < len(symbols) - 1 and (symbols[index], symbols[index + 1]) == pair:
            merged.append(replacement)
            index += 2
        else:
            merged.append(symbols[index])
            index += 1
    return merged


class BPETokenizer:
    """Word-level Byte-Pair Encoding (BPE) subword tokenizer.

    Implements the classic algorithm of Sennrich et al. (2016): each word is
    split into characters plus an end-of-word marker, then the most frequent
    adjacent symbol pair is merged repeatedly until the vocabulary reaches
    ``target_vocab_size``.
    """

    def __init__(self, target_vocab_size: int = 50) -> None:
        if target_vocab_size < 1:
            raise ValueError("target_vocab_size must be positive")
        self.target_vocab_size = target_vocab_size
        self.merges: list[tuple[str, str]] = []
        self.vocab: list[str] = []

    def fit(self, text: str) -> None:
        """Learn BPE merge rules from raw text.

        Calling ``fit`` again discards previously learned state, so repeated
        fits on the same text yield identical merge tables.

        Args:
            text: Whitespace-separated training corpus.

        Raises:
            ValueError: If ``text`` contains no words.
        """
        words = text.split()
        if not words:
            raise ValueError("text must contain at least one word")
        self.merges = []
        corpus = [tuple(list(word) + [END_OF_WORD]) for word in words]

        current_vocab = set()
        for word in corpus:
            current_vocab.update(word)

        while len(current_vocab) < self.target_vocab_size:
            pairs: Counter[tuple[str, str]] = Counter()
            for word in corpus:
                for i in range(len(word) - 1):
                    pairs[(word[i], word[i + 1])] += 1

            if not pairs:
                break

            best_pair = pairs.most_common(1)[0][0]
            self.merges.append(best_pair)
            replacement = "".join(best_pair)
            current_vocab.add(replacement)
            corpus = [tuple(_apply_merge(list(word), best_pair, replacement)) for word in corpus]

        self.vocab = sorted(current_vocab)

    def encode(self, text: str) -> list[str]:
        """Tokenize text by replaying the learned merges in training order.

        Args:
            text: Whitespace-separated input text.

        Returns:
            Subword tokens; word boundaries carry the ``</w>`` marker.

        Raises:
            RuntimeError: If called before ``fit``.
        """
        if not self.vocab:
            raise RuntimeError("fit must be called before encode")
        tokens: list[str] = []
        for word in text.split():
            symbols = list(word) + [END_OF_WORD]
            for pair in self.merges:
                symbols = _apply_merge(symbols, pair, "".join(pair))
            tokens.extend(symbols)
        return tokens

    def decode(self, tokens: list[str]) -> str:
        """Reassemble text from tokens by concatenating and restoring spaces.

        Args:
            tokens: Token sequence produced by ``encode``.

        Returns:
            The reconstructed whitespace-separated text.
        """
        return "".join(tokens).replace(END_OF_WORD, " ").strip()


class LoRALinear:
    """Low-Rank Adaptation (LoRA) linear layer in pure NumPy.

    Represents ``x @ W0 + (alpha / r) * x @ A @ B`` with a frozen base weight
    ``W0`` and a trainable low-rank update ``A @ B`` initialized so the
    adaptation starts at zero (``B = 0``).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        lora_alpha: float = 16.0,
        random_state: int | None = None,
    ) -> None:
        """Initialize the frozen base weight and the low-rank factors.

        Args:
            in_features: Input dimensionality.
            out_features: Output dimensionality.
            r: Rank of the adaptation; must satisfy ``1 <= r <= min(in, out)``.
            lora_alpha: Scaling numerator; the update is scaled by ``alpha / r``.
            random_state: Seed for the isolated random generator.

        Raises:
            ValueError: If dimensions are not positive or ``r`` is out of range.
        """
        if in_features < 1 or out_features < 1:
            raise ValueError("in_features and out_features must be positive")
        if not 1 <= r <= min(in_features, out_features):
            raise ValueError("r must satisfy 1 <= r <= min(in_features, out_features)")
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.scaling = lora_alpha / r

        rng = np.random.default_rng(random_state)
        # Frozen base weight matrix
        self.W0 = rng.standard_normal((in_features, out_features)) * WEIGHT_INIT_SCALE
        # Trainable low-rank adaptation matrices A and B; B starts at zero so the
        # adapted layer initially equals the frozen base layer.
        self.lora_A = rng.standard_normal((in_features, r)) * WEIGHT_INIT_SCALE
        self.lora_B = np.zeros((r, out_features))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply the base weight plus the scaled low-rank adaptation.

        Args:
            x: Input batch of shape ``(n_samples, in_features)``.

        Returns:
            Output batch of shape ``(n_samples, out_features)``.
        """
        base_out = x @ self.W0
        lora_out = (x @ self.lora_A @ self.lora_B) * self.scaling
        return base_out + lora_out


def dpo_loss(
    policy_chosen_logps: np.ndarray,
    policy_rejected_logps: np.ndarray,
    reference_chosen_logps: np.ndarray,
    reference_rejected_logps: np.ndarray,
    beta: float = 0.1,
) -> float:
    r"""Compute the Direct Preference Optimization (DPO) loss.

    Implements
    :math:`-\log\sigma\bigl(\beta[(\pi_c-\pi_r)-(\mathrm{ref}_c-\mathrm{ref}_r)]\bigr)`
    averaged over the batch, evaluated in log space via
    ``softplus(-beta * logits) = logaddexp(0, -beta * logits)`` so that large
    positive or negative logits cannot overflow ``exp``.

    Args:
        policy_chosen_logps: Policy log-probabilities of the preferred responses.
        policy_rejected_logps: Policy log-probabilities of the rejected responses.
        reference_chosen_logps: Reference-model log-probabilities of the preferred responses.
        reference_rejected_logps: Reference-model log-probabilities of the rejected responses.
        beta: Inverse-temperature weighting of the implicit reward margin.

    Returns:
        The mean DPO loss over the batch as a finite float.

    Raises:
        ValueError: If ``beta`` is not positive.
    """
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    pi_logratios = np.asarray(policy_chosen_logps) - np.asarray(policy_rejected_logps)
    ref_logratios = np.asarray(reference_chosen_logps) - np.asarray(reference_rejected_logps)
    logits = pi_logratios - ref_logratios

    # -log(sigmoid(z)) = log(1 + exp(-z)) = logaddexp(0, -z), stable for any z.
    losses = np.logaddexp(0.0, -beta * logits)
    return float(np.mean(losses))
