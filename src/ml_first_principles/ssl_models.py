"""First-principles pure NumPy implementations of self-supervised learning blocks (InfoNCE, MAE)."""

from __future__ import annotations

import numpy as np

NORM_EPS = 1e-12


def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    """Return ``log(sum(exp(values)))`` along ``axis`` with max-subtraction."""
    max_values = np.max(values, axis=axis, keepdims=True)
    stabilized = np.exp(values - max_values)
    return np.squeeze(max_values, axis=axis) + np.log(np.sum(stabilized, axis=axis))


class InfoNCELoss:
    """InfoNCE (NT-Xent) contrastive loss in pure NumPy.

    Both views are L2-normalized and every embedding is contrasted against the
    ``2N - 1`` other embeddings in the concatenated batch. The loss is
    evaluated entirely in log space, so small temperatures cannot overflow
    ``exp``.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        """Store the softmax temperature.

        Args:
            temperature: Positive scaling of cosine similarities.

        Raises:
            ValueError: If ``temperature`` is not positive.
        """
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.maximum(norm, NORM_EPS)

    def forward(self, z_i: np.ndarray, z_j: np.ndarray) -> float:
        """Compute the InfoNCE loss over a batch of positive pairs.

        Args:
            z_i: First-view embeddings of shape ``(n_samples, dim)``.
            z_j: Second-view embeddings of shape ``(n_samples, dim)``.

        Returns:
            The mean contrastive loss over both anchor directions.

        Raises:
            ValueError: If the two views have different shapes.
        """
        z_i = np.asarray(z_i, dtype=float)
        z_j = np.asarray(z_j, dtype=float)
        if z_i.shape != z_j.shape or z_i.ndim != 2:
            raise ValueError("z_i and z_j must be two-dimensional arrays of equal shape")
        z_i = self._normalize(z_i)
        z_j = self._normalize(z_j)

        out = np.concatenate([z_i, z_j], axis=0)
        logits = (out @ out.T) / self.temperature
        # Exclude self-similarity from every denominator.
        np.fill_diagonal(logits, -np.inf)
        log_denominator = _logsumexp(logits, axis=1)

        positive_logit = np.sum(z_i * z_j, axis=-1) / self.temperature
        positive_logit = np.concatenate([positive_logit, positive_logit], axis=0)

        loss = log_denominator - positive_logit
        return float(np.mean(loss))


class PatchMasking:
    """Random patch masking for Masked Autoencoders (MAE) in pure NumPy."""

    def __init__(self, mask_ratio: float = 0.75, random_state: int | None = None) -> None:
        """Store the masking ratio and seed an isolated generator.

        Args:
            mask_ratio: Fraction of patches to mask, strictly inside ``(0, 1)``.
            random_state: Seed for the isolated random generator.

        Raises:
            ValueError: If ``mask_ratio`` is outside ``(0, 1)``.
        """
        if not 0.0 < mask_ratio < 1.0:
            raise ValueError("mask_ratio must lie strictly between 0 and 1")
        self.mask_ratio = mask_ratio
        self._rng = np.random.default_rng(random_state)

    def mask_patches(self, sequence: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Randomly mask patches along the sequence dimension.

        Args:
            sequence: Patch batch of shape ``(batch, seq_len, dim)``.

        Returns:
            ``(sequence_kept, mask, ids_restore)`` where ``sequence_kept`` holds
            the visible patches in shuffled order with the input dtype
            preserved, ``mask`` marks masked positions with ``1`` in the
            original order, and ``ids_restore`` maps shuffled positions back to
            original positions.

        Raises:
            ValueError: If ``sequence`` is not three-dimensional or the mask
                ratio leaves no visible patch.
        """
        sequence = np.asarray(sequence)
        if sequence.ndim != 3:
            raise ValueError("sequence must have shape (batch, seq_len, dim)")
        batch_size, seq_len, _ = sequence.shape
        len_keep = int(seq_len * (1 - self.mask_ratio))
        if len_keep < 1:
            raise ValueError("mask_ratio leaves no visible patch for this sequence length")

        noise = self._rng.random((batch_size, seq_len))
        ids_shuffle = np.argsort(noise, axis=1)
        ids_restore = np.argsort(ids_shuffle, axis=1)

        ids_keep = ids_shuffle[:, :len_keep]
        sequence_kept = np.take_along_axis(sequence, ids_keep[:, :, None], axis=1)

        mask = np.ones((batch_size, seq_len))
        mask[:, :len_keep] = 0
        mask_restored = np.take_along_axis(mask, ids_restore, axis=1)

        return sequence_kept, mask_restored, ids_restore
