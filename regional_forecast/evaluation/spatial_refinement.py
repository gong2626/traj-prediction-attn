from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import numpy as np

try:
    import h3
except ImportError:  # pragma: no cover - depends on local environment
    h3 = None


def _require_h3() -> None:
    if h3 is None:
        raise ImportError("The 'h3' package is required for H3 spatial refinement.")


@lru_cache(maxsize=32)
def _neighbor_lists(region_tokens: tuple[str, ...], ring: int) -> tuple[tuple[int, ...], ...]:
    _require_h3()
    if ring < 0:
        raise ValueError("H3 refinement ring must be non-negative.")

    token_to_id = {str(cell): idx for idx, cell in enumerate(region_tokens)}
    neighbors: list[tuple[int, ...]] = []
    for source_idx, cell in enumerate(region_tokens):
        if ring == 0:
            neighbors.append((source_idx,))
            continue

        disk = [str(item) for item in h3.grid_disk(str(cell), int(ring))]
        target_ids = sorted({token_to_id[item] for item in disk if item in token_to_id})
        if not target_ids:
            target_ids = [source_idx]
        neighbors.append(tuple(target_ids))
    return tuple(neighbors)


def smooth_h3_probabilities(
    probabilities: np.ndarray,
    region_tokens: Sequence[str],
    *,
    alpha: float,
    ring: int,
    preserve_unknown: bool = True,
) -> np.ndarray:
    """Diffuse known H3-cell probability mass to local H3 neighbors."""

    alpha = float(alpha)
    ring = int(ring)
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError("H3 refinement alpha must be in [0, 1].")
    if ring < 0:
        raise ValueError("H3 refinement ring must be non-negative.")

    probs = np.asarray(probabilities)
    if probs.ndim != 2:
        raise ValueError("H3 refinement expects a 2D probability array.")

    region_tokens_tuple = tuple(str(cell) for cell in region_tokens)
    known_count = len(region_tokens_tuple)
    if known_count <= 0 or probs.shape[1] < known_count:
        return probs.copy()
    if alpha == 0.0 or ring == 0:
        return probs.copy()

    output = probs.astype(np.float32, copy=True)
    known = probs[:, :known_count].astype(np.float32, copy=False)
    diffused = np.zeros_like(known, dtype=np.float32)
    neighbors = _neighbor_lists(region_tokens_tuple, ring)

    active_sources = np.flatnonzero(known.sum(axis=0) > 0.0)
    for source_idx in active_sources.tolist():
        target_ids = neighbors[int(source_idx)]
        if not target_ids:
            continue
        source_mass = known[:, int(source_idx)]
        if not np.any(source_mass > 0.0):
            continue
        diffused[:, list(target_ids)] += source_mass[:, None] / float(len(target_ids))

    output[:, :known_count] = (1.0 - alpha) * known + alpha * diffused
    if not preserve_unknown and probs.shape[1] > known_count:
        output[:, known_count:] = probs[:, known_count:] * (1.0 - alpha)

    row_sums = output.sum(axis=1, keepdims=True)
    nonzero = row_sums[:, 0] > 0.0
    if np.any(nonzero):
        output[nonzero] = output[nonzero] / row_sums[nonzero]
    return output


class H3NeighborProbabilityRefiner:
    def __init__(
        self,
        *,
        region_tokens_per_horizon: Sequence[Sequence[str]],
        enabled: bool,
        alpha_per_horizon: Sequence[float],
        ring_per_horizon: Sequence[int],
        preserve_unknown: bool = True,
    ) -> None:
        self.region_tokens_per_horizon = [list(tokens) for tokens in region_tokens_per_horizon]
        self.enabled = bool(enabled)
        self.alpha_per_horizon = [float(value) for value in alpha_per_horizon]
        self.ring_per_horizon = [int(value) for value in ring_per_horizon]
        self.preserve_unknown = bool(preserve_unknown)

        horizon_count = len(self.region_tokens_per_horizon)
        if len(self.alpha_per_horizon) != horizon_count:
            raise ValueError("H3 refinement alpha values must match the number of horizons.")
        if len(self.ring_per_horizon) != horizon_count:
            raise ValueError("H3 refinement ring values must match the number of horizons.")

    def refine_probability_arrays(self, region_probabilities: Sequence[np.ndarray]) -> list[np.ndarray]:
        if not self.enabled:
            return [np.asarray(array).copy() for array in region_probabilities]

        refined: list[np.ndarray] = []
        for horizon_idx, probabilities in enumerate(region_probabilities):
            refined.append(self.refine_horizon(np.asarray(probabilities), horizon_idx))
        return refined

    def refine_horizon(self, probabilities: np.ndarray, horizon_idx: int) -> np.ndarray:
        if not self.enabled:
            return np.asarray(probabilities).copy()
        return smooth_h3_probabilities(
            np.asarray(probabilities),
            self.region_tokens_per_horizon[int(horizon_idx)],
            alpha=float(self.alpha_per_horizon[int(horizon_idx)]),
            ring=int(self.ring_per_horizon[int(horizon_idx)]),
            preserve_unknown=self.preserve_unknown,
        )
