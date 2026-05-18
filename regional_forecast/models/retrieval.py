from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


class TrajectoryMemoryBank:
    """
    Lightweight retrieval scaffold.

    The next step is to replace this with a historical intent module that
    returns future region priors for similar trajectories.
    """

    def __init__(self, top_k: int = 8):
        self.top_k = top_k
        self.keys: torch.Tensor | None = None

    def fit(self, context_vectors: torch.Tensor) -> None:
        self.keys = F.normalize(context_vectors.float(), dim=-1)

    def query(self, queries: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.keys is None or len(self.keys) == 0:
            return {
                "indices": torch.empty((len(queries), 0), dtype=torch.long),
                "pooled": torch.zeros_like(queries),
            }

        q = F.normalize(queries.float(), dim=-1)
        scores = q @ self.keys.T
        k = min(self.top_k, self.keys.shape[0])
        values, indices = torch.topk(scores, k=k, dim=-1)
        pooled = self.keys[indices].mean(dim=1)
        return {"scores": values, "indices": indices, "pooled": pooled}
