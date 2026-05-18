from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch


@dataclass
class RegionalMarkovBaseline:
    input_num_region_classes: int
    num_region_classes_per_horizon: List[int]
    num_status_classes: int
    num_horizons: int
    smoothing: float = 1.0

    def __post_init__(self) -> None:
        self.region_counts = [
            torch.full(
                (self.input_num_region_classes, int(num_region_classes)),
                fill_value=self.smoothing,
                dtype=torch.float32,
            )
            for num_region_classes in self.num_region_classes_per_horizon
        ]
        self.status_counts = [
            torch.full(
                (self.input_num_region_classes, self.num_status_classes),
                fill_value=self.smoothing,
                dtype=torch.float32,
            )
            for _ in range(self.num_horizons)
        ]

    def fit(
        self,
        region_seq: torch.Tensor,
        region_labels: torch.Tensor,
        status_labels: torch.Tensor,
        label_mask: torch.Tensor,
    ) -> None:
        current_regions = region_seq[:, -1]
        for horizon_idx in range(self.num_horizons):
            valid_mask = label_mask[:, horizon_idx].bool()
            if not torch.any(valid_mask):
                continue

            current = current_regions[valid_mask]
            target_region = region_labels[valid_mask, horizon_idx]
            target_status = status_labels[valid_mask, horizon_idx]

            for current_region, region_target, status_target in zip(
                current.tolist(),
                target_region.tolist(),
                target_status.tolist(),
            ):
                self.region_counts[horizon_idx][current_region, region_target] += 1.0
                self.status_counts[horizon_idx][current_region, status_target] += 1.0

    def predict(self, region_seq: torch.Tensor) -> Dict[str, object]:
        current_regions = region_seq[:, -1]
        region_probs = []
        status_probs = []
        for horizon_idx in range(self.num_horizons):
            selected_regions = self.region_counts[horizon_idx][current_regions]
            selected_regions = selected_regions / selected_regions.sum(dim=-1, keepdim=True)
            region_probs.append(selected_regions)

            selected_status = self.status_counts[horizon_idx][current_regions]
            selected_status = selected_status / selected_status.sum(dim=-1, keepdim=True)
            status_probs.append(selected_status)

        return {
            "region_probs": region_probs,
            "status_probs": torch.stack(status_probs, dim=1),
        }

    def state_dict(self) -> Dict[str, object]:
        return {
            "input_num_region_classes": self.input_num_region_classes,
            "num_region_classes_per_horizon": self.num_region_classes_per_horizon,
            "num_status_classes": self.num_status_classes,
            "num_horizons": self.num_horizons,
            "smoothing": self.smoothing,
            "region_counts": self.region_counts,
            "status_counts": self.status_counts,
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, object]) -> "RegionalMarkovBaseline":
        model = cls(
            input_num_region_classes=int(state["input_num_region_classes"]),
            num_region_classes_per_horizon=[int(item) for item in state["num_region_classes_per_horizon"]],
            num_status_classes=int(state["num_status_classes"]),
            num_horizons=int(state["num_horizons"]),
            smoothing=float(state.get("smoothing", 1.0)),
        )
        model.region_counts = [tensor.float() for tensor in state["region_counts"]]
        model.status_counts = [tensor.float() for tensor in state["status_counts"]]
        return model
