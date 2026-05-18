from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from regional_forecast.config import ModelConfig


class LongHorizonRegionalForecaster(nn.Module):
    def __init__(
        self,
        input_num_region_classes: int,
        num_region_classes_per_horizon: list[int],
        num_status_classes: int,
        num_numeric_features: int,
        num_context_features: int,
        config: ModelConfig,
    ):
        super().__init__()
        self.region_embedding = nn.Embedding(input_num_region_classes, config.embedding_dim)
        self.numeric_projection = nn.Linear(num_numeric_features, config.numeric_projection_dim)
        self.context_projection = nn.Linear(num_context_features, config.context_projection_dim)

        encoder_dim = config.embedding_dim + config.numeric_projection_dim
        self.encoder = nn.GRU(
            input_size=encoder_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(config.dropout)
        head_dim = config.hidden_dim + config.context_projection_dim
        self.region_heads = nn.ModuleList(
            [nn.Linear(head_dim, num_classes) for num_classes in num_region_classes_per_horizon]
        )
        self.status_heads = nn.ModuleList(
            [nn.Linear(head_dim, num_status_classes) for _ in range(len(num_region_classes_per_horizon))]
        )

    def forward(
        self,
        region_seq: torch.Tensor,
        numeric_seq: torch.Tensor,
        context: torch.Tensor,
        retrieved_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor | list[torch.Tensor]]:
        region_features = self.region_embedding(region_seq)
        numeric_features = self.numeric_projection(numeric_seq)
        encoded_input = torch.cat([region_features, numeric_features], dim=-1)

        _, hidden = self.encoder(encoded_input)
        summary = hidden[-1]
        context_features = self.context_projection(context)
        features = torch.cat([summary, context_features], dim=-1)

        if retrieved_features is not None:
            features = features + retrieved_features

        features = self.dropout(features)
        region_logits = [head(features) for head in self.region_heads]
        status_logits = [head(features) for head in self.status_heads]

        return {
            "region_logits": region_logits,
            "status_logits": torch.stack(status_logits, dim=1),
        }


class HierarchicalH3RegionalForecaster(nn.Module):
    def __init__(
        self,
        input_num_region_classes: int,
        input_parent_num_region_classes: int,
        num_coarse_region_classes_per_horizon: list[int],
        child_count_by_coarse_per_horizon: list[list[int]],
        max_children_per_horizon: list[int],
        num_status_classes: int,
        num_numeric_features: int,
        num_mask_features: int,
        num_context_features: int,
        config: ModelConfig,
    ):
        super().__init__()
        self.config = config
        self.region_embedding = nn.Embedding(input_num_region_classes, config.embedding_dim)
        self.parent_region_embedding = nn.Embedding(input_parent_num_region_classes, config.parent_embedding_dim)
        self.numeric_projection = nn.Linear(num_numeric_features, config.numeric_projection_dim)
        self.mask_projection = nn.Linear(num_mask_features, config.mask_projection_dim)
        self.context_projection = nn.Linear(num_context_features, config.context_projection_dim)

        encoder_dim = (
            config.embedding_dim
            + config.parent_embedding_dim
            + config.numeric_projection_dim
            + config.mask_projection_dim
        )
        self.encoder = nn.GRU(
            input_size=encoder_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.feature_dim = config.hidden_dim + config.context_projection_dim
        self.coarse_heads = nn.ModuleList(
            [nn.Linear(self.feature_dim, num_classes) for num_classes in num_coarse_region_classes_per_horizon]
        )
        self.status_heads = nn.ModuleList(
            [nn.Linear(self.feature_dim, num_status_classes) for _ in range(len(num_coarse_region_classes_per_horizon))]
        )
        self.child_parent_embeddings = nn.ModuleList(
            [nn.Embedding(num_classes, config.child_context_dim) for num_classes in num_coarse_region_classes_per_horizon]
        )
        self.child_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.feature_dim + config.child_context_dim, config.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.hidden_dim, max_children),
                )
                for max_children in max_children_per_horizon
            ]
        )
        for horizon_idx, (num_classes, max_children, child_counts) in enumerate(
            zip(
                num_coarse_region_classes_per_horizon,
                max_children_per_horizon,
                child_count_by_coarse_per_horizon,
            )
        ):
            counts = list(child_counts)
            if len(counts) < num_classes:
                counts = counts + [1] * (num_classes - len(counts))
            elif len(counts) > num_classes:
                counts = counts[:num_classes]

            valid_mask = torch.zeros((num_classes, max_children), dtype=torch.bool)
            for coarse_id, child_count in enumerate(counts):
                valid_count = max(1, min(int(child_count), max_children))
                valid_mask[coarse_id, :valid_count] = True
            self.register_buffer(f"child_valid_mask_{horizon_idx}", valid_mask, persistent=False)

    def _child_valid_mask(self, horizon_idx: int) -> torch.Tensor:
        return getattr(self, f"child_valid_mask_{horizon_idx}")

    def child_target_is_valid(
        self,
        coarse_ids: torch.Tensor,
        child_ids: torch.Tensor,
        horizon_idx: int,
    ) -> torch.Tensor:
        valid_mask = self._child_valid_mask(horizon_idx)[coarse_ids]
        if child_ids.ndim == 1:
            return valid_mask.gather(1, child_ids.unsqueeze(1)).squeeze(1)
        return valid_mask.gather(-1, child_ids.unsqueeze(-1)).squeeze(-1)

    def encode(
        self,
        region_seq: torch.Tensor,
        parent_region_seq: torch.Tensor,
        numeric_seq: torch.Tensor,
        mask_seq: torch.Tensor,
        context: torch.Tensor,
        retrieved_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        region_features = self.region_embedding(region_seq)
        parent_region_features = self.parent_region_embedding(parent_region_seq)
        numeric_features = self.numeric_projection(numeric_seq)
        mask_features = self.mask_projection(mask_seq)
        encoded_input = torch.cat(
            [region_features, parent_region_features, numeric_features, mask_features],
            dim=-1,
        )

        _, hidden = self.encoder(encoded_input)
        summary = hidden[-1]
        context_features = self.context_projection(context)
        features = torch.cat([summary, context_features], dim=-1)

        if retrieved_features is not None:
            features = features + retrieved_features

        return self.dropout(features)

    def child_logits(self, features: torch.Tensor, coarse_ids: torch.Tensor, horizon_idx: int) -> torch.Tensor:
        original_shape = coarse_ids.shape
        if coarse_ids.ndim == 1:
            flat_coarse_ids = coarse_ids
            flat_features = features
        else:
            batch_size, candidate_count = coarse_ids.shape
            flat_coarse_ids = coarse_ids.reshape(-1)
            flat_features = (
                features.unsqueeze(1)
                .expand(batch_size, candidate_count, features.shape[-1])
                .reshape(-1, features.shape[-1])
            )

        coarse_features = self.child_parent_embeddings[horizon_idx](flat_coarse_ids)
        child_input = torch.cat([flat_features, coarse_features], dim=-1)
        child_logits = self.child_heads[horizon_idx](child_input)
        valid_mask = self._child_valid_mask(horizon_idx)[flat_coarse_ids]
        child_logits = child_logits.masked_fill(~valid_mask, -1.0e4)

        if len(original_shape) == 1:
            return child_logits
        return child_logits.reshape(original_shape[0], original_shape[1], -1)

    def forward(
        self,
        region_seq: torch.Tensor,
        parent_region_seq: torch.Tensor,
        numeric_seq: torch.Tensor,
        mask_seq: torch.Tensor,
        context: torch.Tensor,
        retrieved_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor | list[torch.Tensor]]:
        features = self.encode(
            region_seq=region_seq,
            parent_region_seq=parent_region_seq,
            numeric_seq=numeric_seq,
            mask_seq=mask_seq,
            context=context,
            retrieved_features=retrieved_features,
        )
        coarse_logits = [head(features) for head in self.coarse_heads]
        status_logits = [head(features) for head in self.status_heads]
        return {
            "features": features,
            "coarse_logits": coarse_logits,
            "status_logits": torch.stack(status_logits, dim=1),
        }
