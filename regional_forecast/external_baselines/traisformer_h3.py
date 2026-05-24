from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from regional_forecast.config import ModelConfig
from regional_forecast.external_baselines.common import (
    CausalTransformerBlock,
    masked_last_timestep,
    temporal_valid_mask,
)
from regional_forecast.models.forecast_model import HierarchicalH3RegionalForecaster


class TrAISformerHierarchicalH3RegionalForecaster(HierarchicalH3RegionalForecaster):
    """Adapted H3 regional baseline inspired by TrAISformer causal discrete trajectory modeling."""

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
        max_history_steps: int,
        config: ModelConfig,
    ) -> None:
        super().__init__(
            input_num_region_classes=input_num_region_classes,
            input_parent_num_region_classes=input_parent_num_region_classes,
            num_coarse_region_classes_per_horizon=num_coarse_region_classes_per_horizon,
            child_count_by_coarse_per_horizon=child_count_by_coarse_per_horizon,
            max_children_per_horizon=max_children_per_horizon,
            num_status_classes=num_status_classes,
            num_numeric_features=num_numeric_features,
            num_mask_features=num_mask_features,
            num_context_features=num_context_features,
            config=config,
        )
        self.max_history_steps = int(max_history_steps)
        encoder_input_dim = (
            config.embedding_dim
            + config.parent_embedding_dim
            + config.numeric_projection_dim
            + config.mask_projection_dim
        )
        self.sequence_projection = nn.Linear(encoder_input_dim, config.hidden_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, self.max_history_steps, config.hidden_dim))
        self.sequence_dropout = nn.Dropout(config.transformer_dropout)
        self.blocks = nn.ModuleList(
            [
                CausalTransformerBlock(
                    hidden_dim=config.hidden_dim,
                    num_heads=config.transformer_num_heads,
                    dropout=config.transformer_dropout,
                )
                for _ in range(max(1, int(config.transformer_num_layers)))
            ]
        )
        self.final_layer_norm = nn.LayerNorm(config.hidden_dim)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)

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

        seq_len = int(encoded_input.shape[1])
        if seq_len > self.max_history_steps:
            raise ValueError(f"Sequence length {seq_len} exceeds max_history_steps={self.max_history_steps}.")

        temporal_mask = temporal_valid_mask(region_seq, mask_seq, self.region_embedding.num_embeddings - 1)
        x = self.sequence_projection(encoded_input) + self.position_embedding[:, :seq_len, :]
        x = self.sequence_dropout(x)
        causal_mask = self._causal_mask(seq_len, x.device)
        key_padding_mask = ~temporal_mask
        for block in self.blocks:
            x = block(x, causal_mask=causal_mask, key_padding_mask=key_padding_mask)
        x = self.final_layer_norm(x)
        summary = masked_last_timestep(x, temporal_mask)
        context_features = self.context_projection(context)
        features = torch.cat([summary, context_features], dim=-1)
        if retrieved_features is not None:
            features = features + retrieved_features
        return self.dropout(features)
