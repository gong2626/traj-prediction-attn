from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from regional_forecast.config import ModelConfig
from regional_forecast.external_baselines.common import (
    LearnedAttentionPool,
    TemporalConvResidualBlock,
    temporal_valid_mask,
)
from regional_forecast.models.forecast_model import HierarchicalH3RegionalForecaster


class AttentionTCNGRUHierarchicalH3RegionalForecaster(HierarchicalH3RegionalForecaster):
    """Adapted H3 regional baseline inspired by attention-TCN-GRU trajectory predictors."""

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
        encoder_input_dim = (
            config.embedding_dim
            + config.parent_embedding_dim
            + config.numeric_projection_dim
            + config.mask_projection_dim
        )
        self.sequence_projection = nn.Linear(encoder_input_dim, config.hidden_dim)
        tcn_layers = max(1, int(getattr(config, "external_tcn_num_layers", 3)))
        kernel_size = max(2, int(getattr(config, "external_tcn_kernel_size", 3)))
        tcn_dropout = float(getattr(config, "external_tcn_dropout", config.dropout))
        self.tcn = nn.ModuleList(
            [
                TemporalConvResidualBlock(
                    channels=config.hidden_dim,
                    kernel_size=kernel_size,
                    dilation=2 ** layer_idx,
                    dropout=tcn_dropout,
                )
                for layer_idx in range(tcn_layers)
            ]
        )
        self.gru_encoder = nn.GRU(
            input_size=config.hidden_dim,
            hidden_size=config.hidden_dim,
            num_layers=max(1, int(config.num_layers)),
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.temporal_attention = LearnedAttentionPool(
            hidden_dim=config.hidden_dim,
            attention_dim=max(32, int(config.temporal_attention_dim)),
            dropout=float(config.temporal_attention_dropout),
        )

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

        temporal_mask = temporal_valid_mask(region_seq, mask_seq, self.region_embedding.num_embeddings - 1)
        sequence_features = self.sequence_projection(encoded_input).transpose(1, 2)
        for layer in self.tcn:
            sequence_features = layer(sequence_features)
        sequence_features = sequence_features.transpose(1, 2)
        sequence_features = sequence_features * temporal_mask.unsqueeze(-1).to(sequence_features.dtype)

        encoded_sequence, _ = self.gru_encoder(sequence_features)
        summary, _ = self.temporal_attention(encoded_sequence, temporal_mask)
        context_features = self.context_projection(context)
        features = torch.cat([summary, context_features], dim=-1)
        if retrieved_features is not None:
            features = features + retrieved_features
        return self.dropout(features)
