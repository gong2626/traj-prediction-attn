from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from regional_forecast.config import ModelConfig
from regional_forecast.external_baselines.common import (
    LearnedAttentionPool,
    SinusoidalPositionalEncoding,
    temporal_valid_mask,
)
from regional_forecast.models.forecast_model import HierarchicalH3RegionalForecaster


class _ConvTokenEmbedding(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.token_conv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=hidden_dim,
            kernel_size=3,
            padding=1,
            padding_mode="zeros",
        )
        nn.init.kaiming_normal_(self.token_conv.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.token_conv(x.transpose(1, 2)).transpose(1, 2)


class FlightBERTPPHierarchicalH3RegionalForecaster(HierarchicalH3RegionalForecaster):
    """Adapted H3 regional baseline inspired by FlightBERT++ non-autoregressive multi-horizon decoding."""

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
        self.num_horizons = len(num_coarse_region_classes_per_horizon)
        self.max_history_steps = int(max_history_steps)
        encoder_input_dim = (
            config.embedding_dim
            + config.parent_embedding_dim
            + config.numeric_projection_dim
            + config.mask_projection_dim
        )
        self.encoder_token_embedding = _ConvTokenEmbedding(encoder_input_dim, config.hidden_dim)
        self.decoder_token_embedding = _ConvTokenEmbedding(encoder_input_dim, config.hidden_dim)
        self.encoder_position = SinusoidalPositionalEncoding(
            d_model=config.hidden_dim,
            dropout=config.transformer_dropout,
            max_len=max_history_steps,
        )
        self.decoder_position = SinusoidalPositionalEncoding(
            d_model=config.hidden_dim,
            dropout=config.transformer_dropout,
            max_len=max(self.num_horizons, max_history_steps),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.transformer_num_heads,
            dim_feedforward=config.transformer_ffn_dim,
            dropout=config.transformer_dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=max(1, int(config.transformer_num_layers)),
            enable_nested_tensor=False,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_dim,
            nhead=max(1, int(getattr(config, "external_decoder_num_heads", config.transformer_num_heads))),
            dim_feedforward=config.transformer_ffn_dim,
            dropout=config.transformer_dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=max(1, int(getattr(config, "external_decoder_num_layers", config.transformer_num_layers))),
        )
        self.feature_aggregation = LearnedAttentionPool(
            hidden_dim=config.hidden_dim,
            attention_dim=max(32, int(config.temporal_attention_dim)),
            dropout=float(config.temporal_attention_dropout),
        )
        self.horizon_embedding = nn.Embedding(self.num_horizons, config.hidden_dim)
        self.decoder_seed_projection = nn.Linear(self.feature_dim, config.hidden_dim)
        self.query_fusion = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.transformer_dropout),
        )
        self.decoder_output_projection = nn.Sequential(
            nn.Linear(config.hidden_dim + self.feature_dim, self.feature_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.decoder_norm = nn.LayerNorm(config.hidden_dim)

    def _build_zero_decoder_tokens(self, batch_size: int, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros((batch_size, seq_len, self.encoder_token_embedding.token_conv.in_channels), device=device, dtype=dtype)

    def forward(
        self,
        region_seq: torch.Tensor,
        parent_region_seq: torch.Tensor,
        numeric_seq: torch.Tensor,
        mask_seq: torch.Tensor,
        context: torch.Tensor,
        retrieved_features: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
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

        encoder_tokens = self.encoder_position(self.encoder_token_embedding(encoded_input))
        encoder_memory = self.transformer_encoder(
            encoder_tokens,
            src_key_padding_mask=~temporal_mask,
        )
        summary, attention_weights = self.feature_aggregation(encoder_memory, temporal_mask)
        context_features = self.context_projection(context)
        base_features = torch.cat([summary, context_features], dim=-1)
        if retrieved_features is not None:
            base_features = base_features + retrieved_features
        base_features = self.dropout(base_features)

        horizon_ids = torch.arange(self.num_horizons, device=base_features.device, dtype=torch.long)
        horizon_features = self.horizon_embedding(horizon_ids).unsqueeze(0).expand(base_features.shape[0], -1, -1)
        decoder_seed = self.decoder_seed_projection(base_features).unsqueeze(1).expand(-1, self.num_horizons, -1)
        decoder_queries = self.query_fusion(torch.cat([decoder_seed, horizon_features], dim=-1))

        zero_tokens = self._build_zero_decoder_tokens(
            batch_size=base_features.shape[0],
            seq_len=self.num_horizons,
            device=base_features.device,
            dtype=encoded_input.dtype,
        )
        decoder_tokens = self.decoder_token_embedding(zero_tokens)
        decoder_input = self.decoder_position(decoder_tokens + decoder_queries)
        decoder_hidden = self.transformer_decoder(
            tgt=decoder_input,
            memory=encoder_memory,
            memory_key_padding_mask=~temporal_mask,
        )
        decoder_hidden = self.decoder_norm(decoder_hidden)
        decoder_features = self.decoder_output_projection(
            torch.cat([decoder_hidden, base_features.unsqueeze(1).expand(-1, self.num_horizons, -1)], dim=-1)
        )
        coarse_logits = [
            head(decoder_features[:, horizon_idx, :])
            for horizon_idx, head in enumerate(self.coarse_heads)
        ]
        status_logits = [
            head(decoder_features[:, horizon_idx, :])
            for horizon_idx, head in enumerate(self.status_heads)
        ]
        return {
            "features": base_features,
            "decoder_features": decoder_features,
            "attention_weights": attention_weights,
            "coarse_logits": coarse_logits,
            "status_logits": torch.stack(status_logits, dim=1),
        }
