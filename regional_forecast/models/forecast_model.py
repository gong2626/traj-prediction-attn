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


class AttentionHierarchicalH3RegionalForecaster(nn.Module):
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
        self.num_horizons = len(num_coarse_region_classes_per_horizon)
        self.unknown_input_region_id = int(input_num_region_classes) - 1
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
        self.temporal_attention = nn.Sequential(
            nn.Linear(config.hidden_dim, config.temporal_attention_dim),
            nn.Tanh(),
            nn.Dropout(config.temporal_attention_dropout),
            nn.Linear(config.temporal_attention_dim, 1),
        )
        self.dropout = nn.Dropout(config.dropout)
        self.feature_dim = config.hidden_dim + config.context_projection_dim
        self.horizon_embedding = nn.Embedding(self.num_horizons, config.horizon_embedding_dim)
        self.horizon_feature_projection = nn.Sequential(
            nn.Linear(self.feature_dim + config.horizon_embedding_dim, self.feature_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )
        self.coarse_heads = nn.ModuleList(
            [nn.Linear(self.feature_dim, num_classes) for num_classes in num_coarse_region_classes_per_horizon]
        )
        self.status_heads = nn.ModuleList(
            [nn.Linear(self.feature_dim, num_status_classes) for _ in range(self.num_horizons)]
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

    def _temporal_mask(self, region_seq: torch.Tensor, mask_seq: torch.Tensor) -> torch.Tensor:
        observed_flag = mask_seq[..., 0] > 0.5
        gap_flag = mask_seq[..., 1] > 0.5
        padded_unknown = region_seq == self.unknown_input_region_id
        invalid = padded_unknown & gap_flag & (~observed_flag)
        return ~invalid

    def _attention_pool(
        self,
        encoded_sequence: torch.Tensor,
        region_seq: torch.Tensor,
        mask_seq: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attention_logits = self.temporal_attention(encoded_sequence).squeeze(-1)
        temporal_mask = self._temporal_mask(region_seq, mask_seq)
        mask_fill_value = torch.finfo(attention_logits.dtype).min
        attention_logits = attention_logits.masked_fill(~temporal_mask, mask_fill_value)
        attention_weights = torch.softmax(attention_logits, dim=-1)
        summary = torch.sum(encoded_sequence * attention_weights.unsqueeze(-1), dim=1)
        return summary, attention_weights

    def encode(
        self,
        region_seq: torch.Tensor,
        parent_region_seq: torch.Tensor,
        numeric_seq: torch.Tensor,
        mask_seq: torch.Tensor,
        context: torch.Tensor,
        retrieved_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        region_features = self.region_embedding(region_seq)
        parent_region_features = self.parent_region_embedding(parent_region_seq)
        numeric_features = self.numeric_projection(numeric_seq)
        mask_features = self.mask_projection(mask_seq)
        encoded_input = torch.cat(
            [region_features, parent_region_features, numeric_features, mask_features],
            dim=-1,
        )

        encoded_sequence, _ = self.encoder(encoded_input)
        summary, attention_weights = self._attention_pool(encoded_sequence, region_seq, mask_seq)
        context_features = self.context_projection(context)
        features = torch.cat([summary, context_features], dim=-1)

        if retrieved_features is not None:
            features = features + retrieved_features

        features = self.dropout(features)
        horizon_ids = torch.arange(self.num_horizons, device=features.device, dtype=torch.long)
        horizon_embeds = self.horizon_embedding(horizon_ids).unsqueeze(0).expand(features.shape[0], -1, -1)
        decoder_input = torch.cat(
            [features.unsqueeze(1).expand(-1, self.num_horizons, -1), horizon_embeds],
            dim=-1,
        )
        decoder_features = self.horizon_feature_projection(decoder_input)
        return {
            "features": features,
            "decoder_features": decoder_features,
            "attention_weights": attention_weights,
        }

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
        encoded = self.encode(
            region_seq=region_seq,
            parent_region_seq=parent_region_seq,
            numeric_seq=numeric_seq,
            mask_seq=mask_seq,
            context=context,
            retrieved_features=retrieved_features,
        )
        decoder_features: torch.Tensor = encoded["decoder_features"]
        coarse_logits = [
            head(decoder_features[:, horizon_idx, :])
            for horizon_idx, head in enumerate(self.coarse_heads)
        ]
        status_logits = [
            head(decoder_features[:, horizon_idx, :])
            for horizon_idx, head in enumerate(self.status_heads)
        ]
        return {
            "features": encoded["features"],
            "decoder_features": decoder_features,
            "attention_weights": encoded["attention_weights"],
            "coarse_logits": coarse_logits,
            "status_logits": torch.stack(status_logits, dim=1),
        }


class HorizonDecoderHierarchicalH3RegionalForecaster(nn.Module):
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
        self.num_horizons = len(num_coarse_region_classes_per_horizon)
        self.unknown_input_region_id = int(input_num_region_classes) - 1
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
        self.horizon_embedding = nn.Embedding(self.num_horizons, config.horizon_embedding_dim)
        self.horizon_query_projection = nn.Linear(
            self.feature_dim + config.horizon_embedding_dim,
            config.hidden_dim,
        )
        self.horizon_feature_projection = nn.Sequential(
            nn.Linear(self.feature_dim + config.hidden_dim + config.horizon_embedding_dim, self.feature_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )
        self.coarse_heads = nn.ModuleList(
            [nn.Linear(self.feature_dim, num_classes) for num_classes in num_coarse_region_classes_per_horizon]
        )
        self.status_heads = nn.ModuleList(
            [nn.Linear(self.feature_dim, num_status_classes) for _ in range(self.num_horizons)]
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

    def _temporal_mask(self, region_seq: torch.Tensor, mask_seq: torch.Tensor) -> torch.Tensor:
        observed_flag = mask_seq[..., 0] > 0.5
        gap_flag = mask_seq[..., 1] > 0.5
        padded_unknown = region_seq == self.unknown_input_region_id
        invalid = padded_unknown & gap_flag & (~observed_flag)
        return ~invalid

    def encode(
        self,
        region_seq: torch.Tensor,
        parent_region_seq: torch.Tensor,
        numeric_seq: torch.Tensor,
        mask_seq: torch.Tensor,
        context: torch.Tensor,
        retrieved_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        region_features = self.region_embedding(region_seq)
        parent_region_features = self.parent_region_embedding(parent_region_seq)
        numeric_features = self.numeric_projection(numeric_seq)
        mask_features = self.mask_projection(mask_seq)
        encoded_input = torch.cat(
            [region_features, parent_region_features, numeric_features, mask_features],
            dim=-1,
        )

        encoded_sequence, hidden = self.encoder(encoded_input)
        summary = hidden[-1]
        context_features = self.context_projection(context)
        base_features = torch.cat([summary, context_features], dim=-1)

        if retrieved_features is not None:
            base_features = base_features + retrieved_features

        base_features = self.dropout(base_features)
        temporal_mask = self._temporal_mask(region_seq, mask_seq)

        horizon_ids = torch.arange(self.num_horizons, device=base_features.device, dtype=torch.long)
        horizon_embeds = self.horizon_embedding(horizon_ids).unsqueeze(0).expand(base_features.shape[0], -1, -1)
        query_input = torch.cat(
            [base_features.unsqueeze(1).expand(-1, self.num_horizons, -1), horizon_embeds],
            dim=-1,
        )
        queries = self.horizon_query_projection(query_input)
        attention_logits = torch.einsum("bhd,bsd->bhs", queries, encoded_sequence)
        mask_fill_value = torch.finfo(attention_logits.dtype).min
        attention_logits = attention_logits.masked_fill(~temporal_mask.unsqueeze(1), mask_fill_value)
        attention_weights = torch.softmax(attention_logits, dim=-1)
        attended_sequence = torch.einsum("bhs,bsd->bhd", attention_weights, encoded_sequence)
        decoder_input = torch.cat(
            [
                base_features.unsqueeze(1).expand(-1, self.num_horizons, -1),
                attended_sequence,
                horizon_embeds,
            ],
            dim=-1,
        )
        decoder_features = self.horizon_feature_projection(decoder_input)
        return {
            "features": base_features,
            "decoder_features": decoder_features,
            "attention_weights": attention_weights,
        }

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
        encoded = self.encode(
            region_seq=region_seq,
            parent_region_seq=parent_region_seq,
            numeric_seq=numeric_seq,
            mask_seq=mask_seq,
            context=context,
            retrieved_features=retrieved_features,
        )
        decoder_features: torch.Tensor = encoded["decoder_features"]
        coarse_logits = [
            head(decoder_features[:, horizon_idx, :])
            for horizon_idx, head in enumerate(self.coarse_heads)
        ]
        status_logits = [
            head(decoder_features[:, horizon_idx, :])
            for horizon_idx, head in enumerate(self.status_heads)
        ]
        return {
            "features": encoded["features"],
            "decoder_features": decoder_features,
            "attention_weights": encoded["attention_weights"],
            "coarse_logits": coarse_logits,
            "status_logits": torch.stack(status_logits, dim=1),
        }


class TransformerHorizonDecoderHierarchicalH3RegionalForecaster(HorizonDecoderHierarchicalH3RegionalForecaster):
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
    ):
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
        self.sequence_input_projection = nn.Linear(encoder_input_dim, config.hidden_dim)
        self.position_embedding = nn.Embedding(self.max_history_steps, config.hidden_dim)
        self.sequence_layer_norm = nn.LayerNorm(config.hidden_dim)
        self.sequence_dropout = nn.Dropout(config.transformer_dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.transformer_num_heads,
            dim_feedforward=config.transformer_ffn_dim,
            dropout=config.transformer_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.transformer_num_layers,
            enable_nested_tensor=False,
        )

    def _masked_mean_pool(self, encoded_sequence: torch.Tensor, temporal_mask: torch.Tensor) -> torch.Tensor:
        mask = temporal_mask.unsqueeze(-1).to(encoded_sequence.dtype)
        weighted = encoded_sequence * mask
        denom = mask.sum(dim=1).clamp_min(1.0)
        return weighted.sum(dim=1) / denom

    def encode(
        self,
        region_seq: torch.Tensor,
        parent_region_seq: torch.Tensor,
        numeric_seq: torch.Tensor,
        mask_seq: torch.Tensor,
        context: torch.Tensor,
        retrieved_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
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
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_history_steps={self.max_history_steps}."
            )

        temporal_mask = self._temporal_mask(region_seq, mask_seq)
        position_ids = torch.arange(seq_len, device=encoded_input.device, dtype=torch.long)
        position_features = self.position_embedding(position_ids).unsqueeze(0)
        transformer_input = self.sequence_input_projection(encoded_input) + position_features
        transformer_input = self.sequence_dropout(self.sequence_layer_norm(transformer_input))
        encoded_sequence = self.encoder(
            transformer_input,
            src_key_padding_mask=~temporal_mask,
        )
        summary = self._masked_mean_pool(encoded_sequence, temporal_mask)
        context_features = self.context_projection(context)
        base_features = torch.cat([summary, context_features], dim=-1)

        if retrieved_features is not None:
            base_features = base_features + retrieved_features

        base_features = self.dropout(base_features)
        horizon_ids = torch.arange(self.num_horizons, device=base_features.device, dtype=torch.long)
        horizon_embeds = self.horizon_embedding(horizon_ids).unsqueeze(0).expand(base_features.shape[0], -1, -1)
        query_input = torch.cat(
            [base_features.unsqueeze(1).expand(-1, self.num_horizons, -1), horizon_embeds],
            dim=-1,
        )
        queries = self.horizon_query_projection(query_input)
        attention_logits = torch.einsum("bhd,bsd->bhs", queries, encoded_sequence)
        mask_fill_value = torch.finfo(attention_logits.dtype).min
        attention_logits = attention_logits.masked_fill(~temporal_mask.unsqueeze(1), mask_fill_value)
        attention_weights = torch.softmax(attention_logits, dim=-1)
        attended_sequence = torch.einsum("bhs,bsd->bhd", attention_weights, encoded_sequence)
        decoder_input = torch.cat(
            [
                base_features.unsqueeze(1).expand(-1, self.num_horizons, -1),
                attended_sequence,
                horizon_embeds,
            ],
            dim=-1,
        )
        decoder_features = self.horizon_feature_projection(decoder_input)
        return {
            "features": base_features,
            "decoder_features": decoder_features,
            "attention_weights": attention_weights,
        }


class MultiscaleTransformerH3RegionalForecaster(TransformerHorizonDecoderHierarchicalH3RegionalForecaster):
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
    ):
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
            max_history_steps=max_history_steps,
            config=config,
        )
        self.multiscale_stride = max(1, int(config.multiscale_stride))
        coarse_encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.transformer_num_heads,
            dim_feedforward=config.transformer_ffn_dim,
            dropout=config.transformer_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.coarse_encoder = nn.TransformerEncoder(
            coarse_encoder_layer,
            num_layers=max(1, int(config.multiscale_num_layers)),
            enable_nested_tensor=False,
        )
        self.scale_feature_projection = nn.Sequential(
            nn.Linear(self.feature_dim + config.hidden_dim + config.hidden_dim + config.horizon_embedding_dim, self.feature_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )

    def _pool_multiscale(
        self,
        encoded_sequence: torch.Tensor,
        temporal_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_dim = encoded_sequence.shape
        stride = self.multiscale_stride
        coarse_steps = (seq_len + stride - 1) // stride
        coarse_sequence = []
        coarse_mask = []
        for coarse_idx in range(coarse_steps):
            start = coarse_idx * stride
            end = min(seq_len, start + stride)
            local_features = encoded_sequence[:, start:end, :]
            local_mask = temporal_mask[:, start:end]
            mask = local_mask.unsqueeze(-1).to(encoded_sequence.dtype)
            denom = mask.sum(dim=1).clamp_min(1.0)
            pooled = (local_features * mask).sum(dim=1) / denom
            coarse_sequence.append(pooled)
            coarse_mask.append(local_mask.any(dim=1))
        return torch.stack(coarse_sequence, dim=1), torch.stack(coarse_mask, dim=1)

    def encode(
        self,
        region_seq: torch.Tensor,
        parent_region_seq: torch.Tensor,
        numeric_seq: torch.Tensor,
        mask_seq: torch.Tensor,
        context: torch.Tensor,
        retrieved_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
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
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_history_steps={self.max_history_steps}."
            )

        temporal_mask = self._temporal_mask(region_seq, mask_seq)
        position_ids = torch.arange(seq_len, device=encoded_input.device, dtype=torch.long)
        position_features = self.position_embedding(position_ids).unsqueeze(0)
        transformer_input = self.sequence_input_projection(encoded_input) + position_features
        transformer_input = self.sequence_dropout(self.sequence_layer_norm(transformer_input))
        fine_sequence = self.encoder(
            transformer_input,
            src_key_padding_mask=~temporal_mask,
        )
        coarse_input, coarse_mask = self._pool_multiscale(fine_sequence, temporal_mask)
        coarse_sequence = self.coarse_encoder(
            coarse_input,
            src_key_padding_mask=~coarse_mask,
        )

        summary = self._masked_mean_pool(fine_sequence, temporal_mask)
        context_features = self.context_projection(context)
        base_features = torch.cat([summary, context_features], dim=-1)

        if retrieved_features is not None:
            base_features = base_features + retrieved_features

        base_features = self.dropout(base_features)
        horizon_ids = torch.arange(self.num_horizons, device=base_features.device, dtype=torch.long)
        horizon_embeds = self.horizon_embedding(horizon_ids).unsqueeze(0).expand(base_features.shape[0], -1, -1)
        query_input = torch.cat(
            [base_features.unsqueeze(1).expand(-1, self.num_horizons, -1), horizon_embeds],
            dim=-1,
        )
        queries = self.horizon_query_projection(query_input)

        fine_logits = torch.einsum("bhd,bsd->bhs", queries, fine_sequence)
        fine_logits = fine_logits.masked_fill(~temporal_mask.unsqueeze(1), torch.finfo(fine_logits.dtype).min)
        fine_weights = torch.softmax(fine_logits, dim=-1)
        fine_attended = torch.einsum("bhs,bsd->bhd", fine_weights, fine_sequence)

        coarse_logits = torch.einsum("bhd,btd->bht", queries, coarse_sequence)
        coarse_logits = coarse_logits.masked_fill(~coarse_mask.unsqueeze(1), torch.finfo(coarse_logits.dtype).min)
        coarse_weights = torch.softmax(coarse_logits, dim=-1)
        coarse_attended = torch.einsum("bht,btd->bhd", coarse_weights, coarse_sequence)

        decoder_input = torch.cat(
            [
                base_features.unsqueeze(1).expand(-1, self.num_horizons, -1),
                fine_attended,
                coarse_attended,
                horizon_embeds,
            ],
            dim=-1,
        )
        decoder_features = self.scale_feature_projection(decoder_input)
        return {
            "features": base_features,
            "decoder_features": decoder_features,
            "attention_weights": fine_weights,
            "coarse_attention_weights": coarse_weights,
        }
