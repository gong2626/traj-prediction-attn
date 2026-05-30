import unittest

import torch
import torch.nn as nn

from regional_forecast.config import ModelConfig
from regional_forecast.models.forecast_model import (
    AttentionHierarchicalH3RegionalForecaster,
    HorizonRouterTransformerH3RegionalForecaster,
    HorizonDecoderHierarchicalH3RegionalForecaster,
    MultiscaleTransformerH3RegionalForecaster,
    TransformerHorizonDecoderHierarchicalH3RegionalForecaster,
)


class AttentionHierarchicalModelTests(unittest.TestCase):
    @staticmethod
    def _make_model():
        return AttentionHierarchicalH3RegionalForecaster(
            input_num_region_classes=32,
            input_parent_num_region_classes=16,
            num_coarse_region_classes_per_horizon=[8, 6],
            child_count_by_coarse_per_horizon=[
                [2, 3, 1, 2, 1, 1, 2, 1],
                [2, 1, 2, 1, 1, 1],
            ],
            max_children_per_horizon=[3, 2],
            num_status_classes=2,
            num_numeric_features=5,
            num_mask_features=2,
            num_context_features=4,
            config=ModelConfig(
                model_type="attention_hierarchical_gru",
                embedding_dim=8,
                parent_embedding_dim=4,
                numeric_projection_dim=6,
                mask_projection_dim=3,
                context_projection_dim=5,
                hidden_dim=10,
                num_layers=2,
                dropout=0.1,
                child_context_dim=4,
            ),
        )

    def test_forward_and_child_logits_shapes(self):
        model = self._make_model()

        batch_size = 3
        seq_len = 6
        outputs = model(
            region_seq=torch.randint(0, 32, (batch_size, seq_len)),
            parent_region_seq=torch.randint(0, 16, (batch_size, seq_len)),
            numeric_seq=torch.randn(batch_size, seq_len, 5),
            mask_seq=torch.randn(batch_size, seq_len, 2),
            context=torch.randn(batch_size, 4),
        )

        self.assertEqual(len(outputs["coarse_logits"]), 2)
        self.assertEqual(tuple(outputs["coarse_logits"][0].shape), (batch_size, 8))
        self.assertEqual(tuple(outputs["coarse_logits"][1].shape), (batch_size, 6))
        self.assertEqual(tuple(outputs["status_logits"].shape), (batch_size, 2, 2))
        self.assertEqual(tuple(outputs["decoder_features"].shape[:2]), (batch_size, 2))

        horizon_features = outputs["decoder_features"][:, 0, :]
        coarse_ids = torch.tensor([0, 1, 7], dtype=torch.long)
        child_logits = model.child_logits(horizon_features, coarse_ids, 0)
        self.assertEqual(tuple(child_logits.shape), (batch_size, 3))

    def test_horizon_decoder_forward_and_child_logits_shapes(self):
        model = HorizonDecoderHierarchicalH3RegionalForecaster(
            input_num_region_classes=32,
            input_parent_num_region_classes=16,
            num_coarse_region_classes_per_horizon=[8, 6],
            child_count_by_coarse_per_horizon=[
                [2, 3, 1, 2, 1, 1, 2, 1],
                [2, 1, 2, 1, 1, 1],
            ],
            max_children_per_horizon=[3, 2],
            num_status_classes=2,
            num_numeric_features=5,
            num_mask_features=2,
            num_context_features=4,
            config=ModelConfig(
                model_type="horizon_decoder_gru",
                embedding_dim=8,
                parent_embedding_dim=4,
                numeric_projection_dim=6,
                mask_projection_dim=3,
                context_projection_dim=5,
                hidden_dim=10,
                num_layers=2,
                dropout=0.1,
                child_context_dim=4,
                horizon_embedding_dim=6,
            ),
        )

        batch_size = 3
        seq_len = 6
        outputs = model(
            region_seq=torch.randint(0, 32, (batch_size, seq_len)),
            parent_region_seq=torch.randint(0, 16, (batch_size, seq_len)),
            numeric_seq=torch.randn(batch_size, seq_len, 5),
            mask_seq=torch.randn(batch_size, seq_len, 2),
            context=torch.randn(batch_size, 4),
        )

        self.assertEqual(len(outputs["coarse_logits"]), 2)
        self.assertEqual(tuple(outputs["coarse_logits"][0].shape), (batch_size, 8))
        self.assertEqual(tuple(outputs["coarse_logits"][1].shape), (batch_size, 6))
        self.assertEqual(tuple(outputs["status_logits"].shape), (batch_size, 2, 2))
        self.assertEqual(tuple(outputs["decoder_features"].shape[:2]), (batch_size, 2))
        self.assertEqual(tuple(outputs["attention_weights"].shape[:2]), (batch_size, 2))

        horizon_features = outputs["decoder_features"][:, 1, :]
        coarse_ids = torch.tensor([0, 1, 5], dtype=torch.long)
        child_logits = model.child_logits(horizon_features, coarse_ids, 1)
        self.assertEqual(tuple(child_logits.shape), (batch_size, 2))

    def test_transformer_horizon_decoder_forward_and_child_logits_shapes(self):
        model = TransformerHorizonDecoderHierarchicalH3RegionalForecaster(
            input_num_region_classes=32,
            input_parent_num_region_classes=16,
            num_coarse_region_classes_per_horizon=[8, 6],
            child_count_by_coarse_per_horizon=[
                [2, 3, 1, 2, 1, 1, 2, 1],
                [2, 1, 2, 1, 1, 1],
            ],
            max_children_per_horizon=[3, 2],
            num_status_classes=2,
            num_numeric_features=5,
            num_mask_features=2,
            num_context_features=4,
            max_history_steps=12,
            config=ModelConfig(
                model_type="transformer_horizon_decoder",
                embedding_dim=8,
                parent_embedding_dim=4,
                numeric_projection_dim=6,
                mask_projection_dim=3,
                context_projection_dim=5,
                hidden_dim=16,
                num_layers=2,
                dropout=0.1,
                child_context_dim=4,
                horizon_embedding_dim=6,
                transformer_num_layers=2,
                transformer_num_heads=4,
                transformer_ffn_dim=32,
                transformer_dropout=0.1,
            ),
        )

        batch_size = 3
        seq_len = 10
        outputs = model(
            region_seq=torch.randint(0, 32, (batch_size, seq_len)),
            parent_region_seq=torch.randint(0, 16, (batch_size, seq_len)),
            numeric_seq=torch.randn(batch_size, seq_len, 5),
            mask_seq=torch.randn(batch_size, seq_len, 2),
            context=torch.randn(batch_size, 4),
        )

        self.assertEqual(len(outputs["coarse_logits"]), 2)
        self.assertEqual(tuple(outputs["coarse_logits"][0].shape), (batch_size, 8))
        self.assertEqual(tuple(outputs["coarse_logits"][1].shape), (batch_size, 6))
        self.assertEqual(tuple(outputs["status_logits"].shape), (batch_size, 2, 2))
        self.assertEqual(tuple(outputs["decoder_features"].shape[:2]), (batch_size, 2))
        self.assertEqual(tuple(outputs["attention_weights"].shape[:2]), (batch_size, 2))

        horizon_features = outputs["decoder_features"][:, 0, :]
        coarse_ids = torch.tensor([0, 1, 7], dtype=torch.long)
        child_logits = model.child_logits(horizon_features, coarse_ids, 0)
        self.assertEqual(tuple(child_logits.shape), (batch_size, 3))

    def test_multiscale_transformer_forward_and_child_logits_shapes(self):
        model = MultiscaleTransformerH3RegionalForecaster(
            input_num_region_classes=32,
            input_parent_num_region_classes=16,
            num_coarse_region_classes_per_horizon=[8, 6],
            child_count_by_coarse_per_horizon=[
                [2, 3, 1, 2, 1, 1, 2, 1],
                [2, 1, 2, 1, 1, 1],
            ],
            max_children_per_horizon=[3, 2],
            num_status_classes=2,
            num_numeric_features=5,
            num_mask_features=2,
            num_context_features=4,
            max_history_steps=12,
            config=ModelConfig(
                model_type="multiscale_transformer_h3",
                embedding_dim=8,
                parent_embedding_dim=4,
                numeric_projection_dim=6,
                mask_projection_dim=3,
                context_projection_dim=5,
                hidden_dim=16,
                num_layers=2,
                dropout=0.1,
                child_context_dim=4,
                horizon_embedding_dim=6,
                transformer_num_layers=2,
                transformer_num_heads=4,
                transformer_ffn_dim=32,
                transformer_dropout=0.1,
                multiscale_stride=2,
                multiscale_num_layers=1,
            ),
        )

        batch_size = 3
        seq_len = 10
        outputs = model(
            region_seq=torch.randint(0, 32, (batch_size, seq_len)),
            parent_region_seq=torch.randint(0, 16, (batch_size, seq_len)),
            numeric_seq=torch.randn(batch_size, seq_len, 5),
            mask_seq=torch.randn(batch_size, seq_len, 2),
            context=torch.randn(batch_size, 4),
        )

        self.assertEqual(len(outputs["coarse_logits"]), 2)
        self.assertEqual(tuple(outputs["coarse_logits"][0].shape), (batch_size, 8))
        self.assertEqual(tuple(outputs["coarse_logits"][1].shape), (batch_size, 6))
        self.assertEqual(tuple(outputs["status_logits"].shape), (batch_size, 2, 2))
        self.assertEqual(tuple(outputs["decoder_features"].shape[:2]), (batch_size, 2))
        self.assertEqual(tuple(outputs["attention_weights"].shape[:2]), (batch_size, 2))

        horizon_features = outputs["decoder_features"][:, 1, :]
        coarse_ids = torch.tensor([0, 1, 5], dtype=torch.long)
        child_logits = model.child_logits(horizon_features, coarse_ids, 1)
        self.assertEqual(tuple(child_logits.shape), (batch_size, 2))

    def test_horizon_router_forward_exposes_router_and_expert_features(self):
        model = HorizonRouterTransformerH3RegionalForecaster(
            input_num_region_classes=32,
            input_parent_num_region_classes=16,
            num_coarse_region_classes_per_horizon=[8, 6],
            child_count_by_coarse_per_horizon=[
                [2, 3, 1, 2, 1, 1, 2, 1],
                [2, 1, 2, 1, 1, 1],
            ],
            max_children_per_horizon=[3, 2],
            num_status_classes=2,
            num_numeric_features=5,
            num_mask_features=2,
            num_context_features=4,
            max_history_steps=12,
            config=ModelConfig(
                model_type="horizon_router_transformer_h3",
                embedding_dim=8,
                parent_embedding_dim=4,
                numeric_projection_dim=6,
                mask_projection_dim=3,
                context_projection_dim=5,
                hidden_dim=16,
                num_layers=2,
                dropout=0.1,
                child_context_dim=4,
                horizon_embedding_dim=6,
                transformer_num_layers=2,
                transformer_num_heads=4,
                transformer_ffn_dim=32,
                transformer_dropout=0.1,
                multiscale_stride=2,
                multiscale_num_layers=1,
                short_term_branch_steps=4,
                short_term_branch_horizons=1,
                short_term_branch_num_layers=1,
                router_hidden_dim=12,
            ),
        )

        batch_size = 3
        seq_len = 10
        outputs = model(
            region_seq=torch.randint(0, 32, (batch_size, seq_len)),
            parent_region_seq=torch.randint(0, 16, (batch_size, seq_len)),
            numeric_seq=torch.randn(batch_size, seq_len, 5),
            mask_seq=torch.randn(batch_size, seq_len, 2),
            context=torch.randn(batch_size, 4),
        )

        self.assertEqual(tuple(outputs["router_logits"].shape), (batch_size, 2, 3))
        self.assertEqual(tuple(outputs["router_weights"].shape), (batch_size, 2, 3))
        self.assertEqual(tuple(outputs["global_expert_features"].shape[:2]), (batch_size, 2))
        self.assertEqual(tuple(outputs["coarse_expert_features"].shape[:2]), (batch_size, 2))
        self.assertEqual(tuple(outputs["local_expert_features"].shape[:2]), (batch_size, 2))
        self.assertTrue(torch.allclose(outputs["router_weights"].sum(dim=-1), torch.ones(batch_size, 2), atol=1e-5))

    def test_attention_pool_handles_fp16_mask_fill(self):
        class ZeroAttention(nn.Module):
            def forward(self, encoded_sequence):
                return torch.zeros(
                    encoded_sequence.shape[0],
                    encoded_sequence.shape[1],
                    1,
                    dtype=encoded_sequence.dtype,
                    device=encoded_sequence.device,
                )

        model = self._make_model()
        model.temporal_attention = ZeroAttention()

        encoded_sequence = torch.randn(2, 4, 10, dtype=torch.float16)
        region_seq = torch.tensor(
            [
                [1, 2, 31, 31],
                [3, 4, 5, 31],
            ],
            dtype=torch.long,
        )
        mask_seq = torch.tensor(
            [
                [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
                [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            ],
            dtype=torch.float16,
        )

        summary, attention_weights = model._attention_pool(encoded_sequence, region_seq, mask_seq)
        self.assertEqual(summary.dtype, torch.float16)
        self.assertTrue(torch.isfinite(summary).all())
        self.assertTrue(torch.isfinite(attention_weights).all())


if __name__ == "__main__":
    unittest.main()
