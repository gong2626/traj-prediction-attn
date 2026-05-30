import unittest

import torch
import torch.nn as nn

from regional_forecast.pipeline.train import (
    _build_router_target_distribution,
    _expert_auxiliary_loss,
    _horizon_consistency_loss,
    _masked_hierarchical_loss,
    _router_load_balancing_loss,
)


class HorizonConsistencyLossTests(unittest.TestCase):
    def test_returns_zero_without_decoder_features(self):
        label_mask = torch.tensor([[1, 1, 1]], dtype=torch.bool)

        loss = _horizon_consistency_loss(
            decoder_features=None,
            label_mask=label_mask,
            horizons_minutes=[10, 30, 60],
            weight=0.2,
        )

        self.assertEqual(float(loss.item()), 0.0)

    def test_uses_adjacent_valid_horizons_with_gap_scaling(self):
        decoder_features = torch.tensor(
            [
                [[0.0, 0.0], [2.0, 0.0], [5.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0], [100.0, 100.0]],
            ],
            dtype=torch.float32,
        )
        label_mask = torch.tensor(
            [
                [1, 1, 1],
                [1, 0, 1],
            ],
            dtype=torch.bool,
        )

        loss = _horizon_consistency_loss(
            decoder_features=decoder_features,
            label_mask=label_mask,
            horizons_minutes=[10, 30, 60],
            weight=0.5,
        )

        self.assertAlmostEqual(float(loss.item()), 1.25, places=6)

    def test_hierarchical_loss_adds_consistency_regularizer(self):
        class DummyModel:
            def __init__(self):
                self.child_logits_per_horizon = [
                    torch.tensor([[2.0, 0.0]], dtype=torch.float32),
                    torch.tensor([[0.0, 2.0]], dtype=torch.float32),
                ]

            def child_logits(self, features, coarse_ids, horizon_idx):
                return self.child_logits_per_horizon[horizon_idx]

            def child_target_is_valid(self, coarse_ids, child_ids, horizon_idx):
                return torch.ones_like(child_ids, dtype=torch.bool)

        model = DummyModel()
        region_criterion = nn.CrossEntropyLoss()
        status_criterion = nn.CrossEntropyLoss()
        outputs = {
            "features": torch.zeros((1, 4), dtype=torch.float32),
            "decoder_features": torch.tensor([[[0.0, 0.0], [2.0, 0.0]]], dtype=torch.float32),
            "coarse_logits": [
                torch.tensor([[2.0, 0.0]], dtype=torch.float32),
                torch.tensor([[0.0, 2.0]], dtype=torch.float32),
            ],
            "status_logits": torch.tensor([[[1.5, 0.0], [0.0, 1.5]]], dtype=torch.float32),
        }
        coarse_region_labels = torch.tensor([[0, 1]], dtype=torch.long)
        child_region_labels = torch.tensor([[0, 1]], dtype=torch.long)
        status_labels = torch.tensor([[0, 1]], dtype=torch.long)
        label_mask = torch.tensor([[1, 1]], dtype=torch.bool)

        base_losses = [
            region_criterion(outputs["coarse_logits"][0], coarse_region_labels[:, 0]),
            region_criterion(model.child_logits_per_horizon[0], child_region_labels[:, 0]),
            status_criterion(outputs["status_logits"][:, 0, :], status_labels[:, 0]),
            region_criterion(outputs["coarse_logits"][1], coarse_region_labels[:, 1]),
            region_criterion(model.child_logits_per_horizon[1], child_region_labels[:, 1]),
            status_criterion(outputs["status_logits"][:, 1, :], status_labels[:, 1]),
        ]
        expected = torch.stack(base_losses).mean() + 1.0

        loss = _masked_hierarchical_loss(
            model=model,
            outputs=outputs,
            coarse_region_labels=coarse_region_labels,
            child_region_labels=child_region_labels,
            status_labels=status_labels,
            label_mask=label_mask,
            region_criterion=region_criterion,
            status_criterion=status_criterion,
            horizons_minutes=[10, 30],
            horizon_consistency_weight=0.5,
        )

        self.assertAlmostEqual(float(loss.item()), float(expected.item()), places=6)


class RouterRegularizationTests(unittest.TestCase):
    def test_build_router_target_distribution_masks_inactive_experts(self):
        expert_losses = torch.tensor(
            [
                [0.2, 0.6, 1.4],
                [0.5, 0.1, 9.0],
            ],
            dtype=torch.float32,
        )
        active_mask = torch.tensor(
            [
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        )

        target = _build_router_target_distribution(
            expert_losses=expert_losses,
            active_mask=active_mask,
            temperature=0.5,
        )

        self.assertEqual(tuple(target.shape), (2, 3))
        self.assertAlmostEqual(float(target[0].sum().item()), 1.0, places=6)
        self.assertAlmostEqual(float(target[1].sum().item()), 1.0, places=6)
        self.assertEqual(float(target[1, 2].item()), 0.0)
        self.assertGreater(float(target[0, 0].item()), float(target[0, 1].item()))
        self.assertGreater(float(target[1, 1].item()), float(target[1, 0].item()))

    def test_router_load_balancing_loss_respects_active_expert_mask(self):
        router_weights = torch.tensor(
            [
                [
                    [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                    [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                    [0.5, 0.5, 0.0],
                    [0.5, 0.5, 0.0],
                ]
            ],
            dtype=torch.float32,
        )
        label_mask = torch.tensor([[1, 1, 1, 1]], dtype=torch.bool)
        active_mask = torch.tensor(
            [
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        )

        loss = _router_load_balancing_loss(
            router_weights=router_weights,
            label_mask=label_mask,
            active_mask=active_mask,
        )

        self.assertAlmostEqual(float(loss.item()), 0.0, places=6)

    def test_router_load_balancing_loss_penalizes_global_collapse(self):
        router_weights = torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                ]
            ],
            dtype=torch.float32,
        )
        label_mask = torch.tensor([[1, 1, 1, 1]], dtype=torch.bool)
        active_mask = torch.tensor(
            [
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        )

        loss = _router_load_balancing_loss(
            router_weights=router_weights,
            label_mask=label_mask,
            active_mask=active_mask,
        )

        self.assertGreater(float(loss.item()), 0.1)

    def test_expert_auxiliary_loss_ignores_global_and_inactive_local_paths(self):
        class DummyModel:
            def __init__(self):
                self.coarse_heads = [
                    nn.Linear(2, 2, bias=False),
                    nn.Linear(2, 2, bias=False),
                ]
                self.status_heads = [
                    nn.Linear(2, 2, bias=False),
                    nn.Linear(2, 2, bias=False),
                ]
                for head in [*self.coarse_heads, *self.status_heads]:
                    head.weight.data.copy_(torch.eye(2))

            def child_logits(self, features, coarse_ids, horizon_idx):
                return features

            def child_target_is_valid(self, coarse_ids, child_ids, horizon_idx):
                return torch.ones_like(child_ids, dtype=torch.bool)

        model = DummyModel()
        label_mask = torch.tensor([[1, 1]], dtype=torch.bool)
        coarse_region_labels = torch.tensor([[0, 1]], dtype=torch.long)
        child_region_labels = torch.tensor([[0, 1]], dtype=torch.long)
        status_labels = torch.tensor([[0, 1]], dtype=torch.long)
        outputs = {
            "global_expert_features": torch.tensor([[[3.0, -3.0], [-3.0, 3.0]]], dtype=torch.float32),
            "coarse_expert_features": torch.tensor([[[-2.0, 2.0], [2.0, -2.0]]], dtype=torch.float32),
            "local_expert_features": torch.tensor([[[1.5, -1.5], [9.0, 9.0]]], dtype=torch.float32),
        }
        active_mask = torch.tensor(
            [
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        )

        loss = _expert_auxiliary_loss(
            model=model,
            outputs=outputs,
            coarse_region_labels=coarse_region_labels,
            child_region_labels=child_region_labels,
            status_labels=status_labels,
            label_mask=label_mask,
            region_criterion=nn.CrossEntropyLoss(),
            status_criterion=nn.CrossEntropyLoss(),
            active_mask=active_mask,
        )

        self.assertGreater(float(loss.item()), 0.5)


if __name__ == "__main__":
    unittest.main()
