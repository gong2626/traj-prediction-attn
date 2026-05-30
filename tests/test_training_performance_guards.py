import unittest
from unittest import mock

import torch
import torch.nn as nn

import regional_forecast.pipeline.train as train_module
from regional_forecast.config import (
    AOIConfig,
    ColumnConfig,
    CatalogConfig,
    DatasetConfig,
    EvaluationConfig,
    GridConfig,
    ModelConfig,
    OutputConfig,
    ProjectConfig,
    TrainingConfig,
)
from regional_forecast.data.dataset import ChunkAwareBatchSampler


class ChunkAwareBatchSamplerPerformanceTests(unittest.TestCase):
    def test_shuffle_path_avoids_python_shuffle_over_full_chunk_indices(self):
        sampler = ChunkAwareBatchSampler(
            chunk_offsets=[100],
            chunk_lengths=[16],
            batch_size=4,
            shuffle=True,
            drop_last=False,
        )

        observed_shuffle_lengths = []
        original_shuffle = train_module.random.shuffle

        def guarded_shuffle(values):
            observed_shuffle_lengths.append(len(values))
            if len(values) > 1:
                raise AssertionError("per-sample chunk indices should not be shuffled in Python")
            original_shuffle(values)

        with mock.patch("regional_forecast.data.dataset.random.shuffle", side_effect=guarded_shuffle):
            batches = list(iter(sampler))

        flattened = [index for batch in batches for index in batch]
        self.assertEqual(sorted(flattened), list(range(100, 116)))
        self.assertEqual(observed_shuffle_lengths, [1])


class LossAccumulationPerformanceTests(unittest.TestCase):
    def test_weighted_loss_accumulator_exists_and_detaches_graph(self):
        self.assertTrue(hasattr(train_module, "_accumulate_weighted_loss"))
        accumulate = getattr(train_module, "_accumulate_weighted_loss")

        leaf = torch.tensor(2.0, requires_grad=True)
        total = accumulate(None, leaf * 3.0, 4)
        self.assertFalse(total.requires_grad)
        self.assertAlmostEqual(float(total.item()), 24.0, places=6)

        total = accumulate(total, leaf * 1.5, 2)
        self.assertFalse(total.requires_grad)
        self.assertAlmostEqual(float(total.item()), 30.0, places=6)


class DistillationPerformanceGuardTests(unittest.TestCase):
    def _make_cfg(self, *, teacher_weight=0.08, status_weight=0.01, teacher_interval=2):
        return ProjectConfig(
            columns=ColumnConfig(),
            aoi=AOIConfig(enabled=False, name="global", mode="global"),
            grid=GridConfig(),
            dataset=DatasetConfig(),
            model=ModelConfig(model_type="horizon_router_transformer_h3"),
            training=TrainingConfig(
                teacher_distillation_weight=teacher_weight,
                teacher_status_distillation_weight=status_weight,
                teacher_distillation_interval_steps=teacher_interval,
            ),
            evaluation=EvaluationConfig(),
            catalog=CatalogConfig(),
            output=OutputConfig(root_dir="artifacts", run_name="perf_guard"),
        )

    def _make_batch(self):
        return {
            "region_seq": torch.zeros((2, 3), dtype=torch.long),
            "parent_region_seq": torch.zeros((2, 3), dtype=torch.long),
            "numeric_seq": torch.zeros((2, 3, 4), dtype=torch.float32),
            "mask_seq": torch.zeros((2, 3, 2), dtype=torch.float32),
            "context": torch.zeros((2, 5), dtype=torch.float32),
            "region_labels": torch.zeros((2, 2), dtype=torch.long),
            "coarse_region_labels": torch.zeros((2, 2), dtype=torch.long),
            "child_region_labels": torch.zeros((2, 2), dtype=torch.long),
            "status_labels": torch.zeros((2, 2), dtype=torch.long),
            "label_mask": torch.ones((2, 2), dtype=torch.bool),
        }

    def test_should_compute_teacher_distillation_skips_validation(self):
        cfg = self._make_cfg()

        self.assertFalse(train_module._should_compute_teacher_distillation(cfg, is_training=False, global_step=1))
        self.assertTrue(train_module._should_compute_teacher_distillation(cfg, is_training=True, global_step=1))

    def test_should_compute_teacher_distillation_respects_step_interval(self):
        cfg = self._make_cfg(teacher_interval=3)

        self.assertTrue(train_module._should_compute_teacher_distillation(cfg, is_training=True, global_step=1))
        self.assertFalse(train_module._should_compute_teacher_distillation(cfg, is_training=True, global_step=2))
        self.assertFalse(train_module._should_compute_teacher_distillation(cfg, is_training=True, global_step=3))
        self.assertTrue(train_module._should_compute_teacher_distillation(cfg, is_training=True, global_step=4))

    def test_compute_loss_skips_teacher_and_router_regularization_for_validation(self):
        cfg = self._make_cfg()
        batch = self._make_batch()
        region_criterion = nn.CrossEntropyLoss()
        status_criterion = nn.CrossEntropyLoss()

        class DummyModel:
            def __call__(self, region_seq, parent_region_seq, numeric_seq, mask_seq, context):
                return {
                    "features": torch.zeros((2, 8), dtype=torch.float32),
                    "decoder_features": torch.zeros((2, 2, 8), dtype=torch.float32),
                    "coarse_logits": [
                        torch.zeros((2, 3), dtype=torch.float32),
                        torch.zeros((2, 3), dtype=torch.float32),
                    ],
                    "status_logits": torch.zeros((2, 2, 2), dtype=torch.float32),
                }

            def child_logits(self, features, coarse_ids, horizon_idx):
                return torch.zeros((features.shape[0], 3), dtype=torch.float32)

            def child_target_is_valid(self, coarse_ids, child_ids, horizon_idx):
                return torch.ones_like(child_ids, dtype=torch.bool)

        with (
            mock.patch("regional_forecast.pipeline.train._teacher_distillation_loss", return_value=torch.tensor(7.0)) as teacher_loss,
            mock.patch("regional_forecast.pipeline.train._router_regularization_loss", return_value=torch.tensor(5.0)) as router_loss,
        ):
            _, loss = train_module._compute_loss_for_batch(
                cfg=cfg,
                model_type="horizon_router_transformer_h3",
                model=DummyModel(),
                batch=batch,
                region_criterion=region_criterion,
                status_criterion=status_criterion,
                is_training=False,
                global_step=10,
            )

        self.assertTrue(torch.isfinite(loss))
        teacher_loss.assert_not_called()
        router_loss.assert_not_called()


if __name__ == "__main__":
    unittest.main()
