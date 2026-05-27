import unittest
from unittest import mock

import torch

import regional_forecast.pipeline.train as train_module
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


if __name__ == "__main__":
    unittest.main()
