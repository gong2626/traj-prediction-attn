import unittest
import tempfile
import warnings
from pathlib import Path

import torch

from regional_forecast.data.dataset import RegionalForecastDataset
from regional_forecast.data.subset_sampling import (
    choose_covering_train_indices,
    collect_required_labels,
    compute_eligible_indices,
)


class SubsetSamplingTests(unittest.TestCase):
    def test_monolithic_tensor_dataset_getitem_does_not_warn(self):
        artifact = {
            "format": "monolithic",
            "metadata": {"horizons_minutes": [10]},
            "region_seq": torch.zeros((1, 2), dtype=torch.long),
            "parent_region_seq": torch.zeros((1, 2), dtype=torch.long),
            "numeric_seq": torch.zeros((1, 2, 3), dtype=torch.float32),
            "mask_seq": torch.zeros((1, 2, 2), dtype=torch.float32),
            "context": torch.zeros((1, 4), dtype=torch.float32),
            "region_labels": torch.zeros((1, 1), dtype=torch.long),
            "coarse_region_labels": torch.zeros((1, 1), dtype=torch.long),
            "child_region_labels": torch.zeros((1, 1), dtype=torch.long),
            "status_labels": torch.zeros((1, 1), dtype=torch.long),
            "label_mask": torch.ones((1, 1), dtype=torch.bool),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pt"
            torch.save(artifact, path)
            dataset = RegionalForecastDataset(str(path))
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                dataset[0]
            self.assertEqual(captured, [])

    def test_compute_eligible_indices_requires_support_for_all_valid_horizons(self):
        region_labels = torch.tensor(
            [
                [1, 5],
                [2, 5],
                [3, 6],
                [4, 0],
            ],
            dtype=torch.long,
        )
        label_mask = torch.tensor(
            [
                [1, 1],
                [1, 1],
                [1, 1],
                [1, 0],
            ],
            dtype=torch.bool,
        )
        support_counts = [
            torch.tensor([0, 12, 3, 20, 15], dtype=torch.long),
            torch.tensor([0, 0, 0, 0, 0, 10, 11], dtype=torch.long),
        ]

        indices = compute_eligible_indices(
            region_labels=region_labels,
            label_mask=label_mask,
            support_counts=support_counts,
            min_support=10,
        )

        self.assertEqual(indices.tolist(), [0, 2, 3])

    def test_choose_covering_train_indices_covers_required_labels(self):
        train_region_labels = torch.tensor(
            [
                [1, 10],
                [2, 10],
                [1, 11],
                [3, 12],
                [4, 13],
            ],
            dtype=torch.long,
        )
        train_label_mask = torch.tensor(
            [
                [1, 1],
                [1, 1],
                [1, 1],
                [1, 1],
                [1, 0],
            ],
            dtype=torch.bool,
        )
        required_labels = [
            {1, 3},
            {10, 12},
        ]

        indices = choose_covering_train_indices(
            train_region_labels=train_region_labels,
            train_label_mask=train_label_mask,
            required_labels_per_horizon=required_labels,
            examples_per_label=1,
            seed=7,
        )

        self.assertGreaterEqual(len(indices), 2)
        selected = train_region_labels[indices]
        selected_mask = train_label_mask[indices]
        covered = collect_required_labels(
            region_labels=selected,
            label_mask=selected_mask,
            indices=list(range(len(indices))),
        )
        self.assertTrue({1, 3}.issubset(covered[0]))
        self.assertTrue({10, 12}.issubset(covered[1]))


if __name__ == "__main__":
    unittest.main()
