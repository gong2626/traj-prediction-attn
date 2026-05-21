import unittest

import h3
import numpy as np

from regional_forecast.evaluation.metrics import StreamingMultihorizonMetricAccumulator, _candidate_region_set_metrics


class CandidateMetricsTests(unittest.TestCase):
    def test_candidate_metrics_report_budget_coverage_size_and_area(self):
        region_tokens = [
            str(h3.latlng_to_cell(0.0, 0.0, 3)),
            str(h3.latlng_to_cell(10.0, 10.0, 3)),
            str(h3.latlng_to_cell(20.0, 20.0, 3)),
        ]
        probabilities = np.asarray(
            [
                [0.6, 0.3, 0.1, 0.0],
                [0.1, 0.2, 0.3, 0.4],
            ],
            dtype=np.float32,
        )
        targets = np.asarray([0, 2], dtype=np.int64)

        metrics = _candidate_region_set_metrics(
            region_probs=probabilities,
            region_targets=targets,
            region_tokens=region_tokens,
            top_ks=[1, 3],
            cumulative_mass=0.5,
            max_cells=2,
        )

        self.assertAlmostEqual(metrics["candidate_union_cover"], 1.0)
        self.assertAlmostEqual(metrics["candidate_union_cell_count_mean"], 1.5)
        self.assertGreater(metrics["candidate_union_area_km2_mean"], 0.0)
        self.assertAlmostEqual(metrics["candidate_cover_at1_cells"], 1.0)
        self.assertAlmostEqual(metrics["candidate_cover_at2_cells"], 1.0)
        self.assertAlmostEqual(metrics["candidate_cover_at3_cells"], 1.0)
        self.assertAlmostEqual(metrics["candidate_cover_at_budget_cells"], 1.0)
        self.assertEqual(metrics["candidate_budget_cells"], 2.0)

    def test_candidate_metrics_return_zero_for_empty_targets(self):
        metrics = _candidate_region_set_metrics(
            region_probs=np.empty((0, 4), dtype=np.float32),
            region_targets=np.empty((0,), dtype=np.int64),
            region_tokens=["83754efffffffff", "83754afffffffff", "837541fffffffff"],
            top_ks=[1, 3],
            cumulative_mass=0.8,
            max_cells=2,
        )

        self.assertEqual(metrics["candidate_union_cover"], 0.0)
        self.assertEqual(metrics["candidate_union_cell_count_mean"], 0.0)
        self.assertEqual(metrics["candidate_union_area_km2_mean"], 0.0)
        self.assertEqual(metrics["candidate_cover_at_budget_cells"], 0.0)

    def test_streaming_accumulator_exposes_candidate_set_metrics(self):
        region_tokens = [
            str(h3.latlng_to_cell(0.0, 0.0, 3)),
            str(h3.latlng_to_cell(10.0, 10.0, 3)),
            str(h3.latlng_to_cell(20.0, 20.0, 3)),
        ]
        accumulator = StreamingMultihorizonMetricAccumulator(
            top_ks=[1, 3],
            calibration_bins=5,
            horizons_minutes=[10],
            region_tokens_per_horizon=[region_tokens],
            neighbor_hit_rings=[0],
            candidate_probability_mass=[0.5],
            candidate_top_cells=[2],
        )

        accumulator.update(
            region_probabilities=[
                np.asarray(
                    [
                        [0.6, 0.3, 0.1, 0.0],
                        [0.1, 0.2, 0.3, 0.4],
                    ],
                    dtype=np.float32,
                )
            ],
            region_labels=np.asarray([[0], [2]], dtype=np.int64),
            status_probabilities=np.asarray([[[1.0, 0.0]], [[1.0, 0.0]]], dtype=np.float32),
            status_labels=np.asarray([[0], [0]], dtype=np.int64),
            label_mask=np.asarray([[1], [1]], dtype=bool),
        )

        row = accumulator.finalize()[0]
        self.assertAlmostEqual(row["candidate_union_cell_count_mean"], 1.5)
        self.assertGreater(row["candidate_union_area_km2_mean"], 0.0)
        self.assertAlmostEqual(row["candidate_cover_at_budget_cells"], 1.0)


if __name__ == "__main__":
    unittest.main()
