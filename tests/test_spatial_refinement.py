import unittest

import h3
import numpy as np

from regional_forecast.evaluation.spatial_refinement import (
    H3NeighborProbabilityRefiner,
    smooth_h3_probabilities,
)


class SpatialRefinementTests(unittest.TestCase):
    def test_smoothing_diffuses_mass_to_known_h3_neighbors(self):
        center = str(h3.latlng_to_cell(0.0, 0.0, 3))
        neighbor = next(cell for cell in h3.grid_disk(center, 1) if str(cell) != center)
        far = str(h3.latlng_to_cell(60.0, 60.0, 3))
        region_tokens = [center, str(neighbor), far]
        probabilities = np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)

        smoothed = smooth_h3_probabilities(
            probabilities,
            region_tokens,
            alpha=0.2,
            ring=1,
        )

        self.assertAlmostEqual(float(smoothed[0, 0]), 0.9, places=6)
        self.assertAlmostEqual(float(smoothed[0, 1]), 0.1, places=6)
        self.assertAlmostEqual(float(smoothed[0, 2]), 0.0, places=6)
        self.assertAlmostEqual(float(smoothed[0, 3]), 0.0, places=6)
        self.assertAlmostEqual(float(smoothed.sum()), 1.0, places=6)

    def test_smoothing_preserves_unknown_mass(self):
        center = str(h3.latlng_to_cell(0.0, 0.0, 3))
        neighbor = next(cell for cell in h3.grid_disk(center, 1) if str(cell) != center)
        region_tokens = [center, str(neighbor)]
        probabilities = np.asarray([[0.6, 0.0, 0.4]], dtype=np.float32)

        smoothed = smooth_h3_probabilities(
            probabilities,
            region_tokens,
            alpha=0.5,
            ring=1,
            preserve_unknown=True,
        )

        self.assertAlmostEqual(float(smoothed[0, 0]), 0.45, places=6)
        self.assertAlmostEqual(float(smoothed[0, 1]), 0.15, places=6)
        self.assertAlmostEqual(float(smoothed[0, 2]), 0.4, places=6)
        self.assertAlmostEqual(float(smoothed.sum()), 1.0, places=6)

    def test_refiner_is_noop_when_disabled(self):
        probabilities = [np.asarray([[0.7, 0.3]], dtype=np.float32)]
        refiner = H3NeighborProbabilityRefiner(
            region_tokens_per_horizon=[["83754efffffffff"]],
            enabled=False,
            alpha_per_horizon=[0.5],
            ring_per_horizon=[1],
        )

        refined = refiner.refine_probability_arrays(probabilities)

        self.assertTrue(np.allclose(refined[0], probabilities[0]))
        self.assertIsNot(refined[0], probabilities[0])


if __name__ == "__main__":
    unittest.main()
