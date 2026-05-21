import unittest

import torch

from regional_forecast.pipeline.evaluate import _build_child_lookup_tensors, _build_sparse_fine_probs


class EvaluateSparseFineProbsTests(unittest.TestCase):
    def test_unknown_coarse_id_does_not_index_past_lookup_tables(self):
        coarse_probs = torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.float32)
        top_parent_ids = torch.tensor([[2]], dtype=torch.long)
        child_probs_top = torch.tensor([[[1.0]]], dtype=torch.float32)
        hierarchy = {
            "coarse_to_children": [
                [0],
                [1],
            ]
        }

        lookup_indices, lookup_valid = _build_child_lookup_tensors(
            [hierarchy],
            device=torch.device("cpu"),
            unknown_fine_ids=[2],
        )[0]

        self.assertEqual(tuple(lookup_indices.shape), (3, 1))
        self.assertEqual(tuple(lookup_valid.shape), (3, 1))
        self.assertEqual(int(lookup_indices[2, 0].item()), 2)
        self.assertFalse(bool(lookup_valid[2, 0].item()))

        fine_probs = _build_sparse_fine_probs(
            coarse_probs=coarse_probs,
            top_parent_ids=top_parent_ids,
            child_probs_top=child_probs_top,
            hierarchy=hierarchy,
            fine_class_count=3,
            unknown_fine_id=2,
            child_lookup_indices=lookup_indices,
            child_lookup_valid=lookup_valid,
        )

        self.assertEqual(tuple(fine_probs.shape), (1, 3))
        self.assertAlmostEqual(float(fine_probs[0, 2].item()), 1.0, places=6)
        self.assertAlmostEqual(float(fine_probs[0, 0].item()), 0.0, places=6)
        self.assertAlmostEqual(float(fine_probs[0, 1].item()), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
