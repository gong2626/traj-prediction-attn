import unittest
from pathlib import Path


class RunFullCurrentSchemeScriptTests(unittest.TestCase):
    def test_script_declares_parallel_group_helper_and_dependency_phases(self):
        script_path = Path("tools/run_full_current_scheme.sh")
        content = script_path.read_text(encoding="utf-8")

        self.assertIn("run_parallel_group_if_needed()", content)
        self.assertIn("phase1_train_tuned_vs_traisformer", content)
        self.assertIn("phase2_train_gru_vs_flightbertpp", content)
        self.assertIn("phase3_train_gated_local", content)

        tuned_pos = content.index("phase1_train_tuned_vs_traisformer")
        gated_pos = content.index("phase3_train_gated_local")
        self.assertLess(tuned_pos, gated_pos)


if __name__ == "__main__":
    unittest.main()
