import unittest
from pathlib import Path


class RunFullRouterRegScriptTests(unittest.TestCase):
    def test_script_and_full_config_exist_with_expected_commands(self):
        script_path = Path("tools/run_full_routerreg.sh")
        config_path = Path("configs/region_forecast_adrr_full_horizon_router_distill_routerreg.yaml")

        self.assertTrue(script_path.exists(), f"missing script: {script_path}")
        self.assertTrue(config_path.exists(), f"missing config: {config_path}")

        script_content = script_path.read_text(encoding="utf-8")
        config_content = config_path.read_text(encoding="utf-8")

        self.assertIn('CONFIG_PATH="configs/region_forecast_adrr_full_horizon_router_distill_routerreg.yaml"', script_content)
        self.assertIn('"${PYTHON_BIN}" training.py --config "${CONFIG_PATH}"', script_content)
        self.assertIn('"${PYTHON_BIN}" evaluation.py --config "${CONFIG_PATH}"', script_content)
        self.assertIn("global_multiscale_transformer_h3_tuned.pt", script_content)
        self.assertIn("global_horizon_expert_blend_full_v1_report.json", script_content)
        self.assertIn('"${PYTHON_BIN}" build_dataset.py --config "${CONFIG_PATH}"', script_content)

        self.assertIn('model_type: "horizon_router_transformer_h3"', config_content)
        self.assertIn('init_from_checkpoint: "artifacts/checkpoints/global_multiscale_transformer_h3_tuned.pt"', config_content)
        self.assertIn('teacher_blend_report_path: "artifacts/evaluations/global_horizon_expert_blend_full_v1_report.json"', config_content)
        self.assertIn('run_name: "global_horizon_router_transformer_h3_distill_routerreg"', config_content)

    def test_force_mode_restarts_training_from_scratch(self):
        script_path = Path("tools/run_full_routerreg.sh")
        script_content = script_path.read_text(encoding="utf-8")

        self.assertIn('if [[ "${FORCE}" == "1" ]]; then', script_content)
        self.assertIn('train_args+=(--restart)', script_content)


if __name__ == "__main__":
    unittest.main()
