import unittest
from pathlib import Path


class RouterRegRollbackTests(unittest.TestCase):
    def test_full_config_and_script_do_not_use_offline_teacher_cache(self):
        script_content = Path("tools/run_full_routerreg.sh").read_text(encoding="utf-8")
        config_content = Path(
            "configs/region_forecast_adrr_full_horizon_router_distill_routerreg.yaml"
        ).read_text(encoding="utf-8")

        self.assertNotIn("--build-teacher-cache-only", script_content)
        self.assertNotIn("TEACHER_CACHE=", script_content)
        self.assertNotIn("teacher cache manifest", script_content)
        self.assertNotIn("teacher_target_cache_path:", config_content)


if __name__ == "__main__":
    unittest.main()
