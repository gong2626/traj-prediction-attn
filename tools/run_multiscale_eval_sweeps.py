from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent.parent


DEVSWEEP_CASES = [
    {
        "name": "base",
        "parent_top_k": [64, 48, 48, 12],
        "coarse_temp": [1.0, 1.0, 1.0, 1.0],
        "child_temp": [1.0, 1.0, 1.0, 1.0],
    },
    {
        "name": "parentk_short_up",
        "parent_top_k": [96, 72, 48, 12],
        "coarse_temp": [1.0, 1.0, 1.0, 1.0],
        "child_temp": [1.0, 1.0, 1.0, 1.0],
    },
    {
        "name": "parentk_all_up",
        "parent_top_k": [96, 72, 64, 16],
        "coarse_temp": [1.0, 1.0, 1.0, 1.0],
        "child_temp": [1.0, 1.0, 1.0, 1.0],
    },
    {
        "name": "temp_sharp_short",
        "parent_top_k": [64, 48, 48, 12],
        "coarse_temp": [0.95, 0.95, 1.0, 1.0],
        "child_temp": [0.95, 0.95, 1.0, 1.0],
    },
    {
        "name": "temp_smooth_short",
        "parent_top_k": [64, 48, 48, 12],
        "coarse_temp": [1.05, 1.05, 1.0, 1.0],
        "child_temp": [1.05, 1.05, 1.0, 1.0],
    },
    {
        "name": "parentk_up_temp_sharp",
        "parent_top_k": [96, 72, 48, 12],
        "coarse_temp": [0.95, 0.95, 1.0, 1.0],
        "child_temp": [0.95, 0.95, 1.0, 1.0],
    },
]


FULLSWEEP_CASES = [
    {
        "name": "base",
        "parent_top_k": [64, 48, 48, 12],
        "coarse_temp": [1.0, 1.0, 1.0, 1.0],
        "child_temp": [1.0, 1.0, 1.0, 1.0],
    },
    {
        "name": "parentk_short_up",
        "parent_top_k": [96, 72, 48, 12],
        "coarse_temp": [1.0, 1.0, 1.0, 1.0],
        "child_temp": [1.0, 1.0, 1.0, 1.0],
    },
    {
        "name": "temp_sharp_short",
        "parent_top_k": [64, 48, 48, 12],
        "coarse_temp": [0.95, 0.95, 1.0, 1.0],
        "child_temp": [0.95, 0.95, 1.0, 1.0],
    },
    {
        "name": "parentk_up_temp_sharp",
        "parent_top_k": [96, 72, 48, 12],
        "coarse_temp": [0.95, 0.95, 1.0, 1.0],
        "child_temp": [0.95, 0.95, 1.0, 1.0],
    },
]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _prepare_case_config(base_cfg_path: Path, case: dict[str, Any], output_dir: Path) -> tuple[Path, str]:
    cfg = _load_yaml(base_cfg_path)
    cfg.setdefault("evaluation", {})
    cfg.setdefault("output", {})

    cfg["evaluation"]["hierarchical_parent_top_k"] = list(case["parent_top_k"])
    cfg["evaluation"]["hierarchical_coarse_temperature"] = list(case["coarse_temp"])
    cfg["evaluation"]["hierarchical_child_temperature"] = list(case["child_temp"])
    run_name = f"{cfg['output']['run_name']}_{case['name']}"
    cfg["output"]["run_name"] = run_name

    output_dir.mkdir(parents=True, exist_ok=True)
    case_cfg_path = output_dir / f"{run_name}.yaml"
    with case_cfg_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False, allow_unicode=True)
    return case_cfg_path, run_name


def _run_case(case_cfg_path: Path, python_bin: str) -> None:
    cmd = [python_bin, "evaluation.py", "--config", str(case_cfg_path)]
    subprocess.run(cmd, cwd=ROOT, check=True)


def _collect_metrics(summary_csv: Path) -> dict[str, float]:
    with summary_csv.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No rows found in summary CSV: {summary_csv}")
    macro = {
        "top1": sum(float(r["region_top1"]) for r in rows) / len(rows),
        "top3": sum(float(r["region_top3"]) for r in rows) / len(rows),
        "top5": sum(float(r["region_top5"]) for r in rows) / len(rows),
        "nll": sum(float(r["region_nll"]) for r in rows) / len(rows),
        "brier": sum(float(r["region_brier"]) for r in rows) / len(rows),
        "ece": sum(float(r["region_ece"]) for r in rows) / len(rows),
        "union_cover": sum(float(r["candidate_union_cover"]) for r in rows) / len(rows),
        "union_cells": sum(float(r["candidate_union_cell_count_mean"]) for r in rows) / len(rows),
        "union_area": sum(float(r["candidate_union_area_km2_mean"]) for r in rows) / len(rows),
    }
    return macro


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evaluation-only sweeps for multiscale hierarchical decoding.")
    parser.add_argument("--mode", choices=["devsplit", "full"], required=True)
    parser.add_argument("--python", default="python")
    args = parser.parse_args()

    if args.mode == "devsplit":
        base_cfg = ROOT / "configs/region_forecast_adrr_devsplit_multiscale_eval_parentk_temp_base.yaml"
        output_root = ROOT / "artifacts/research_devsplit/evaluations/sweeps"
        cases = DEVSWEEP_CASES
        artifact_root = ROOT / "artifacts/research_devsplit/evaluations"
    else:
        base_cfg = ROOT / "configs/region_forecast_adrr_full_multiscale_eval_parentk_temp_base.yaml"
        output_root = ROOT / "artifacts/evaluations/sweeps"
        cases = FULLSWEEP_CASES
        artifact_root = ROOT / "artifacts/evaluations"

    results: list[dict[str, Any]] = []
    case_cfg_paths: list[str] = []
    for case in cases:
        case_cfg_path, run_name = _prepare_case_config(base_cfg, case, output_root)
        case_cfg_paths.append(str(case_cfg_path))
        summary_csv = artifact_root / f"{run_name}_summary.csv"
        if summary_csv.exists():
            print(f"Skipping existing case: {run_name}")
        else:
            _run_case(case_cfg_path, args.python)
        metrics = _collect_metrics(summary_csv)
        results.append(
            {
                "case": case["name"],
                "run_name": run_name,
                "parent_top_k": case["parent_top_k"],
                "coarse_temp": case["coarse_temp"],
                "child_temp": case["child_temp"],
                **metrics,
            }
        )

    out_json = output_root / f"{args.mode}_multiscale_eval_sweep_results.json"
    out_md = output_root / f"{args.mode}_multiscale_eval_sweep_results.md"
    with out_json.open("w", encoding="utf-8") as handle:
        json.dump({"case_configs": case_cfg_paths, "results": results}, handle, ensure_ascii=False, indent=2)

    lines = [
        "# Multiscale Eval Sweep Results",
        "",
        "| case | top1 | top3 | top5 | nll | brier | ece | union_cover | union_cells | union_area |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in results:
        lines.append(
            f"| {item['case']} | {item['top1']:.4f} | {item['top3']:.4f} | {item['top5']:.4f} | "
            f"{item['nll']:.4f} | {item['brier']:.4f} | {item['ece']:.4f} | "
            f"{item['union_cover']:.4f} | {item['union_cells']:.4f} | {item['union_area']:.0f} |"
        )
    with out_md.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    print(f"Wrote sweep summary: {out_json}")
    print(f"Wrote markdown summary: {out_md}")


if __name__ == "__main__":
    main()
