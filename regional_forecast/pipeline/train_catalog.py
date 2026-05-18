from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from regional_forecast.config import ProjectConfig, load_config, load_config_dict
from regional_forecast.data.preprocess import build_dataset_artifacts, summarize_aircraft_type_catalog
from regional_forecast.pipeline.evaluate import run_evaluation
from regional_forecast.pipeline.train import run_training
from regional_forecast.registry import register_skipped_type_fallback


def _clone_cfg(cfg: ProjectConfig) -> ProjectConfig:
    return load_config_dict(cfg.to_dict(), source_path=getattr(cfg, "_config_path", None))


def _minimum_aircraft_for_non_empty_splits(cfg: ProjectConfig) -> int:
    for count in range(1, 10000):
        train_count = int(count * cfg.dataset.train_ratio)
        val_count = int(count * cfg.dataset.val_ratio)
        test_count = count - train_count - val_count
        if train_count > 0 and val_count > 0 and test_count > 0:
            return count
    return 1


def _save_summary(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _resolve_selected_types(
    available: Sequence[Dict[str, Any]],
    requested: Optional[Sequence[str]],
    max_types: Optional[int],
) -> List[Dict[str, Any]]:
    if requested:
        requested_keys = {str(item).strip().casefold() for item in requested if str(item).strip()}
        filtered = [
            item for item in available if str(item["aircraft_type"]).strip().casefold() in requested_keys
        ]
    else:
        filtered = list(available)

    if max_types is not None:
        filtered = filtered[: max(0, int(max_types))]
    return filtered


def _global_cfg(cfg: ProjectConfig) -> ProjectConfig:
    cloned = _clone_cfg(cfg)
    cloned.dataset.scope = "global"
    cloned.dataset.name = None
    return cloned


def _type_cfg(cfg: ProjectConfig, aircraft_type: str) -> ProjectConfig:
    cloned = _clone_cfg(cfg)
    cloned.dataset.scope = "type"
    cloned.dataset.name = aircraft_type
    return cloned


def _build_reason(stats: Dict[str, Any], *, min_rows: int, min_aircraft: int) -> Optional[str]:
    if int(stats["rows"]) < int(min_rows):
        return f"rows_below_threshold({stats['rows']}<{min_rows})"
    if int(stats["aircraft_count"]) < int(min_aircraft):
        return f"aircraft_below_threshold({stats['aircraft_count']}<{min_aircraft})"
    return None


def _checkpoint_path(cfg: ProjectConfig) -> Path:
    return cfg.checkpoint_dir / f"{cfg.output.run_name}.pt"


def _run_single_training_bundle(
    cfg: ProjectConfig,
    *,
    evaluate_after_training: bool,
) -> Dict[str, Any]:
    counts = build_dataset_artifacts(cfg)
    split_issue = []
    if int(counts.get("train", 0)) <= 0:
        split_issue.append("train=0")
    if int(counts.get("val", 0)) <= 0:
        split_issue.append("val=0")
    if evaluate_after_training and int(counts.get("test", 0)) <= 0:
        split_issue.append("test=0")

    result: Dict[str, Any] = {
        "dataset_dir": str(cfg.dataset_dir),
        "checkpoint_dir": str(cfg.checkpoint_dir),
        "evaluation_dir": str(cfg.evaluation_dir),
        "dataset_counts": counts,
    }
    if split_issue:
        result["status"] = "skipped_fallback"
        result["reason"] = "insufficient_split_samples(" + ",".join(split_issue) + ")"
        return result

    run_training(cfg)
    result["status"] = "trained"
    result["checkpoint_path"] = str(_checkpoint_path(cfg))

    if evaluate_after_training:
        run_evaluation(cfg)
        result["evaluated"] = True
        result["evaluation_summary_path"] = str(cfg.evaluation_dir / f"{cfg.output.run_name}_summary.json")
    else:
        result["evaluated"] = False

    return result


def run_catalog_training(
    cfg: ProjectConfig,
    *,
    include_global: Optional[bool] = None,
    evaluate_after_training: Optional[bool] = None,
    type_names: Optional[Sequence[str]] = None,
    max_types: Optional[int] = None,
    skip_existing: Optional[bool] = None,
) -> Dict[str, Any]:
    include_global = cfg.catalog.include_global if include_global is None else bool(include_global)
    evaluate_after_training = (
        cfg.catalog.evaluate_after_training
        if evaluate_after_training is None
        else bool(evaluate_after_training)
    )
    max_types = cfg.catalog.max_types if max_types is None else max_types
    skip_existing = cfg.catalog.skip_existing if skip_existing is None else bool(skip_existing)
    min_rows = int(cfg.catalog.min_rows_per_type)
    min_aircraft = (
        int(cfg.catalog.min_aircraft_per_type)
        if cfg.catalog.min_aircraft_per_type is not None
        else _minimum_aircraft_for_non_empty_splits(cfg)
    )

    summary: Dict[str, Any] = {
        "run_name": cfg.output.run_name,
        "include_global": include_global,
        "evaluate_after_training": evaluate_after_training,
        "min_rows_per_type": min_rows,
        "min_aircraft_per_type": min_aircraft,
        "requested_types": list(type_names or []),
        "global": None,
        "types": [],
    }

    if include_global:
        print("Preparing global model...")
        global_cfg = _global_cfg(cfg)
        try:
            global_result = _run_single_training_bundle(
                global_cfg,
                evaluate_after_training=evaluate_after_training,
            )
        except Exception as exc:  # pragma: no cover - runtime failure path
            global_result = {"status": "failed", "error": str(exc)}
        summary["global"] = global_result

    catalog = summarize_aircraft_type_catalog(cfg)
    selected_types = _resolve_selected_types(catalog, type_names, max_types)

    for stats in selected_types:
        aircraft_type = str(stats["aircraft_type"])
        print(f"Processing aircraft type: {aircraft_type}")
        reason = _build_reason(stats, min_rows=min_rows, min_aircraft=min_aircraft)
        type_result: Dict[str, Any] = {
            "aircraft_type": aircraft_type,
            **stats,
        }

        if reason is not None:
            register_skipped_type_fallback(
                cfg,
                aircraft_type=aircraft_type,
                reason=reason,
                stats=stats,
            )
            type_result["status"] = "skipped_fallback"
            type_result["reason"] = reason
            summary["types"].append(type_result)
            print(f"  skipped -> fallback to global ({reason})")
            continue

        type_cfg = _type_cfg(cfg, aircraft_type)
        checkpoint_path = _checkpoint_path(type_cfg)
        if skip_existing and checkpoint_path.exists():
            type_result["status"] = "skipped_existing"
            type_result["checkpoint_path"] = str(checkpoint_path)
            summary["types"].append(type_result)
            print("  skipped -> existing checkpoint")
            continue

        try:
            result = _run_single_training_bundle(
                type_cfg,
                evaluate_after_training=evaluate_after_training,
            )
        except Exception as exc:  # pragma: no cover - runtime failure path
            result = {"status": "failed", "error": str(exc)}

        if result.get("status") == "skipped_fallback":
            register_skipped_type_fallback(
                cfg,
                aircraft_type=aircraft_type,
                reason=str(result.get("reason", "insufficient_split_samples")),
                stats={**stats, **result.get("dataset_counts", {})},
            )
        type_result.update(result)
        summary["types"].append(type_result)
        print(f"  {type_result['status']}")

    summary_path = cfg.catalog_dir / f"{cfg.output.run_name}_catalog_summary.json"
    _save_summary(summary_path, summary)
    summary["summary_path"] = str(summary_path)
    print(f"Catalog summary saved: {summary_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-train per-aircraft-type forecasting models.")
    parser.add_argument("--config", type=str, default="config/region_forecast.yaml")
    parser.add_argument("--type", dest="type_names", action="append", default=None, help="Train only the specified aircraft type. Can be repeated.")
    parser.add_argument("--max-types", type=int, default=None, help="Maximum number of aircraft types to train.")
    parser.add_argument("--skip-global", action="store_true", help="Skip global model build/train/evaluate.")
    parser.add_argument("--skip-evaluation", action="store_true", help="Train models without running evaluation.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip aircraft types that already have a checkpoint.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_catalog_training(
        cfg,
        include_global=not args.skip_global,
        evaluate_after_training=not args.skip_evaluation,
        type_names=args.type_names,
        max_types=args.max_types,
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    main()
