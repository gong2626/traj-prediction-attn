from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any, Dict, List, Sequence

from regional_forecast.config import ProjectConfig


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_registry(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"version": 1, "updated_at": _now_iso(), "models": []}
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        return {"version": 1, "updated_at": _now_iso(), "models": []}
    payload.setdefault("version", 1)
    payload.setdefault("models", [])
    return payload


def _save_registry(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload["updated_at"] = _now_iso()
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _portable_path(cfg: ProjectConfig, path: str | Path | None) -> str | None:
    if path is None:
        return None

    resolved = cfg.resolve_path(path)
    if resolved is None:
        return None

    try:
        return str(resolved.relative_to(cfg.base_dir))
    except ValueError:
        return str(resolved)


def _resolve_entry_paths(cfg: ProjectConfig, entry: Dict[str, Any]) -> Dict[str, Any]:
    resolved = dict(entry)
    for key in (
        "artifact_root",
        "dataset_dir",
        "checkpoint_path",
        "train_summary_path",
        "evaluation_summary_csv_path",
        "evaluation_summary_json_path",
        "evaluation_examples_path",
    ):
        raw_value = entry.get(key)
        if raw_value:
            normalized = cfg.resolve_path(raw_value)
            if normalized is not None:
                resolved[key] = str(normalized)
    return resolved


def slugify_name(name: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z]+", "_", str(name)).strip("_").lower()
    return slug or "unknown"


def _entry_identity_from_parts(scope: str, name: str | None, run_name: str) -> Dict[str, str]:
    resolved_name = name or "global"
    return {
        "scope": scope,
        "name": resolved_name,
        "name_slug": slugify_name(resolved_name),
        "run_name": run_name,
    }


def _entry_identity(cfg: ProjectConfig) -> Dict[str, str]:
    return _entry_identity_from_parts(cfg.dataset_scope, cfg.dataset_name, cfg.output.run_name)


def _upsert_entry(path: Path, entry: Dict[str, Any]) -> Dict[str, Any]:
    registry = _load_registry(path)
    models = registry.setdefault("models", [])

    target_idx = None
    for idx, item in enumerate(models):
        if (
            item.get("scope") == entry.get("scope")
            and item.get("name_slug") == entry.get("name_slug")
            and item.get("run_name") == entry.get("run_name")
        ):
            target_idx = idx
            break

    if target_idx is None:
        models.append(entry)
    else:
        merged = dict(models[target_idx])
        merged.update(entry)
        models[target_idx] = merged

    models.sort(
        key=lambda item: (
            str(item.get("scope", "")),
            str(item.get("name_slug", "")),
            str(item.get("run_name", "")),
        )
    )
    _save_registry(path, registry)
    return registry


def read_model_registry(path: Path) -> Dict[str, Any]:
    return _load_registry(path)


def _dataset_summary(metadata: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "dataset_scope": metadata.get("dataset_scope"),
        "dataset_name": metadata.get("dataset_name"),
        "dataset_name_slug": metadata.get("dataset_name_slug"),
        "selected_aircraft_types": metadata.get("selected_aircraft_types", []),
        "history_minutes": metadata.get("history_minutes"),
        "horizons_minutes": metadata.get("horizons_minutes"),
        "input_h3_resolution": metadata.get("input_h3_resolution"),
        "input_parent_resolution": metadata.get("input_parent_resolution"),
        "target_h3_resolutions": metadata.get("target_h3_resolutions"),
        "coarse_h3_resolutions": metadata.get("coarse_h3_resolutions"),
        "split_unit": metadata.get("split_unit"),
    }


def _metrics_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    summary: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        horizon = int(row["horizon_minutes"])
        summary[f"{horizon}m"] = {
            "samples": row.get("samples"),
            "region_top1": row.get("region_top1"),
            "region_top3": row.get("region_top3"),
            "region_top5": row.get("region_top5"),
            "region_top1_unknown_rate": row.get("region_top1_unknown_rate"),
            "candidate_union_cover": row.get("candidate_union_cover"),
            "status_accuracy": row.get("status_accuracy"),
        }
    return summary


def register_training_run(
    cfg: ProjectConfig,
    *,
    model_type: str,
    metadata: Dict[str, Any],
    checkpoint_path: Path,
    summary_path: Path,
    best_val_loss: float,
    sample_counts: Dict[str, int],
) -> Dict[str, Any]:
    entry: Dict[str, Any] = {
        **_entry_identity(cfg),
        "status": "trained",
        "model_type": model_type,
        "artifact_root": _portable_path(cfg, cfg.artifact_root),
        "dataset_dir": _portable_path(cfg, cfg.dataset_dir),
        "checkpoint_path": _portable_path(cfg, checkpoint_path),
        "train_summary_path": _portable_path(cfg, summary_path),
        "best_val_loss": float(best_val_loss),
        "sample_counts": sample_counts,
        "dataset_summary": _dataset_summary(metadata),
        "config": {
            "model_type": cfg.model.model_type,
            "training": {
                "batch_size": cfg.training.batch_size,
                "epochs": cfg.training.epochs,
                "learning_rate": cfg.training.learning_rate,
                "device": cfg.training.device,
                "amp": cfg.training.amp,
            },
        },
        "updated_at": _now_iso(),
    }
    return _upsert_entry(cfg.model_registry_path, entry)


def register_evaluation_run(
    cfg: ProjectConfig,
    *,
    metadata: Dict[str, Any],
    summary_csv_path: Path,
    summary_json_path: Path,
    examples_path: Path,
    rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    entry: Dict[str, Any] = {
        **_entry_identity(cfg),
        "status": "evaluated",
        "artifact_root": _portable_path(cfg, cfg.artifact_root),
        "dataset_dir": _portable_path(cfg, cfg.dataset_dir),
        "checkpoint_path": _portable_path(cfg, cfg.checkpoint_dir / f"{cfg.output.run_name}.pt"),
        "evaluation_summary_csv_path": _portable_path(cfg, summary_csv_path),
        "evaluation_summary_json_path": _portable_path(cfg, summary_json_path),
        "evaluation_examples_path": _portable_path(cfg, examples_path),
        "evaluation_metrics": _metrics_summary(rows),
        "dataset_summary": _dataset_summary(metadata),
        "updated_at": _now_iso(),
    }
    return _upsert_entry(cfg.model_registry_path, entry)


def register_skipped_type_fallback(
    cfg: ProjectConfig,
    *,
    aircraft_type: str,
    reason: str,
    stats: Dict[str, Any],
    fallback_scope: str = "global",
    fallback_name: str = "global",
) -> Dict[str, Any]:
    entry: Dict[str, Any] = {
        **_entry_identity_from_parts("type", aircraft_type, cfg.output.run_name),
        "status": "skipped_fallback",
        "artifact_root": _portable_path(cfg, cfg.artifact_root),
        "skip_reason": reason,
        "fallback_scope": fallback_scope,
        "fallback_name": fallback_name,
        "fallback_name_slug": slugify_name(fallback_name),
        "sample_counts": stats,
        "updated_at": _now_iso(),
    }
    return _upsert_entry(cfg.model_registry_path, entry)


def resolve_registered_model(
    cfg: ProjectConfig,
    *,
    aircraft_type: str | None = None,
    scope: str | None = None,
    name: str | None = None,
) -> tuple[Dict[str, Any], bool]:
    registry = _load_registry(cfg.model_registry_path)
    models = list(registry.get("models", []))
    run_name = cfg.output.run_name

    scoped_models = [item for item in models if item.get("run_name") == run_name]
    if not scoped_models:
        raise FileNotFoundError(f"No registered models found for run_name={run_name}")

    if scope is not None:
        requested_scope = str(scope).strip().lower()
        requested_name_slug = slugify_name(name or "global")
        for item in scoped_models:
            if (
                item.get("scope") == requested_scope
                and item.get("name_slug") == requested_name_slug
                and item.get("checkpoint_path")
                and item.get("status") in {"trained", "evaluated"}
            ):
                return _resolve_entry_paths(cfg, item), False
        raise FileNotFoundError(
            f"No registered model found for scope={requested_scope}, name={name or 'global'}, run_name={run_name}"
        )

    if aircraft_type:
        requested_name_slug = slugify_name(aircraft_type)
        for item in scoped_models:
            if (
                item.get("scope") == "type"
                and item.get("name_slug") == requested_name_slug
                and item.get("checkpoint_path")
                and item.get("status") in {"trained", "evaluated"}
            ):
                return _resolve_entry_paths(cfg, item), False

    for item in scoped_models:
        if (
            item.get("scope") == "global"
            and item.get("checkpoint_path")
            and item.get("status") in {"trained", "evaluated"}
        ):
            return _resolve_entry_paths(cfg, item), aircraft_type is not None

    raise FileNotFoundError(
        f"No suitable registered model found for aircraft_type={aircraft_type or 'n/a'}, run_name={run_name}"
    )
