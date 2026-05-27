from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
import sys
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from regional_forecast.config import ProjectConfig, load_config
from regional_forecast.data.dataset import RegionalForecastDataset
from regional_forecast.evaluation.metrics import compute_multihorizon_metrics
from regional_forecast.pipeline.evaluate import _load_model, _predict_probs


DEFAULT_SCORE_METRICS: List[Tuple[str, bool]] = [
    ("region_top1", True),
    ("region_top3", True),
    ("region_top5", True),
    ("region_nll", False),
    ("region_brier", False),
    ("region_ece", False),
    ("candidate_cover_at_budget_cells", True),
]


def _parse_expert_specs(values: Sequence[str]) -> Dict[str, Path]:
    experts: Dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expert spec must be name=config_path, got: {value}")
        name, config_path = value.split("=", 1)
        name = name.strip()
        config_path = config_path.strip()
        if not name:
            raise ValueError(f"Expert name is empty in spec: {value}")
        experts[name] = Path(config_path)
    return experts


def _prepare_cfg(path: Path, *, batch_size: int, num_workers: int) -> ProjectConfig:
    cfg = load_config(str(path))
    cfg.training.device = "cpu"
    cfg.training.batch_size = int(batch_size)
    cfg.training.num_workers = int(num_workers)
    cfg.training.pin_memory = False
    cfg.training.persistent_workers = False
    return cfg


def _prediction_cache_path(cache_dir: Path, expert_name: str, split: str) -> Path:
    return cache_dir / f"{expert_name}_{split}_predictions.pt"


def _dataset_cache_path(cache_dir: Path, split: str) -> Path:
    return cache_dir / f"{split}_dataset_meta.pt"


def _load_or_predict_split(
    cfg: ProjectConfig,
    *,
    expert_name: str,
    split: str,
    cache_dir: Path,
) -> tuple[RegionalForecastDataset, Dict[str, object]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset = RegionalForecastDataset(str(cfg.dataset_dir / f"{split}.pt"))
    pred_cache = _prediction_cache_path(cache_dir, expert_name, split)
    data_cache = _dataset_cache_path(cache_dir, split)

    if pred_cache.exists() and data_cache.exists():
        cached_pred = torch.load(pred_cache, map_location="cpu", weights_only=False)
        cached_meta = torch.load(data_cache, map_location="cpu", weights_only=False)
        if cached_meta.get("path") == str(dataset.path):
            return dataset, cached_pred

    model_type, model = _load_model(cfg, dataset)
    predictions = _predict_probs(cfg, dataset, model_type, model)

    serializable_predictions = {
        "region_probs": [tensor.cpu() for tensor in predictions["region_probs"]],
        "region_true_probs": predictions["region_true_probs"],
        "status_probs": predictions["status_probs"].cpu(),
        "coarse_probs": None
        if predictions.get("coarse_probs") is None
        else [tensor.cpu() for tensor in predictions["coarse_probs"]],
    }
    torch.save(serializable_predictions, pred_cache)
    torch.save({"path": str(dataset.path)}, data_cache)
    return dataset, serializable_predictions


def _region_tokens_per_horizon(dataset: RegionalForecastDataset) -> List[List[str]]:
    return [list(meta["region_tokens"]) for meta in dataset.metadata["target_regionizers"]]


def _compute_rows(
    dataset: RegionalForecastDataset,
    *,
    region_probabilities: Sequence[np.ndarray],
    status_probabilities: np.ndarray,
    region_true_probabilities: Sequence[np.ndarray] | None,
    candidate_probability_mass: Sequence[float],
    candidate_top_cells: Sequence[int],
) -> List[Dict[str, float]]:
    return compute_multihorizon_metrics(
        region_probabilities=region_probabilities,
        region_labels=dataset.tensor("region_labels").numpy(),
        region_true_probabilities=region_true_probabilities,
        status_probabilities=status_probabilities,
        status_labels=dataset.tensor("status_labels").numpy(),
        label_mask=dataset.tensor("label_mask").numpy(),
        top_ks=[1, 3, 5],
        calibration_bins=10,
        horizons_minutes=dataset.metadata["horizons_minutes"],
        region_tokens_per_horizon=_region_tokens_per_horizon(dataset),
        neighbor_hit_rings=[1, 1, 1, 1],
        candidate_probability_mass=candidate_probability_mass,
        candidate_top_cells=candidate_top_cells,
    )


def _simplex_weights(num_experts: int, step: float) -> List[Tuple[float, ...]]:
    if num_experts < 2:
        raise ValueError("Need at least two experts for blending.")
    if step <= 0.0 or step > 1.0:
        raise ValueError("weight step must be in (0, 1].")
    denom = round(1.0 / step)
    if abs(denom * step - 1.0) > 1.0e-8:
        raise ValueError("weight step must divide 1.0 exactly, e.g. 0.1 or 0.05.")

    weights: List[Tuple[float, ...]] = []

    def _recurse(prefix: List[int], remaining: int, slots: int) -> None:
        if slots == 1:
            combo = prefix + [remaining]
            weights.append(tuple(value / denom for value in combo))
            return
        for value in range(remaining + 1):
            _recurse(prefix + [value], remaining - value, slots - 1)

    _recurse([], int(denom), num_experts)
    return weights


def _blend_region_probs(predictions_by_expert: Dict[str, Dict[str, object]], expert_names: Sequence[str], weights: Sequence[float]) -> List[np.ndarray]:
    num_horizons = len(predictions_by_expert[expert_names[0]]["region_probs"])
    blended: List[np.ndarray] = []
    for horizon_idx in range(num_horizons):
        arr = sum(
            float(weights[expert_idx]) * predictions_by_expert[name]["region_probs"][horizon_idx].numpy()
            for expert_idx, name in enumerate(expert_names)
        )
        arr = np.clip(arr, 1.0e-12, None)
        arr = arr / arr.sum(axis=1, keepdims=True)
        blended.append(arr)
    return blended


def _blend_status_probs(predictions_by_expert: Dict[str, Dict[str, object]], expert_names: Sequence[str], weights: Sequence[float]) -> np.ndarray:
    arr = sum(
        float(weights[expert_idx]) * predictions_by_expert[name]["status_probs"].numpy()
        for expert_idx, name in enumerate(expert_names)
    )
    arr = np.clip(arr, 1.0e-12, None)
    return arr / arr.sum(axis=-1, keepdims=True)


def _temperature_scale_probabilities(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")
    if abs(temperature - 1.0) < 1.0e-12:
        return probabilities
    scaled = np.power(np.clip(probabilities, 1.0e-12, 1.0), 1.0 / temperature)
    return scaled / scaled.sum(axis=1, keepdims=True)


def _score_rows(
    rows: Sequence[Dict[str, float]],
    metric_specs: Sequence[Tuple[str, bool]],
    horizon_idx: int,
) -> np.ndarray:
    values: List[np.ndarray] = []
    for metric_name, higher_is_better in metric_specs:
        metric_values = np.asarray([row[horizon_idx][metric_name] for row in rows], dtype=np.float64)
        low = float(metric_values.min())
        high = float(metric_values.max())
        if abs(high - low) < 1.0e-12:
            normalized = np.ones_like(metric_values)
        else:
            normalized = (metric_values - low) / (high - low)
        if not higher_is_better:
            normalized = 1.0 - normalized
        values.append(normalized)
    return np.sum(np.stack(values, axis=0), axis=0)


def _search_best_weights(
    dataset: RegionalForecastDataset,
    *,
    predictions_by_expert: Dict[str, Dict[str, object]],
    expert_names: Sequence[str],
    weight_candidates: Sequence[Tuple[float, ...]],
    metric_specs: Sequence[Tuple[str, bool]],
    candidate_probability_mass: Sequence[float],
    candidate_top_cells: Sequence[int],
) -> tuple[List[Tuple[float, ...]], List[Dict[str, float]]]:
    blended_rows: List[List[Dict[str, float]]] = []
    for weights in weight_candidates:
        region_probs = _blend_region_probs(predictions_by_expert, expert_names, weights)
        status_probs = _blend_status_probs(predictions_by_expert, expert_names, weights)
        blended_rows.append(
            _compute_rows(
                dataset,
                region_probabilities=region_probs,
                status_probabilities=status_probs,
                region_true_probabilities=None,
                candidate_probability_mass=candidate_probability_mass,
                candidate_top_cells=candidate_top_cells,
            )
        )

    best_weights: List[Tuple[float, ...]] = []
    best_rows: List[Dict[str, float]] = []
    for horizon_idx in range(len(dataset.metadata["horizons_minutes"])):
        scores = _score_rows(blended_rows, metric_specs, horizon_idx)
        best_idx = int(scores.argmax())
        best_weights.append(tuple(weight_candidates[best_idx]))
        best_rows.append(dict(blended_rows[best_idx][horizon_idx]))
    return best_weights, best_rows


def _apply_per_horizon_weights(
    predictions_by_expert: Dict[str, Dict[str, object]],
    *,
    expert_names: Sequence[str],
    per_horizon_weights: Sequence[Tuple[float, ...]],
) -> tuple[List[np.ndarray], np.ndarray]:
    num_horizons = len(predictions_by_expert[expert_names[0]]["region_probs"])
    if len(per_horizon_weights) != num_horizons:
        raise ValueError("per_horizon_weights length does not match number of horizons.")

    region_probs: List[np.ndarray] = []
    status_shape = predictions_by_expert[expert_names[0]]["status_probs"].numpy().shape
    status_probs = np.zeros(status_shape, dtype=np.float64)

    for horizon_idx in range(num_horizons):
        weights = per_horizon_weights[horizon_idx]
        blended_region = sum(
            float(weights[expert_idx]) * predictions_by_expert[name]["region_probs"][horizon_idx].numpy()
            for expert_idx, name in enumerate(expert_names)
        )
        blended_region = np.clip(blended_region, 1.0e-12, None)
        blended_region = blended_region / blended_region.sum(axis=1, keepdims=True)
        region_probs.append(blended_region)

        blended_status = sum(
            float(weights[expert_idx]) * predictions_by_expert[name]["status_probs"].numpy()[:, horizon_idx, :]
            for expert_idx, name in enumerate(expert_names)
        )
        blended_status = np.clip(blended_status, 1.0e-12, None)
        status_probs[:, horizon_idx, :] = blended_status / blended_status.sum(axis=-1, keepdims=True)

    return region_probs, status_probs


def _search_best_temperatures(
    dataset: RegionalForecastDataset,
    *,
    region_probabilities: Sequence[np.ndarray],
    status_probabilities: np.ndarray,
    metric_specs: Sequence[Tuple[str, bool]],
    candidate_probability_mass: Sequence[float],
    candidate_top_cells: Sequence[int],
    temperature_candidates: Sequence[float],
) -> tuple[List[float], List[Dict[str, float]]]:
    num_horizons = len(region_probabilities)
    best_temperatures: List[float] = []
    best_rows: List[Dict[str, float]] = []

    for horizon_idx in range(num_horizons):
        candidate_rows: List[Dict[str, float]] = []
        for temperature in temperature_candidates:
            scaled_region_probs = [np.asarray(arr) for arr in region_probabilities]
            scaled_region_probs[horizon_idx] = _temperature_scale_probabilities(region_probabilities[horizon_idx], temperature)
            rows = _compute_rows(
                dataset,
                region_probabilities=scaled_region_probs,
                status_probabilities=status_probabilities,
                region_true_probabilities=None,
                candidate_probability_mass=candidate_probability_mass,
                candidate_top_cells=candidate_top_cells,
            )
            candidate_rows.append(rows[horizon_idx])

        scores = _score_rows([[row] for row in candidate_rows], metric_specs, 0)
        best_idx = int(scores.argmax())
        best_temperatures.append(float(temperature_candidates[best_idx]))
        best_rows.append(dict(candidate_rows[best_idx]))
    return best_temperatures, best_rows


def _apply_per_horizon_temperatures(
    region_probabilities: Sequence[np.ndarray],
    temperatures: Sequence[float],
) -> List[np.ndarray]:
    return [
        _temperature_scale_probabilities(np.asarray(region_probabilities[horizon_idx]), float(temperatures[horizon_idx]))
        for horizon_idx in range(len(region_probabilities))
    ]


def _macro_summary(rows: Sequence[Dict[str, float]], metrics: Sequence[Tuple[str, bool]]) -> Dict[str, float]:
    return {metric_name: float(mean(row[metric_name] for row in rows)) for metric_name, _ in metrics}


def _write_rows_csv(path: Path, rows: Sequence[Dict[str, float]]) -> None:
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Search a horizon-specialized expert blend on a devsplit proxy.")
    parser.add_argument(
        "--expert",
        action="append",
        required=True,
        help="Expert spec in the form name=config_path. Repeat for multiple experts.",
    )
    parser.add_argument("--val-split", type=str, default="val")
    parser.add_argument("--test-split", type=str, default="test")
    parser.add_argument("--weight-step", type=float, default=0.1)
    parser.add_argument("--temperatures", type=str, default="0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--candidate-probability-mass", type=str, default="0.8,0.8,0.8,0.8")
    parser.add_argument("--candidate-top-cells", type=str, default="24,18,18,24")
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="artifacts/research_devsplit/evaluations/adrr_devsplit_horizon_expert_blend",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="artifacts/research_devsplit/ensemble_cache",
    )
    args = parser.parse_args()

    experts = _parse_expert_specs(args.expert)
    expert_names = list(experts.keys())
    candidate_probability_mass = [float(value) for value in args.candidate_probability_mass.split(",")]
    candidate_top_cells = [int(value) for value in args.candidate_top_cells.split(",")]
    temperature_candidates = [float(value) for value in args.temperatures.split(",") if value.strip()]

    val_predictions: Dict[str, Dict[str, object]] = {}
    test_predictions: Dict[str, Dict[str, object]] = {}
    val_dataset: RegionalForecastDataset | None = None
    test_dataset: RegionalForecastDataset | None = None
    cache_dir = Path(args.cache_dir)

    for expert_name, config_path in experts.items():
        cfg = _prepare_cfg(config_path, batch_size=args.batch_size, num_workers=args.num_workers)
        val_dataset, val_predictions[expert_name] = _load_or_predict_split(
            cfg,
            expert_name=expert_name,
            split=args.val_split,
            cache_dir=cache_dir,
        )
        test_dataset, test_predictions[expert_name] = _load_or_predict_split(
            cfg,
            expert_name=expert_name,
            split=args.test_split,
            cache_dir=cache_dir,
        )
        print(f"Loaded predictions for expert={expert_name}")

    if val_dataset is None or test_dataset is None:
        raise RuntimeError("No datasets were loaded.")

    weight_candidates = _simplex_weights(len(expert_names), args.weight_step)
    best_weights, best_val_rows = _search_best_weights(
        val_dataset,
        predictions_by_expert=val_predictions,
        expert_names=expert_names,
        weight_candidates=weight_candidates,
        metric_specs=DEFAULT_SCORE_METRICS,
        candidate_probability_mass=candidate_probability_mass,
        candidate_top_cells=candidate_top_cells,
    )

    val_region_probs, val_status_probs = _apply_per_horizon_weights(
        val_predictions,
        expert_names=expert_names,
        per_horizon_weights=best_weights,
    )
    best_temperatures, best_temp_val_rows = _search_best_temperatures(
        val_dataset,
        region_probabilities=val_region_probs,
        status_probabilities=val_status_probs,
        metric_specs=DEFAULT_SCORE_METRICS,
        candidate_probability_mass=candidate_probability_mass,
        candidate_top_cells=candidate_top_cells,
        temperature_candidates=temperature_candidates,
    )

    test_region_probs, test_status_probs = _apply_per_horizon_weights(
        test_predictions,
        expert_names=expert_names,
        per_horizon_weights=best_weights,
    )
    scaled_test_region_probs = _apply_per_horizon_temperatures(test_region_probs, best_temperatures)
    test_rows = _compute_rows(
        test_dataset,
        region_probabilities=scaled_test_region_probs,
        status_probabilities=test_status_probs,
        region_true_probabilities=None,
        candidate_probability_mass=candidate_probability_mass,
        candidate_top_cells=candidate_top_cells,
    )

    report = {
        "experts": {name: str(path) for name, path in experts.items()},
        "score_metrics": [{"name": name, "higher_is_better": higher} for name, higher in DEFAULT_SCORE_METRICS],
        "weight_step": args.weight_step,
        "temperature_candidates": temperature_candidates,
        "selected_weights_per_horizon": [
            {
                "horizon_minutes": int(val_dataset.metadata["horizons_minutes"][idx]),
                "weights": {expert_names[expert_idx]: float(weight) for expert_idx, weight in enumerate(weights)},
            }
            for idx, weights in enumerate(best_weights)
        ],
        "selected_temperatures_per_horizon": [
            {
                "horizon_minutes": int(val_dataset.metadata["horizons_minutes"][idx]),
                "temperature": float(best_temperatures[idx]),
            }
            for idx in range(len(best_temperatures))
        ],
        "validation_rows_after_weight_search": best_val_rows,
        "validation_rows_after_temperature_search": best_temp_val_rows,
        "test_rows": test_rows,
        "test_macro": _macro_summary(
            test_rows,
            [
                *DEFAULT_SCORE_METRICS,
                ("candidate_union_cover", True),
                ("candidate_union_cell_count_mean", False),
                ("candidate_union_area_km2_mean", False),
            ],
        ),
    }

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    with open(output_prefix.with_name(f"{output_prefix.name}_report.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    with open(output_prefix.with_name(f"{output_prefix.name}_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(test_rows, handle, ensure_ascii=False, indent=2)
    _write_rows_csv(output_prefix.with_name(f"{output_prefix.name}_summary.csv"), test_rows)

    print("Selected weights:")
    for item in report["selected_weights_per_horizon"]:
        print(item)
    print("Selected temperatures:")
    for item in report["selected_temperatures_per_horizon"]:
        print(item)
    print("Test macro:")
    for key, value in report["test_macro"].items():
        print(f"{key}={value:.6f}")


if __name__ == "__main__":
    main()
