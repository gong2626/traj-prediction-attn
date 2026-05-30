from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from regional_forecast.config import ProjectConfig, load_config
from regional_forecast.data.dataset import RegionalForecastDataset
from regional_forecast.evaluation.metrics import (
    StreamingMultihorizonMetricAccumulator,
    compute_multihorizon_metrics,
)
from regional_forecast.pipeline.evaluate import (
    _build_child_lookup_tensors,
    _iter_eval_batches,
    _load_model,
    _predict_batch_outputs,
)


DEFAULT_SCORE_METRICS: List[Tuple[str, bool]] = [
    ("region_top1", True),
    ("region_top3", True),
    ("region_top5", True),
    ("region_nll", False),
    ("region_brier", False),
    ("region_ece", False),
    ("candidate_cover_at_budget_cells", True),
]

HIERARCHICAL_MODEL_TYPES = {
    "hierarchical_gru",
    "attention_hierarchical_gru",
    "horizon_decoder_gru",
    "transformer_horizon_decoder",
    "multiscale_transformer_h3",
    "multiscale_transformer_h3_local_causal",
    "multiscale_transformer_h3_gated_local",
    "horizon_router_transformer_h3",
    "attention_tcn_gru_h3",
    "traisformer_h3",
    "flightbertpp_h3",
}


@dataclass
class ExpertRuntime:
    name: str
    cfg: ProjectConfig
    model_type: str
    model: object
    device: torch.device
    hierarchy_per_horizon: Sequence[Dict[str, object]] | None
    fine_class_counts: Sequence[int] | None
    unknown_fine_ids: Sequence[int] | None
    child_lookup_tables: List[tuple[torch.Tensor, torch.Tensor]] | None


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


def _prepare_cfg(path: Path, *, device: str) -> ProjectConfig:
    cfg = load_config(str(path))
    cfg.training.device = str(device)
    return cfg


def _region_tokens_per_horizon(dataset: RegionalForecastDataset) -> List[List[str]]:
    return [list(meta["region_tokens"]) for meta in dataset.metadata["target_regionizers"]]


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


def _simplex_weights(num_experts: int, step: float) -> List[Tuple[float, ...]]:
    if num_experts < 2:
        raise ValueError("Need at least two experts for blending.")
    if step <= 0.0 or step > 1.0:
        raise ValueError("weight step must be in (0, 1].")
    denom = round(1.0 / step)
    if abs(denom * step - 1.0) > 1.0e-8:
        raise ValueError("weight step must divide 1.0 exactly, e.g. 0.2 or 0.1.")

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


def _temperature_scale_probabilities(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")
    if abs(temperature - 1.0) < 1.0e-12:
        return probabilities
    scaled = np.power(np.clip(probabilities, 1.0e-12, 1.0), 1.0 / float(temperature))
    return scaled / scaled.sum(axis=1, keepdims=True)


def _compute_rows_from_arrays(
    *,
    dataset: RegionalForecastDataset,
    region_probabilities: Sequence[np.ndarray],
    status_probabilities: np.ndarray,
    region_labels: np.ndarray,
    status_labels: np.ndarray,
    label_mask: np.ndarray,
    candidate_probability_mass: Sequence[float],
    candidate_top_cells: Sequence[int],
) -> List[Dict[str, float]]:
    return compute_multihorizon_metrics(
        region_probabilities=region_probabilities,
        region_labels=region_labels,
        region_true_probabilities=None,
        status_probabilities=status_probabilities,
        status_labels=status_labels,
        label_mask=label_mask,
        top_ks=[1, 3, 5],
        calibration_bins=10,
        horizons_minutes=dataset.metadata["horizons_minutes"],
        region_tokens_per_horizon=_region_tokens_per_horizon(dataset),
        neighbor_hit_rings=[1, 1, 1, 1],
        candidate_probability_mass=candidate_probability_mass,
        candidate_top_cells=candidate_top_cells,
    )


def _score_horizon_rows(
    rows_by_candidate: Sequence[Sequence[Dict[str, float]]],
    *,
    metric_specs: Sequence[Tuple[str, bool]],
    horizon_idx: int,
) -> np.ndarray:
    values: List[np.ndarray] = []
    for metric_name, higher_is_better in metric_specs:
        metric_values = np.asarray(
            [rows[horizon_idx][metric_name] for rows in rows_by_candidate],
            dtype=np.float64,
        )
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


def _macro_summary(rows: Sequence[Dict[str, float]], metrics: Sequence[Tuple[str, bool]]) -> Dict[str, float]:
    return {metric_name: float(mean(row[metric_name] for row in rows)) for metric_name, _ in metrics}


def _collect_sample_indices(dataset: RegionalForecastDataset, sample_size: int, seed: int) -> List[int]:
    sample_size = min(int(sample_size), len(dataset))
    rng = random.Random(int(seed))
    if sample_size >= len(dataset):
        return list(range(len(dataset)))
    indices = rng.sample(range(len(dataset)), sample_size)
    indices.sort()
    return indices


def _prepare_expert_runtime(
    *,
    expert_name: str,
    cfg: ProjectConfig,
    dataset: RegionalForecastDataset,
) -> ExpertRuntime:
    model_type, model = _load_model(cfg, dataset)
    device = torch.device("cuda" if cfg.training.device == "cuda" and torch.cuda.is_available() else "cpu")
    if model_type != "markov":
        model = model.to(device)
        model.eval()

    hierarchy_per_horizon = dataset.metadata.get("hierarchy_per_horizon") if model_type in HIERARCHICAL_MODEL_TYPES else None
    fine_class_counts = dataset.metadata.get("num_region_classes_per_horizon") if model_type in HIERARCHICAL_MODEL_TYPES else None
    unknown_fine_ids = dataset.metadata.get("unknown_region_ids_per_horizon") if model_type in HIERARCHICAL_MODEL_TYPES else None
    child_lookup_tables = (
        _build_child_lookup_tensors(
            hierarchy_per_horizon,
            device=device,
            unknown_fine_ids=unknown_fine_ids,
        )
        if hierarchy_per_horizon is not None and fine_class_counts is not None and unknown_fine_ids is not None
        else None
    )
    return ExpertRuntime(
        name=expert_name,
        cfg=cfg,
        model_type=model_type,
        model=model,
        device=device,
        hierarchy_per_horizon=hierarchy_per_horizon,
        fine_class_counts=fine_class_counts,
        unknown_fine_ids=unknown_fine_ids,
        child_lookup_tables=child_lookup_tables,
    )


def _predict_batch_for_expert(expert: ExpertRuntime, batch_cpu: Dict[str, torch.Tensor]) -> tuple[List[np.ndarray], np.ndarray]:
    region_probs, status_probs, _ = _predict_batch_outputs(
        expert.cfg,
        batch_cpu,
        expert.model_type,
        expert.model,
        expert.device,
        expert.hierarchy_per_horizon,
        expert.fine_class_counts,
        expert.unknown_fine_ids,
        expert.child_lookup_tables,
    )
    return region_probs, status_probs


def _collect_sample_predictions(
    *,
    dataset: RegionalForecastDataset,
    sample_indices: Sequence[int],
    experts: Sequence[ExpertRuntime],
    batch_size: int,
) -> tuple[Dict[str, Dict[str, object]], Dict[str, np.ndarray]]:
    if not sample_indices:
        raise ValueError("sample_indices is empty.")

    num_horizons = len(dataset.metadata["horizons_minutes"])
    prediction_store: Dict[str, Dict[str, object]] = {}
    for expert in experts:
        prediction_store[expert.name] = {
            "region_probs": [[] for _ in range(num_horizons)],
            "status_probs": [],
        }

    labels_store = {
        "region_labels": [],
        "status_labels": [],
        "label_mask": [],
    }

    for start in range(0, len(sample_indices), int(batch_size)):
        batch_indices = list(sample_indices[start:start + int(batch_size)])
        batch_cpu = dataset.get_batch(batch_indices)
        labels_store["region_labels"].append(batch_cpu["region_labels"].numpy())
        labels_store["status_labels"].append(batch_cpu["status_labels"].numpy())
        labels_store["label_mask"].append(batch_cpu["label_mask"].numpy())

        for expert in experts:
            region_probs, status_probs = _predict_batch_for_expert(expert, batch_cpu)
            prediction_store[expert.name]["status_probs"].append(status_probs)
            for horizon_idx in range(num_horizons):
                prediction_store[expert.name]["region_probs"][horizon_idx].append(region_probs[horizon_idx])

    packed_predictions: Dict[str, Dict[str, object]] = {}
    for expert in experts:
        packed_predictions[expert.name] = {
            "region_probs": [
                np.concatenate(prediction_store[expert.name]["region_probs"][horizon_idx], axis=0)
                for horizon_idx in range(num_horizons)
            ],
            "status_probs": np.concatenate(prediction_store[expert.name]["status_probs"], axis=0),
        }

    labels = {
        key: np.concatenate(chunks, axis=0)
        for key, chunks in labels_store.items()
    }
    return packed_predictions, labels


def _blend_region_probs_cached(
    predictions_by_expert: Dict[str, Dict[str, object]],
    expert_names: Sequence[str],
    weights: Sequence[float],
) -> List[np.ndarray]:
    num_horizons = len(predictions_by_expert[expert_names[0]]["region_probs"])
    blended: List[np.ndarray] = []
    for horizon_idx in range(num_horizons):
        arr = sum(
            float(weights[expert_idx]) * predictions_by_expert[name]["region_probs"][horizon_idx]
            for expert_idx, name in enumerate(expert_names)
        )
        arr = np.clip(arr, 1.0e-12, None)
        arr = arr / arr.sum(axis=1, keepdims=True)
        blended.append(arr.astype(np.float32, copy=False))
    return blended


def _blend_status_probs_cached(
    predictions_by_expert: Dict[str, Dict[str, object]],
    expert_names: Sequence[str],
    weights: Sequence[float],
) -> np.ndarray:
    arr = sum(
        float(weights[expert_idx]) * predictions_by_expert[name]["status_probs"]
        for expert_idx, name in enumerate(expert_names)
    )
    arr = np.clip(arr, 1.0e-12, None)
    return (arr / arr.sum(axis=-1, keepdims=True)).astype(np.float32, copy=False)


def _search_weights_on_sample(
    *,
    dataset: RegionalForecastDataset,
    predictions_by_expert: Dict[str, Dict[str, object]],
    labels: Dict[str, np.ndarray],
    expert_names: Sequence[str],
    weight_candidates: Sequence[Tuple[float, ...]],
    metric_specs: Sequence[Tuple[str, bool]],
    candidate_probability_mass: Sequence[float],
    candidate_top_cells: Sequence[int],
) -> tuple[List[Tuple[float, ...]], List[Dict[str, float]]]:
    rows_by_candidate: List[List[Dict[str, float]]] = []
    for weights in weight_candidates:
        blended_region_probs = _blend_region_probs_cached(predictions_by_expert, expert_names, weights)
        blended_status_probs = _blend_status_probs_cached(predictions_by_expert, expert_names, weights)
        rows_by_candidate.append(
            _compute_rows_from_arrays(
                dataset=dataset,
                region_probabilities=blended_region_probs,
                status_probabilities=blended_status_probs,
                region_labels=labels["region_labels"],
                status_labels=labels["status_labels"],
                label_mask=labels["label_mask"],
                candidate_probability_mass=candidate_probability_mass,
                candidate_top_cells=candidate_top_cells,
            )
        )

    num_horizons = len(dataset.metadata["horizons_minutes"])
    best_weights: List[Tuple[float, ...]] = []
    best_rows: List[Dict[str, float]] = []
    for horizon_idx in range(num_horizons):
        scores = _score_horizon_rows(rows_by_candidate, metric_specs=metric_specs, horizon_idx=horizon_idx)
        best_idx = int(scores.argmax())
        best_weights.append(tuple(weight_candidates[best_idx]))
        best_rows.append(dict(rows_by_candidate[best_idx][horizon_idx]))
    return best_weights, best_rows


def _search_temperatures_on_sample(
    *,
    dataset: RegionalForecastDataset,
    blended_region_probs: Sequence[np.ndarray],
    blended_status_probs: np.ndarray,
    labels: Dict[str, np.ndarray],
    candidate_probability_mass: Sequence[float],
    candidate_top_cells: Sequence[int],
    metric_specs: Sequence[Tuple[str, bool]],
    temperature_candidates: Sequence[float],
) -> tuple[List[float], List[Dict[str, float]]]:
    num_horizons = len(dataset.metadata["horizons_minutes"])
    best_temperatures: List[float] = []
    best_rows: List[Dict[str, float]] = []
    for horizon_idx in range(num_horizons):
        rows_for_temps: List[Dict[str, float]] = []
        for temperature in temperature_candidates:
            scaled_region_probs = [np.asarray(arr) for arr in blended_region_probs]
            scaled_region_probs[horizon_idx] = _temperature_scale_probabilities(
                np.asarray(blended_region_probs[horizon_idx]),
                float(temperature),
            )
            rows = _compute_rows_from_arrays(
                dataset=dataset,
                region_probabilities=scaled_region_probs,
                status_probabilities=blended_status_probs,
                region_labels=labels["region_labels"],
                status_labels=labels["status_labels"],
                label_mask=labels["label_mask"],
                candidate_probability_mass=candidate_probability_mass,
                candidate_top_cells=candidate_top_cells,
            )
            rows_for_temps.append(rows[horizon_idx])

        pseudo_rows = [[row] for row in rows_for_temps]
        scores = _score_horizon_rows(pseudo_rows, metric_specs=metric_specs, horizon_idx=0)
        best_idx = int(scores.argmax())
        best_temperatures.append(float(temperature_candidates[best_idx]))
        best_rows.append(dict(rows_for_temps[best_idx]))
    return best_temperatures, best_rows


def _evaluate_full_test_streaming(
    *,
    dataset: RegionalForecastDataset,
    experts: Sequence[ExpertRuntime],
    expert_names: Sequence[str],
    per_horizon_weights: Sequence[Tuple[float, ...]],
    per_horizon_temperatures: Sequence[float],
    batch_size: int,
    candidate_probability_mass: Sequence[float],
    candidate_top_cells: Sequence[int],
) -> List[Dict[str, float]]:
    accumulator = StreamingMultihorizonMetricAccumulator(
        top_ks=[1, 3, 5],
        calibration_bins=10,
        horizons_minutes=dataset.metadata["horizons_minutes"],
        region_tokens_per_horizon=_region_tokens_per_horizon(dataset),
        neighbor_hit_rings=[1, 1, 1, 1],
        candidate_probability_mass=candidate_probability_mass,
        candidate_top_cells=candidate_top_cells,
    )

    batch_counter = 0
    for _, batch_cpu in _iter_eval_batches(dataset, int(batch_size)):
        region_probs_by_expert: Dict[str, List[np.ndarray]] = {}
        status_probs_by_expert: Dict[str, np.ndarray] = {}
        for expert in experts:
            region_probs, status_probs = _predict_batch_for_expert(expert, batch_cpu)
            region_probs_by_expert[expert.name] = region_probs
            status_probs_by_expert[expert.name] = status_probs

        blended_region_probs: List[np.ndarray] = []
        num_horizons = len(dataset.metadata["horizons_minutes"])
        batch_size_current = int(batch_cpu["region_labels"].shape[0])
        blended_status_probs = np.zeros_like(next(iter(status_probs_by_expert.values())), dtype=np.float32)
        for horizon_idx in range(num_horizons):
            weights = per_horizon_weights[horizon_idx]
            region_arr = sum(
                float(weights[expert_idx]) * region_probs_by_expert[name][horizon_idx]
                for expert_idx, name in enumerate(expert_names)
            )
            region_arr = np.clip(region_arr, 1.0e-12, None)
            region_arr = region_arr / region_arr.sum(axis=1, keepdims=True)
            region_arr = _temperature_scale_probabilities(region_arr, float(per_horizon_temperatures[horizon_idx]))
            blended_region_probs.append(region_arr.astype(np.float32, copy=False))

            status_arr = sum(
                float(weights[expert_idx]) * status_probs_by_expert[name][:, horizon_idx, :]
                for expert_idx, name in enumerate(expert_names)
            )
            status_arr = np.clip(status_arr, 1.0e-12, None)
            status_arr = status_arr / status_arr.sum(axis=-1, keepdims=True)
            blended_status_probs[:, horizon_idx, :] = status_arr.astype(np.float32, copy=False)

        accumulator.update(
            region_probabilities=blended_region_probs,
            region_labels=batch_cpu["region_labels"].numpy(),
            status_probabilities=blended_status_probs,
            status_labels=batch_cpu["status_labels"].numpy(),
            label_mask=batch_cpu["label_mask"].numpy(),
            region_true_probabilities=None,
        )
        batch_counter += 1
        if batch_counter % 50 == 0:
            print(f"Full-test streaming progress: {batch_counter} batches processed.")

    return accumulator.finalize()


def main() -> None:
    parser = argparse.ArgumentParser(description="Full-scale horizon expert blend with sampled validation search and streaming test evaluation.")
    parser.add_argument(
        "--expert",
        action="append",
        required=True,
        help="Expert spec in the form name=config_path. Repeat for multiple experts.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--val-sample-size", type=int, default=50000)
    parser.add_argument("--search-seed", type=int, default=42)
    parser.add_argument("--weight-step", type=float, default=0.2)
    parser.add_argument("--temperatures", type=str, default="0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5")
    parser.add_argument("--search-batch-size", type=int, default=2048)
    parser.add_argument("--test-batch-size", type=int, default=4096)
    parser.add_argument("--candidate-probability-mass", type=str, default="0.8,0.8,0.8,0.8")
    parser.add_argument("--candidate-top-cells", type=str, default="24,18,18,24")
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="artifacts/evaluations/global_horizon_expert_blend",
    )
    args = parser.parse_args()

    expert_specs = _parse_expert_specs(args.expert)
    expert_names = list(expert_specs.keys())
    candidate_probability_mass = [float(value) for value in args.candidate_probability_mass.split(",")]
    candidate_top_cells = [int(value) for value in args.candidate_top_cells.split(",")]
    temperature_candidates = [float(value) for value in args.temperatures.split(",") if value.strip()]
    weight_candidates = _simplex_weights(len(expert_names), float(args.weight_step))

    base_cfg = _prepare_cfg(next(iter(expert_specs.values())), device=args.device)
    val_dataset = RegionalForecastDataset(str(base_cfg.dataset_dir / "val.pt"))
    test_dataset = RegionalForecastDataset(str(base_cfg.dataset_dir / "test.pt"))

    experts: List[ExpertRuntime] = []
    for expert_name, config_path in expert_specs.items():
        cfg = _prepare_cfg(config_path, device=args.device)
        experts.append(_prepare_expert_runtime(expert_name=expert_name, cfg=cfg, dataset=val_dataset))
        print(f"Loaded expert runtime: {expert_name}")

    sample_indices = _collect_sample_indices(val_dataset, args.val_sample_size, args.search_seed)
    sampled_predictions, sampled_labels = _collect_sample_predictions(
        dataset=val_dataset,
        sample_indices=sample_indices,
        experts=experts,
        batch_size=args.search_batch_size,
    )
    print(f"Collected validation sample predictions: {len(sample_indices)} rows")

    best_weights, best_val_rows = _search_weights_on_sample(
        dataset=val_dataset,
        predictions_by_expert=sampled_predictions,
        labels=sampled_labels,
        expert_names=expert_names,
        weight_candidates=weight_candidates,
        metric_specs=DEFAULT_SCORE_METRICS,
        candidate_probability_mass=candidate_probability_mass,
        candidate_top_cells=candidate_top_cells,
    )

    blended_val_region_probs: List[np.ndarray] = []
    num_horizons = len(val_dataset.metadata["horizons_minutes"])
    blended_val_status_probs = np.zeros_like(next(iter(sampled_predictions.values()))["status_probs"], dtype=np.float32)
    for horizon_idx in range(num_horizons):
        weights = best_weights[horizon_idx]
        arr = sum(
            float(weights[expert_idx]) * sampled_predictions[name]["region_probs"][horizon_idx]
            for expert_idx, name in enumerate(expert_names)
        )
        arr = np.clip(arr, 1.0e-12, None)
        arr = arr / arr.sum(axis=1, keepdims=True)
        blended_val_region_probs.append(arr.astype(np.float32, copy=False))
        status_arr = sum(
            float(weights[expert_idx]) * sampled_predictions[name]["status_probs"][:, horizon_idx, :]
            for expert_idx, name in enumerate(expert_names)
        )
        status_arr = np.clip(status_arr, 1.0e-12, None)
        status_arr = status_arr / status_arr.sum(axis=-1, keepdims=True)
        blended_val_status_probs[:, horizon_idx, :] = status_arr.astype(np.float32, copy=False)

    best_temperatures, best_temp_val_rows = _search_temperatures_on_sample(
        dataset=val_dataset,
        blended_region_probs=blended_val_region_probs,
        blended_status_probs=blended_val_status_probs,
        labels=sampled_labels,
        candidate_probability_mass=candidate_probability_mass,
        candidate_top_cells=candidate_top_cells,
        metric_specs=DEFAULT_SCORE_METRICS,
        temperature_candidates=temperature_candidates,
    )

    test_rows = _evaluate_full_test_streaming(
        dataset=test_dataset,
        experts=experts,
        expert_names=expert_names,
        per_horizon_weights=best_weights,
        per_horizon_temperatures=best_temperatures,
        batch_size=args.test_batch_size,
        candidate_probability_mass=candidate_probability_mass,
        candidate_top_cells=candidate_top_cells,
    )

    report = {
        "experts": {name: str(path) for name, path in expert_specs.items()},
        "device": args.device,
        "val_sample_size": int(len(sample_indices)),
        "search_seed": int(args.search_seed),
        "score_metrics": [{"name": name, "higher_is_better": higher} for name, higher in DEFAULT_SCORE_METRICS],
        "weight_step": float(args.weight_step),
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
