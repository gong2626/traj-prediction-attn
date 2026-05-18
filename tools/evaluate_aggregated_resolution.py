from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

try:
    import h3
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise ImportError("The 'h3' package is required. Install it with 'pip install h3'.") from exc

from regional_forecast.config import ProjectConfig, load_config
from regional_forecast.data.dataset import RegionalForecastDataset
from regional_forecast.data.regionizer import H3Regionizer
from regional_forecast.evaluation.metrics import compute_multihorizon_metrics
from regional_forecast.pipeline.evaluate import _load_model, _predict_probs


def _parse_overrides(items: Sequence[str]) -> Dict[int, int]:
    overrides: Dict[int, int] = {}
    for item in items:
        try:
            horizon_text, resolution_text = item.split("=", maxsplit=1)
            horizon = int(horizon_text.strip())
            resolution = int(resolution_text.strip())
        except ValueError as exc:  # pragma: no cover - validated at runtime
            raise ValueError(
                f"Invalid override '{item}'. Expected format like '30=3' or '60=2'."
            ) from exc
        overrides[horizon] = resolution
    return overrides


def _build_parent_regionizer(region_tokens: Sequence[str], target_resolution: int) -> H3Regionizer:
    parent_tokens = sorted({str(h3.cell_to_parent(str(cell), target_resolution)) for cell in region_tokens})
    return H3Regionizer.from_metadata(
        {
            "kind": "h3",
            "resolution": int(target_resolution),
            "region_tokens": parent_tokens,
        }
    )


def _aggregate_labels_to_resolution(
    labels: np.ndarray,
    source_tokens: Sequence[str],
    target_regionizer: H3Regionizer,
    target_resolution: int,
) -> np.ndarray:
    aggregated = np.full(labels.shape, target_regionizer.unknown_region_id, dtype=np.int64)
    for source_id, source_cell in enumerate(source_tokens):
        parent_cell = str(h3.cell_to_parent(str(source_cell), target_resolution))
        aggregated[labels == source_id] = target_regionizer.encode_cell(parent_cell)
    return aggregated


def _aggregate_probs_to_resolution(
    probabilities: np.ndarray,
    source_tokens: Sequence[str],
    target_regionizer: H3Regionizer,
    target_resolution: int,
) -> np.ndarray:
    aggregated = np.zeros((probabilities.shape[0], target_regionizer.unknown_region_id + 1), dtype=np.float32)
    aggregated[:, target_regionizer.unknown_region_id] = probabilities[:, len(source_tokens)]

    for source_id, source_cell in enumerate(source_tokens):
        parent_cell = str(h3.cell_to_parent(str(source_cell), target_resolution))
        target_id = target_regionizer.encode_cell(parent_cell)
        aggregated[:, target_id] += probabilities[:, source_id]

    return aggregated


def _resolve_horizon_payload(
    dataset: RegionalForecastDataset,
    predictions: Dict[str, object],
    region_labels_all: torch.Tensor,
    coarse_region_labels_all: torch.Tensor,
    horizon_idx: int,
    target_resolution: int,
) -> Dict[str, object]:
    horizon_minutes = int(dataset.metadata["horizons_minutes"][horizon_idx])
    fine_regionizer_meta = dataset.metadata["target_regionizers"][horizon_idx]
    fine_regionizer = H3Regionizer.from_metadata(fine_regionizer_meta)
    fine_resolution = int(fine_regionizer_meta["resolution"])

    fine_probs = predictions["region_probs"][horizon_idx].numpy()
    fine_labels = region_labels_all.numpy()[:, horizon_idx]

    if target_resolution == fine_resolution:
        return {
            "horizon_minutes": horizon_minutes,
            "source_resolution": fine_resolution,
            "target_resolution": target_resolution,
            "aggregation_source": "fine_head",
            "regionizer": fine_regionizer,
            "region_probs": fine_probs,
            "region_labels": fine_labels,
            "region_true_probs": None if predictions["region_true_probs"] is None else predictions["region_true_probs"][horizon_idx],
        }

    coarse_regionizers = dataset.metadata.get("coarse_regionizers", [])
    coarse_probs = predictions.get("coarse_probs")
    if coarse_regionizers and coarse_probs is not None:
        coarse_regionizer_meta = coarse_regionizers[horizon_idx]
        coarse_resolution = int(coarse_regionizer_meta["resolution"])
        if target_resolution == coarse_resolution:
            coarse_regionizer = H3Regionizer.from_metadata(coarse_regionizer_meta)
            region_probs = coarse_probs[horizon_idx].numpy()
            region_labels = coarse_region_labels_all.numpy()[:, horizon_idx]
            true_probs = region_probs[np.arange(region_probs.shape[0]), region_labels]
            return {
                "horizon_minutes": horizon_minutes,
                "source_resolution": fine_resolution,
                "target_resolution": target_resolution,
                "aggregation_source": "coarse_head",
                "regionizer": coarse_regionizer,
                "region_probs": region_probs,
                "region_labels": region_labels,
                "region_true_probs": true_probs,
            }

    if target_resolution >= fine_resolution:
        raise ValueError(
            f"Horizon {horizon_minutes} min currently uses fine resolution {fine_resolution}; "
            f"target resolution {target_resolution} must be coarser."
        )

    target_regionizer = _build_parent_regionizer(fine_regionizer.region_tokens, target_resolution)
    region_probs = _aggregate_probs_to_resolution(
        probabilities=fine_probs,
        source_tokens=fine_regionizer.region_tokens,
        target_regionizer=target_regionizer,
        target_resolution=target_resolution,
    )
    region_labels = _aggregate_labels_to_resolution(
        labels=fine_labels,
        source_tokens=fine_regionizer.region_tokens,
        target_regionizer=target_regionizer,
        target_resolution=target_resolution,
    )
    true_probs = region_probs[np.arange(region_probs.shape[0]), region_labels]
    return {
        "horizon_minutes": horizon_minutes,
        "source_resolution": fine_resolution,
        "target_resolution": target_resolution,
        "aggregation_source": "aggregated_from_fine",
        "regionizer": target_regionizer,
        "region_probs": region_probs,
        "region_labels": region_labels,
        "region_true_probs": true_probs,
    }


def run_aggregated_evaluation(cfg: ProjectConfig, overrides: Dict[int, int], output_name: str | None) -> None:
    dataset = RegionalForecastDataset(str(cfg.dataset_dir / "test.pt"))
    model_type, model = _load_model(cfg, dataset)
    predictions = _predict_probs(cfg, dataset, model_type, model)
    region_labels_all = dataset.tensor("region_labels").long()
    coarse_region_labels_all = dataset.tensor("coarse_region_labels").long()
    status_labels_all = dataset.tensor("status_labels").long()
    label_mask_all = dataset.tensor("label_mask").bool()

    horizon_payloads: List[Dict[str, object]] = []
    for horizon_idx, horizon_minutes in enumerate(dataset.metadata["horizons_minutes"]):
        target_resolution = int(overrides.get(int(horizon_minutes), dataset.metadata["target_h3_resolutions"][horizon_idx]))
        horizon_payloads.append(
            _resolve_horizon_payload(
                dataset=dataset,
                predictions=predictions,
                region_labels_all=region_labels_all,
                coarse_region_labels_all=coarse_region_labels_all,
                horizon_idx=horizon_idx,
                target_resolution=target_resolution,
            )
        )

    region_probabilities = [payload["region_probs"] for payload in horizon_payloads]
    region_labels = np.stack([payload["region_labels"] for payload in horizon_payloads], axis=1)
    region_true_probabilities = [payload["region_true_probs"] for payload in horizon_payloads]
    region_tokens_per_horizon = [payload["regionizer"].region_tokens for payload in horizon_payloads]  # type: ignore[index]

    rows = compute_multihorizon_metrics(
        region_probabilities=region_probabilities,
        region_labels=region_labels,
        region_true_probabilities=region_true_probabilities,
        status_probabilities=predictions["status_probs"].numpy(),
        status_labels=status_labels_all.numpy(),
        label_mask=label_mask_all.numpy(),
        top_ks=cfg.evaluation.top_ks,
        calibration_bins=cfg.evaluation.calibration_bins,
        horizons_minutes=dataset.metadata["horizons_minutes"],
        region_tokens_per_horizon=region_tokens_per_horizon,
        neighbor_hit_rings=cfg.evaluation.neighbor_hit_rings,
        candidate_probability_mass=cfg.candidate_probability_mass_per_horizon,
        candidate_top_cells=cfg.candidate_top_cells_per_horizon,
    )

    for row, payload in zip(rows, horizon_payloads):
        row["source_resolution"] = float(payload["source_resolution"])
        row["target_resolution"] = float(payload["target_resolution"])
        row["aggregation_source"] = str(payload["aggregation_source"])

    cfg.evaluation_dir.mkdir(parents=True, exist_ok=True)
    suffix = output_name or (
        "aggregated_"
        + "_".join(
            f"{int(item['horizon_minutes'])}m_r{int(item['target_resolution'])}"
            for item in horizon_payloads
            if int(item["target_resolution"]) != int(item["source_resolution"])
        )
    )
    if not suffix:
        suffix = "aggregated_no_change"

    csv_path = cfg.evaluation_dir / f"{cfg.output.run_name}_{suffix}_summary.csv"
    json_path = cfg.evaluation_dir / f"{cfg.output.run_name}_{suffix}_summary.json"

    import pandas as pd

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)

    print(f"Aggregated evaluation done: {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Re-evaluate an existing checkpoint at coarser H3 resolutions without retraining. "
            "Example: --override 30=3 --override 60=3"
        )
    )
    parser.add_argument("--config", type=str, default="config/region_forecast_synthetic.yaml")
    parser.add_argument(
        "--override",
        type=str,
        action="append",
        default=[],
        help="Per-horizon target resolution override in the form '<minutes>=<resolution>'. Can be repeated.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Optional output suffix. Defaults to an auto-generated name based on overrides.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    overrides = _parse_overrides(args.override)
    run_aggregated_evaluation(cfg, overrides, args.output_name)


if __name__ == "__main__":
    main()
