from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from regional_forecast.config import load_config
from regional_forecast.data.dataset import RegionalForecastDataset
from regional_forecast.models.baseline import RegionalMarkovBaseline
from regional_forecast.registry import register_training_run


def _save_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _fit_batch_vectorized(model: RegionalMarkovBaseline, batch: Dict[str, torch.Tensor]) -> int:
    current_regions = batch["region_seq"][:, -1].long()
    label_mask = batch["label_mask"].bool()
    region_labels = batch["region_labels"].long()
    status_labels = batch["status_labels"].long()
    update_count = 0

    for horizon_idx in range(model.num_horizons):
        valid = label_mask[:, horizon_idx]
        if not torch.any(valid):
            continue

        current = current_regions[valid]
        region_target = region_labels[valid, horizon_idx]
        status_target = status_labels[valid, horizon_idx]
        update_count += int(current.shape[0])

        region_width = int(model.region_counts[horizon_idx].shape[1])
        region_flat = current * region_width + region_target
        region_values = torch.ones_like(region_flat, dtype=model.region_counts[horizon_idx].dtype)
        model.region_counts[horizon_idx].view(-1).index_add_(0, region_flat.cpu(), region_values.cpu())

        status_width = int(model.status_counts[horizon_idx].shape[1])
        status_flat = current * status_width + status_target
        status_values = torch.ones_like(status_flat, dtype=model.status_counts[horizon_idx].dtype)
        model.status_counts[horizon_idx].view(-1).index_add_(0, status_flat.cpu(), status_values.cpu())

    return update_count


def train_markov_baseline(config_path: str, batch_size: int) -> None:
    cfg = load_config(config_path)
    cfg.model.model_type = "markov"
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_ds = RegionalForecastDataset(str(cfg.dataset_dir / "train.pt"))
    model = RegionalMarkovBaseline(
        input_num_region_classes=int(train_ds.metadata["input_num_region_classes"]),
        num_region_classes_per_horizon=[
            int(value)
            for value in train_ds.metadata["num_region_classes_per_horizon"]
        ],
        num_status_classes=int(train_ds.metadata["num_status_classes"]),
        num_horizons=len(train_ds.metadata["horizons_minutes"]),
    )

    sampler = (
        train_ds.build_batch_sampler(batch_size, shuffle=False, drop_last=False)
        if train_ds.supports_chunk_batching
        else range(0, len(train_ds), batch_size)
    )

    total_updates = 0
    total_batches = 0
    if train_ds.supports_chunk_batching:
        for batch_indices in sampler:
            batch = train_ds.get_batch([int(index) for index in batch_indices])
            total_updates += _fit_batch_vectorized(model, batch)
            total_batches += 1
            if total_batches % 100 == 0:
                print(f"processed_batches={total_batches} updates={total_updates}", flush=True)
    else:
        for start in sampler:
            indices = list(range(int(start), min(int(start) + int(batch_size), len(train_ds))))
            batch = train_ds.get_batch(indices)
            total_updates += _fit_batch_vectorized(model, batch)
            total_batches += 1
            if total_batches % 100 == 0:
                print(f"processed_batches={total_batches} updates={total_updates}", flush=True)

    checkpoint_path = cfg.checkpoint_dir / f"{cfg.output.run_name}.pt"
    summary_path = cfg.checkpoint_dir / f"{cfg.output.run_name}_train_summary.json"
    torch.save(
        {
            "model_type": "markov",
            "model_state": model.state_dict(),
            "metadata": train_ds.metadata,
            "config": cfg.to_dict(),
        },
        checkpoint_path,
    )
    summary = {
        "mode": "markov",
        "train_samples": len(train_ds),
        "total_batches": total_batches,
        "total_label_updates": total_updates,
        "checkpoint_path": str(checkpoint_path),
    }
    _save_json(summary_path, summary)
    register_training_run(
        cfg,
        model_type="markov",
        metadata=train_ds.metadata,
        checkpoint_path=checkpoint_path,
        summary_path=summary_path,
        best_val_loss=0.0,
        sample_counts={"train": len(train_ds)},
    )
    print(f"Saved Markov checkpoint: {checkpoint_path}", flush=True)
    print(f"Saved summary: {summary_path}", flush=True)
    print(f"Updated registry: {cfg.model_registry_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a scalable Markov regional baseline.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=65536)
    args = parser.parse_args()
    train_markov_baseline(args.config, args.batch_size)


if __name__ == "__main__":
    main()
