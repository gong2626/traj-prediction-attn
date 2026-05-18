from __future__ import annotations

import argparse

from regional_forecast.config import load_config
from regional_forecast.data.preprocess import build_dataset_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build regional forecasting dataset artifacts.")
    parser.add_argument("--config", type=str, default="config/region_forecast.yaml")
    parser.add_argument("--scope", type=str, choices=["global", "type"], default=None)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--workers", type=int, default=1, help="Worker processes for dataset loading, shard scanning, and segment resampling.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.scope is not None:
        cfg.dataset.scope = args.scope
    if args.name is not None:
        cfg.dataset.name = args.name

    counts = build_dataset_artifacts(cfg, workers=args.workers)

    print("Dataset artifacts ready:")
    print(f"  scope: {cfg.dataset_scope}")
    print(f"  workers: {max(1, int(args.workers))}")
    if cfg.dataset_name is not None:
        print(f"  name: {cfg.dataset_name}")
    for split_name, count in counts.items():
        print(f"  {split_name}: {count}")
    print(f"  saved_to: {cfg.dataset_dir}")
