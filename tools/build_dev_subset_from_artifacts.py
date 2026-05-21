from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from regional_forecast.data.dataset import RegionalForecastDataset
from regional_forecast.data.subset_sampling import (
    choose_covering_train_indices,
    collect_required_labels,
    compute_eligible_indices,
    compute_support_counts,
    compute_unseen_rates,
    write_subset_dataset,
)


def _sample_indices(indices: List[int], count: int, rng: random.Random) -> List[int]:
    if count >= len(indices):
        return sorted(indices)
    return sorted(rng.sample(indices, count))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a split-preserving dev subset from full dataset artifacts.")
    parser.add_argument("--source-dataset-dir", type=str, default="artifacts/datasets")
    parser.add_argument("--output-root", type=str, default="artifacts/research_devsplit")
    parser.add_argument("--train-size", type=int, default=30000)
    parser.add_argument("--val-size", type=int, default=2000)
    parser.add_argument("--test-size", type=int, default=2000)
    parser.add_argument("--min-train-support", type=int, default=10)
    parser.add_argument("--train-examples-per-label", type=int, default=4)
    parser.add_argument("--materialize-batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source_dataset_dir = Path(args.source_dataset_dir).resolve()
    output_root = Path(args.output_root).resolve()
    rng = random.Random(int(args.seed))

    train_ds = RegionalForecastDataset(str(source_dataset_dir / "train.pt"))
    val_ds = RegionalForecastDataset(str(source_dataset_dir / "val.pt"))
    test_ds = RegionalForecastDataset(str(source_dataset_dir / "test.pt"))

    train_region_labels = train_ds.tensor("region_labels").long()
    train_label_mask = train_ds.tensor("label_mask").bool()
    val_region_labels = val_ds.tensor("region_labels").long()
    val_label_mask = val_ds.tensor("label_mask").bool()
    test_region_labels = test_ds.tensor("region_labels").long()
    test_label_mask = test_ds.tensor("label_mask").bool()

    horizons_minutes = list(train_ds.metadata["horizons_minutes"])
    support_counts = compute_support_counts(
        train_region_labels,
        train_label_mask,
        num_classes_per_horizon=train_ds.metadata["num_region_classes_per_horizon"],
    )

    eligible_train_indices = compute_eligible_indices(
        region_labels=train_region_labels,
        label_mask=train_label_mask,
        support_counts=support_counts,
        min_support=args.min_train_support,
    ).tolist()
    eligible_val_indices = compute_eligible_indices(
        region_labels=val_region_labels,
        label_mask=val_label_mask,
        support_counts=support_counts,
        min_support=args.min_train_support,
    ).tolist()
    eligible_test_indices = compute_eligible_indices(
        region_labels=test_region_labels,
        label_mask=test_label_mask,
        support_counts=support_counts,
        min_support=args.min_train_support,
    ).tolist()

    selected_val_indices = _sample_indices(eligible_val_indices, int(args.val_size), rng)
    selected_test_indices = _sample_indices(eligible_test_indices, int(args.test_size), rng)

    required_labels_val = collect_required_labels(
        region_labels=val_region_labels,
        label_mask=val_label_mask,
        indices=selected_val_indices,
    )
    required_labels_test = collect_required_labels(
        region_labels=test_region_labels,
        label_mask=test_label_mask,
        indices=selected_test_indices,
    )
    required_labels = [
        set(required_labels_val[horizon_idx]).union(required_labels_test[horizon_idx])
        for horizon_idx in range(len(horizons_minutes))
    ]

    covering_train_indices = choose_covering_train_indices(
        train_region_labels=train_region_labels,
        train_label_mask=train_label_mask,
        required_labels_per_horizon=required_labels,
        examples_per_label=args.train_examples_per_label,
        seed=args.seed,
    )

    train_selected = set(int(index) for index in covering_train_indices)
    remaining_train_pool = [int(index) for index in eligible_train_indices if int(index) not in train_selected]
    target_train_size = max(int(args.train_size), len(train_selected))
    additional_needed = max(0, target_train_size - len(train_selected))
    if additional_needed > 0:
        if additional_needed >= len(remaining_train_pool):
            train_selected.update(remaining_train_pool)
        else:
            train_selected.update(rng.sample(remaining_train_pool, additional_needed))
    selected_train_indices = sorted(train_selected)

    split_indices = {
        "train": selected_train_indices,
        "val": selected_val_indices,
        "test": selected_test_indices,
    }
    counts = write_subset_dataset(
        source_root=source_dataset_dir,
        output_root=output_root,
        split_indices=split_indices,
        batch_size=int(args.materialize_batch_size),
    )

    subset_train_ds = RegionalForecastDataset(str(output_root / "datasets" / "train.pt"))
    subset_val_ds = RegionalForecastDataset(str(output_root / "datasets" / "val.pt"))
    subset_test_ds = RegionalForecastDataset(str(output_root / "datasets" / "test.pt"))

    val_unseen = compute_unseen_rates(
        train_region_labels=subset_train_ds.tensor("region_labels").long(),
        train_label_mask=subset_train_ds.tensor("label_mask").bool(),
        eval_region_labels=subset_val_ds.tensor("region_labels").long(),
        eval_label_mask=subset_val_ds.tensor("label_mask").bool(),
        horizons_minutes=horizons_minutes,
    )
    test_unseen = compute_unseen_rates(
        train_region_labels=subset_train_ds.tensor("region_labels").long(),
        train_label_mask=subset_train_ds.tensor("label_mask").bool(),
        eval_region_labels=subset_test_ds.tensor("region_labels").long(),
        eval_label_mask=subset_test_ds.tensor("label_mask").bool(),
        horizons_minutes=horizons_minutes,
    )

    report: Dict[str, object] = {
        "source_dataset_dir": str(source_dataset_dir),
        "output_root": str(output_root),
        "seed": int(args.seed),
        "sampling": {
            "train_size_target": int(args.train_size),
            "val_size_target": int(args.val_size),
            "test_size_target": int(args.test_size),
            "min_train_support": int(args.min_train_support),
            "train_examples_per_label": int(args.train_examples_per_label),
        },
        "eligible_pool_sizes": {
            "train": int(len(eligible_train_indices)),
            "val": int(len(eligible_val_indices)),
            "test": int(len(eligible_test_indices)),
        },
        "selected_counts": counts,
        "required_label_counts_per_horizon": {
            str(horizons_minutes[horizon_idx]): int(len(required_labels[horizon_idx]))
            for horizon_idx in range(len(horizons_minutes))
        },
        "val_unseen_rates": val_unseen,
        "test_unseen_rates": test_unseen,
    }

    output_root.mkdir(parents=True, exist_ok=True)
    with open(output_root / "subset_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print("Dev subset dataset ready:")
    print(f"  source: {source_dataset_dir}")
    print(f"  output: {output_root / 'datasets'}")
    print(f"  train: {counts['train']}")
    print(f"  val: {counts['val']}")
    print(f"  test: {counts['test']}")
    print(f"  eligible_train: {len(eligible_train_indices)}")
    print(f"  eligible_val: {len(eligible_val_indices)}")
    print(f"  eligible_test: {len(eligible_test_indices)}")
    print(f"  report: {output_root / 'subset_report.json'}")


if __name__ == "__main__":
    main()
