from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from regional_forecast.config import load_config
from regional_forecast.data.cleaning import (
    TrajectoryCleaningConfig,
    clean_trajectory_directory,
    config_from_project,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Clean raw aircraft trajectory CSV files into a new CSV directory. "
            "The pipeline performs anomaly detection, mission splitting, in-mission smoothing, "
            "speed/heading/vertical-rate recomputation, and conservative same-flight gap imputation."
        )
    )
    parser.add_argument("--config", type=str, default="config/region_forecast.yaml")
    parser.add_argument("--input", type=str, default=None, help="Input directory. Defaults to dataset.input_path.")
    parser.add_argument("--output", type=str, default="artifacts/cleaned_csv", help="Output directory for cleaned CSV files.")
    parser.add_argument("--file-pattern", type=str, default=None, help="Input glob pattern. Defaults to dataset.file_pattern.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for debugging on the first N files.")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes for directory cleaning.")
    parser.add_argument("--sample-interval-minutes", type=int, default=None, help="Imputed point interval. Defaults to dataset.sample_interval_minutes.")
    parser.add_argument("--mission-gap-minutes", type=float, default=None, help="Gap above this starts a new mission unless bridgeable.")
    parser.add_argument("--max-smoothing-gap-minutes", type=float, default=10.0, help="Do not smooth across larger gaps.")
    parser.add_argument("--max-impute-gap-minutes", type=float, default=None, help="Largest missing gap to fill inside a same-flight mission.")
    parser.add_argument("--max-point-speed-kts", type=float, default=None, help="Adjacent speed above this is treated as impossible.")
    parser.add_argument("--max-bridge-speed-kts", type=float, default=None, help="Average speed limit for reconnecting/filling a gap.")
    parser.add_argument("--spike-speed-kts", type=float, default=1200.0, help="Two-sided spike speed threshold.")
    parser.add_argument("--smoothing-window", type=int, default=None, help="Centered rolling smoothing window in points.")
    parser.add_argument("--minimal-output", action="store_true", help="Drop raw_* diagnostic columns from output CSVs.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    input_dir = cfg.resolve_path(args.input or cfg.dataset.input_path)
    output_dir = cfg.resolve_path(args.output)
    if input_dir is None:
        raise ValueError("Input directory could not be resolved.")
    if output_dir is None:
        raise ValueError("Output directory could not be resolved.")

    cleaning: TrajectoryCleaningConfig = config_from_project(
        cfg,
        max_impute_gap_minutes=args.max_impute_gap_minutes,
        max_point_speed_kts=args.max_point_speed_kts,
        max_bridge_speed_kts=args.max_bridge_speed_kts,
        smoothing_window=args.smoothing_window,
    )
    if args.sample_interval_minutes is not None:
        cleaning.sample_interval_minutes = int(args.sample_interval_minutes)
    if args.mission_gap_minutes is not None:
        cleaning.mission_gap_minutes = float(args.mission_gap_minutes)
    cleaning.max_smoothing_gap_minutes = float(args.max_smoothing_gap_minutes)
    cleaning.spike_speed_kts = float(args.spike_speed_kts)
    cleaning.keep_diagnostics = not bool(args.minimal_output)

    summary = clean_trajectory_directory(
        input_dir=Path(input_dir),
        output_dir=Path(output_dir),
        cfg=cfg,
        cleaning=cleaning,
        file_pattern=args.file_pattern or cfg.dataset.file_pattern,
        limit=args.limit,
        workers=args.workers,
    )

    print(f"Cleaned files: {len(summary)}")
    print(f"Output directory: {output_dir}")
    print(f"Summary: {Path(output_dir) / '_cleaning_summary.csv'}")
    print(
        "Totals: "
        f"input_rows={int(summary['input_rows'].sum())}, "
        f"output_rows={int(summary['output_rows'].sum())}, "
        f"spike_rows_removed={int(summary['spike_rows_removed'].sum())}, "
        f"imputed_rows={int(summary['imputed_rows'].sum())}, "
        f"missions={int(summary['missions'].sum())}"
    )


if __name__ == "__main__":
    main()
