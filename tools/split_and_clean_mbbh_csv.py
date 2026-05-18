from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import DefaultDict, Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from regional_forecast.config import load_config
from regional_forecast.data.cleaning import TrajectoryCleaningConfig, clean_trajectory_directory, config_from_project


RENAME_MAP = {
    "mbbh": "target_id",
    "positiontime": "timestamp",
    "course": "rotation",
}


def _safe_component(value: object, fallback: str = "unknown") -> str:
    if value is None or pd.isna(value):
        return fallback
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return fallback
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "_", text)
    cleaned = cleaned.strip("._-")
    return cleaned or fallback


def _resolve_inputs(cfg, input_arg: str | None, file_pattern: str) -> List[Path]:
    raw_path = input_arg or cfg.dataset.input_path
    input_path = cfg.resolve_path(raw_path)
    if input_path is None:
        raise ValueError("Input path could not be resolved.")

    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        paths = sorted(input_path.glob(file_pattern))
        if paths:
            return paths
        raise FileNotFoundError(f"No files matched {file_pattern!r} under {input_path}")

    raise FileNotFoundError(f"Input path not found: {input_path}")


def _standardize_chunk(chunk: pd.DataFrame, *, copy_target_type: bool) -> pd.DataFrame:
    frame = chunk.rename(columns=RENAME_MAP).copy()
    if copy_target_type:
        if "target_type" in frame.columns:
            frame["aircraft_type"] = frame["target_type"].fillna("").astype(str).str.strip().replace("", "unknown_type")
        else:
            frame["aircraft_type"] = "unknown_type"

    preferred = [
        "target_id",
        "timestamp",
        "longitude",
        "latitude",
        "altitude",
        "speed",
        "rotation",
        "aircraft_type",
        "target_type",
    ]
    ordered = [column for column in preferred if column in frame.columns]
    remaining = [column for column in frame.columns if column not in ordered]
    return frame[ordered + remaining]


def _split_one_source(
    source_path: Path,
    split_dir: Path,
    *,
    group_column: str,
    chunksize: int,
    copy_target_type: bool,
    prefix_with_source: bool,
) -> Dict[str, object]:
    rows_in = 0
    rows_out = 0
    groups_seen = 0
    files_written = 0

    file_prefix = f"{_safe_component(source_path.stem)}__" if prefix_with_source else ""
    output_rows: DefaultDict[Path, int] = defaultdict(int)
    output_sources: DefaultDict[Path, set[str]] = defaultdict(set)
    output_group_ids: Dict[Path, str] = {}
    initialized_outputs: set[Path] = set()

    header = set(pd.read_csv(source_path, nrows=0).columns)
    dtype_map = {
        column: "string"
        for column in {group_column, "target_type"}
        if column in header
    }
    reader = pd.read_csv(
        source_path,
        chunksize=max(1, int(chunksize)),
        dtype=dtype_map or None,
    )
    for chunk in reader:
        rows_in += int(len(chunk))
        if group_column not in chunk.columns:
            raise KeyError(f"Missing required group column {group_column!r} in {source_path.name}")

        chunk[group_column] = (
            chunk[group_column]
            .fillna("")
            .astype(str)
            .str.strip()
            .replace("", "unknown_mbbh")
        )

        for group_value, group in chunk.groupby(group_column, sort=False, dropna=False):
            groups_seen += 1
            safe_group = _safe_component(group_value, fallback="unknown_mbbh")
            output_path = split_dir / f"{file_prefix}{safe_group}.csv"
            standardized = _standardize_chunk(group, copy_target_type=copy_target_type)
            first_write = output_path not in initialized_outputs
            standardized.to_csv(
                output_path,
                index=False,
                mode="w" if first_write else "a",
                header=first_write,
            )
            initialized_outputs.add(output_path)
            output_rows[output_path] += int(len(standardized))
            output_sources[output_path].add(source_path.name)
            output_group_ids.setdefault(output_path, str(group_value))
            rows_out += int(len(standardized))
            files_written = len(output_rows)

    details = [
        {
            "output_file": str(path),
            "rows": int(rows),
            "group_id": output_group_ids.get(path),
            "source_files": sorted(output_sources.get(path, set())),
        }
        for path, rows in sorted(output_rows.items(), key=lambda item: str(item[0]))
    ]

    return {
        "source_file": source_path.name,
        "rows_in": rows_in,
        "rows_out": rows_out,
        "groups_seen": groups_seen,
        "files_written": files_written,
        "outputs": details,
    }


def _build_split_clean_config(cfg):
    clean_cfg = replace(
        cfg,
        columns=replace(
            cfg.columns,
            aircraft_id="target_id",
            trajectory_id="target_id",
            time="timestamp",
            heading="rotation",
            aircraft_type="aircraft_type",
        ),
    )
    if hasattr(cfg, "_config_path"):
        setattr(clean_cfg, "_config_path", getattr(cfg, "_config_path"))
    return clean_cfg


def _build_cleaning_config(
    cfg,
    *,
    sample_interval_minutes: int | None,
    mission_gap_minutes: float | None,
    max_smoothing_gap_minutes: float,
    max_impute_gap_minutes: float | None,
    max_point_speed_kts: float | None,
    max_bridge_speed_kts: float | None,
    spike_speed_kts: float,
    smoothing_window: int | None,
) -> TrajectoryCleaningConfig:
    cleaning = config_from_project(
        cfg,
        max_impute_gap_minutes=max_impute_gap_minutes,
        max_point_speed_kts=max_point_speed_kts,
        max_bridge_speed_kts=max_bridge_speed_kts,
        smoothing_window=smoothing_window,
    )
    if sample_interval_minutes is not None:
        cleaning.sample_interval_minutes = int(sample_interval_minutes)
    if mission_gap_minutes is not None:
        cleaning.mission_gap_minutes = float(mission_gap_minutes)
    cleaning.max_smoothing_gap_minutes = float(max_smoothing_gap_minutes)
    cleaning.spike_speed_kts = float(spike_speed_kts)
    return cleaning


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Split a mbbh-based single CSV into per-flight CSV files, normalize columns to the project schema, "
            "and then run the existing cleaning pipeline on the split directory."
        )
    )
    parser.add_argument("--config", type=str, default="config/region_forecast.yaml")
    parser.add_argument("--input", type=str, default=None, help="Input CSV file or directory. Defaults to dataset.input_path.")
    parser.add_argument("--split-output", type=str, default="data/split_csv", help="Directory for per-flight split CSV files.")
    parser.add_argument("--clean-output", type=str, default="data/cleaned_csv", help="Directory for cleaned CSV files.")
    parser.add_argument("--file-pattern", type=str, default=None, help="Used when --input points to a directory.")
    parser.add_argument("--chunksize", type=int, default=200000, help="Rows per read chunk while splitting a large CSV.")
    parser.add_argument("--copy-target-type", action="store_true", help="Copy target_type into aircraft_type for the cleaned output.")
    parser.add_argument("--split-only", action="store_true", help="Only split files; skip the cleaning stage.")
    parser.add_argument("--workers", type=int, default=1, help="Worker processes for the cleaning stage.")
    parser.add_argument("--sample-interval-minutes", type=int, default=None)
    parser.add_argument("--mission-gap-minutes", type=float, default=None)
    parser.add_argument("--max-smoothing-gap-minutes", type=float, default=10.0)
    parser.add_argument("--max-impute-gap-minutes", type=float, default=None)
    parser.add_argument("--max-point-speed-kts", type=float, default=None)
    parser.add_argument("--max-bridge-speed-kts", type=float, default=None)
    parser.add_argument("--spike-speed-kts", type=float, default=1200.0)
    parser.add_argument("--smoothing-window", type=int, default=None)
    parser.add_argument("--minimal-output", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    input_paths = _resolve_inputs(cfg, args.input, args.file_pattern or cfg.dataset.file_pattern)

    split_dir = cfg.resolve_path(args.split_output)
    clean_dir = cfg.resolve_path(args.clean_output)
    if split_dir is None or clean_dir is None:
        raise ValueError("Output directory could not be resolved.")
    if split_dir.resolve() == clean_dir.resolve():
        raise ValueError("--split-output and --clean-output must point to different directories.")

    split_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)

    prefix_with_source = len(input_paths) > 1
    source_summaries: List[Dict[str, object]] = []
    total_rows_in = 0
    total_rows_out = 0

    for source_path in input_paths:
        summary = _split_one_source(
            source_path,
            split_dir,
            group_column="mbbh",
            chunksize=args.chunksize,
            copy_target_type=bool(args.copy_target_type),
            prefix_with_source=prefix_with_source,
        )
        source_summaries.append(summary)
        total_rows_in += int(summary["rows_in"])
        total_rows_out += int(summary["rows_out"])

    split_summary_path = split_dir / "_split_summary.json"
    with open(split_summary_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "input_paths": [str(path) for path in input_paths],
                "split_output_dir": str(split_dir),
                "copy_target_type": bool(args.copy_target_type),
                "rows_in": total_rows_in,
                "rows_out": total_rows_out,
                "source_summaries": source_summaries,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Split files written to: {split_dir}")
    print(f"Split summary: {split_summary_path}")
    print(f"Split rows: input_rows={total_rows_in}, output_rows={total_rows_out}")

    if args.split_only:
        return

    cleaning = _build_cleaning_config(
        cfg,
        sample_interval_minutes=args.sample_interval_minutes,
        mission_gap_minutes=args.mission_gap_minutes,
        max_smoothing_gap_minutes=args.max_smoothing_gap_minutes,
        max_impute_gap_minutes=args.max_impute_gap_minutes,
        max_point_speed_kts=args.max_point_speed_kts,
        max_bridge_speed_kts=args.max_bridge_speed_kts,
        spike_speed_kts=args.spike_speed_kts,
        smoothing_window=args.smoothing_window,
    )
    cleaning.keep_diagnostics = not bool(args.minimal_output)

    summary = clean_trajectory_directory(
        input_dir=split_dir,
        output_dir=clean_dir,
        cfg=_build_split_clean_config(cfg),
        cleaning=cleaning,
        file_pattern="*.csv",
        workers=args.workers,
    )

    print(f"Cleaned files: {len(summary)}")
    print(f"Output directory: {clean_dir}")
    print(f"Summary: {clean_dir / '_cleaning_summary.csv'}")
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
