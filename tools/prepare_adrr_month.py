from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import json
import shutil
import sys
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from regional_forecast.config import load_config
from regional_forecast.data.adrr_sharded import SHARD_DATASET_FORMAT, SHARD_DATASET_MANIFEST
from regional_forecast.data.trajectory_ops import (
    FT_TO_M,
    collapse_duplicate_times,
    numeric_series,
    recompute_motion,
    remove_spikes,
)


POINTS_CHUNK_SIZE = 500_000
DEFAULT_NUM_SHARDS = 128
ADRR_TIME_FORMAT = "%d-%m-%Y %H:%M:%S"
AIRAC_DATE_FORMAT = "%d-%b-%y"
DEFAULT_INPUT_DIR = "data/202403"
DEFAULT_OUTPUT_DIR = "data/adrr_202403_shards"
UNKNOWN_AIRCRAFT_TYPE = "unknown_type"
RAW_POINT_COLUMNS: Sequence[str] = (
    "ECTRL ID",
    "Sequence Number",
    "Time Over",
    "Flight Level",
    "Latitude",
    "Longitude",
)
STANDARDIZED_COLUMNS: Sequence[str] = (
    "target_id",
    "timestamp",
    "latitude",
    "longitude",
    "altitude",
    "speed",
    "rotation",
    "vrate",
    "aircraft_type",
    "callsign",
    "nationality",
    "observed_flag",
    "gap_flag",
)
GAP_BUCKETS: Sequence[Tuple[str, float, float]] = (
    ("gap_0_1", 0.0, 1.0),
    ("gap_1_2", 1.0, 2.0),
    ("gap_2_5", 2.0, 5.0),
    ("gap_5_10", 5.0, 10.0),
    ("gap_10_15", 10.0, 15.0),
    ("gap_15_30", 15.0, 30.0),
    ("gap_30_plus", 30.0, float("inf")),
)

_WORKER_METADATA_PATH: str | None = None
_WORKER_METADATA_DF: pd.DataFrame | None = None
_WORKER_AIRAC_PATH: str | None = None
_WORKER_AIRAC_WINDOWS: List[Tuple[str, pd.Timestamp, pd.Timestamp]] | None = None


@dataclass
class FlightSummary:
    input_rows: int = 0
    invalid_rows: int = 0
    duplicate_time_groups: int = 0
    duplicate_rows_removed: int = 0
    spike_rows_removed: int = 0
    output_rows: int = 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare one month of ADRR source tables into standardized Parquet shards for the regional forecaster."
        )
    )
    parser.add_argument("--config", type=str, default="config/region_forecast_adrr.yaml")
    parser.add_argument("--input-dir", type=str, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for standardized Parquet shards.")
    parser.add_argument("--clean-output", type=str, default=None, help="Alias for --output-dir.")
    parser.add_argument("--workers", type=int, default=1, help="Worker processes for shard standardization.")
    parser.add_argument("--num-shards", type=int, default=DEFAULT_NUM_SHARDS, help="Number of standardized shard files.")
    parser.add_argument("--chunksize", type=int, default=POINTS_CHUNK_SIZE, help="Rows per CSV chunk while partitioning raw points.")
    parser.add_argument("--max-flights", type=int, default=None, help="Optional smoke-test limit on the first N flights.")
    return parser.parse_args()


def _resolve_single_file(input_dir: Path, pattern: str) -> Path:
    matches = sorted(input_dir.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one file matching {pattern!r} under {input_dir}, found {len(matches)}.")
    return matches[0]


def _infer_month_tag(input_dir: Path, source_paths: Iterable[Path]) -> str:
    if re.fullmatch(r"\d{6}", input_dir.name):
        return input_dir.name
    for path in source_paths:
        match = re.search(r"(20\d{4})", path.name)
        if match:
            return match.group(1)
    raise ValueError(f"Could not infer YYYYMM month tag from {input_dir}.")


def _clean_text(value: object, default: str = "") -> str:
    if value is None or pd.isna(value):
        return default
    text = str(value).strip()
    return text if text else default


def _parse_adrr_series(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format=ADRR_TIME_FORMAT, utc=True, errors="coerce")


def _load_flight_metadata(path: Path) -> pd.DataFrame:
    usecols = [
        "ECTRL ID",
        "ADEP",
        "ADES",
        "ACTUAL OFF BLOCK TIME",
        "ACTUAL ARRIVAL TIME",
        "AC Type",
        "AC Operator",
        "AC Registration",
        "ICAO Flight Type",
        "STATFOR Market Segment",
        "Requested FL",
        "Actual Distance Flown (nm)",
    ]
    frame = pd.read_csv(path, usecols=usecols, dtype={"ECTRL ID": "string"}).rename(
        columns={
            "ECTRL ID": "target_id",
            "ADEP": "adrr_adep",
            "ADES": "adrr_ades",
            "ACTUAL OFF BLOCK TIME": "adrr_actual_off_block_time",
            "ACTUAL ARRIVAL TIME": "adrr_actual_arrival_time",
            "AC Type": "aircraft_type",
            "AC Operator": "adrr_operator",
            "AC Registration": "adrr_registration",
            "ICAO Flight Type": "adrr_icao_flight_type",
            "STATFOR Market Segment": "adrr_market_segment",
            "Requested FL": "adrr_requested_fl",
            "Actual Distance Flown (nm)": "adrr_actual_distance_nm",
        }
    ).copy()
    frame["target_id"] = frame["target_id"].fillna("").astype(str).str.strip()
    frame = frame[frame["target_id"].ne("")].copy()
    frame["aircraft_type"] = frame["aircraft_type"].fillna("").astype(str).str.strip().replace("", UNKNOWN_AIRCRAFT_TYPE)
    for column in (
        "adrr_adep",
        "adrr_ades",
        "adrr_operator",
        "adrr_registration",
        "adrr_icao_flight_type",
        "adrr_market_segment",
        "adrr_requested_fl",
    ):
        frame[column] = frame[column].fillna("").astype(str).str.strip()
    frame["adrr_actual_off_block_time"] = _parse_adrr_series(frame["adrr_actual_off_block_time"])
    frame["adrr_actual_arrival_time"] = _parse_adrr_series(frame["adrr_actual_arrival_time"])
    frame["adrr_actual_distance_nm"] = pd.to_numeric(frame["adrr_actual_distance_nm"], errors="coerce").astype(np.float32)
    frame = frame.drop_duplicates(subset=["target_id"], keep="last").reset_index(drop=True)
    return frame


def _load_airac_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path).rename(
        columns={
            "External ID": "airac_id",
            "Date From": "date_from",
            "Date To": "date_to",
        }
    )
    frame["airac_id"] = frame["airac_id"].fillna("").astype(str).str.strip()
    frame["date_from"] = pd.to_datetime(frame["date_from"], format=AIRAC_DATE_FORMAT, utc=True, errors="coerce")
    frame["date_to_exclusive"] = pd.to_datetime(frame["date_to"], format=AIRAC_DATE_FORMAT, utc=True, errors="coerce") + pd.Timedelta(days=1)
    frame = frame.dropna(subset=["date_from", "date_to_exclusive"]).copy()
    return frame[["airac_id", "date_from", "date_to_exclusive"]].sort_values("date_from").reset_index(drop=True)


def _get_worker_metadata(metadata_path: Path) -> pd.DataFrame:
    global _WORKER_METADATA_PATH, _WORKER_METADATA_DF
    resolved = str(metadata_path.resolve())
    if _WORKER_METADATA_DF is None or _WORKER_METADATA_PATH != resolved:
        frame = pd.read_parquet(metadata_path)
        _WORKER_METADATA_DF = frame.set_index("target_id", drop=False)
        _WORKER_METADATA_PATH = resolved
    return _WORKER_METADATA_DF


def _get_worker_airac_windows(airac_path: Path) -> List[Tuple[str, pd.Timestamp, pd.Timestamp]]:
    global _WORKER_AIRAC_PATH, _WORKER_AIRAC_WINDOWS
    resolved = str(airac_path.resolve())
    if _WORKER_AIRAC_WINDOWS is None or _WORKER_AIRAC_PATH != resolved:
        frame = pd.read_parquet(airac_path)
        _WORKER_AIRAC_WINDOWS = [
            (str(row.airac_id), pd.Timestamp(row.date_from), pd.Timestamp(row.date_to_exclusive))
            for row in frame.itertuples(index=False)
        ]
        _WORKER_AIRAC_PATH = resolved
    return _WORKER_AIRAC_WINDOWS


def _assign_airac_ids(timestamps: pd.Series, windows: Sequence[Tuple[str, pd.Timestamp, pd.Timestamp]]) -> pd.Series:
    result = pd.Series("", index=timestamps.index, dtype=object)
    for airac_id, start, end_exclusive in windows:
        mask = timestamps.ge(start) & timestamps.lt(end_exclusive)
        if mask.any():
            result.loc[mask] = airac_id
    return result


def _default_metadata_row() -> pd.Series:
    return pd.Series(
        {
            "target_id": "",
            "aircraft_type": UNKNOWN_AIRCRAFT_TYPE,
            "adrr_adep": "",
            "adrr_ades": "",
            "adrr_actual_off_block_time": pd.NaT,
            "adrr_actual_arrival_time": pd.NaT,
            "adrr_operator": "",
            "adrr_registration": "",
            "adrr_icao_flight_type": "",
            "adrr_market_segment": "",
            "adrr_requested_fl": "",
            "adrr_actual_distance_nm": np.float32(np.nan),
        }
    )


def _resolve_output_dir(cfg, args: argparse.Namespace, month_tag: str) -> Path:
    raw_output = args.output_dir or args.clean_output or DEFAULT_OUTPUT_DIR.replace("202403", month_tag)
    output_dir = cfg.resolve_path(raw_output)
    if output_dir is None:
        raise ValueError("Output directory could not be resolved.")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _clean_output_dir(output_dir: Path) -> None:
    for path in output_dir.glob("adrr_*_shard_*.parquet"):
        path.unlink()
    for path in output_dir.glob("*.pt"):
        path.unlink()
    for path in output_dir.glob("*.json"):
        path.unlink()
    for path in output_dir.glob("*.csv"):
        path.unlink()
    for path in output_dir.glob("*.parquet"):
        if path.name in {"flights_metadata.parquet", "airac_windows.parquet"}:
            path.unlink()
    tmp_dir = output_dir / "_tmp_raw_shards"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)


def _limit_chunk_by_flights(
    frame: pd.DataFrame,
    *,
    max_flights: Optional[int],
    groups_seen: int,
    current_flight_id: Optional[str],
) -> Tuple[pd.DataFrame, int, Optional[str], bool]:
    if max_flights is None or frame.empty:
        if not frame.empty:
            current_flight_id = str(frame["ECTRL ID"].iloc[-1])
        return frame, groups_seen, current_flight_id, False

    ids = frame["ECTRL ID"].to_numpy(dtype=object, copy=False)
    starts = np.concatenate(([0], np.flatnonzero(ids[1:] != ids[:-1]) + 1))
    ends = np.concatenate((starts[1:], [len(ids)]))
    keep_end = len(ids)
    truncated = False

    for start, end in zip(starts.tolist(), ends.tolist()):
        flight_id = str(ids[start])
        if flight_id != current_flight_id:
            if groups_seen >= int(max_flights):
                keep_end = int(start)
                truncated = True
                break
            groups_seen += 1
            current_flight_id = flight_id
        keep_end = int(end)

    return frame.iloc[:keep_end].copy(), groups_seen, current_flight_id, truncated


def _partition_points_to_raw_shards(
    points_path: Path,
    raw_dir: Path,
    *,
    month_tag: str,
    num_shards: int,
    chunksize: int,
    max_flights: Optional[int],
) -> Dict[str, object]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows_in = 0
    rows_out = 0
    groups_seen = 0
    current_flight_id: Optional[str] = None
    truncated = False

    read_options = {
        "usecols": list(RAW_POINT_COLUMNS),
        "chunksize": int(chunksize),
        "dtype": {
            "ECTRL ID": "string",
            "Sequence Number": "string",
            "Time Over": "string",
            "Flight Level": "string",
            "Latitude": "string",
            "Longitude": "string",
        },
    }

    for chunk in pd.read_csv(points_path, **read_options):
        rows_in += int(len(chunk))
        chunk["ECTRL ID"] = chunk["ECTRL ID"].fillna("").astype(str).str.strip()
        chunk = chunk[chunk["ECTRL ID"].ne("")].copy()
        if chunk.empty:
            continue

        chunk, groups_seen, current_flight_id, hit_limit = _limit_chunk_by_flights(
            chunk,
            max_flights=max_flights,
            groups_seen=groups_seen,
            current_flight_id=current_flight_id,
        )
        if chunk.empty and hit_limit:
            truncated = True
            break
        if chunk.empty:
            continue

        shard_ids = (
            pd.util.hash_pandas_object(chunk["ECTRL ID"], index=False)
            .to_numpy(dtype=np.uint64, copy=False)
            % np.uint64(max(1, int(num_shards)))
        ).astype(np.int32, copy=False)
        chunk["_shard_id"] = shard_ids
        rows_out += int(len(chunk))

        for shard_id, shard_frame in chunk.groupby("_shard_id", sort=False):
            output_path = raw_dir / f"adrr_{month_tag}_raw_shard_{int(shard_id):03d}.csv"
            shard_frame = shard_frame[list(RAW_POINT_COLUMNS)]
            shard_frame.to_csv(output_path, mode="a", header=not output_path.exists(), index=False)

        if hit_limit:
            truncated = True
            break

    files_written = len(list(raw_dir.glob(f"adrr_{month_tag}_raw_shard_*.csv")))
    return {
        "source_file": points_path.name,
        "rows_in": int(rows_in),
        "rows_out": int(rows_out),
        "groups_seen": int(groups_seen),
        "files_written": int(files_written),
        "truncated_by_max_flights": bool(truncated),
    }


def _gap_bucket_stats(gap_minutes: pd.Series) -> Dict[str, object]:
    positive = pd.to_numeric(gap_minutes, errors="coerce").fillna(0.0)
    positive = positive[positive > 0.0]
    stats: Dict[str, object] = {
        "gap_observations": int(len(positive)),
        "gap_minutes_sum": float(positive.sum()) if not positive.empty else 0.0,
        "gap_minutes_max": float(positive.max()) if not positive.empty else 0.0,
    }
    for name, left, right in GAP_BUCKETS:
        if positive.empty:
            stats[name] = 0
        elif np.isinf(right):
            stats[name] = int((positive >= left).sum())
        else:
            stats[name] = int(((positive >= left) & (positive < right)).sum())
    return stats


def _standardize_flight_group(
    frame: pd.DataFrame,
    metadata: pd.Series,
    airac_windows: Sequence[Tuple[str, pd.Timestamp, pd.Timestamp]],
) -> Tuple[pd.DataFrame, FlightSummary]:
    summary = FlightSummary(input_rows=int(len(frame)))
    working = frame.copy()
    flight_id = _clean_text(working["ECTRL ID"].iloc[0], "")
    working["timestamp"] = _parse_adrr_series(working["Time Over"])
    working["latitude"] = numeric_series(working, "Latitude")
    working["longitude"] = numeric_series(working, "Longitude")
    working["flight_level_raw"] = numeric_series(working, "Flight Level", default=np.nan)
    working["sequence_number"] = pd.to_numeric(working["Sequence Number"], errors="coerce")
    working["altitude"] = np.clip(working["flight_level_raw"].fillna(0.0) * 100.0 * FT_TO_M, 0.0, None)

    valid_mask = (
        working["timestamp"].notna()
        & working["latitude"].between(-90.0, 90.0)
        & working["longitude"].between(-180.0, 180.0)
    )
    summary.invalid_rows = int((~valid_mask).sum())
    working = working.loc[valid_mask].copy()
    if working.empty:
        return pd.DataFrame(columns=list(STANDARDIZED_COLUMNS)), summary

    working = working.sort_values(["sequence_number", "timestamp"]).reset_index(drop=True)
    working, duplicate_groups, duplicate_rows_removed = collapse_duplicate_times(
        working,
        group_cols=["ECTRL ID"],
        time_col="timestamp",
        numeric_cols=["sequence_number", "flight_level_raw", "latitude", "longitude", "altitude"],
        quality_col="quality_flags",
    )
    summary.duplicate_time_groups = int(duplicate_groups)
    summary.duplicate_rows_removed = int(duplicate_rows_removed)

    working, spikes_removed = remove_spikes(
        working,
        time_col="timestamp",
        lat_col="latitude",
        lon_col="longitude",
    )
    summary.spike_rows_removed = int(spikes_removed)
    if working.empty:
        return pd.DataFrame(columns=list(STANDARDIZED_COLUMNS)), summary

    working = recompute_motion(
        working,
        time_col="timestamp",
        lat_col="latitude",
        lon_col="longitude",
        altitude_col="altitude",
        speed_col="speed",
        heading_col="rotation",
        vrate_col="vrate",
    )
    working = working.sort_values("timestamp").reset_index(drop=True)
    working["observed_flag"] = np.float32(1.0)
    working["gap_flag"] = np.float32(0.0)
    working["aircraft_type"] = _clean_text(metadata.get("aircraft_type", UNKNOWN_AIRCRAFT_TYPE), UNKNOWN_AIRCRAFT_TYPE)
    working["callsign"] = ""
    working["nationality"] = ""
    working["target_id"] = flight_id
    working["timestamp"] = pd.to_datetime(working["timestamp"], utc=True, errors="coerce")
    working["aircraft_type"] = working["aircraft_type"].astype(str)
    summary.output_rows = int(len(working))

    standardized = working[list(STANDARDIZED_COLUMNS)].copy()
    standardized["latitude"] = standardized["latitude"].astype(np.float32)
    standardized["longitude"] = standardized["longitude"].astype(np.float32)
    standardized["altitude"] = standardized["altitude"].astype(np.float32)
    standardized["speed"] = standardized["speed"].astype(np.float32)
    standardized["rotation"] = standardized["rotation"].astype(np.float32)
    standardized["vrate"] = standardized["vrate"].astype(np.float32)
    standardized["observed_flag"] = standardized["observed_flag"].astype(np.float32)
    standardized["gap_flag"] = standardized["gap_flag"].astype(np.float32)
    return standardized, summary


def _standardize_raw_shard(
    raw_path: Path,
    output_path: Path,
    metadata_df: pd.DataFrame,
    airac_windows: Sequence[Tuple[str, pd.Timestamp, pd.Timestamp]],
) -> Dict[str, object]:
    frame = pd.read_csv(raw_path, dtype="string")
    if frame.empty:
        return {
            "raw_file": raw_path.name,
            "shard_file": output_path.name,
            "input_rows": 0,
            "output_rows": 0,
            "processed_flights": 0,
            "written_flights": 0,
            "invalid_rows": 0,
            "duplicate_time_groups": 0,
            "duplicate_rows_removed": 0,
            "spike_rows_removed": 0,
            "gap_observations": 0,
            "gap_minutes_sum": 0.0,
            "gap_minutes_max": 0.0,
        }

    frame["ECTRL ID"] = frame["ECTRL ID"].fillna("").astype(str).str.strip()
    frame = frame[frame["ECTRL ID"].ne("")].copy()
    if frame.empty:
        return {
            "raw_file": raw_path.name,
            "shard_file": output_path.name,
            "input_rows": 0,
            "output_rows": 0,
            "processed_flights": 0,
            "written_flights": 0,
            "invalid_rows": 0,
            "duplicate_time_groups": 0,
            "duplicate_rows_removed": 0,
            "spike_rows_removed": 0,
            "gap_observations": 0,
            "gap_minutes_sum": 0.0,
            "gap_minutes_max": 0.0,
        }

    standardized_groups: List[pd.DataFrame] = []
    summary_rows: List[FlightSummary] = []
    gap_stats_total = {
        "gap_observations": 0,
        "gap_minutes_sum": 0.0,
        "gap_minutes_max": 0.0,
        **{name: 0 for name, _, _ in GAP_BUCKETS},
    }

    default_metadata = _default_metadata_row()
    for flight_id, group in frame.groupby("ECTRL ID", sort=False):
        metadata = metadata_df.loc[flight_id] if flight_id in metadata_df.index else default_metadata
        standardized, stats = _standardize_flight_group(group, metadata, airac_windows)
        summary_rows.append(stats)
        if standardized.empty:
            continue
        gap_minutes = standardized["timestamp"].diff().dt.total_seconds().div(60.0).fillna(0.0).clip(lower=0.0)
        gap_stats = _gap_bucket_stats(gap_minutes)
        for key, value in gap_stats.items():
            if key == "gap_minutes_max":
                gap_stats_total[key] = max(float(gap_stats_total[key]), float(value))
            else:
                gap_stats_total[key] = float(gap_stats_total.get(key, 0.0)) + float(value)
        standardized_groups.append(standardized)

    if standardized_groups:
        standardized_frame = pd.concat(standardized_groups, ignore_index=True)
        standardized_frame.to_parquet(output_path, index=False)
        output_rows = int(len(standardized_frame))
    else:
        if output_path.exists():
            output_path.unlink()
        output_rows = 0

    aggregate = {
        "raw_file": raw_path.name,
        "shard_file": output_path.name,
        "input_rows": int(sum(item.input_rows for item in summary_rows)),
        "output_rows": int(output_rows),
        "processed_flights": int(len(summary_rows)),
        "written_flights": int(len(standardized_groups)),
        "invalid_rows": int(sum(item.invalid_rows for item in summary_rows)),
        "duplicate_time_groups": int(sum(item.duplicate_time_groups for item in summary_rows)),
        "duplicate_rows_removed": int(sum(item.duplicate_rows_removed for item in summary_rows)),
        "spike_rows_removed": int(sum(item.spike_rows_removed for item in summary_rows)),
        "gap_observations": int(gap_stats_total["gap_observations"]),
        "gap_minutes_sum": float(gap_stats_total["gap_minutes_sum"]),
        "gap_minutes_max": float(gap_stats_total["gap_minutes_max"]),
    }
    for name, _, _ in GAP_BUCKETS:
        aggregate[name] = int(gap_stats_total[name])
    return aggregate


def _standardize_raw_shard_worker(args: Tuple[str, str, str, str]) -> Dict[str, object]:
    raw_path_text, output_path_text, metadata_path_text, airac_path_text = args
    metadata_df = _get_worker_metadata(Path(metadata_path_text))
    airac_windows = _get_worker_airac_windows(Path(airac_path_text))
    return _standardize_raw_shard(Path(raw_path_text), Path(output_path_text), metadata_df, airac_windows)


def _standardize_raw_shards(
    raw_dir: Path,
    output_dir: Path,
    *,
    month_tag: str,
    workers: int,
) -> pd.DataFrame:
    raw_paths = sorted(raw_dir.glob(f"adrr_{month_tag}_raw_shard_*.csv"))
    if not raw_paths:
        raise FileNotFoundError(f"No raw shard files were created under {raw_dir}")

    metadata_path = output_dir / "flights_metadata.parquet"
    airac_path = output_dir / "airac_windows.parquet"
    tasks = []
    for raw_path in raw_paths:
        shard_suffix = raw_path.stem.split("_")[-1]
        output_path = output_dir / f"adrr_{month_tag}_shard_{shard_suffix}.parquet"
        tasks.append((str(raw_path), str(output_path), str(metadata_path), str(airac_path)))

    worker_count = max(1, int(workers))
    if worker_count <= 1 or len(tasks) <= 1:
        rows = [_standardize_raw_shard_worker(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            rows = list(executor.map(_standardize_raw_shard_worker, tasks))

    summary = pd.DataFrame(rows).sort_values("raw_file").reset_index(drop=True)
    summary.to_csv(output_dir / "_prepare_summary.csv", index=False)
    return summary


def _build_dataset_manifest(
    output_dir: Path,
    *,
    month_tag: str,
    flights_path: Path,
    points_path: Path,
    airac_path: Path,
    num_shards: int,
) -> Dict[str, object]:
    manifest = {
        "format": SHARD_DATASET_FORMAT,
        "month_tag": month_tag,
        "input_format": "parquet",
        "file_pattern": f"adrr_{month_tag}_shard_*.parquet",
        "num_shards": int(num_shards),
        "standardized_columns": list(STANDARDIZED_COLUMNS),
        "flights_metadata_path": "flights_metadata.parquet",
        "airac_windows_path": "airac_windows.parquet",
        "source_files": {
            "flights": flights_path.name,
            "points": points_path.name,
            "airac": airac_path.name,
        },
    }
    with open(output_dir / SHARD_DATASET_MANIFEST, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return manifest


def _write_prepare_summary(
    output_dir: Path,
    *,
    month_tag: str,
    workers: int,
    num_shards: int,
    max_flights: Optional[int],
    partition_summary: Dict[str, object],
    standardize_summary: pd.DataFrame,
    manifest: Dict[str, object],
) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "month_tag": month_tag,
        "output_dir": str(output_dir),
        "workers": int(max(1, int(workers))),
        "num_shards": int(num_shards),
        "max_flights": int(max_flights) if max_flights is not None else None,
        "partition_rows_in": int(partition_summary["rows_in"]),
        "partition_rows_out": int(partition_summary["rows_out"]),
        "partition_groups_seen": int(partition_summary["groups_seen"]),
        "partition_files_written": int(partition_summary["files_written"]),
        "truncated_by_max_flights": bool(partition_summary["truncated_by_max_flights"]),
        "processed_shards": int(len(standardize_summary)),
        "processed_flights": int(standardize_summary["processed_flights"].sum()) if not standardize_summary.empty else 0,
        "written_flights": int(standardize_summary["written_flights"].sum()) if not standardize_summary.empty else 0,
        "input_rows": int(standardize_summary["input_rows"].sum()) if not standardize_summary.empty else 0,
        "output_rows": int(standardize_summary["output_rows"].sum()) if not standardize_summary.empty else 0,
        "invalid_rows": int(standardize_summary["invalid_rows"].sum()) if not standardize_summary.empty else 0,
        "duplicate_time_groups": int(standardize_summary["duplicate_time_groups"].sum()) if not standardize_summary.empty else 0,
        "duplicate_rows_removed": int(standardize_summary["duplicate_rows_removed"].sum()) if not standardize_summary.empty else 0,
        "spike_rows_removed": int(standardize_summary["spike_rows_removed"].sum()) if not standardize_summary.empty else 0,
        "dataset_manifest": manifest,
        "adrr_config_hint": {
            "dataset.input_path": str(output_dir).replace("\\", "/"),
            "dataset.input_format": "parquet",
            "dataset.file_pattern": manifest["file_pattern"],
        },
    }
    if not standardize_summary.empty:
        summary["gap_observations"] = int(standardize_summary["gap_observations"].sum())
        summary["gap_minutes_sum"] = float(standardize_summary["gap_minutes_sum"].sum())
        summary["gap_minutes_max"] = float(standardize_summary["gap_minutes_max"].max())
        for name, _, _ in GAP_BUCKETS:
            summary[name] = int(standardize_summary[name].sum())
    else:
        summary["gap_observations"] = 0
        summary["gap_minutes_sum"] = 0.0
        summary["gap_minutes_max"] = 0.0
        for name, _, _ in GAP_BUCKETS:
            summary[name] = 0

    summary["mean_gap_minutes"] = (
        float(summary["gap_minutes_sum"]) / float(summary["gap_observations"])
        if int(summary["gap_observations"]) > 0
        else 0.0
    )

    with open(output_dir / "_prepare_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return summary


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    input_dir = cfg.resolve_path(args.input_dir)
    if input_dir is None or not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    flights_path = _resolve_single_file(Path(input_dir), "Flights_*.csv")
    points_path = _resolve_single_file(Path(input_dir), "Flight_Points_Actual_*.csv")
    airac_path = _resolve_single_file(Path(input_dir), "AIRAC_*.csv")
    month_tag = _infer_month_tag(Path(input_dir), [flights_path, points_path, airac_path])

    output_dir = _resolve_output_dir(cfg, args, month_tag)
    _clean_output_dir(output_dir)
    raw_dir = output_dir / "_tmp_raw_shards"

    metadata_df = _load_flight_metadata(flights_path)
    metadata_df.to_parquet(output_dir / "flights_metadata.parquet", index=False)
    airac_df = _load_airac_frame(airac_path)
    airac_df.to_parquet(output_dir / "airac_windows.parquet", index=False)

    partition_summary = _partition_points_to_raw_shards(
        points_path=points_path,
        raw_dir=raw_dir,
        month_tag=month_tag,
        num_shards=args.num_shards,
        chunksize=args.chunksize,
        max_flights=args.max_flights,
    )
    print(
        "Partition totals: "
        f"rows_in={partition_summary['rows_in']}, "
        f"rows_out={partition_summary['rows_out']}, "
        f"flights={partition_summary['groups_seen']}, "
        f"raw_shards={partition_summary['files_written']}"
    )

    standardize_summary = _standardize_raw_shards(
        raw_dir=raw_dir,
        output_dir=output_dir,
        month_tag=month_tag,
        workers=args.workers,
    )
    shutil.rmtree(raw_dir, ignore_errors=True)

    manifest = _build_dataset_manifest(
        output_dir,
        month_tag=month_tag,
        flights_path=flights_path,
        points_path=points_path,
        airac_path=airac_path,
        num_shards=args.num_shards,
    )
    prepare_summary = _write_prepare_summary(
        output_dir,
        month_tag=month_tag,
        workers=args.workers,
        num_shards=args.num_shards,
        max_flights=args.max_flights,
        partition_summary=partition_summary,
        standardize_summary=standardize_summary,
        manifest=manifest,
    )

    print(f"Standardized shards: {len(list(output_dir.glob(manifest['file_pattern'])))}")
    print(f"Output directory: {output_dir}")
    print(f"Prepare summary: {output_dir / '_prepare_summary.json'}")
    print(f"Dataset manifest: {output_dir / SHARD_DATASET_MANIFEST}")
    print(f"Dataset file pattern: {manifest['file_pattern']}")
    print(f"Output rows: {prepare_summary['output_rows']}")


if __name__ == "__main__":
    main()
