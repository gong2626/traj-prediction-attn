from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from regional_forecast.config import ProjectConfig, load_config, load_config_dict
from regional_forecast.data.aoi import hash_fraction
from regional_forecast.data.preprocess import (
    CONTEXT_FEATURES,
    MASK_FEATURES,
    NUMERIC_FEATURES,
    _is_ground_state,
    _numeric_series,
    _resample_single_segment,
    attach_aircraft_metadata,
)
from regional_forecast.data.regionizer import H3Regionizer
from regional_forecast.evaluation.spatial_refinement import H3NeighborProbabilityRefiner
from regional_forecast.models.baseline import RegionalMarkovBaseline
from regional_forecast.models.forecast_model import (
    AttentionHierarchicalH3RegionalForecaster,
    HierarchicalH3RegionalForecaster,
    HorizonRouterTransformerH3RegionalForecaster,
    HorizonDecoderHierarchicalH3RegionalForecaster,
    LongHorizonRegionalForecaster,
    MultiscaleGatedLocalTransformerH3RegionalForecaster,
    MultiscaleLocalCausalTransformerH3RegionalForecaster,
    MultiscaleTransformerH3RegionalForecaster,
    TransformerHorizonDecoderHierarchicalH3RegionalForecaster,
)
from regional_forecast.reachability import ReachabilityFilter
from regional_forecast.registry import resolve_registered_model


def _resolve_device(cfg: ProjectConfig) -> torch.device:
    if cfg.training.device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _read_input_file(path: Path, input_format: str | None) -> pd.DataFrame:
    if input_format and input_format.lower() != "auto":
        fmt = input_format.lower()
    else:
        suffix = path.suffix.lower()
        fmt = {
            ".csv": "csv",
            ".parquet": "parquet",
            ".json": "json",
        }.get(suffix, "csv")

    if fmt == "csv":
        return pd.read_csv(path)
    if fmt == "parquet":
        return pd.read_parquet(path)
    if fmt == "json":
        return pd.read_json(path)
    raise ValueError(f"Unsupported input format: {fmt}")


def _infer_aircraft_type(frame: pd.DataFrame, cfg: ProjectConfig, aircraft_type: Optional[str]) -> Optional[str]:
    if aircraft_type is not None and str(aircraft_type).strip():
        return str(aircraft_type).strip()
    if cfg.columns.aircraft_type in frame.columns:
        non_empty_types = frame[cfg.columns.aircraft_type].dropna().astype(str).str.strip()
        non_empty_types = non_empty_types[non_empty_types.ne("")]
        if not non_empty_types.empty:
            return str(non_empty_types.iloc[-1])
    return None


def _child_count_by_coarse_per_horizon(metadata: Dict[str, object]) -> List[List[int]]:
    if "child_count_by_coarse_per_horizon" in metadata:
        return [list(map(int, counts)) for counts in metadata["child_count_by_coarse_per_horizon"]]  # type: ignore[index]
    return [
        [max(1, len(children)) for children in horizon_meta["coarse_to_children"]] + [1]
        for horizon_meta in metadata["hierarchy_per_horizon"]  # type: ignore[index]
    ]


def _load_model_bundle(
    runtime_cfg: ProjectConfig,
    entry: Dict[str, Any],
) -> tuple[str, object, Dict[str, Any], ProjectConfig]:
    checkpoint_path = Path(entry["checkpoint_path"])
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata: Dict[str, Any] = checkpoint["metadata"]
    train_cfg = load_config_dict(checkpoint.get("config", {}))
    train_cfg.columns = runtime_cfg.columns
    train_cfg.dataset.aircraft_type_map_path = runtime_cfg.dataset.aircraft_type_map_path
    train_cfg.dataset.default_aircraft_type = runtime_cfg.dataset.default_aircraft_type
    train_cfg.evaluation = runtime_cfg.evaluation
    train_cfg.training.device = runtime_cfg.training.device

    model_type = checkpoint["model_type"]
    if model_type == "markov":
        model = RegionalMarkovBaseline.from_state_dict(checkpoint["model_state"])
        return model_type, model, metadata, train_cfg

    if model_type == "gru":
        model = LongHorizonRegionalForecaster(
            input_num_region_classes=metadata["input_num_region_classes"],
            num_region_classes_per_horizon=metadata["num_region_classes_per_horizon"],
            num_status_classes=metadata["num_status_classes"],
            num_numeric_features=metadata["num_numeric_features"],
            num_context_features=metadata["num_context_features"],
            config=train_cfg.model,
        )
    elif model_type == "hierarchical_gru":
        model = HierarchicalH3RegionalForecaster(
            input_num_region_classes=metadata["input_num_region_classes"],
            input_parent_num_region_classes=metadata["input_parent_num_region_classes"],
            num_coarse_region_classes_per_horizon=metadata["num_coarse_region_classes_per_horizon"],
            child_count_by_coarse_per_horizon=_child_count_by_coarse_per_horizon(metadata),
            max_children_per_horizon=metadata["max_children_per_horizon"],
            num_status_classes=metadata["num_status_classes"],
            num_numeric_features=metadata["num_numeric_features"],
            num_mask_features=metadata["num_mask_features"],
            num_context_features=metadata["num_context_features"],
            config=train_cfg.model,
        )
    elif model_type == "attention_hierarchical_gru":
        model = AttentionHierarchicalH3RegionalForecaster(
            input_num_region_classes=metadata["input_num_region_classes"],
            input_parent_num_region_classes=metadata["input_parent_num_region_classes"],
            num_coarse_region_classes_per_horizon=metadata["num_coarse_region_classes_per_horizon"],
            child_count_by_coarse_per_horizon=_child_count_by_coarse_per_horizon(metadata),
            max_children_per_horizon=metadata["max_children_per_horizon"],
            num_status_classes=metadata["num_status_classes"],
            num_numeric_features=metadata["num_numeric_features"],
            num_mask_features=metadata["num_mask_features"],
            num_context_features=metadata["num_context_features"],
            config=train_cfg.model,
        )
    elif model_type == "horizon_decoder_gru":
        model = HorizonDecoderHierarchicalH3RegionalForecaster(
            input_num_region_classes=metadata["input_num_region_classes"],
            input_parent_num_region_classes=metadata["input_parent_num_region_classes"],
            num_coarse_region_classes_per_horizon=metadata["num_coarse_region_classes_per_horizon"],
            child_count_by_coarse_per_horizon=_child_count_by_coarse_per_horizon(metadata),
            max_children_per_horizon=metadata["max_children_per_horizon"],
            num_status_classes=metadata["num_status_classes"],
            num_numeric_features=metadata["num_numeric_features"],
            num_mask_features=metadata["num_mask_features"],
            num_context_features=metadata["num_context_features"],
            config=train_cfg.model,
        )
    elif model_type == "transformer_horizon_decoder":
        model = TransformerHorizonDecoderHierarchicalH3RegionalForecaster(
            input_num_region_classes=metadata["input_num_region_classes"],
            input_parent_num_region_classes=metadata["input_parent_num_region_classes"],
            num_coarse_region_classes_per_horizon=metadata["num_coarse_region_classes_per_horizon"],
            child_count_by_coarse_per_horizon=_child_count_by_coarse_per_horizon(metadata),
            max_children_per_horizon=metadata["max_children_per_horizon"],
            num_status_classes=metadata["num_status_classes"],
            num_numeric_features=metadata["num_numeric_features"],
            num_mask_features=metadata["num_mask_features"],
            num_context_features=metadata["num_context_features"],
            max_history_steps=int(metadata["history_steps"]),
            config=train_cfg.model,
        )
    elif model_type == "multiscale_transformer_h3":
        model = MultiscaleTransformerH3RegionalForecaster(
            input_num_region_classes=metadata["input_num_region_classes"],
            input_parent_num_region_classes=metadata["input_parent_num_region_classes"],
            num_coarse_region_classes_per_horizon=metadata["num_coarse_region_classes_per_horizon"],
            child_count_by_coarse_per_horizon=_child_count_by_coarse_per_horizon(metadata),
            max_children_per_horizon=metadata["max_children_per_horizon"],
            num_status_classes=metadata["num_status_classes"],
            num_numeric_features=metadata["num_numeric_features"],
            num_mask_features=metadata["num_mask_features"],
            num_context_features=metadata["num_context_features"],
            max_history_steps=int(metadata["history_steps"]),
            config=train_cfg.model,
        )
    elif model_type == "multiscale_transformer_h3_local_causal":
        model = MultiscaleLocalCausalTransformerH3RegionalForecaster(
            input_num_region_classes=metadata["input_num_region_classes"],
            input_parent_num_region_classes=metadata["input_parent_num_region_classes"],
            num_coarse_region_classes_per_horizon=metadata["num_coarse_region_classes_per_horizon"],
            child_count_by_coarse_per_horizon=_child_count_by_coarse_per_horizon(metadata),
            max_children_per_horizon=metadata["max_children_per_horizon"],
            num_status_classes=metadata["num_status_classes"],
            num_numeric_features=metadata["num_numeric_features"],
            num_mask_features=metadata["num_mask_features"],
            num_context_features=metadata["num_context_features"],
            max_history_steps=int(metadata["history_steps"]),
            config=train_cfg.model,
        )
    elif model_type == "multiscale_transformer_h3_gated_local":
        model = MultiscaleGatedLocalTransformerH3RegionalForecaster(
            input_num_region_classes=metadata["input_num_region_classes"],
            input_parent_num_region_classes=metadata["input_parent_num_region_classes"],
            num_coarse_region_classes_per_horizon=metadata["num_coarse_region_classes_per_horizon"],
            child_count_by_coarse_per_horizon=_child_count_by_coarse_per_horizon(metadata),
            max_children_per_horizon=metadata["max_children_per_horizon"],
            num_status_classes=metadata["num_status_classes"],
            num_numeric_features=metadata["num_numeric_features"],
            num_mask_features=metadata["num_mask_features"],
            num_context_features=metadata["num_context_features"],
            max_history_steps=int(metadata["history_steps"]),
            config=train_cfg.model,
        )
    elif model_type == "horizon_router_transformer_h3":
        model = HorizonRouterTransformerH3RegionalForecaster(
            input_num_region_classes=metadata["input_num_region_classes"],
            input_parent_num_region_classes=metadata["input_parent_num_region_classes"],
            num_coarse_region_classes_per_horizon=metadata["num_coarse_region_classes_per_horizon"],
            child_count_by_coarse_per_horizon=_child_count_by_coarse_per_horizon(metadata),
            max_children_per_horizon=metadata["max_children_per_horizon"],
            num_status_classes=metadata["num_status_classes"],
            num_numeric_features=metadata["num_numeric_features"],
            num_mask_features=metadata["num_mask_features"],
            num_context_features=metadata["num_context_features"],
            max_history_steps=int(metadata["history_steps"]),
            config=train_cfg.model,
        )
    else:
        raise ValueError(f"Unsupported checkpoint model_type: {model_type}")

    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model_type, model, metadata, train_cfg


def _select_trajectory(
    frame: pd.DataFrame,
    cfg: ProjectConfig,
    input_path: Path,
    trajectory_id: Optional[str],
) -> tuple[pd.DataFrame, str]:
    traj_col = cfg.columns.trajectory_id
    if traj_col not in frame.columns:
        frame = frame.copy()
        frame[traj_col] = input_path.stem

    frame[traj_col] = frame[traj_col].fillna(input_path.stem).astype(str)
    frame[cfg.columns.time] = pd.to_datetime(frame[cfg.columns.time], utc=True, errors="coerce")
    frame = frame.dropna(subset=[cfg.columns.time, cfg.columns.lat, cfg.columns.lon]).copy()
    if frame.empty:
        raise RuntimeError("No valid trajectory rows remain after parsing timestamps and coordinates.")

    if trajectory_id is not None:
        selected = frame[frame[traj_col].astype(str) == str(trajectory_id)].copy()
        if selected.empty:
            raise RuntimeError(f"No rows found for trajectory_id={trajectory_id}")
        return selected, str(trajectory_id)

    unique_ids = frame[traj_col].drop_duplicates().astype(str).tolist()
    if len(unique_ids) == 1:
        return frame.copy(), unique_ids[0]

    latest_by_traj = (
        frame.groupby(traj_col, sort=False)[cfg.columns.time]
        .max()
        .sort_values(ascending=False)
    )
    selected_id = str(latest_by_traj.index[0])
    return frame[frame[traj_col].astype(str) == selected_id].copy(), selected_id


def _resample_for_inference(frame: pd.DataFrame, cfg: ProjectConfig, source_name: str) -> pd.DataFrame:
    frame = frame.sort_values(cfg.columns.time).copy()
    frame["_source_file"] = source_name
    frame["_aircraft_id"] = frame.get("_aircraft_id", pd.Series(source_name, index=frame.index)).astype(str)

    if len(frame) < 2:
        single = frame.copy().reset_index(drop=True)
        single["observed_flag"] = 1.0
        single["gap_flag"] = 0.0
        return single

    resampled = _resample_single_segment(frame, cfg)
    if resampled.empty:
        raise RuntimeError("Input trajectory could not be resampled into a usable sequence.")
    return resampled


def _attach_region_and_features(
    frame: pd.DataFrame,
    cfg: ProjectConfig,
    metadata: Dict[str, Any],
) -> pd.DataFrame:
    frame = frame.copy()
    input_regionizer = H3Regionizer.from_metadata(metadata["input_regionizer"])
    parent_regionizer = H3Regionizer.from_metadata(metadata["input_parent_regionizer"])

    transformed = input_regionizer.transform_frame(frame, cfg.columns.lat, cfg.columns.lon)
    frame["region_id"] = transformed["region_id"].astype(int)
    frame["h3_cell"] = transformed["h3_cell"].astype(str)

    parent_transformed = parent_regionizer.transform_frame(frame, cfg.columns.lat, cfg.columns.lon)
    frame["parent_region_id"] = parent_transformed["region_id"].astype(int)
    frame["parent_h3_cell"] = parent_transformed["h3_cell"].astype(str)

    lat = _numeric_series(frame, cfg.columns.lat)
    lon = _numeric_series(frame, cfg.columns.lon)
    altitude = _numeric_series(frame, cfg.columns.altitude)
    speed = _numeric_series(frame, cfg.columns.speed)
    heading = _numeric_series(frame, cfg.columns.heading)
    vrate = _numeric_series(frame, cfg.columns.vertical_rate)

    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    heading_rad = np.deg2rad(heading)

    frame["lat_sin"] = np.sin(lat_rad)
    frame["lat_cos"] = np.cos(lat_rad)
    frame["lon_sin"] = np.sin(lon_rad)
    frame["lon_cos"] = np.cos(lon_rad)

    prev_lat = lat.groupby(frame[cfg.columns.trajectory_id]).shift(1).fillna(lat)
    prev_lon = lon.groupby(frame[cfg.columns.trajectory_id]).shift(1).fillna(lon)
    delta_lat_deg = lat - prev_lat
    delta_lon_deg = ((lon - prev_lon + 180.0) % 360.0) - 180.0
    mean_lat_rad = np.deg2rad((lat + prev_lat) / 2.0)
    delta_north_km = delta_lat_deg * 111.32
    delta_east_km = delta_lon_deg * 111.32 * np.cos(mean_lat_rad)
    step_distance_km = np.sqrt(np.square(delta_north_km) + np.square(delta_east_km))
    track_bearing_rad = np.arctan2(delta_east_km, delta_north_km)

    frame["delta_north_km_norm"] = np.clip(delta_north_km / 150.0, -3.0, 3.0)
    frame["delta_east_km_norm"] = np.clip(delta_east_km / 150.0, -3.0, 3.0)
    frame["step_distance_norm"] = np.clip(step_distance_km / 100.0, 0.0, 6.0)
    frame["track_bearing_sin"] = np.sin(track_bearing_rad)
    frame["track_bearing_cos"] = np.cos(track_bearing_rad)
    frame["region_changed"] = (
        frame.groupby(cfg.columns.trajectory_id)["region_id"].diff().fillna(0).ne(0).astype(float)
    )
    frame["parent_region_changed"] = (
        frame.groupby(cfg.columns.trajectory_id)["parent_region_id"].diff().fillna(0).ne(0).astype(float)
    )

    frame["altitude_km"] = altitude / 1000.0
    frame["speed_norm"] = speed / 500.0
    frame["heading_sin"] = np.sin(heading_rad)
    frame["heading_cos"] = np.cos(heading_rad)
    frame["vrate_norm"] = np.clip(vrate / 2000.0, -1.0, 1.0)

    delta_minutes = (
        frame.groupby(cfg.columns.trajectory_id)[cfg.columns.time]
        .diff()
        .dt.total_seconds()
        .div(60.0)
        .fillna(float(cfg.dataset.sample_interval_minutes))
    )
    frame["delta_minutes_norm"] = np.clip(
        delta_minutes / max(float(cfg.dataset.sample_interval_minutes), 1.0),
        0.0,
        6.0,
    )

    frame["observed_flag"] = np.clip(_numeric_series(frame, "observed_flag", default=1.0), 0.0, 1.0)
    frame["gap_flag"] = np.clip(_numeric_series(frame, "gap_flag", default=0.0), 0.0, 1.0)
    frame["is_ground"] = _is_ground_state(frame, cfg).astype(bool)
    frame["ground_flag"] = frame["is_ground"].astype(float)
    frame["status_id"] = frame["is_ground"].astype(int)

    hours = frame[cfg.columns.time].dt.hour + frame[cfg.columns.time].dt.minute / 60.0
    weekdays = frame[cfg.columns.time].dt.dayofweek.astype(float)
    frame["hour_sin"] = np.sin(2.0 * np.pi * hours / 24.0)
    frame["hour_cos"] = np.cos(2.0 * np.pi * hours / 24.0)
    frame["weekday_sin"] = np.sin(2.0 * np.pi * weekdays / 7.0)
    frame["weekday_cos"] = np.cos(2.0 * np.pi * weekdays / 7.0)

    frame["current_lat_sin"] = frame["lat_sin"]
    frame["current_lat_cos"] = frame["lat_cos"]
    frame["current_lon_sin"] = frame["lon_sin"]
    frame["current_lon_cos"] = frame["lon_cos"]
    frame["target_hash"] = frame["_aircraft_id"].astype(str).apply(hash_fraction)
    return frame


def _build_inference_batch(
    frame: pd.DataFrame,
    cfg: ProjectConfig,
    metadata: Dict[str, Any],
    aircraft_id: str,
    trajectory_id: str,
) -> tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    history_steps = int(metadata["history_steps"])
    unknown_region_id = int(metadata["input_regionizer"]["unknown_region_id"])
    unknown_parent_region_id = int(metadata["input_parent_regionizer"]["unknown_region_id"])

    frame = frame.sort_values(cfg.columns.time).reset_index(drop=True)
    airborne_indices = np.flatnonzero(~frame["is_ground"].to_numpy(dtype=bool))
    if airborne_indices.size > 0:
        current_idx = int(airborne_indices[-1])
        used_latest_airborne = current_idx != len(frame) - 1
    else:
        current_idx = len(frame) - 1
        used_latest_airborne = False

    hist = frame.iloc[max(0, current_idx - history_steps + 1): current_idx + 1].copy()
    observed_steps = len(hist)
    pad_count = max(0, history_steps - observed_steps)

    pad_numeric = np.zeros((pad_count, len(NUMERIC_FEATURES)), dtype=np.float32)
    pad_mask = np.zeros((pad_count, len(MASK_FEATURES)), dtype=np.float32)
    if pad_count > 0:
        pad_mask[:, 1] = 1.0

    region_seq = hist["region_id"].to_numpy(dtype=np.int64)
    parent_region_seq = hist["parent_region_id"].to_numpy(dtype=np.int64)
    numeric_seq = hist[NUMERIC_FEATURES].to_numpy(dtype=np.float32)
    mask_seq = hist[MASK_FEATURES].to_numpy(dtype=np.float32)

    if pad_count > 0:
        region_seq = np.concatenate([np.full(pad_count, unknown_region_id, dtype=np.int64), region_seq], axis=0)
        parent_region_seq = np.concatenate(
            [np.full(pad_count, unknown_parent_region_id, dtype=np.int64), parent_region_seq],
            axis=0,
        )
        numeric_seq = np.concatenate([pad_numeric, numeric_seq], axis=0)
        mask_seq = np.concatenate([pad_mask, mask_seq], axis=0)

    current = frame.iloc[current_idx]
    context = np.asarray(
        [
            float(current["hour_sin"]),
            float(current["hour_cos"]),
            float(current["weekday_sin"]),
            float(current["weekday_cos"]),
            float(current["current_lat_sin"]),
            float(current["current_lat_cos"]),
            float(current["current_lon_sin"]),
            float(current["current_lon_cos"]),
            float(current["target_hash"]),
        ],
        dtype=np.float32,
    )

    batch = {
        "region_seq": torch.tensor(region_seq[None, :], dtype=torch.long),
        "parent_region_seq": torch.tensor(parent_region_seq[None, :], dtype=torch.long),
        "numeric_seq": torch.tensor(numeric_seq[None, :, :], dtype=torch.float32),
        "mask_seq": torch.tensor(mask_seq[None, :, :], dtype=torch.float32),
        "context": torch.tensor(context[None, :], dtype=torch.float32),
    }
    meta = {
        "aircraft_id": aircraft_id,
        "trajectory_id": trajectory_id,
        "current_timestamp": current[cfg.columns.time].isoformat(),
        "current_cell": str(current["h3_cell"]),
        "current_region_id": int(current["region_id"]),
        "current_on_ground": bool(current["is_ground"]),
        "used_latest_airborne": bool(used_latest_airborne),
        "history_steps_required": history_steps,
        "history_steps_observed": observed_steps,
        "history_insufficient": observed_steps < history_steps,
        "resampled_points": int(len(frame)),
        "padding_steps": pad_count,
    }
    return batch, meta


def _build_sparse_fine_probs(
    coarse_probs: torch.Tensor,
    top_parent_ids: torch.Tensor,
    child_probs_top: torch.Tensor,
    hierarchy: Dict[str, object],
    fine_class_count: int,
    unknown_fine_id: int,
) -> torch.Tensor:
    batch_size, top_k = top_parent_ids.shape
    fine_probs = torch.zeros((batch_size, fine_class_count), dtype=torch.float32)
    used_mass = torch.zeros(batch_size, dtype=torch.float32)
    coarse_to_children: Sequence[Sequence[int]] = hierarchy["coarse_to_children"]  # type: ignore[assignment]

    parent_ids_np = top_parent_ids.cpu().numpy()
    parent_probs_cpu = coarse_probs.gather(1, top_parent_ids).cpu()
    child_probs_cpu = child_probs_top.cpu()

    for sample_idx in range(batch_size):
        for parent_rank in range(top_k):
            coarse_id = int(parent_ids_np[sample_idx, parent_rank])
            if coarse_id < 0 or coarse_id >= len(coarse_to_children):
                continue
            children = list(coarse_to_children[coarse_id])
            if not children:
                continue
            valid_child_count = min(len(children), int(child_probs_cpu.shape[-1]))
            if valid_child_count <= 0:
                continue
            local_probs = (
                float(parent_probs_cpu[sample_idx, parent_rank].item())
                * child_probs_cpu[sample_idx, parent_rank, :valid_child_count]
            )
            fine_probs[sample_idx, children[:valid_child_count]] += local_probs
            used_mass[sample_idx] += float(local_probs.sum().item())

    fine_probs[:, unknown_fine_id] = torch.clamp(1.0 - used_mass, min=0.0)
    return fine_probs


def _predict_single(
    cfg: ProjectConfig,
    model_type: str,
    model,
    metadata: Dict[str, Any],
    batch: Dict[str, torch.Tensor],
) -> Dict[str, Any]:
    device = _resolve_device(cfg)
    batch = {key: value.to(device) for key, value in batch.items()}

    if model_type == "markov":
        predictions = model.predict(batch["region_seq"].cpu())
        return {
            "region_probs": [tensor for tensor in predictions["region_probs"]],
            "status_probs": predictions["status_probs"],
            "coarse_probs": None,
        }

    model = model.to(device)
    with torch.no_grad():
        if model_type == "gru":
            outputs = model(batch["region_seq"], batch["numeric_seq"], batch["context"])
            return {
                "region_probs": [torch.softmax(logits, dim=-1).cpu() for logits in outputs["region_logits"]],
                "status_probs": torch.softmax(outputs["status_logits"], dim=-1).cpu(),
                "coarse_probs": None,
            }

        outputs = model(
            batch["region_seq"],
            batch["parent_region_seq"],
            batch["numeric_seq"],
            batch["mask_seq"],
            batch["context"],
        )
        features: torch.Tensor = outputs["features"]  # type: ignore[assignment]
        decoder_features: torch.Tensor | None = outputs.get("decoder_features")  # type: ignore[assignment]
        coarse_logits_list: List[torch.Tensor] = outputs["coarse_logits"]  # type: ignore[assignment]
        fine_probs_out: List[torch.Tensor] = []
        coarse_probs_out: List[torch.Tensor] = []

        top_parent_k_per_horizon = [max(1, int(value)) for value in cfg.hierarchical_parent_top_k_per_horizon]
        fine_class_counts = metadata["num_region_classes_per_horizon"]
        unknown_fine_ids = metadata["unknown_region_ids_per_horizon"]
        hierarchy_per_horizon = metadata["hierarchy_per_horizon"]

        for horizon_idx, coarse_logits in enumerate(coarse_logits_list):
            coarse_prob = torch.softmax(coarse_logits, dim=-1)
            coarse_probs_out.append(coarse_prob.cpu())
            current_top_k = min(top_parent_k_per_horizon[horizon_idx], int(coarse_prob.shape[-1]))
            _, top_parent_ids = torch.topk(coarse_prob, k=current_top_k, dim=-1)
            horizon_features = (
                decoder_features[:, horizon_idx, :]
                if decoder_features is not None
                else features
            )
            child_probs_top = torch.softmax(model.child_logits(horizon_features, top_parent_ids, horizon_idx), dim=-1)
            fine_probs = _build_sparse_fine_probs(
                coarse_probs=coarse_prob,
                top_parent_ids=top_parent_ids,
                child_probs_top=child_probs_top,
                hierarchy=hierarchy_per_horizon[horizon_idx],
                fine_class_count=int(fine_class_counts[horizon_idx]),
                unknown_fine_id=int(unknown_fine_ids[horizon_idx]),
            )
            fine_probs_out.append(fine_probs)

        return {
            "region_probs": fine_probs_out,
            "status_probs": torch.softmax(outputs["status_logits"], dim=-1).cpu(),
            "coarse_probs": coarse_probs_out,
        }


def _status_name(status_id: int) -> str:
    return "landed" if int(status_id) == 1 else "airborne"


def _cell_payload(regionizer: H3Regionizer, region_id: int, probability: float, rank: int, include_geometry: bool) -> Dict[str, Any]:
    cell = regionizer.decode_region_id(int(region_id))
    payload: Dict[str, Any] = {
        "rank": rank,
        "region_id": int(region_id),
        "cell": cell,
        "probability": float(probability),
        "is_unknown": cell is None,
    }
    if include_geometry and cell is not None:
        payload["cell_geometry"] = regionizer.cell_geometry(cell)
    return payload


def _feature_collection(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "type": "FeatureCollection",
        "features": features,
    }


def _cell_feature(
    boundary_lonlat: Sequence[Sequence[float]],
    properties: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [list(boundary_lonlat)],
        },
        "properties": properties,
    }


def _candidate_region_geojson(
    candidate: Dict[str, Any],
    *,
    horizon_minutes: int,
    candidate_rank: int,
    predicted_status: str,
    status_probabilities: Dict[str, Any],
) -> Dict[str, Any]:
    features: List[Dict[str, Any]] = []
    center_cell = candidate.get("center_cell")
    for geometry in candidate.get("cell_geometries", []) or []:
        if not isinstance(geometry, dict):
            continue
        cell = geometry.get("cell")
        boundary_lonlat = geometry.get("boundary_lonlat")
        if cell is None or not boundary_lonlat:
            continue
        features.append(
            _cell_feature(
                boundary_lonlat,
                {
                    "horizon_minutes": int(horizon_minutes),
                    "candidate_rank": int(candidate_rank),
                    "candidate_confidence": float(candidate.get("confidence", 0.0)),
                    "predicted_status": predicted_status,
                    "airborne_probability": float(status_probabilities.get("airborne", 0.0)),
                    "landed_probability": float(status_probabilities.get("landed", 0.0)),
                    "center_cell": center_cell,
                    "cell": str(cell),
                    "is_center_cell": str(cell) == str(center_cell),
                },
            )
        )
    return _feature_collection(features)


def _prediction_geojson(horizons: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_horizon: List[Dict[str, Any]] = []
    all_features: List[Dict[str, Any]] = []

    for horizon in horizons:
        horizon_minutes = int(horizon["horizon_minutes"])
        horizon_features: List[Dict[str, Any]] = []
        for candidate_rank, candidate in enumerate(horizon.get("candidate_regions", []) or [], start=1):
            feature_collection = candidate.get("cells_geojson")
            if not isinstance(feature_collection, dict):
                continue
            features = feature_collection.get("features", [])
            if not isinstance(features, list):
                continue
            horizon_features.extend(features)
            all_features.extend(features)

        by_horizon.append(
            {
                "horizon_minutes": horizon_minutes,
                "feature_collection": _feature_collection(horizon_features),
            }
        )

    return {
        "by_horizon": by_horizon,
        "all_horizons": _feature_collection(all_features),
    }


def _write_json_payload(path_like: str | Path, payload: Dict[str, Any], label: str) -> None:
    destination = Path(path_like)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"{label}: {destination}")


def _write_geojson_outputs(
    prediction: Dict[str, Any],
    *,
    geojson_output_path: Optional[str],
    geojson_output_dir: Optional[str],
) -> None:
    geojson = prediction.get("geojson")
    if not isinstance(geojson, dict):
        raise RuntimeError("GeoJSON export requested, but prediction output does not contain GeoJSON data.")

    if geojson_output_path is not None:
        all_horizons = geojson.get("all_horizons", _feature_collection([]))
        _write_json_payload(geojson_output_path, all_horizons, "GeoJSON saved")

    if geojson_output_dir is not None:
        output_dir = Path(geojson_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        by_horizon = geojson.get("by_horizon", [])
        if isinstance(by_horizon, list):
            for item in by_horizon:
                if not isinstance(item, dict):
                    continue
                horizon_minutes = int(item.get("horizon_minutes", 0))
                feature_collection = item.get("feature_collection", _feature_collection([]))
                destination = output_dir / f"horizon_{horizon_minutes}min.geojson"
                _write_json_payload(destination, feature_collection, "GeoJSON saved")


def _build_prediction_output(
    cfg: ProjectConfig,
    metadata: Dict[str, Any],
    batch_meta: Dict[str, Any],
    predictions: Dict[str, Any],
    include_geometry: bool = False,
    include_geojson: bool = False,
) -> List[Dict[str, Any]]:
    target_regionizers = [H3Regionizer.from_metadata(item) for item in metadata["target_regionizers"]]
    coarse_regionizers = [H3Regionizer.from_metadata(item) for item in metadata["coarse_regionizers"]]
    reachability_filter = ReachabilityFilter(cfg, metadata)
    spatial_refiner = H3NeighborProbabilityRefiner(
        region_tokens_per_horizon=[
            item["region_tokens"]
            for item in metadata["target_regionizers"]
        ],
        enabled=bool(cfg.evaluation.h3_refinement_enabled),
        alpha_per_horizon=cfg.h3_refinement_alpha_per_horizon,
        ring_per_horizon=cfg.h3_refinement_ring_per_horizon,
        preserve_unknown=bool(cfg.evaluation.h3_refinement_preserve_unknown),
    )
    region_prob_arrays = [tensor.numpy()[0] for tensor in predictions["region_probs"]]
    status_prob_array = predictions["status_probs"].numpy()[0]
    coarse_prob_arrays = None
    if predictions["coarse_probs"] is not None:
        coarse_prob_arrays = [tensor.numpy()[0] for tensor in predictions["coarse_probs"]]

    top_k_export = max(int(k) for k in cfg.evaluation.top_ks) if cfg.evaluation.top_ks else 5
    candidate_probability_mass = cfg.candidate_probability_mass_per_horizon
    candidate_top_cells = cfg.candidate_top_cells_per_horizon
    parent_top_k_per_horizon = cfg.hierarchical_parent_top_k_per_horizon
    horizons: List[Dict[str, Any]] = []

    for horizon_idx, horizon_minutes in enumerate(metadata["horizons_minutes"]):
        target_regionizer = target_regionizers[horizon_idx]
        model_probs = region_prob_arrays[horizon_idx]
        model_top_region_ids = np.argsort(-model_probs)[:top_k_export].tolist()
        model_raw_top1_region_id = int(model_top_region_ids[0]) if model_top_region_ids else None
        model_raw_top1_cell = (
            target_regionizer.decode_region_id(model_raw_top1_region_id)
            if model_raw_top1_region_id is not None
            else None
        )
        model_raw_top1_probability = (
            float(model_probs[model_raw_top1_region_id])
            if model_raw_top1_region_id is not None
            else 0.0
        )
        refined_model_probs = spatial_refiner.refine_horizon(model_probs[None, :], horizon_idx)[0]
        probs, reachability = reachability_filter.apply(
            refined_model_probs,
            current_cell=batch_meta.get("current_cell"),
            horizon_idx=horizon_idx,
            horizon_minutes=int(horizon_minutes),
        )
        top_region_ids = np.argsort(-probs)[:top_k_export].tolist()
        candidates = target_regionizer.top_candidate_regions(
            probabilities=probs,
            cumulative_mass=float(candidate_probability_mass[horizon_idx]),
            max_cells=int(candidate_top_cells[horizon_idx]),
            include_geometry=(include_geometry or include_geojson),
        )
        raw_top1_region_id = int(top_region_ids[0]) if top_region_ids else None
        raw_top1_cell = target_regionizer.decode_region_id(raw_top1_region_id) if raw_top1_region_id is not None else None
        raw_top1_probability = float(probs[raw_top1_region_id]) if raw_top1_region_id is not None else 0.0
        status_probabilities = {
            "airborne": float(status_prob_array[horizon_idx, 0]),
            "landed": float(status_prob_array[horizon_idx, 1]),
        }
        predicted_status = _status_name(int(status_prob_array[horizon_idx].argmax()))

        horizon_payload: Dict[str, Any] = {
            "horizon_minutes": int(horizon_minutes),
            "status_probabilities": status_probabilities,
            "predicted_status": predicted_status,
            "raw_top1_region_id": raw_top1_region_id,
            "raw_top1_cell": raw_top1_cell,
            "raw_top1_probability": raw_top1_probability,
            "raw_top1_is_unknown": bool(raw_top1_cell is None) if raw_top1_region_id is not None else False,
            "model_raw_top1_region_id": model_raw_top1_region_id,
            "model_raw_top1_cell": model_raw_top1_cell,
            "model_raw_top1_probability": model_raw_top1_probability,
            "model_raw_top1_is_unknown": bool(model_raw_top1_cell is None) if model_raw_top1_region_id is not None else False,
            "reachability_filter": reachability,
            "top_region_predictions": [
                _cell_payload(
                    target_regionizer,
                    int(region_id),
                    float(probs[int(region_id)]),
                    rank,
                    include_geometry,
                )
                for rank, region_id in enumerate(top_region_ids, start=1)
            ],
            "candidate_regions": candidates,
        }
        if include_geometry and raw_top1_cell is not None:
            horizon_payload["raw_top1_geometry"] = target_regionizer.cell_geometry(raw_top1_cell)

        if include_geojson:
            for candidate_rank, candidate in enumerate(horizon_payload["candidate_regions"], start=1):
                candidate["cells_geojson"] = _candidate_region_geojson(
                    candidate,
                    horizon_minutes=int(horizon_minutes),
                    candidate_rank=candidate_rank,
                    predicted_status=predicted_status,
                    status_probabilities=status_probabilities,
                )
            horizon_features: List[Dict[str, Any]] = []
            for candidate in horizon_payload["candidate_regions"]:
                feature_collection = candidate.get("cells_geojson", {})
                if isinstance(feature_collection, dict):
                    features = feature_collection.get("features", [])
                    if isinstance(features, list):
                        horizon_features.extend(features)
            horizon_payload["candidate_regions_geojson"] = _feature_collection(horizon_features)

        if coarse_prob_arrays is not None:
            coarse_regionizer = coarse_regionizers[horizon_idx]
            coarse_probs = coarse_prob_arrays[horizon_idx]
            coarse_ids = np.argsort(-coarse_probs)[: int(parent_top_k_per_horizon[horizon_idx])].tolist()
            horizon_payload["top_coarse_predictions"] = [
                _cell_payload(
                    coarse_regionizer,
                    int(region_id),
                    float(coarse_probs[int(region_id)]),
                    rank,
                    include_geometry,
                )
                for rank, region_id in enumerate(coarse_ids, start=1)
            ]

        horizons.append(horizon_payload)

    return horizons


def _summarize_prediction(result: Dict[str, Any]) -> Dict[str, Any]:
    horizons = result.get("horizons", [])
    return {
        "model_scope": result["model"].get("scope"),
        "model_name": result["model"].get("name"),
        "fallback_used": bool(result["model"].get("fallback_used")),
        "trajectory_id": result["input"].get("trajectory_id"),
        "aircraft_type": result["input"].get("aircraft_type"),
        "current_timestamp": result["input"].get("current_timestamp"),
        "history_insufficient": bool(result["input"].get("history_insufficient")),
        "history_steps_required": result["input"].get("history_steps_required"),
        "history_steps_observed": result["input"].get("history_steps_observed"),
        "padding_steps": result["input"].get("padding_steps"),
        "num_horizons": len(horizons),
        "horizon_minutes": [item.get("horizon_minutes") for item in horizons],
    }


def _horizon_records(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for horizon in result.get("horizons", []):
        candidate_regions = horizon.get("candidate_regions", []) or []
        top_predictions = horizon.get("top_region_predictions", []) or []
        top_prediction = top_predictions[0] if top_predictions else {}
        records.append(
            {
                "horizon_minutes": horizon.get("horizon_minutes"),
                "predicted_status": horizon.get("predicted_status"),
                "airborne_probability": horizon.get("status_probabilities", {}).get("airborne"),
                "landed_probability": horizon.get("status_probabilities", {}).get("landed"),
                "top_region_cell": top_prediction.get("cell"),
                "top_region_probability": top_prediction.get("probability"),
                "candidate_region_count": len(candidate_regions),
                "candidate_confidence_sum": float(
                    sum(float(item.get("confidence", 0.0)) for item in candidate_regions)
                ),
            }
        )
    return records


def _run_prediction_core(
    cfg: ProjectConfig,
    *,
    frame: pd.DataFrame,
    source_name: str,
    trajectory_id: Optional[str],
    aircraft_type: Optional[str],
    scope: Optional[str],
    name: Optional[str],
    output_path: Optional[str],
    include_geometry: bool = False,
    include_geojson: bool = False,
) -> Dict[str, Any]:
    working = attach_aircraft_metadata(frame, cfg, fallback_aircraft_id=Path(source_name).stem)
    resolved_aircraft_type = _infer_aircraft_type(working, cfg, aircraft_type)

    entry, fallback_used = resolve_registered_model(
        cfg,
        aircraft_type=resolved_aircraft_type,
        scope=scope,
        name=name,
    )
    model_type, model, metadata, train_cfg = _load_model_bundle(cfg, entry)

    input_ref = Path(source_name)
    selected_frame, selected_trajectory_id = _select_trajectory(working, cfg, input_ref, trajectory_id)
    selected_frame = selected_frame.copy()
    inferred_aircraft_id = (
        str(selected_frame["_aircraft_id"].iloc[0]) if "_aircraft_id" in selected_frame.columns else input_ref.stem
    )
    selected_frame["_aircraft_id"] = str(inferred_aircraft_id)
    selected_frame = _resample_for_inference(selected_frame, train_cfg, source_name)
    selected_frame = _attach_region_and_features(selected_frame, train_cfg, metadata)

    batch, batch_meta = _build_inference_batch(
        selected_frame,
        train_cfg,
        metadata,
        aircraft_id=str(selected_frame["_aircraft_id"].iloc[0]),
        trajectory_id=selected_trajectory_id,
    )
    predictions = _predict_single(cfg, model_type, model, metadata, batch)
    horizons = _build_prediction_output(
        cfg,
        metadata,
        batch_meta,
        predictions,
        include_geometry=include_geometry,
        include_geojson=include_geojson,
    )

    if include_geometry or include_geojson:
        input_regionizer = H3Regionizer.from_metadata(metadata["input_regionizer"])
        batch_meta["current_cell_geometry"] = input_regionizer.cell_geometry(batch_meta["current_cell"])

    result: Dict[str, Any] = {
        "model": {
            "scope": entry.get("scope"),
            "name": entry.get("name"),
            "name_slug": entry.get("name_slug"),
            "run_name": entry.get("run_name"),
            "checkpoint_path": entry.get("checkpoint_path"),
            "fallback_used": bool(fallback_used),
        },
        "input": {
            "path": str(source_name),
            "trajectory_id": selected_trajectory_id,
            "aircraft_type": resolved_aircraft_type,
            **batch_meta,
        },
        "horizons": horizons,
    }
    if include_geojson:
        result["geojson"] = _prediction_geojson(horizons)

    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with open(destination, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
        print(f"Prediction saved: {destination}")

    return result


def run_prediction_from_dataframe(
    cfg: ProjectConfig,
    *,
    dataframe: pd.DataFrame,
    traj_name: str = "in_memory_track",
    trajectory_id: Optional[str] = None,
    aircraft_type: Optional[str] = None,
    scope: Optional[str] = None,
    name: Optional[str] = None,
    output_path: Optional[str] = None,
    sort_by_timestamp: bool = True,
    include_dataframe: bool = False,
    include_records: bool = True,
    include_geometry: bool = False,
    include_geojson: bool = False,
    geojson_output_path: Optional[str] = None,
    geojson_output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    frame = dataframe.copy()
    if sort_by_timestamp and cfg.columns.time in frame.columns:
        frame = frame.sort_values(cfg.columns.time).reset_index(drop=True)

    include_geojson = bool(include_geojson or geojson_output_path is not None or geojson_output_dir is not None)

    prediction = _run_prediction_core(
        cfg,
        frame=frame,
        source_name=traj_name,
        trajectory_id=trajectory_id,
        aircraft_type=aircraft_type,
        scope=scope,
        name=name,
        output_path=output_path,
        include_geometry=include_geometry,
        include_geojson=include_geojson,
    )

    if geojson_output_path is not None or geojson_output_dir is not None:
        _write_geojson_outputs(
            prediction,
            geojson_output_path=geojson_output_path,
            geojson_output_dir=geojson_output_dir,
        )

    result: Dict[str, Any] = {
        "prediction": prediction,
        "summary": _summarize_prediction(prediction),
    }
    if include_dataframe:
        result["dataframe"] = frame
    if include_records:
        result["records"] = _horizon_records(prediction)
    return result


def run_prediction(
    cfg: ProjectConfig,
    *,
    input_path: str,
    input_format: Optional[str],
    trajectory_id: Optional[str],
    aircraft_type: Optional[str],
    scope: Optional[str],
    name: Optional[str],
    output_path: Optional[str],
    include_geometry: bool = False,
    include_geojson: bool = False,
    geojson_output_path: Optional[str] = None,
    geojson_output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    input_file = Path(input_path)
    frame = _read_input_file(input_file, input_format)
    include_geojson = bool(include_geojson or geojson_output_path is not None or geojson_output_dir is not None)
    prediction = _run_prediction_core(
        cfg,
        frame=frame,
        source_name=str(input_file),
        trajectory_id=trajectory_id,
        aircraft_type=aircraft_type,
        scope=scope,
        name=name,
        output_path=output_path,
        include_geometry=include_geometry,
        include_geojson=include_geojson,
    )
    if geojson_output_path is not None or geojson_output_dir is not None:
        _write_geojson_outputs(
            prediction,
            geojson_output_path=geojson_output_path,
            geojson_output_dir=geojson_output_dir,
        )
    return prediction


def main() -> None:
    parser = argparse.ArgumentParser(description="Run single-trajectory movement prediction from a registered model.")
    parser.add_argument("--config", type=str, default="config/region_forecast.yaml")
    parser.add_argument("--input", type=str, required=True, help="Path to a CSV/Parquet/JSON trajectory file.")
    parser.add_argument("--input-format", type=str, default=None, help="Optional input format override: csv/parquet/json.")
    parser.add_argument("--trajectory-id", type=str, default=None, help="Optional trajectory id to select from the input.")
    parser.add_argument("--aircraft-type", type=str, default=None, help="Aircraft type used for automatic model routing.")
    parser.add_argument("--scope", type=str, choices=["global", "type"], default=None, help="Optional explicit model scope override.")
    parser.add_argument("--name", type=str, default=None, help="Optional explicit model name override.")
    parser.add_argument("--output", type=str, default=None, help="Optional output JSON path.")
    parser.add_argument("--include-geometry", action="store_true", help="Include H3 cell boundary coordinates in the output JSON.")
    parser.add_argument("--include-geojson", action="store_true", help="Include GeoJSON candidate-cell output for all horizons.")
    parser.add_argument("--geojson-output", type=str, default=None, help="Optional pure GeoJSON output path for all horizons.")
    parser.add_argument("--geojson-output-dir", type=str, default=None, help="Optional directory for one GeoJSON file per horizon.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    result = run_prediction(
        cfg,
        input_path=args.input,
        input_format=args.input_format,
        trajectory_id=args.trajectory_id,
        aircraft_type=args.aircraft_type,
        scope=args.scope,
        name=args.name,
        output_path=args.output,
        include_geometry=args.include_geometry,
        include_geojson=args.include_geojson,
        geojson_output_path=args.geojson_output,
        geojson_output_dir=args.geojson_output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
