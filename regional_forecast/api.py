from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from regional_forecast.config import ProjectConfig, load_config
from regional_forecast.pipeline.predict import run_prediction_from_dataframe


def _resolve_config(
    *,
    config: Optional[ProjectConfig],
    config_path: Optional[str],
) -> ProjectConfig:
    if config is not None:
        return config
    return load_config(config_path or "config/region_forecast.yaml")


def predict_from_dataframe(
    pos_pd: pd.DataFrame,
    *,
    config: Optional[ProjectConfig] = None,
    config_path: Optional[str] = None,
    traj_name: str = "demo_track",
    trajectory_id: Optional[str] = None,
    aircraft_type: Optional[str] = None,
    scope: Optional[str] = None,
    name: Optional[str] = None,
    sort_by_timestamp: bool = True,
    include_dataframe: bool = False,
    include_records: bool = True,
    include_geometry: bool = False,
    include_geojson: bool = False,
    geojson_output_path: Optional[str] = None,
    geojson_output_dir: Optional[str] = None,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    cfg = _resolve_config(config=config, config_path=config_path)
    return run_prediction_from_dataframe(
        cfg,
        dataframe=pos_pd,
        traj_name=traj_name,
        trajectory_id=trajectory_id,
        aircraft_type=aircraft_type,
        scope=scope,
        name=name,
        output_path=output_path,
        sort_by_timestamp=sort_by_timestamp,
        include_dataframe=include_dataframe,
        include_records=include_records,
        include_geometry=include_geometry,
        include_geojson=include_geojson,
        geojson_output_path=geojson_output_path,
        geojson_output_dir=geojson_output_dir,
    )


def predict_geojson_from_dataframe(
    pos_pd: pd.DataFrame,
    *,
    config: Optional[ProjectConfig] = None,
    config_path: Optional[str] = None,
    traj_name: str = "demo_track",
    trajectory_id: Optional[str] = None,
    aircraft_type: Optional[str] = None,
    scope: Optional[str] = None,
    name: Optional[str] = None,
    sort_by_timestamp: bool = True,
    include_prediction: bool = False,
    output_path: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    cfg = _resolve_config(config=config, config_path=config_path)
    result = run_prediction_from_dataframe(
        cfg,
        dataframe=pos_pd,
        traj_name=traj_name,
        trajectory_id=trajectory_id,
        aircraft_type=aircraft_type,
        scope=scope,
        name=name,
        output_path=None,
        sort_by_timestamp=sort_by_timestamp,
        include_dataframe=False,
        include_records=False,
        include_geometry=False,
        include_geojson=True,
        geojson_output_path=output_path,
        geojson_output_dir=output_dir,
    )

    payload: Dict[str, Any] = {
        "summary": result["summary"],
        "geojson": result["prediction"].get("geojson", {}),
    }
    if include_prediction:
        payload["prediction"] = result["prediction"]
    return payload
