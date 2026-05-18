from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

try:
    import h3
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise ImportError("The 'h3' package is required. Install it with 'pip install h3'.") from exc

try:
    import folium
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise ImportError("The 'folium' package is required. Install it with 'pip install folium'.") from exc


HORIZON_COLORS = {
    30: "#e63946",
    60: "#f4a261",
    120: "#2a9d8f",
    300: "#1d3557",
}
FALLBACK_COLORS = [
    "#6a4c93",
    "#ff006e",
    "#1982c4",
    "#8ac926",
    "#ffca3a",
]
CURRENT_COLOR = "#111111"
POLYGON_PANE = "prediction-polygons"
MARKER_PANE = "prediction-markers"


def _horizon_color(horizon_minutes: int, horizon_idx: int) -> str:
    if int(horizon_minutes) in HORIZON_COLORS:
        return HORIZON_COLORS[int(horizon_minutes)]
    return FALLBACK_COLORS[horizon_idx % len(FALLBACK_COLORS)]


def _cell_center(cell: str) -> Tuple[float, float]:
    lat, lon = h3.cell_to_latlng(cell)
    return float(lat), float(lon)


def _cell_polygon(cell: str) -> List[List[float]]:
    boundary = h3.cell_to_boundary(cell)
    ring = [[float(lat), float(lon)] for lat, lon in boundary]
    if ring[0] != ring[-1]:
        ring.append(ring[0])
    return ring


def _point_icon(color: str, size_px: int, border_color: str = "#ffffff") -> folium.DivIcon:
    return folium.DivIcon(
        html=(
            f"<div style='width:{size_px}px;height:{size_px}px;"
            "display:flex;align-items:center;justify-content:center;'>"
            f"<div style='width:{size_px - 4}px;height:{size_px - 4}px;border-radius:50%;"
            f"background:{color};border:2px solid {border_color};"
            "box-shadow:0 0 0 1px rgba(0,0,0,0.35);'></div></div>"
        ),
        icon_size=(size_px, size_px),
        icon_anchor=(size_px // 2, size_px // 2),
        class_name="custom-point-icon",
    )


def _candidate_opacity(candidate_rank: int) -> float:
    return max(0.14, 0.38 - (candidate_rank - 1) * 0.06)


def _iter_prediction_coordinates(prediction: Dict[str, object], max_candidate_regions: int) -> Iterable[Tuple[float, float]]:
    current_cell = prediction.get("input", {}).get("current_cell")
    if current_cell:
        yield _cell_center(str(current_cell))

    for horizon in prediction.get("horizons", []):
        for candidate in horizon.get("candidate_regions", [])[:max_candidate_regions]:
            for cell in candidate.get("cells", []):
                for lat, lon in _cell_polygon(str(cell)):
                    yield float(lat), float(lon)


def _map_bounds(
    prediction: Dict[str, object],
    max_candidate_regions: int,
) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    latitudes: List[float] = []
    longitudes: List[float] = []

    for lat, lon in _iter_prediction_coordinates(prediction, max_candidate_regions):
        latitudes.append(float(lat))
        longitudes.append(float(lon))

    if not latitudes or not longitudes:
        return (20.0, 120.0), (-60.0, 60.0), (60.0, 180.0)

    center = (float(sum(latitudes) / len(latitudes)), float(sum(longitudes) / len(longitudes)))
    bounds = (
        max(-85.0, min(latitudes) - 3.0),
        min(85.0, max(latitudes) + 3.0),
        max(-180.0, min(longitudes) - 3.0),
        min(180.0, max(longitudes) + 3.0),
    )
    return center, (bounds[0], bounds[2]), (bounds[1], bounds[3])


def _legend_html(prediction: Dict[str, object]) -> str:
    items = "".join(
        (
            f"<div style='margin-bottom:4px;'>"
            f"<span style='display:inline-block;width:12px;height:12px;background:{color};"
            f"margin-right:6px;border:1px solid #333;'></span>{minutes} min"
            f"</div>"
        )
        for minutes, color in HORIZON_COLORS.items()
    )
    input_meta = prediction.get("input", {})
    header = (
        f"traj={input_meta.get('trajectory_id', 'n/a')}<br>"
        f"time={str(input_meta.get('current_timestamp', ''))[:16].replace('T', ' ')}"
    )
    return (
        "<div style=\"position: fixed; bottom: 24px; left: 24px; z-index: 9999; "
        "background: rgba(255,255,255,0.92); border: 1px solid #999; border-radius: 6px; "
        "padding: 10px 12px; font-size: 12px; line-height: 1.3;\">"
        "<div style='font-weight:600; margin-bottom:6px;'>Prediction</div>"
        f"<div style='margin-bottom:6px;'>{header}</div>"
        f"{items}"
        "<div style='margin-top:6px;'>Black = current point<br>Filled polygons = candidate regions</div>"
        "</div>"
    )


def _current_popup(prediction: Dict[str, object]) -> str:
    model_meta = prediction.get("model", {})
    input_meta = prediction.get("input", {})
    return (
        f"run_name={model_meta.get('run_name')}<br>"
        f"scope={model_meta.get('scope')}<br>"
        f"aircraft_id={input_meta.get('aircraft_id')}<br>"
        f"trajectory_id={input_meta.get('trajectory_id')}<br>"
        f"aircraft_type={input_meta.get('aircraft_type')}<br>"
        f"timestamp={input_meta.get('current_timestamp')}<br>"
        f"current_cell={input_meta.get('current_cell')}<br>"
        f"history_steps={input_meta.get('history_steps_observed')}/{input_meta.get('history_steps_required')}<br>"
        f"resampled_points={input_meta.get('resampled_points')}"
    )


def _candidate_popup(
    prediction: Dict[str, object],
    horizon: Dict[str, object],
    candidate_rank: int,
    candidate: Dict[str, object],
) -> str:
    input_meta = prediction.get("input", {})
    status_probabilities = horizon.get("status_probabilities", {})
    return (
        f"aircraft_id={input_meta.get('aircraft_id')}<br>"
        f"trajectory_id={input_meta.get('trajectory_id')}<br>"
        f"timestamp={input_meta.get('current_timestamp')}<br>"
        f"horizon={horizon.get('horizon_minutes')} min<br>"
        f"predicted_status={horizon.get('predicted_status')}<br>"
        f"airborne_prob={float(status_probabilities.get('airborne', 0.0)):.3f}<br>"
        f"landed_prob={float(status_probabilities.get('landed', 0.0)):.3f}<br>"
        f"candidate_rank={candidate_rank}<br>"
        f"confidence={float(candidate.get('confidence', 0.0)):.3f}<br>"
        f"center_cell={candidate.get('center_cell')}<br>"
        f"cell_count={len(candidate.get('cells', []))}"
    )


def _layer_name(horizon: Dict[str, object], candidate_count: int) -> str:
    horizon_minutes = int(horizon.get("horizon_minutes", 0))
    predicted_status = str(horizon.get("predicted_status", "unknown"))
    return f"H{horizon_minutes} candidate_regions | {predicted_status} | regions={candidate_count}"


def _add_current_marker(group: folium.FeatureGroup, prediction: Dict[str, object]) -> None:
    current_cell = prediction.get("input", {}).get("current_cell")
    if not current_cell:
        return

    lat, lon = _cell_center(str(current_cell))
    folium.Marker(
        location=[lat, lon],
        icon=_point_icon(CURRENT_COLOR, 16),
        popup=_current_popup(prediction),
        tooltip="current",
        pane=MARKER_PANE,
    ).add_to(group)


def _add_horizon_candidates(
    group: folium.FeatureGroup,
    prediction: Dict[str, object],
    horizon: Dict[str, object],
    horizon_idx: int,
    max_candidate_regions: int,
) -> None:
    color = _horizon_color(int(horizon.get("horizon_minutes", 0)), horizon_idx)

    for candidate_rank, candidate in enumerate(horizon.get("candidate_regions", [])[:max_candidate_regions], start=1):
        popup = _candidate_popup(prediction, horizon, candidate_rank, candidate)
        opacity = _candidate_opacity(candidate_rank)

        for cell in candidate.get("cells", []):
            folium.Polygon(
                locations=_cell_polygon(str(cell)),
                color=color,
                weight=1,
                fill=True,
                fill_color=color,
                fill_opacity=opacity,
                popup=popup,
                tooltip=(
                    f"H{int(horizon.get('horizon_minutes', 0))} "
                    f"candidate#{candidate_rank} conf={float(candidate.get('confidence', 0.0)):.3f}"
                ),
                pane=POLYGON_PANE,
            ).add_to(group)


def build_prediction_map(
    prediction: Dict[str, object],
    max_candidate_regions: int,
) -> folium.Map:
    center, southwest, northeast = _map_bounds(prediction, max_candidate_regions)
    fmap = folium.Map(location=list(center), zoom_start=5, tiles="CartoDB positron")
    folium.map.CustomPane(POLYGON_PANE, z_index=400).add_to(fmap)
    folium.map.CustomPane(MARKER_PANE, z_index=650).add_to(fmap)

    current_group = folium.FeatureGroup(name="current point", show=True)
    _add_current_marker(current_group, prediction)
    current_group.add_to(fmap)

    horizons = prediction.get("horizons", [])
    for horizon_idx, horizon in enumerate(horizons):
        candidate_count = len(horizon.get("candidate_regions", [])[:max_candidate_regions])
        feature_group = folium.FeatureGroup(
            name=_layer_name(horizon, candidate_count),
            show=True,
        )
        _add_horizon_candidates(
            group=feature_group,
            prediction=prediction,
            horizon=horizon,
            horizon_idx=horizon_idx,
            max_candidate_regions=max_candidate_regions,
        )
        feature_group.add_to(fmap)

    fmap.fit_bounds([list(southwest), list(northeast)])
    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.get_root().html.add_child(folium.Element(_legend_html(prediction)))
    return fmap


def _default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_candidate_map.html")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Render one predict.py output JSON into an interactive HTML map. "
            "Each horizon's candidate_regions are exported as a separate toggleable layer."
        )
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the prediction JSON file generated by predict.py.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to the output HTML map. Defaults to <input_stem>_candidate_map.html.",
    )
    parser.add_argument(
        "--max-candidate-regions",
        type=int,
        default=5,
        help="Maximum number of candidate regions to render per horizon.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    with input_path.open("r", encoding="utf-8") as handle:
        prediction = json.load(handle)

    fmap = build_prediction_map(
        prediction=prediction,
        max_candidate_regions=max(1, int(args.max_candidate_regions)),
    )

    output_path = Path(args.output) if args.output else _default_output_path(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(output_path))
    print(f"Saved prediction map to: {output_path}")


if __name__ == "__main__":
    main()
