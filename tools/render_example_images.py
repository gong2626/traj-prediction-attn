from __future__ import annotations

import argparse
import html
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Sequence, Tuple

try:
    import h3
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise ImportError("The 'h3' package is required. Install it with 'pip install h3'.") from exc

try:
    import folium
except ImportError:  # pragma: no cover - runtime dependency
    folium = None


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
GROUND_TRUTH_COLOR = "#6a4c93"
PATH_COLORS = ["#111111", "#2563eb", "#f97316", "#16a34a", "#db2777"]
NEIGHBOR_BORDER_DASH = "4, 6"
POLYGON_PANE = "prediction-polygons"
PATH_PANE = "representative-paths"
INTERACTION_PANE = "prediction-interaction"
MARKER_PANE = "prediction-markers"
HIT_PANE = "prediction-hit-markers"
OFFLINE_WIDTH = 1440
OFFLINE_HEIGHT = 760
OFFLINE_LAT_MIN = -85.0
OFFLINE_LAT_MAX = 85.0
VisualMode = Literal["candidate", "neighbor", "both", "topk"]


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


def _hit_icon() -> folium.DivIcon:
    return folium.DivIcon(
        html=(
            "<div style='transform: translateY(-14px);'>"
            "<span style=\"display:inline-block;padding:1px 6px;border-radius:10px;"
            "background:#ffffff;border:1px solid #d62828;color:#d62828;"
            "font-size:10px;font-weight:700;line-height:1.2;white-space:nowrap;\">"
            "HIT</span></div>"
        ),
        icon_size=(32, 18),
        icon_anchor=(16, 18),
        class_name="custom-hit-icon",
    )


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


def _grid_disk(cell: str, ring: int) -> List[str]:
    if ring <= 0:
        return [str(cell)]
    return sorted(str(item) for item in h3.grid_disk(str(cell), int(ring)))


def _sample_examples(examples: Sequence[Dict[str, object]], num_examples: int, seed: int) -> List[Dict[str, object]]:
    if num_examples <= 0 or num_examples >= len(examples):
        return list(examples)
    rng = random.Random(seed)
    return rng.sample(list(examples), int(num_examples))


def _truth_cell(horizon: Dict[str, object]) -> str | None:
    truth = horizon.get("ground_truth")
    if not truth or not truth.get("cell"):
        return None
    return str(truth["cell"])


def _top_region_predictions(horizon: Dict[str, object], top_region_k: int) -> List[Dict[str, object]]:
    predictions: List[Dict[str, object]] = []
    for prediction in horizon.get("top_region_predictions", []):
        cell = prediction.get("cell")
        if not cell:
            continue
        lat, lon = _cell_center(str(cell))
        predictions.append(
            {
                "horizon_minutes": int(horizon.get("horizon_minutes", 0)),
                "rank": int(prediction.get("rank", 0)),
                "cell": str(cell),
                "probability": float(prediction.get("probability", 0.0)),
                "lat": float(lat),
                "lon": float(lon),
            }
        )
        if len(predictions) >= max(1, int(top_region_k)):
            break
    return predictions


def _horizon_hit(
    horizon: Dict[str, object],
    visual_mode: VisualMode,
    max_candidate_regions: int,
    neighbor_top_k: int,
    top_region_k: int,
) -> bool:
    truth_cell = _truth_cell(horizon)
    if truth_cell is None:
        return False

    if visual_mode in ("candidate", "both"):
        for candidate in horizon.get("candidate_regions", [])[:max_candidate_regions]:
            if float(candidate.get("confidence", 0.0)) <= 0.0:
                continue
            if truth_cell in {str(cell) for cell in candidate.get("cells", [])}:
                return True

    if visual_mode in ("neighbor", "both"):
        neighbor_ring = int(horizon.get("neighbor_ring", 0))
        top_predictions = horizon.get("top_region_predictions", [])[:max(1, int(neighbor_top_k))]
        for prediction in top_predictions:
            cell = prediction.get("cell")
            if not cell:
                continue
            if truth_cell in _grid_disk(str(cell), neighbor_ring):
                return True

    if visual_mode == "topk":
        for prediction in _top_region_predictions(horizon, top_region_k):
            if truth_cell == str(prediction["cell"]):
                return True

    return False


def _count_hit_horizons(
    example: Dict[str, object],
    visual_mode: VisualMode,
    max_candidate_regions: int,
    neighbor_top_k: int,
    top_region_k: int,
) -> int:
    hit_count = 0
    for horizon in example.get("horizons", []):
        if _horizon_hit(
            horizon=horizon,
            visual_mode=visual_mode,
            max_candidate_regions=max_candidate_regions,
            neighbor_top_k=neighbor_top_k,
            top_region_k=top_region_k,
        ):
            hit_count += 1
    return hit_count


def _filter_examples_by_hit_horizons(
    examples: Sequence[Dict[str, object]],
    min_hit_horizons: int,
    visual_mode: VisualMode,
    max_candidate_regions: int,
    neighbor_top_k: int,
    top_region_k: int,
) -> List[Dict[str, object]]:
    if min_hit_horizons <= 0:
        return list(examples)

    filtered: List[Dict[str, object]] = []
    for example in examples:
        hit_count = _count_hit_horizons(
            example=example,
            visual_mode=visual_mode,
            max_candidate_regions=max_candidate_regions,
            neighbor_top_k=neighbor_top_k,
            top_region_k=top_region_k,
        )
        if hit_count >= min_hit_horizons:
            filtered.append(example)
    return filtered


def _iter_example_coordinates(example: Dict[str, object]) -> Iterable[Tuple[float, float]]:
    current_cell = example.get("current_cell")
    if current_cell:
        yield _cell_center(str(current_cell))

    for horizon in example.get("horizons", []):
        truth = horizon.get("ground_truth")
        if truth and truth.get("cell"):
            yield _cell_center(str(truth["cell"]))

        for candidate in horizon.get("candidate_regions", []):
            if float(candidate.get("confidence", 0.0)) <= 0.0:
                continue
            for cell in candidate.get("cells", []):
                for lat, lon in _cell_polygon(str(cell)):
                    yield float(lat), float(lon)

        for prediction in horizon.get("top_region_predictions", []):
            cell = prediction.get("cell")
            if not cell:
                continue
            for neighbor_cell in _grid_disk(str(cell), int(horizon.get("neighbor_ring", 0))):
                for lat, lon in _cell_polygon(neighbor_cell):
                    yield float(lat), float(lon)


def _map_bounds(examples: Sequence[Dict[str, object]]) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    latitudes: List[float] = []
    longitudes: List[float] = []

    for example in examples:
        for lat, lon in _iter_example_coordinates(example):
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


def _legend_html() -> str:
    items = "".join(
        (
            f"<div style='margin-bottom:4px;'>"
            f"<span style='display:inline-block;width:12px;height:12px;background:{color};"
            f"margin-right:6px;border:1px solid #333;'></span>{minutes} min"
            f"</div>"
        )
        for minutes, color in HORIZON_COLORS.items()
    )
    return (
        "<div style=\"position: fixed; bottom: 24px; left: 24px; z-index: 9999; "
        "background: rgba(255,255,255,0.92); border: 1px solid #999; border-radius: 6px; "
        "padding: 10px 12px; font-size: 12px; line-height: 1.3;\">"
        "<div style='font-weight:600; margin-bottom:6px;'>Horizons</div>"
        f"{items}"
        "<div style='margin-top:6px;'>Black = current<br>Purple = ground truth<br>"
        "Dashed outline = top-k neighbor ring<br>Solid/dashed line = representative path<br>Badge = hit cell</div>"
        "</div>"
    )


def _candidate_opacity(candidate_rank: int) -> float:
    return max(0.12, 0.36 - (candidate_rank - 1) * 0.05)


def _top_region_opacity(rank: int) -> float:
    return max(0.12, 0.34 - (max(1, int(rank)) - 1) * 0.04)


def _distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    earth_radius_km = 6371.0
    phi1 = math.radians(float(lat1))
    phi2 = math.radians(float(lat2))
    delta_phi = math.radians(float(lat2) - float(lat1))
    delta_lambda = math.radians(float(lon2) - float(lon1))
    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    return 2.0 * earth_radius_km * math.asin(math.sqrt(a))


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    delta_lambda = math.radians(float(lon2) - float(lon1))
    phi1 = math.radians(float(lat1))
    phi2 = math.radians(float(lat2))
    y = math.sin(delta_lambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(delta_lambda)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def _angle_diff_deg(left: float, right: float) -> float:
    diff = abs(float(left) - float(right)) % 360.0
    return min(diff, 360.0 - diff)


def _build_representative_paths(
    example: Dict[str, object],
    *,
    top_region_k: int,
    path_count: int,
    max_speed_kts: float,
) -> List[Dict[str, object]]:
    if path_count <= 0:
        return []

    current_cell = example.get("current_cell")
    if not current_cell:
        return []

    current_lat, current_lon = _cell_center(str(current_cell))
    paths: List[Dict[str, object]] = [
        {
            "points": [
                {
                    "horizon_minutes": 0,
                    "rank": 0,
                    "cell": str(current_cell),
                    "probability": 1.0,
                    "lat": float(current_lat),
                    "lon": float(current_lon),
                }
            ],
            "score": 0.0,
            "last_bearing": None,
        }
    ]

    horizons = sorted(example.get("horizons", []), key=lambda item: int(item.get("horizon_minutes", 0)))
    beam_width = max(30, int(path_count) * 20)

    for horizon in horizons:
        candidates = _top_region_predictions(horizon, top_region_k)
        if not candidates:
            continue

        new_paths: List[Dict[str, object]] = []
        for path in paths:
            points = list(path["points"])  # type: ignore[index]
            last_point = points[-1]
            last_lat = float(last_point["lat"])  # type: ignore[index]
            last_lon = float(last_point["lon"])  # type: ignore[index]
            last_minutes = int(last_point["horizon_minutes"])  # type: ignore[index]
            last_bearing = path.get("last_bearing")

            for candidate in candidates:
                candidate_lat = float(candidate["lat"])
                candidate_lon = float(candidate["lon"])
                horizon_minutes = int(candidate["horizon_minutes"])
                dt_minutes = max(1, horizon_minutes - last_minutes)
                distance_km = _distance_km(last_lat, last_lon, candidate_lat, candidate_lon)
                speed_kts = distance_km / (dt_minutes / 60.0) / 1.852

                probability = max(float(candidate["probability"]), 1.0e-6)
                probability_score = math.log(probability)
                speed_penalty = max(0.0, speed_kts - float(max_speed_kts)) / 80.0

                turn_penalty = 0.0
                next_bearing = last_bearing
                if distance_km > 5.0:
                    bearing = _bearing_deg(last_lat, last_lon, candidate_lat, candidate_lon)
                    if isinstance(last_bearing, (int, float)):
                        turn_penalty = _angle_diff_deg(float(last_bearing), bearing) / 180.0
                    next_bearing = bearing

                score = float(path["score"]) + probability_score - speed_penalty - 0.8 * turn_penalty
                new_paths.append(
                    {
                        "points": points + [candidate],
                        "score": float(score),
                        "last_bearing": next_bearing,
                    }
                )

        new_paths.sort(key=lambda item: float(item["score"]), reverse=True)
        paths = new_paths[:beam_width]

    return paths[: max(0, int(path_count))]


def _path_locations(path: Dict[str, object]) -> List[Tuple[float, float]]:
    locations: List[Tuple[float, float]] = []
    last_cell: str | None = None
    for point in path.get("points", []):  # type: ignore[union-attr]
        cell = str(point.get("cell", ""))  # type: ignore[union-attr]
        lat = float(point["lat"])  # type: ignore[index]
        lon = float(point["lon"])  # type: ignore[index]
        if locations and cell == last_cell:
            continue
        locations.append((lat, lon))
        last_cell = cell
    return locations


def _path_popup(example: Dict[str, object], path: Dict[str, object], path_rank: int) -> str:
    lines = [
        f"aircraft={example['aircraft_id']}",
        f"trajectory={example['trajectory_id']}",
        f"timestamp={example['timestamp']}",
        f"path_rank={path_rank}",
        f"score={float(path.get('score', 0.0)):.3f}",
    ]
    for point in path.get("points", []):  # type: ignore[union-attr]
        lines.append(
            " -> "
            f"H{int(point.get('horizon_minutes', 0))} "  # type: ignore[union-attr]
            f"rank={int(point.get('rank', 0))} "  # type: ignore[union-attr]
            f"p={float(point.get('probability', 0.0)):.3f} "  # type: ignore[union-attr]
            f"cell={point.get('cell')}"  # type: ignore[union-attr]
        )
    return "<br>".join(lines)


def _candidate_popup(example: Dict[str, object], horizon: Dict[str, object], candidate_rank: int, candidate: Dict[str, object]) -> str:
    return (
        f"aircraft={example['aircraft_id']}<br>"
        f"trajectory={example['trajectory_id']}<br>"
        f"timestamp={example['timestamp']}<br>"
        f"horizon={horizon['horizon_minutes']} min<br>"
        f"predicted_status={horizon['predicted_status']}<br>"
        f"candidate_rank={candidate_rank}<br>"
        f"confidence={float(candidate['confidence']):.3f}<br>"
        f"center_cell={candidate['center_cell']}"
    )


def _truth_popup(example: Dict[str, object], horizon: Dict[str, object]) -> str:
    truth = horizon.get("ground_truth", {})
    return (
        f"aircraft={example['aircraft_id']}<br>"
        f"trajectory={example['trajectory_id']}<br>"
        f"timestamp={example['timestamp']}<br>"
        f"horizon={horizon['horizon_minutes']} min<br>"
        f"status={truth.get('status')}<br>"
        f"cell={truth.get('cell')}"
    )


def _cell_interaction_popup(cell: str, details: Sequence[str]) -> str:
    detail_lines = "".join(f"<li>{item}</li>" for item in details)
    return (
        f"<b>cell={cell}</b><br>"
        f"<div style='margin-top:4px;'>overlaps:</div>"
        f"<ul style='margin:4px 0 0 16px; padding:0;'>{detail_lines}</ul>"
    )


def _cell_interaction_tooltip(cell: str, details: Sequence[str]) -> str:
    detail_lines = "<br>".join(details)
    return (
        "<div style='font-size:12px; line-height:1.35;'>"
        f"<b>cell={cell}</b><br>"
        f"{detail_lines}"
        "</div>"
    )


def _unique_details(details: Sequence[str]) -> List[str]:
    ordered: List[str] = []
    seen = set()
    for item in details:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _collect_cell_memberships(
    example: Dict[str, object],
    visual_mode: VisualMode,
    max_candidate_regions: int,
    neighbor_top_k: int,
    top_region_k: int,
) -> Dict[str, List[str]]:
    memberships: Dict[str, List[str]] = {}

    for horizon in example.get("horizons", []):
        horizon_minutes = int(horizon["horizon_minutes"])

        if visual_mode in ("candidate", "both"):
            for candidate_rank, candidate in enumerate(horizon.get("candidate_regions", [])[:max_candidate_regions], start=1):
                confidence = float(candidate.get("confidence", 0.0))
                if confidence <= 0.0:
                    continue
                for cell in candidate.get("cells", []):
                    memberships.setdefault(str(cell), []).append(
                        f"H{horizon_minutes} candidate#{candidate_rank} conf={confidence:.3f}"
                    )

        if visual_mode in ("neighbor", "both"):
            neighbor_ring = int(horizon.get("neighbor_ring", 0))
            top_predictions = horizon.get("top_region_predictions", [])[:max(1, int(neighbor_top_k))]
            for prediction in top_predictions:
                cell = prediction.get("cell")
                if not cell:
                    continue
                rank = int(prediction.get("rank", 0))
                probability = float(prediction.get("probability", 0.0))
                for neighbor_cell in _grid_disk(str(cell), neighbor_ring):
                    memberships.setdefault(str(neighbor_cell), []).append(
                        f"H{horizon_minutes} top{rank} ring={neighbor_ring} p={probability:.3f}"
                    )

        if visual_mode == "topk":
            for prediction in _top_region_predictions(horizon, top_region_k):
                rank = int(prediction.get("rank", 0))
                probability = float(prediction.get("probability", 0.0))
                memberships.setdefault(str(prediction["cell"]), []).append(
                    f"H{horizon_minutes} top{rank} p={probability:.3f}"
                )

        truth = horizon.get("ground_truth")
        if truth and truth.get("cell"):
            memberships.setdefault(str(truth["cell"]), []).append(
                f"H{horizon_minutes} ground_truth"
            )

    return memberships


def _add_cell_interaction_layer(
    group: folium.FeatureGroup,
    example: Dict[str, object],
    visual_mode: VisualMode,
    max_candidate_regions: int,
    neighbor_top_k: int,
    top_region_k: int,
) -> None:
    memberships = _collect_cell_memberships(
        example=example,
        visual_mode=visual_mode,
        max_candidate_regions=max_candidate_regions,
        neighbor_top_k=neighbor_top_k,
        top_region_k=top_region_k,
    )

    for cell, details in memberships.items():
        unique_details = _unique_details(details)
        popup = (
            f"aircraft={example['aircraft_id']}<br>"
            f"trajectory={example['trajectory_id']}<br>"
            f"timestamp={example['timestamp']}<br>"
            f"{_cell_interaction_popup(cell, unique_details)}"
        )
        tooltip = folium.Tooltip(
            _cell_interaction_tooltip(cell, unique_details),
            sticky=True,
        )
        folium.Polygon(
            locations=_cell_polygon(cell),
            color="#000000",
            weight=0,
            fill=True,
            fill_color="#000000",
            fill_opacity=0.01,
            popup=popup,
            tooltip=tooltip,
            pane=INTERACTION_PANE,
        ).add_to(group)


def _layer_name(example_index: int, example: Dict[str, object]) -> str:
    aircraft_id = str(example.get("aircraft_id", "unknown"))
    trajectory_id = str(example.get("trajectory_id", "trajectory"))
    timestamp = str(example.get("timestamp", ""))[:16].replace("T", " ")
    return f"ex{example_index:02d} | {aircraft_id} | {trajectory_id} | {timestamp}"


def _add_current_marker(group: folium.FeatureGroup, example: Dict[str, object]) -> None:
    current_cell = example.get("current_cell")
    if not current_cell:
        return
    lat, lon = _cell_center(str(current_cell))
    popup = (
        f"aircraft={example['aircraft_id']}<br>"
        f"trajectory={example['trajectory_id']}<br>"
        f"timestamp={example['timestamp']}<br>"
        f"current_cell={current_cell}"
    )
    folium.Marker(
        location=[lat, lon],
        icon=_point_icon(CURRENT_COLOR, 16),
        popup=popup,
        tooltip="current",
        pane=MARKER_PANE,
    ).add_to(group)


def _add_hit_marker(
    group: folium.FeatureGroup,
    lat: float,
    lon: float,
    example: Dict[str, object],
    horizon: Dict[str, object],
    hit_sources: List[str],
) -> None:
    popup = (
        f"aircraft={example['aircraft_id']}<br>"
        f"trajectory={example['trajectory_id']}<br>"
        f"timestamp={example['timestamp']}<br>"
        f"horizon={horizon['horizon_minutes']} min<br>"
        f"hit_sources={', '.join(hit_sources)}"
    )
    folium.Marker(
        location=[lat, lon],
        icon=_hit_icon(),
        popup=popup,
        tooltip=f"HIT ({', '.join(hit_sources)})",
        pane=HIT_PANE,
    ).add_to(group)


def _add_horizon_content(
    group: folium.FeatureGroup,
    example: Dict[str, object],
    horizon: Dict[str, object],
    horizon_idx: int,
    max_candidate_regions: int,
    visual_mode: VisualMode,
    neighbor_top_k: int,
    top_region_k: int,
) -> None:
    horizon_minutes = int(horizon["horizon_minutes"])
    color = _horizon_color(horizon_minutes, horizon_idx)
    truth = horizon.get("ground_truth")
    truth_cell = str(truth["cell"]) if truth and truth.get("cell") else None
    hit_sources: List[str] = []

    if visual_mode in ("candidate", "both"):
        for candidate_rank, candidate in enumerate(horizon.get("candidate_regions", [])[:max_candidate_regions], start=1):
            if float(candidate.get("confidence", 0.0)) <= 0.0:
                continue
            popup = _candidate_popup(example, horizon, candidate_rank, candidate)
            opacity = _candidate_opacity(candidate_rank)

            for cell in candidate.get("cells", []):
                cell = str(cell)
                is_hit_cell = truth_cell is not None and cell == truth_cell
                if is_hit_cell and "candidate" not in hit_sources:
                    hit_sources.append("candidate")
                folium.Polygon(
                    locations=_cell_polygon(cell),
                    color=color,
                    weight=1,
                    fill=True,
                    fill_color=color,
                    fill_opacity=opacity,
                    interactive=False,
                    pane=POLYGON_PANE,
                ).add_to(group)

    if visual_mode == "topk":
        for prediction in _top_region_predictions(horizon, top_region_k):
            cell = str(prediction["cell"])
            rank = int(prediction["rank"])
            probability = float(prediction["probability"])
            is_hit_cell = truth_cell is not None and cell == truth_cell
            if is_hit_cell and "topk" not in hit_sources:
                hit_sources.append("topk")
            popup = (
                f"aircraft={example['aircraft_id']}<br>"
                f"trajectory={example['trajectory_id']}<br>"
                f"timestamp={example['timestamp']}<br>"
                f"horizon={horizon_minutes} min<br>"
                f"top_rank={rank}<br>"
                f"probability={probability:.3f}<br>"
                f"cell={cell}"
            )
            folium.Polygon(
                locations=_cell_polygon(cell),
                color=color,
                weight=2,
                fill=True,
                fill_color=color,
                fill_opacity=_top_region_opacity(rank),
                popup=popup,
                tooltip=f"H{horizon_minutes} top{rank} p={probability:.3f}",
                pane=POLYGON_PANE,
            ).add_to(group)

    if visual_mode in ("neighbor", "both"):
        neighbor_ring = int(horizon.get("neighbor_ring", 0))
        top_predictions = horizon.get("top_region_predictions", [])[:max(1, int(neighbor_top_k))]

        for prediction in top_predictions:
            cell = prediction.get("cell")
            if not cell:
                continue
            rank = int(prediction.get("rank", 0))
            probability = float(prediction.get("probability", 0.0))
            popup = (
                f"aircraft={example['aircraft_id']}<br>"
                f"trajectory={example['trajectory_id']}<br>"
                f"timestamp={example['timestamp']}<br>"
                f"horizon={horizon_minutes} min<br>"
                f"top_rank={rank}<br>"
                f"probability={probability:.3f}<br>"
                f"cell={cell}<br>"
                f"neighbor_ring={neighbor_ring}"
            )

            ring_cells = _grid_disk(str(cell), neighbor_ring)
            for neighbor_cell in ring_cells:
                is_hit_cell = truth_cell is not None and str(neighbor_cell) == truth_cell
                if is_hit_cell and "neighbor" not in hit_sources:
                    hit_sources.append("neighbor")
                folium.Polygon(
                    locations=_cell_polygon(neighbor_cell),
                    color=color,
                    weight=2,
                    dash_array=NEIGHBOR_BORDER_DASH,
                    fill=True,
                    fill_color=color,
                    fill_opacity=max(0.05, 0.16 - (rank - 1) * 0.02),
                    interactive=False,
                    pane=POLYGON_PANE,
                ).add_to(group)

    if truth and truth.get("cell"):
        lat, lon = _cell_center(str(truth["cell"]))
        folium.Marker(
            location=[lat, lon],
            icon=_point_icon(GROUND_TRUTH_COLOR, 18),
            popup=_truth_popup(example, horizon),
            tooltip=f"GT {horizon_minutes}m",
            pane=MARKER_PANE,
        ).add_to(group)
        if hit_sources:
            _add_hit_marker(group, lat, lon, example, horizon, hit_sources)


def _add_representative_paths(
    group: folium.FeatureGroup,
    example: Dict[str, object],
    *,
    top_region_k: int,
    path_count: int,
    max_speed_kts: float,
) -> None:
    paths = _build_representative_paths(
        example,
        top_region_k=top_region_k,
        path_count=path_count,
        max_speed_kts=max_speed_kts,
    )
    for path_rank, path in enumerate(paths, start=1):
        locations = _path_locations(path)
        if len(locations) < 2:
            continue
        color = PATH_COLORS[(path_rank - 1) % len(PATH_COLORS)]
        folium.PolyLine(
            locations=[[lat, lon] for lat, lon in locations],
            color=color,
            weight=4 if path_rank == 1 else 3,
            opacity=0.9 if path_rank == 1 else 0.72,
            dash_array=None if path_rank == 1 else "8, 6",
            popup=_path_popup(example, path, path_rank),
            tooltip=f"representative path #{path_rank}",
            pane=PATH_PANE,
        ).add_to(group)


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def _offline_mercator_y(lat: float) -> float:
    clipped = max(OFFLINE_LAT_MIN, min(OFFLINE_LAT_MAX, float(lat)))
    radians = math.radians(clipped)
    return math.log(math.tan(math.pi / 4.0 + radians / 2.0))


def _offline_project(lat: float, lon: float) -> Tuple[float, float]:
    normalized_lon = ((float(lon) + 180.0) % 360.0) - 180.0
    x = (normalized_lon + 180.0) / 360.0 * OFFLINE_WIDTH
    y_max = _offline_mercator_y(OFFLINE_LAT_MAX)
    y_min = _offline_mercator_y(OFFLINE_LAT_MIN)
    y = (y_max - _offline_mercator_y(float(lat))) / (y_max - y_min) * OFFLINE_HEIGHT
    return x, y


def _offline_polygon_points(cell: str) -> str:
    points = []
    for lat, lon in _cell_polygon(str(cell)):
        x, y = _offline_project(lat, lon)
        points.append(f"{x:.2f},{y:.2f}")
    return " ".join(points)


def _offline_title(lines: Sequence[object]) -> str:
    return _escape("\n".join(str(line) for line in lines))


def _offline_polygon(
    cell: str,
    *,
    color: str,
    fill_opacity: float,
    stroke_width: float,
    dash_array: str | None,
    title_lines: Sequence[object],
) -> str:
    dash = f" stroke-dasharray=\"{_escape(dash_array)}\"" if dash_array else ""
    return (
        f"<polygon points=\"{_offline_polygon_points(cell)}\" "
        f"fill=\"{_escape(color)}\" fill-opacity=\"{float(fill_opacity):.3f}\" "
        f"stroke=\"{_escape(color)}\" stroke-width=\"{float(stroke_width):.2f}\"{dash}>"
        f"<title>{_offline_title(title_lines)}</title>"
        "</polygon>"
    )


def _offline_polyline(
    locations: Sequence[Tuple[float, float]],
    *,
    color: str,
    stroke_width: float,
    dash_array: str | None,
    title_lines: Sequence[object],
) -> str:
    points = []
    for lat, lon in locations:
        x, y = _offline_project(lat, lon)
        points.append(f"{x:.2f},{y:.2f}")
    dash = f" stroke-dasharray=\"{_escape(dash_array)}\"" if dash_array else ""
    return (
        f"<polyline points=\"{' '.join(points)}\" fill=\"none\" "
        f"stroke=\"{_escape(color)}\" stroke-width=\"{float(stroke_width):.2f}\" "
        f"stroke-linecap=\"round\" stroke-linejoin=\"round\"{dash}>"
        f"<title>{_offline_title(title_lines)}</title>"
        "</polyline>"
    )


def _offline_marker(
    lat: float,
    lon: float,
    *,
    color: str,
    radius: float,
    stroke: str,
    title_lines: Sequence[object],
) -> str:
    x, y = _offline_project(lat, lon)
    return (
        f"<circle cx=\"{x:.2f}\" cy=\"{y:.2f}\" r=\"{float(radius):.2f}\" "
        f"fill=\"{_escape(color)}\" stroke=\"{_escape(stroke)}\" stroke-width=\"2\">"
        f"<title>{_offline_title(title_lines)}</title>"
        "</circle>"
    )


def _offline_hit_badge(lat: float, lon: float, hit_sources: Sequence[str]) -> str:
    x, y = _offline_project(lat, lon)
    title = _offline_title(["HIT", f"sources={', '.join(hit_sources)}"])
    return (
        f"<g transform=\"translate({x + 8:.2f},{y - 24:.2f})\">"
        "<rect width=\"38\" height=\"18\" rx=\"8\" fill=\"#ffffff\" stroke=\"#d62828\" stroke-width=\"1\"/>"
        "<text x=\"19\" y=\"12\" text-anchor=\"middle\" "
        "font-family=\"Arial, sans-serif\" font-size=\"10\" font-weight=\"700\" fill=\"#d62828\">HIT</text>"
        f"<title>{title}</title>"
        "</g>"
    )


def _offline_grid_svg() -> str:
    elements: List[str] = [
        f"<rect x=\"0\" y=\"0\" width=\"{OFFLINE_WIDTH}\" height=\"{OFFLINE_HEIGHT}\" fill=\"#eef6f9\"/>"
    ]
    for lon in range(-180, 181, 30):
        x, _ = _offline_project(0.0, float(lon))
        elements.append(
            f"<line x1=\"{x:.2f}\" y1=\"0\" x2=\"{x:.2f}\" y2=\"{OFFLINE_HEIGHT}\" "
            "stroke=\"#c6d3d8\" stroke-width=\"1\"/>"
        )
        if lon not in (-180, 180):
            elements.append(
                f"<text x=\"{x + 3:.2f}\" y=\"18\" font-size=\"10\" fill=\"#73828a\">{lon}</text>"
            )
    for lat in range(-60, 61, 30):
        _, y = _offline_project(float(lat), 0.0)
        elements.append(
            f"<line x1=\"0\" y1=\"{y:.2f}\" x2=\"{OFFLINE_WIDTH}\" y2=\"{y:.2f}\" "
            "stroke=\"#c6d3d8\" stroke-width=\"1\"/>"
        )
        elements.append(
            f"<text x=\"8\" y=\"{y - 4:.2f}\" font-size=\"10\" fill=\"#73828a\">{lat}</text>"
        )
    return "\n".join(elements)


def _offline_horizon_elements(
    example: Dict[str, object],
    horizon: Dict[str, object],
    horizon_idx: int,
    max_candidate_regions: int,
    visual_mode: VisualMode,
    neighbor_top_k: int,
    top_region_k: int,
) -> str:
    elements: List[str] = []
    horizon_minutes = int(horizon["horizon_minutes"])
    color = _horizon_color(horizon_minutes, horizon_idx)
    truth = horizon.get("ground_truth")
    truth_cell = str(truth["cell"]) if truth and truth.get("cell") else None
    hit_sources: List[str] = []

    if visual_mode in ("candidate", "both"):
        for candidate_rank, candidate in enumerate(horizon.get("candidate_regions", [])[:max_candidate_regions], start=1):
            confidence = float(candidate.get("confidence", 0.0))
            if confidence <= 0.0:
                continue
            for cell in candidate.get("cells", []):
                cell = str(cell)
                if truth_cell is not None and cell == truth_cell and "candidate" not in hit_sources:
                    hit_sources.append("candidate")
                elements.append(
                    _offline_polygon(
                        cell,
                        color=color,
                        fill_opacity=_candidate_opacity(candidate_rank),
                        stroke_width=1.0,
                        dash_array=None,
                        title_lines=[
                            f"aircraft={example.get('aircraft_id')}",
                            f"trajectory={example.get('trajectory_id')}",
                            f"timestamp={example.get('timestamp')}",
                            f"horizon={horizon_minutes} min",
                            f"candidate_rank={candidate_rank}",
                            f"confidence={confidence:.3f}",
                            f"cell={cell}",
                            f"center_cell={candidate.get('center_cell')}",
                        ],
                    )
                )

    if visual_mode == "topk":
        for prediction in _top_region_predictions(horizon, top_region_k):
            cell = str(prediction["cell"])
            rank = int(prediction["rank"])
            probability = float(prediction["probability"])
            if truth_cell is not None and cell == truth_cell and "topk" not in hit_sources:
                hit_sources.append("topk")
            elements.append(
                _offline_polygon(
                    cell,
                    color=color,
                    fill_opacity=_top_region_opacity(rank),
                    stroke_width=2.0,
                    dash_array=None,
                    title_lines=[
                        f"aircraft={example.get('aircraft_id')}",
                        f"trajectory={example.get('trajectory_id')}",
                        f"timestamp={example.get('timestamp')}",
                        f"horizon={horizon_minutes} min",
                        f"top_rank={rank}",
                        f"probability={probability:.3f}",
                        f"cell={cell}",
                    ],
                )
            )

    if visual_mode in ("neighbor", "both"):
        neighbor_ring = int(horizon.get("neighbor_ring", 0))
        top_predictions = horizon.get("top_region_predictions", [])[:max(1, int(neighbor_top_k))]
        for prediction in top_predictions:
            cell = prediction.get("cell")
            if not cell:
                continue
            rank = int(prediction.get("rank", 0))
            probability = float(prediction.get("probability", 0.0))
            fill_opacity = max(0.05, 0.16 - (rank - 1) * 0.02)
            for neighbor_cell in _grid_disk(str(cell), neighbor_ring):
                neighbor_cell = str(neighbor_cell)
                if truth_cell is not None and neighbor_cell == truth_cell and "neighbor" not in hit_sources:
                    hit_sources.append("neighbor")
                elements.append(
                    _offline_polygon(
                        neighbor_cell,
                        color=color,
                        fill_opacity=fill_opacity,
                        stroke_width=1.5,
                        dash_array=NEIGHBOR_BORDER_DASH,
                        title_lines=[
                            f"aircraft={example.get('aircraft_id')}",
                            f"trajectory={example.get('trajectory_id')}",
                            f"timestamp={example.get('timestamp')}",
                            f"horizon={horizon_minutes} min",
                            f"top_rank={rank}",
                            f"probability={probability:.3f}",
                            f"cell={cell}",
                            f"neighbor_cell={neighbor_cell}",
                            f"neighbor_ring={neighbor_ring}",
                        ],
                    )
                )

    if truth_cell:
        lat, lon = _cell_center(truth_cell)
        elements.append(
            _offline_marker(
                lat,
                lon,
                color=GROUND_TRUTH_COLOR,
                radius=7.0,
                stroke="#ffffff",
                title_lines=[
                    f"ground_truth horizon={horizon_minutes} min",
                    f"status={truth.get('status') if truth else ''}",
                    f"cell={truth_cell}",
                ],
            )
        )
        if hit_sources:
            elements.append(_offline_hit_badge(lat, lon, hit_sources))

    return "\n".join(elements)


def _offline_path_elements(
    example: Dict[str, object],
    *,
    top_region_k: int,
    path_count: int,
    max_speed_kts: float,
) -> str:
    elements: List[str] = []
    paths = _build_representative_paths(
        example,
        top_region_k=top_region_k,
        path_count=path_count,
        max_speed_kts=max_speed_kts,
    )
    for path_rank, path in enumerate(paths, start=1):
        locations = _path_locations(path)
        if len(locations) < 2:
            continue
        color = PATH_COLORS[(path_rank - 1) % len(PATH_COLORS)]
        elements.append(
            _offline_polyline(
                locations,
                color=color,
                stroke_width=4.0 if path_rank == 1 else 3.0,
                dash_array=None if path_rank == 1 else "8 6",
                title_lines=[
                    f"representative path #{path_rank}",
                    f"aircraft={example.get('aircraft_id')}",
                    f"trajectory={example.get('trajectory_id')}",
                    f"timestamp={example.get('timestamp')}",
                    f"score={float(path.get('score', 0.0)):.3f}",
                ],
            )
        )
    return "\n".join(elements)


def _offline_example_group(
    example_index: int,
    example: Dict[str, object],
    *,
    visible: bool,
    max_candidate_regions: int,
    visual_mode: VisualMode,
    neighbor_top_k: int,
    top_region_k: int,
    path_count: int,
    max_speed_kts: float,
) -> str:
    layer_id = f"example-layer-{example_index}"
    elements: List[str] = []
    current_cell = example.get("current_cell")
    if current_cell:
        lat, lon = _cell_center(str(current_cell))
        elements.append(
            _offline_marker(
                lat,
                lon,
                color=CURRENT_COLOR,
                radius=6.0,
                stroke="#ffffff",
                title_lines=[
                    "current",
                    f"aircraft={example.get('aircraft_id')}",
                    f"trajectory={example.get('trajectory_id')}",
                    f"timestamp={example.get('timestamp')}",
                    f"cell={current_cell}",
                ],
            )
        )

    for horizon_idx, horizon in enumerate(example.get("horizons", [])):
        elements.append(
            _offline_horizon_elements(
                example=example,
                horizon=horizon,
                horizon_idx=horizon_idx,
                max_candidate_regions=max_candidate_regions,
                visual_mode=visual_mode,
                neighbor_top_k=neighbor_top_k,
                top_region_k=top_region_k,
            )
        )

    elements.append(
        _offline_path_elements(
            example,
            top_region_k=top_region_k,
            path_count=path_count,
            max_speed_kts=max_speed_kts,
        )
    )

    display = "inline" if visible else "none"
    return (
        f"<g id=\"{layer_id}\" class=\"example-layer\" style=\"display:{display}\">"
        f"<title>{_escape(_layer_name(example_index, example))}</title>"
        f"{''.join(elements)}"
        "</g>"
    )


def build_offline_examples_html(
    examples: Sequence[Dict[str, object]],
    max_candidate_regions: int,
    show_first_n: int,
    visual_mode: VisualMode,
    neighbor_top_k: int,
    top_region_k: int,
    path_count: int,
    max_speed_kts: float,
) -> str:
    visible_count = max(0, int(show_first_n))
    if visible_count == 0 and examples:
        visible_count = 1

    controls: List[str] = []
    layers: List[str] = []
    for example_index, example in enumerate(examples, start=1):
        layer_id = f"example-layer-{example_index}"
        checked = example_index <= visible_count
        label = _escape(_layer_name(example_index, example))
        controls.append(
            "<label class=\"layer-row\">"
            f"<input type=\"checkbox\" {'checked' if checked else ''} "
            f"onchange=\"toggleLayer('{layer_id}', this.checked)\">"
            f"<span>{label}</span>"
            "</label>"
        )
        layers.append(
            _offline_example_group(
                example_index,
                example,
                visible=checked,
                max_candidate_regions=max_candidate_regions,
                visual_mode=visual_mode,
                neighbor_top_k=neighbor_top_k,
                top_region_k=top_region_k,
                path_count=path_count,
                max_speed_kts=max_speed_kts,
            )
        )

    horizon_items = "".join(
        (
            "<div class=\"legend-row\">"
            f"<span class=\"swatch\" style=\"background:{_escape(color)}\"></span>"
            f"<span>{int(minutes)} min</span>"
            "</div>"
        )
        for minutes, color in HORIZON_COLORS.items()
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Offline prediction examples</title>
  <style>
    html, body {{
      margin: 0;
      height: 100%;
      font-family: Arial, sans-serif;
      background: #f5f7f8;
      color: #1f2933;
    }}
    .page {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 360px;
      height: 100vh;
    }}
    .map-wrap {{
      position: relative;
      overflow: auto;
      background: #d9e5e9;
    }}
    svg {{
      width: 100%;
      height: 100%;
      min-width: 980px;
      min-height: 620px;
      display: block;
    }}
    polygon, circle, rect, text {{
      vector-effect: non-scaling-stroke;
    }}
    polygon:hover, circle:hover {{
      stroke: #000000;
      stroke-width: 2.5;
    }}
    .side {{
      border-left: 1px solid #c7d0d5;
      background: #ffffff;
      display: flex;
      flex-direction: column;
      min-width: 0;
    }}
    .panel {{
      padding: 14px 16px;
      border-bottom: 1px solid #e1e7eb;
    }}
    .title {{
      font-weight: 700;
      font-size: 15px;
      margin-bottom: 8px;
    }}
    .meta {{
      font-size: 12px;
      color: #52616b;
      line-height: 1.45;
    }}
    .actions {{
      display: flex;
      gap: 8px;
      margin-top: 10px;
    }}
    button {{
      border: 1px solid #b7c2ca;
      background: #f8fafb;
      border-radius: 4px;
      padding: 6px 9px;
      cursor: pointer;
      font-size: 12px;
    }}
    button:hover {{
      background: #eef3f6;
    }}
    .layers {{
      overflow: auto;
      padding: 8px 12px 16px;
    }}
    .layer-row {{
      display: flex;
      gap: 8px;
      align-items: flex-start;
      font-size: 12px;
      line-height: 1.35;
      padding: 5px 4px;
      border-radius: 4px;
    }}
    .layer-row:hover {{
      background: #f2f6f8;
    }}
    .legend-row {{
      display: flex;
      gap: 7px;
      align-items: center;
      font-size: 12px;
      margin-bottom: 5px;
    }}
    .swatch {{
      display: inline-block;
      width: 12px;
      height: 12px;
      border: 1px solid #333333;
    }}
    @media (max-width: 900px) {{
      .page {{
        grid-template-columns: 1fr;
        grid-template-rows: minmax(0, 1fr) 260px;
      }}
      .side {{
        border-left: 0;
        border-top: 1px solid #c7d0d5;
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <main class="map-wrap">
      <svg viewBox="0 0 {OFFLINE_WIDTH} {OFFLINE_HEIGHT}" role="img" aria-label="Offline prediction examples map">
        {_offline_grid_svg()}
        {''.join(layers)}
      </svg>
    </main>
    <aside class="side">
      <section class="panel">
        <div class="title">Offline Examples</div>
        <div class="meta">
          No external JavaScript, CSS, or map tiles are loaded. The map uses a Web Mercator grid without a basemap.
          Hover over cells and points for details.
        </div>
        <div class="actions">
          <button type="button" onclick="setAllLayers(true)">Show all</button>
          <button type="button" onclick="setAllLayers(false)">Hide all</button>
        </div>
      </section>
      <section class="panel">
        <div class="title">Horizons</div>
        {horizon_items}
        <div class="meta">Black = current<br>Purple = ground truth<br>Dashed outline = top-k neighbor ring<br>Solid/dashed lines = representative paths<br>Badge = hit cell</div>
      </section>
      <section class="layers">
        {''.join(controls)}
      </section>
    </aside>
  </div>
  <script>
    function toggleLayer(id, visible) {{
      var el = document.getElementById(id);
      if (el) {{
        el.style.display = visible ? 'inline' : 'none';
      }}
    }}
    function setAllLayers(visible) {{
      var inputs = document.querySelectorAll('.layer-row input[type="checkbox"]');
      inputs.forEach(function(input) {{
        input.checked = visible;
        var match = input.getAttribute('onchange').match(/'([^']+)'/);
        if (match) {{
          toggleLayer(match[1], visible);
        }}
      }});
    }}
  </script>
</body>
</html>
"""


def build_examples_map(
    examples: Sequence[Dict[str, object]],
    max_candidate_regions: int,
    show_first_n: int,
    visual_mode: VisualMode,
    neighbor_top_k: int,
    top_region_k: int,
    path_count: int,
    max_speed_kts: float,
) -> folium.Map:
    if folium is None:
        raise ImportError("The 'folium' package is required for --render-mode folium. Use --offline for no-CDN HTML.")
    center, southwest, northeast = _map_bounds(examples)
    fmap = folium.Map(location=list(center), zoom_start=3, tiles="CartoDB positron")
    folium.map.CustomPane(POLYGON_PANE, z_index=400).add_to(fmap)
    folium.map.CustomPane(INTERACTION_PANE, z_index=500).add_to(fmap)
    folium.map.CustomPane(PATH_PANE, z_index=620).add_to(fmap)
    folium.map.CustomPane(MARKER_PANE, z_index=650).add_to(fmap)
    folium.map.CustomPane(HIT_PANE, z_index=680).add_to(fmap)

    for example_index, example in enumerate(examples, start=1):
        feature_group = folium.FeatureGroup(
            name=_layer_name(example_index, example),
            show=example_index <= max(1, show_first_n),
        )
        _add_current_marker(feature_group, example)
        for horizon_idx, horizon in enumerate(example.get("horizons", [])):
            _add_horizon_content(
                group=feature_group,
                example=example,
                horizon=horizon,
                horizon_idx=horizon_idx,
                max_candidate_regions=max_candidate_regions,
                visual_mode=visual_mode,
                neighbor_top_k=neighbor_top_k,
                top_region_k=top_region_k,
            )
        _add_representative_paths(
            feature_group,
            example,
            top_region_k=top_region_k,
            path_count=path_count,
            max_speed_kts=max_speed_kts,
        )
        _add_cell_interaction_layer(
            group=feature_group,
            example=example,
            visual_mode=visual_mode,
            max_candidate_regions=max_candidate_regions,
            neighbor_top_k=neighbor_top_k,
            top_region_k=top_region_k,
        )
        feature_group.add_to(fmap)

    fmap.fit_bounds([list(southwest), list(northeast)])
    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.get_root().html.add_child(folium.Element(_legend_html()))
    return fmap


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render evaluation examples.json into an interactive HTML map with one layer per example."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="artifacts/evaluations/global_h3_movement_gru_examples.json",
        help="Path to the examples JSON file generated by evaluation.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/example_maps/examples_map.html",
        help="Path to the output HTML map.",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=8,
        help="How many examples to render. Use 0 for all examples in the file.",
    )
    parser.add_argument(
        "--max-candidate-regions",
        type=int,
        default=5,
        help="Maximum number of candidate regions to show per horizon for each example.",
    )
    parser.add_argument(
        "--show-first-n",
        type=int,
        default=1,
        help="How many sample layers are visible by default when the map opens.",
    )
    parser.add_argument(
        "--visual-mode",
        type=str,
        default="candidate",
        choices=["candidate", "neighbor", "both", "topk"],
        help="Which prediction layers to show: candidate regions, neighbor rings, both, or top-k cells.",
    )
    parser.add_argument(
        "--neighbor-top-k",
        type=int,
        default=5,
        help="How many top predicted cells to visualize with neighbor rings.",
    )
    parser.add_argument(
        "--top-region-k",
        type=int,
        default=5,
        help="How many top predicted cells to draw per horizon when --visual-mode topk is used.",
    )
    parser.add_argument(
        "--path-count",
        type=int,
        default=0,
        help="How many representative connected paths to draw from current cell through top-k horizon cells.",
    )
    parser.add_argument(
        "--path-max-speed-kts",
        type=float,
        default=750.0,
        help="Speed above this is penalized when connecting top-k cells into representative paths.",
    )
    parser.add_argument(
        "--min-hit-horizons",
        type=int,
        default=0,
        help=(
            "Only visualize examples whose displayed predictions hit at least this many horizons. "
            "Set to 4 to keep only windows that hit all four horizons."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for example sampling.",
    )
    parser.add_argument(
        "--render-mode",
        type=str,
        default="folium",
        choices=["folium", "offline"],
        help=(
            "HTML rendering mode. 'folium' uses Leaflet/CDN/online map tiles; "
            "'offline' writes a self-contained SVG HTML with no external assets."
        ),
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Shortcut for --render-mode offline.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    with input_path.open("r", encoding="utf-8") as handle:
        examples = json.load(handle)

    visual_mode = str(args.visual_mode)
    max_candidate_regions = max(1, int(args.max_candidate_regions))
    neighbor_top_k = max(1, int(args.neighbor_top_k))
    top_region_k = max(1, int(args.top_region_k))
    path_count = max(0, int(args.path_count))
    path_max_speed_kts = max(1.0, float(args.path_max_speed_kts))
    min_hit_horizons = max(0, int(args.min_hit_horizons))

    filtered = _filter_examples_by_hit_horizons(
        examples=examples,
        min_hit_horizons=min_hit_horizons,
        visual_mode=visual_mode,
        max_candidate_regions=max_candidate_regions,
        neighbor_top_k=neighbor_top_k,
        top_region_k=top_region_k,
    )
    if min_hit_horizons > 0:
        print(
            "Hit-horizon filter:"
            f" kept {len(filtered)} / {len(examples)} examples"
            f" with min_hit_horizons={min_hit_horizons}"
        )
    if not filtered:
        raise ValueError(
            "No examples remain after applying the hit-horizon filter. "
            "Lower --min-hit-horizons or relax the displayed candidate settings."
        )

    selected = _sample_examples(filtered, args.num_examples, args.seed)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    render_mode = "offline" if bool(args.offline) else str(args.render_mode)
    if render_mode == "offline":
        output_path.write_text(
            build_offline_examples_html(
                examples=selected,
                max_candidate_regions=max_candidate_regions,
                show_first_n=max(0, int(args.show_first_n)),
                visual_mode=visual_mode,
                neighbor_top_k=neighbor_top_k,
                top_region_k=top_region_k,
                path_count=path_count,
                max_speed_kts=path_max_speed_kts,
            ),
            encoding="utf-8",
        )
    else:
        fmap = build_examples_map(
            examples=selected,
            max_candidate_regions=max_candidate_regions,
            show_first_n=max(0, int(args.show_first_n)),
            visual_mode=visual_mode,
            neighbor_top_k=neighbor_top_k,
            top_region_k=top_region_k,
            path_count=path_count,
            max_speed_kts=path_max_speed_kts,
        )
        fmap.save(str(output_path))
    print(f"Saved example map to: {output_path}")


if __name__ == "__main__":
    main()
