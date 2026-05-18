from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

try:
    import folium
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise ImportError("The 'folium' package is required. Install it with 'pip install folium'.") from exc


ROUTE_COLORS = [
    "#e63946",
    "#1d3557",
    "#2a9d8f",
    "#f4a261",
    "#6a4c93",
    "#ff006e",
    "#1982c4",
    "#8ac926",
    "#ffca3a",
    "#ff595e",
    "#577590",
    "#b56576",
]


def _load_frame(input_dir: Path) -> pd.DataFrame:
    paths = sorted(input_dir.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["_source_file"] = path.name
        frames.append(frame)

    merged = pd.concat(frames, ignore_index=True)
    merged["timestamp"] = pd.to_datetime(merged["timestamp"], utc=True, errors="coerce")
    merged = merged.dropna(subset=["timestamp", "latitude", "longitude"])
    return merged


def _route_palette(route_ids: List[str]) -> Dict[str, str]:
    palette = {}
    for idx, route_id in enumerate(sorted(route_ids)):
        palette[route_id] = ROUTE_COLORS[idx % len(ROUTE_COLORS)]
    return palette


def build_map(
    frame: pd.DataFrame,
    max_aircraft: int,
    max_missions_per_aircraft: int,
) -> folium.Map:
    aircraft_ids = frame["target_id"].drop_duplicates().tolist()
    if max_aircraft > 0:
        aircraft_ids = aircraft_ids[:max_aircraft]
        frame = frame[frame["target_id"].isin(aircraft_ids)].copy()

    if max_missions_per_aircraft > 0:
        keep_keys: List[Tuple[str, int]] = []
        for aircraft_id, group in frame.groupby("target_id", sort=False):
            mission_ids = sorted(group["mission_id"].drop_duplicates().tolist())[:max_missions_per_aircraft]
            keep_keys.extend((aircraft_id, mission_id) for mission_id in mission_ids)
        keep_df = pd.DataFrame(keep_keys, columns=["target_id", "mission_id"])
        frame = frame.merge(keep_df, on=["target_id", "mission_id"], how="inner")

    center_lat = float(frame["latitude"].mean())
    center_lon = float(frame["longitude"].mean())
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=3, tiles="CartoDB positron")

    route_palette = _route_palette(frame["route_id"].drop_duplicates().astype(str).tolist())

    for route_id, route_group in frame.groupby("route_id", sort=True):
        feature_group = folium.FeatureGroup(name=str(route_id), show=True)
        for (aircraft_id, mission_id), mission in route_group.groupby(["target_id", "mission_id"], sort=False):
            mission = mission.sort_values("timestamp")
            coords = mission[["latitude", "longitude"]].to_numpy().tolist()
            if len(coords) < 2:
                continue
            tooltip = (
                f"aircraft={aircraft_id}<br>"
                f"mission={mission_id}<br>"
                f"route={route_id}<br>"
                f"start={mission['timestamp'].iloc[0]}<br>"
                f"end={mission['timestamp'].iloc[-1]}"
            )
            folium.PolyLine(
                locations=coords,
                color=route_palette[str(route_id)],
                weight=2,
                opacity=0.55,
                tooltip=tooltip,
            ).add_to(feature_group)

            start = mission.iloc[0]
            end = mission.iloc[-1]
            folium.CircleMarker(
                location=[float(start["latitude"]), float(start["longitude"])],
                radius=3,
                color=route_palette[str(route_id)],
                fill=True,
                fill_opacity=0.9,
                tooltip=f"DEP {aircraft_id} mission={mission_id}",
            ).add_to(feature_group)
            folium.CircleMarker(
                location=[float(end["latitude"]), float(end["longitude"])],
                radius=3,
                color=route_palette[str(route_id)],
                fill=True,
                fill_opacity=0.9,
                tooltip=f"ARR {aircraft_id} mission={mission_id}",
            ).add_to(feature_group)

        feature_group.add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)
    return fmap


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize synthetic trajectories on a basemap.")
    parser.add_argument("--input-dir", type=str, default="data/sim_csv")
    parser.add_argument("--output", type=str, default="artifacts/synthetic_preview/synthetic_trajectories_map.html")
    parser.add_argument("--max-aircraft", type=int, default=40)
    parser.add_argument("--max-missions-per-aircraft", type=int, default=6)
    args = parser.parse_args()

    frame = _load_frame(Path(args.input_dir))
    fmap = build_map(
        frame=frame,
        max_aircraft=args.max_aircraft,
        max_missions_per_aircraft=args.max_missions_per_aircraft,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(output_path))
    print(f"Saved map to: {output_path}")


if __name__ == "__main__":
    main()
