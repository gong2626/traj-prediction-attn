from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


EARTH_RADIUS_KM = 6371.0


@dataclass(frozen=True)
class Airport:
    code: str
    lat: float
    lon: float


@dataclass(frozen=True)
class RouteProfile:
    name: str
    airports: Dict[str, Airport]
    route_weights: Dict[str, Dict[str, float]]
    route_templates: Dict[Tuple[str, str], List[Tuple[float, float]]]


def _build_profile(
    name: str,
    *,
    airports: Dict[str, Airport],
    route_weights: Dict[str, Dict[str, float]],
    route_templates: Dict[Tuple[str, str], List[Tuple[float, float]]],
) -> RouteProfile:
    return RouteProfile(
        name=name,
        airports={str(code): airport for code, airport in airports.items()},
        route_weights={
            str(origin): {str(dest): float(weight) for dest, weight in destinations.items()}
            for origin, destinations in route_weights.items()
        },
        route_templates={
            (str(origin), str(destination)): [(float(lat), float(lon)) for lat, lon in points]
            for (origin, destination), points in route_templates.items()
        },
    )


EAST_ASIA_AIRPORTS: Dict[str, Airport] = {
    "RJTT": Airport("RJTT", 35.5494, 139.7798),
    "RJAA": Airport("RJAA", 35.7720, 140.3929),
    "WMKK": Airport("WMKK", 2.7456, 101.7099),
    "WSSS": Airport("WSSS", 1.3644, 103.9915),
    "RPLL": Airport("RPLL", 14.5086, 121.0198),
    "YPDN": Airport("YPDN", -12.4083, 130.8727),
    "YBBN": Airport("YBBN", -27.3842, 153.1175),
    "YSSY": Airport("YSSY", -33.9399, 151.1753),
}


EAST_ASIA_ROUTE_WEIGHTS: Dict[str, Dict[str, float]] = {
    "RJTT": {"YSSY": 0.55, "WMKK": 0.30, "RPLL": 0.15},
    "RJAA": {"YBBN": 0.45, "WSSS": 0.35, "RPLL": 0.20},
    "WMKK": {"RJTT": 0.45, "YPDN": 0.30, "WSSS": 0.25},
    "WSSS": {"RJAA": 0.40, "YPDN": 0.35, "RPLL": 0.25},
    "RPLL": {"RJTT": 0.50, "WMKK": 0.30, "WSSS": 0.20},
    "YPDN": {"YSSY": 0.45, "RJTT": 0.30, "WMKK": 0.25},
    "YBBN": {"RJAA": 0.45, "WSSS": 0.30, "YPDN": 0.25},
    "YSSY": {"RJTT": 0.45, "WSSS": 0.30, "YPDN": 0.25},
}


EAST_ASIA_ROUTE_TEMPLATES: Dict[Tuple[str, str], List[Tuple[float, float]]] = {
    ("RJTT", "YSSY"): [(31.0, 143.0), (14.0, 145.0), (-12.4, 130.9), (-24.5, 145.0)],
    ("RJTT", "WMKK"): [(28.0, 135.0), (20.0, 126.0), (14.5, 121.0), (7.0, 110.0)],
    ("RJTT", "RPLL"): [(27.0, 133.0), (21.0, 125.0)],
    ("RJAA", "YBBN"): [(30.0, 146.0), (16.0, 147.0), (-12.4, 130.9), (-22.0, 146.0)],
    ("RJAA", "WSSS"): [(28.0, 136.0), (18.5, 126.0), (9.5, 114.0)],
    ("RJAA", "RPLL"): [(26.0, 133.0), (18.0, 125.0)],
    ("WMKK", "RJTT"): [(7.0, 110.0), (14.5, 121.0), (20.0, 126.0), (28.0, 135.0)],
    ("WMKK", "YPDN"): [(4.0, 108.0), (-2.0, 116.0), (-8.0, 124.0)],
    ("WMKK", "WSSS"): [(2.0, 103.0)],
    ("WSSS", "RJAA"): [(9.5, 114.0), (18.5, 126.0), (28.0, 136.0)],
    ("WSSS", "YPDN"): [(-2.0, 113.0), (-7.0, 121.0)],
    ("WSSS", "RPLL"): [(6.0, 111.0), (10.0, 116.0)],
    ("RPLL", "RJTT"): [(21.0, 125.0), (27.0, 133.0)],
    ("RPLL", "WMKK"): [(10.0, 116.0), (6.0, 111.0)],
    ("RPLL", "WSSS"): [(9.0, 116.0), (6.0, 110.0)],
    ("YPDN", "YSSY"): [(-18.0, 137.0), (-25.0, 145.0)],
    ("YPDN", "RJTT"): [(5.0, 138.0), (20.0, 143.0), (31.0, 143.0)],
    ("YPDN", "WMKK"): [(-7.0, 124.0), (-1.0, 116.0), (4.0, 108.0)],
    ("YBBN", "RJAA"): [(-22.0, 146.0), (-12.4, 130.9), (16.0, 147.0), (30.0, 146.0)],
    ("YBBN", "WSSS"): [(-18.0, 139.0), (-6.0, 124.0), (2.0, 112.0)],
    ("YBBN", "YPDN"): [(-20.0, 142.0)],
    ("YSSY", "RJTT"): [(-24.5, 145.0), (-12.4, 130.9), (14.0, 145.0), (31.0, 143.0)],
    ("YSSY", "WSSS"): [(-23.0, 145.0), (-12.4, 130.9), (-2.0, 116.0)],
    ("YSSY", "YPDN"): [(-25.0, 145.0), (-18.0, 137.0)],
}


EUROPE_MIDEAST_AIRPORTS: Dict[str, Airport] = {
    "EGLL": Airport("EGLL", 51.4700, -0.4543),
    "LFPG": Airport("LFPG", 49.0097, 2.5479),
    "EDDF": Airport("EDDF", 50.0379, 8.5622),
    "LIRF": Airport("LIRF", 41.8003, 12.2389),
    "LTBA": Airport("LTBA", 40.9769, 28.8146),
    "HECA": Airport("HECA", 30.1219, 31.4056),
    "OMDB": Airport("OMDB", 25.2532, 55.3657),
    "OTHH": Airport("OTHH", 25.2731, 51.6081),
}


EUROPE_MIDEAST_ROUTE_WEIGHTS: Dict[str, Dict[str, float]] = {
    "EGLL": {"OMDB": 0.45, "HECA": 0.30, "LTBA": 0.25},
    "LFPG": {"OMDB": 0.40, "HECA": 0.30, "LIRF": 0.30},
    "EDDF": {"OMDB": 0.40, "OTHH": 0.35, "LTBA": 0.25},
    "LIRF": {"HECA": 0.35, "OMDB": 0.35, "LTBA": 0.30},
    "LTBA": {"OMDB": 0.40, "HECA": 0.35, "EDDF": 0.25},
    "HECA": {"LIRF": 0.35, "OMDB": 0.35, "EGLL": 0.30},
    "OMDB": {"EGLL": 0.35, "EDDF": 0.35, "OTHH": 0.30},
    "OTHH": {"EDDF": 0.40, "LTBA": 0.30, "HECA": 0.30},
}


EUROPE_MIDEAST_ROUTE_TEMPLATES: Dict[Tuple[str, str], List[Tuple[float, float]]] = {
    ("EGLL", "OMDB"): [(50.0, 6.0), (47.0, 14.0), (41.0, 26.0), (34.0, 41.0)],
    ("EGLL", "HECA"): [(50.5, 5.0), (46.0, 12.0), (39.0, 20.0), (34.0, 28.0)],
    ("EGLL", "LTBA"): [(51.0, 5.0), (48.0, 12.0), (44.0, 20.0)],
    ("LFPG", "OMDB"): [(48.0, 7.0), (45.0, 15.0), (39.0, 27.0), (33.0, 42.0)],
    ("LFPG", "HECA"): [(47.0, 6.0), (43.0, 13.0), (36.0, 24.0)],
    ("LFPG", "LIRF"): [(46.0, 6.0), (44.0, 10.0)],
    ("EDDF", "OMDB"): [(48.0, 12.0), (43.0, 21.0), (36.0, 35.0)],
    ("EDDF", "OTHH"): [(47.0, 12.0), (42.0, 22.0), (33.0, 41.0)],
    ("EDDF", "LTBA"): [(48.0, 12.0), (44.0, 20.0)],
    ("LIRF", "HECA"): [(38.0, 18.0), (34.0, 24.0)],
    ("LIRF", "OMDB"): [(38.0, 18.0), (33.0, 28.0), (29.0, 40.0)],
    ("LIRF", "LTBA"): [(41.0, 18.0), (41.0, 24.0)],
    ("LTBA", "OMDB"): [(38.0, 34.0), (33.0, 44.0)],
    ("LTBA", "HECA"): [(37.0, 31.0), (33.0, 31.0)],
    ("LTBA", "EDDF"): [(44.0, 20.0), (48.0, 12.0)],
    ("HECA", "LIRF"): [(34.0, 24.0), (38.0, 18.0)],
    ("HECA", "OMDB"): [(28.0, 36.0), (25.5, 46.0)],
    ("HECA", "EGLL"): [(34.0, 28.0), (39.0, 20.0), (46.0, 10.0)],
    ("OMDB", "EGLL"): [(34.0, 41.0), (41.0, 26.0), (47.0, 14.0), (50.0, 6.0)],
    ("OMDB", "EDDF"): [(36.0, 35.0), (43.0, 21.0), (48.0, 12.0)],
    ("OMDB", "OTHH"): [(25.3, 53.0)],
    ("OTHH", "EDDF"): [(32.0, 40.0), (42.0, 22.0), (47.0, 12.0)],
    ("OTHH", "LTBA"): [(33.0, 40.0), (37.0, 31.0)],
    ("OTHH", "HECA"): [(27.0, 44.0), (28.0, 36.0)],
}


BUILTIN_PROFILES: Dict[str, RouteProfile] = {
    "east_asia": _build_profile(
        "east_asia",
        airports=EAST_ASIA_AIRPORTS,
        route_weights=EAST_ASIA_ROUTE_WEIGHTS,
        route_templates=EAST_ASIA_ROUTE_TEMPLATES,
    ),
    "europe_mideast": _build_profile(
        "europe_mideast",
        airports=EUROPE_MIDEAST_AIRPORTS,
        route_weights=EUROPE_MIDEAST_ROUTE_WEIGHTS,
        route_templates=EUROPE_MIDEAST_ROUTE_TEMPLATES,
    ),
}


def _load_profile_from_json(path: Path) -> RouteProfile:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    airports = {
        str(code): Airport(str(code), float(item["lat"]), float(item["lon"]))
        for code, item in payload["airports"].items()
    }
    route_weights = {
        str(origin): {str(dest): float(weight) for dest, weight in destinations.items()}
        for origin, destinations in payload["route_weights"].items()
    }
    route_templates = {}
    for key, points in payload["route_templates"].items():
        if "->" not in str(key):
            raise ValueError("Custom route template keys must use the form ORIGIN->DEST.")
        origin, destination = [part.strip() for part in str(key).split("->", maxsplit=1)]
        route_templates[(origin, destination)] = [(float(lat), float(lon)) for lat, lon in points]

    return _build_profile(
        str(payload.get("name") or path.stem),
        airports=airports,
        route_weights=route_weights,
        route_templates=route_templates,
    )


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
    lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1_rad, lat2_rad = math.radians(lat1), math.radians(lat2)
    dlon_rad = math.radians(lon2 - lon1)
    y = math.sin(dlon_rad) * math.cos(lat2_rad)
    x = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon_rad)
    bearing = math.degrees(math.atan2(y, x))
    return (bearing + 360.0) % 360.0


def resolve_route_profile(profile_name: str, profile_json: Optional[Path] = None) -> RouteProfile:
    if profile_json is not None:
        return _load_profile_from_json(profile_json)
    if profile_name not in BUILTIN_PROFILES:
        raise ValueError(
            f"Unknown profile '{profile_name}'. Available profiles: {sorted(BUILTIN_PROFILES)}"
        )
    return BUILTIN_PROFILES[profile_name]


def route_points(profile: RouteProfile, origin: str, destination: str) -> List[Tuple[float, float]]:
    origin_airport = profile.airports[origin]
    destination_airport = profile.airports[destination]
    key = (origin, destination)

    if key in profile.route_templates:
        intermediates = profile.route_templates[key]
    elif (destination, origin) in profile.route_templates:
        intermediates = list(reversed(profile.route_templates[(destination, origin)]))
    else:
        intermediates = []

    points = [(origin_airport.lat, origin_airport.lon)]
    points.extend(intermediates)
    points.append((destination_airport.lat, destination_airport.lon))
    return points


def polyline_distances(points: Sequence[Tuple[float, float]]) -> Tuple[List[float], float]:
    cumulative = [0.0]
    total = 0.0
    for start, end in zip(points[:-1], points[1:]):
        total += haversine_km(start[0], start[1], end[0], end[1])
        cumulative.append(total)
    return cumulative, total


def interpolate_route(points: Sequence[Tuple[float, float]], fraction: float) -> Tuple[float, float, float]:
    cumulative, total = polyline_distances(points)
    if total <= 1e-6:
        lat, lon = points[0]
        return lat, lon, 0.0

    distance = min(max(fraction, 0.0), 1.0) * total
    for idx in range(len(points) - 1):
        left, right = cumulative[idx], cumulative[idx + 1]
        if right >= distance:
            leg_fraction = 0.0 if right <= left else (distance - left) / (right - left)
            start_lat, start_lon = points[idx]
            end_lat, end_lon = points[idx + 1]
            lat = start_lat + leg_fraction * (end_lat - start_lat)
            lon = start_lon + leg_fraction * (end_lon - start_lon)
            heading = bearing_deg(start_lat, start_lon, end_lat, end_lon)
            return lat, lon, heading

    lat, lon = points[-1]
    prev_lat, prev_lon = points[-2]
    return lat, lon, bearing_deg(prev_lat, prev_lon, lat, lon)


def apply_lateral_jitter(
    lat: float,
    lon: float,
    track_bearing_deg: float,
    offset_km: float,
) -> Tuple[float, float]:
    perpendicular = math.radians(track_bearing_deg + 90.0)
    lat_offset = (offset_km / 111.0) * math.cos(perpendicular)
    lon_scale = max(0.1, 111.0 * math.cos(math.radians(lat)))
    lon_offset = (offset_km / lon_scale) * math.sin(perpendicular)
    return lat + lat_offset, lon + lon_offset


def altitude_profile(fraction: float, cruise_altitude_m: float) -> float:
    if fraction <= 0.05:
        return 0.0
    if fraction <= 0.22:
        return cruise_altitude_m * (fraction - 0.05) / 0.17
    if fraction <= 0.78:
        return cruise_altitude_m
    if fraction <= 0.95:
        return cruise_altitude_m * (0.95 - fraction) / 0.17
    return 0.0


def choose_destination(rng: np.random.Generator, profile: RouteProfile, departure: str) -> str:
    choices = profile.route_weights[departure]
    destinations = list(choices.keys())
    probs = np.asarray(list(choices.values()), dtype=float)
    probs = probs / probs.sum()
    return str(rng.choice(destinations, p=probs))


def drop_airborne_points(
    rows: List[Dict[str, object]],
    rng: np.random.Generator,
    drop_rate: float,
    missing_segment_prob: float,
    max_missing_segment_points: int,
) -> List[Dict[str, object]]:
    airborne_indices = [idx for idx, row in enumerate(rows) if row["phase"] == "airborne"]
    keep = np.ones(len(rows), dtype=bool)

    for idx in airborne_indices:
        if rng.random() < drop_rate:
            keep[idx] = False

    if airborne_indices and rng.random() < missing_segment_prob:
        max_len = min(max_missing_segment_points, max(1, len(airborne_indices) // 5))
        if max_len > 0:
            segment_len = int(rng.integers(1, max_len + 1))
            start_pos = int(rng.integers(0, max(1, len(airborne_indices) - segment_len + 1)))
            for idx in airborne_indices[start_pos:start_pos + segment_len]:
                keep[idx] = False

    trimmed = [row for idx, row in enumerate(rows) if keep[idx]]
    return trimmed if len(trimmed) >= 6 else rows


def recompute_kinematics(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    if not rows:
        return rows

    rows = sorted(rows, key=lambda item: item["timestamp"])
    previous = None
    for row in rows:
        if previous is None:
            row["speed"] = float(row.get("speed", 0.0))
            row["rotation"] = float(row.get("rotation", 0.0))
            row["vrate"] = float(row.get("vrate", 0.0))
            previous = row
            continue

        dt_seconds = max(1.0, (row["timestamp"] - previous["timestamp"]).total_seconds())
        distance_km = haversine_km(previous["latitude"], previous["longitude"], row["latitude"], row["longitude"])
        speed_knots = (distance_km / (dt_seconds / 3600.0)) / 1.852
        row["speed"] = round(speed_knots, 2)
        row["rotation"] = round(
            bearing_deg(previous["latitude"], previous["longitude"], row["latitude"], row["longitude"]),
            2,
        )
        row["vrate"] = round((row["altitude"] - previous["altitude"]) / dt_seconds * 60.0, 2)
        previous = row

    return rows


def simulate_mission(
    profile: RouteProfile,
    aircraft_id: str,
    mission_idx: int,
    origin: str,
    destination: str,
    mission_start: datetime,
    rng: np.random.Generator,
    min_step_minutes: int,
    max_step_minutes: int,
    jitter_km: float,
    drop_rate: float,
    missing_segment_prob: float,
    max_missing_segment_points: int,
) -> Tuple[List[Dict[str, object]], datetime]:
    route = route_points(profile, origin, destination)
    _, total_distance_km = polyline_distances(route)
    cruise_altitude_m = float(rng.uniform(9000.0, 11500.0))
    cruise_speed_kmh = float(rng.uniform(760.0, 860.0))
    airborne_minutes = total_distance_km / cruise_speed_kmh * 60.0
    airborne_minutes *= float(rng.uniform(1.05, 1.20))

    mean_step = float(rng.uniform(min_step_minutes, max_step_minutes))
    airborne_steps = max(18, int(math.ceil(airborne_minutes / mean_step)))

    rows: List[Dict[str, object]] = []
    timestamp = mission_start

    for ground_idx in range(2):
        rows.append(
            {
                "target_id": aircraft_id,
                "timestamp": timestamp,
                "longitude": profile.airports[origin].lon,
                "latitude": profile.airports[origin].lat,
                "altitude": 0.0,
                "speed": 0.0,
                "rotation": 0.0,
                "vrate": 0.0,
                "route_id": f"{origin}_{destination}",
                "departure_airport": origin,
                "destination_airport": destination,
                "mission_id": mission_idx,
                "phase": "ground",
            }
        )
        timestamp += timedelta(minutes=5)

    for step_idx in range(airborne_steps):
        fraction = step_idx / max(1, airborne_steps - 1)
        lat, lon, heading = interpolate_route(route, fraction)
        phase_altitude = altitude_profile(fraction, cruise_altitude_m)

        if 0.04 < fraction < 0.96:
            offset = float(rng.normal(0.0, jitter_km))
            lat, lon = apply_lateral_jitter(lat, lon, heading, offset)

        rows.append(
            {
                "target_id": aircraft_id,
                "timestamp": timestamp,
                "longitude": lon,
                "latitude": lat,
                "altitude": phase_altitude,
                "speed": 0.0,
                "rotation": heading,
                "vrate": 0.0,
                "route_id": f"{origin}_{destination}",
                "departure_airport": origin,
                "destination_airport": destination,
                "mission_id": mission_idx,
                "phase": "airborne",
            }
        )
        step_minutes = float(rng.uniform(min_step_minutes, max_step_minutes))
        timestamp += timedelta(minutes=step_minutes)

    arrival_time = timestamp
    for delta_minutes in (5, 12, 20):
        rows.append(
            {
                "target_id": aircraft_id,
                "timestamp": arrival_time + timedelta(minutes=delta_minutes),
                "longitude": profile.airports[destination].lon,
                "latitude": profile.airports[destination].lat,
                "altitude": 0.0,
                "speed": 0.0,
                "rotation": 0.0,
                "vrate": 0.0,
                "route_id": f"{origin}_{destination}",
                "departure_airport": origin,
                "destination_airport": destination,
                "mission_id": mission_idx,
                "phase": "ground",
            }
        )

    rows = drop_airborne_points(rows, rng, drop_rate, missing_segment_prob, max_missing_segment_points)
    rows = recompute_kinematics(rows)
    next_start = rows[-1]["timestamp"] + timedelta(hours=float(rng.uniform(3.0, 24.0)))
    return rows, next_start


def write_aircraft_history(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "target_id",
        "timestamp",
        "longitude",
        "latitude",
        "altitude",
        "speed",
        "rotation",
        "vrate",
        "route_id",
        "departure_airport",
        "destination_airport",
        "mission_id",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = {key: row[key] for key in fieldnames}
            payload["timestamp"] = row["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow(payload)


def build_aircraft_ids(num_aircraft: int, id_prefix: str) -> List[str]:
    return [f"{id_prefix}{index:04d}" for index in range(num_aircraft)]


def generate_dataset(
    output_dir: Path,
    route_profile: RouteProfile,
    id_prefix: str,
    num_aircraft: int,
    missions_min: int,
    missions_max: int,
    seed: int,
    min_step_minutes: int,
    max_step_minutes: int,
    jitter_km: float,
    drop_rate: float,
    missing_segment_prob: float,
    max_missing_segment_points: int,
) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, object]] = []
    route_counts: Dict[str, int] = {}
    aircraft_ids = build_aircraft_ids(num_aircraft, id_prefix)

    for aircraft_id in aircraft_ids:
        home = str(rng.choice(list(route_profile.route_weights.keys())))
        current_airport = home
        mission_count = int(rng.integers(missions_min, missions_max + 1))
        current_time = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(
            hours=float(rng.uniform(0.0, 24.0 * 14.0))
        )
        all_rows: List[Dict[str, object]] = []

        for mission_idx in range(mission_count):
            destination = choose_destination(rng, route_profile, current_airport)
            mission_rows, current_time = simulate_mission(
                profile=route_profile,
                aircraft_id=aircraft_id,
                mission_idx=mission_idx,
                origin=current_airport,
                destination=destination,
                mission_start=current_time,
                rng=rng,
                min_step_minutes=min_step_minutes,
                max_step_minutes=max_step_minutes,
                jitter_km=jitter_km,
                drop_rate=drop_rate,
                missing_segment_prob=missing_segment_prob,
                max_missing_segment_points=max_missing_segment_points,
            )
            all_rows.extend(mission_rows)
            summary_rows.append(
                {
                    "aircraft_id": aircraft_id,
                    "mission_id": mission_idx,
                    "origin": current_airport,
                    "destination": destination,
                    "points": len(mission_rows),
                    "start_time": mission_rows[0]["timestamp"].isoformat(),
                    "end_time": mission_rows[-1]["timestamp"].isoformat(),
                }
            )
            route_key = f"{current_airport}->{destination}"
            route_counts[route_key] = route_counts.get(route_key, 0) + 1
            current_airport = destination

        all_rows = sorted(all_rows, key=lambda item: item["timestamp"])
        write_aircraft_history(output_dir / f"{aircraft_id}.csv", all_rows)

    summary = {
        "num_aircraft": num_aircraft,
        "missions_total": len(summary_rows),
        "missions_min": missions_min,
        "missions_max": missions_max,
        "seed": seed,
        "profile": route_profile.name,
        "id_prefix": id_prefix,
        "directed_route_count": len(route_counts),
        "directed_routes_available": len(route_profile.route_templates),
        "route_counts": dict(sorted(route_counts.items(), key=lambda item: (-item[1], item[0]))),
        "airports": {code: vars(airport) for code, airport in route_profile.airports.items()},
        "route_weights": route_profile.route_weights,
        "missions": summary_rows,
    }
    with (output_dir / "synthetic_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic multi-mission aircraft trajectories.")
    parser.add_argument("--output-dir", type=str, default="data/sim_csv")
    parser.add_argument(
        "--profile",
        type=str,
        default="east_asia",
        choices=sorted(BUILTIN_PROFILES.keys()),
        help="Built-in route profile to use.",
    )
    parser.add_argument(
        "--profile-json",
        type=str,
        default=None,
        help="Optional custom route profile JSON path. When provided, it overrides --profile.",
    )
    parser.add_argument(
        "--id-prefix",
        type=str,
        default="sim_",
        help="Prefix used for generated aircraft ids and file names.",
    )
    parser.add_argument("--num-aircraft", type=int, default=80)
    parser.add_argument("--missions-min", type=int, default=4)
    parser.add_argument("--missions-max", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-step-minutes", type=int, default=3)
    parser.add_argument("--max-step-minutes", type=int, default=8)
    parser.add_argument("--jitter-km", type=float, default=8.0)
    parser.add_argument("--drop-rate", type=float, default=0.03)
    parser.add_argument("--missing-segment-prob", type=float, default=0.20)
    parser.add_argument("--max-missing-segment-points", type=int, default=6)
    args = parser.parse_args()

    route_profile = resolve_route_profile(
        profile_name=args.profile,
        profile_json=Path(args.profile_json) if args.profile_json else None,
    )
    summary = generate_dataset(
        output_dir=Path(args.output_dir),
        route_profile=route_profile,
        id_prefix=args.id_prefix,
        num_aircraft=args.num_aircraft,
        missions_min=args.missions_min,
        missions_max=args.missions_max,
        seed=args.seed,
        min_step_minutes=args.min_step_minutes,
        max_step_minutes=args.max_step_minutes,
        jitter_km=args.jitter_km,
        drop_rate=args.drop_rate,
        missing_segment_prob=args.missing_segment_prob,
        max_missing_segment_points=args.max_missing_segment_points,
    )

    print(
        "Synthetic dataset ready:",
        f"profile={summary['profile']}",
        f"id_prefix={summary['id_prefix']}",
        f"aircraft={summary['num_aircraft']}",
        f"missions={summary['missions_total']}",
        f"output_dir={args.output_dir}",
    )


if __name__ == "__main__":
    main()
