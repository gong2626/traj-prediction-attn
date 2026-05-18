from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from regional_forecast.config import ProjectConfig
from regional_forecast.data.regionizer import H3Regionizer


EARTH_RADIUS_KM = 6371.0088
KNOT_KMH = 1.852


def _great_circle_km(
    lat1: float,
    lon1: float,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    lat1_rad = np.radians(float(lat1))
    lat2_rad = np.radians(lat2)
    dlat = lat2_rad - lat1_rad
    dlon = np.radians(((lon2 - float(lon1) + 180.0) % 360.0) - 180.0)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _cell_center_arrays(regionizer: H3Regionizer) -> Tuple[np.ndarray, np.ndarray]:
    latitudes: List[float] = []
    longitudes: List[float] = []
    for cell in regionizer.region_tokens:
        geometry = regionizer.cell_geometry(cell)
        if geometry is None:
            latitudes.append(np.nan)
            longitudes.append(np.nan)
        else:
            latitudes.append(float(geometry["center_lat"]))
            longitudes.append(float(geometry["center_lon"]))
    return np.asarray(latitudes, dtype=float), np.asarray(longitudes, dtype=float)


class ReachabilityFilter:
    def __init__(self, cfg: ProjectConfig, metadata: Dict[str, Any]):
        self.cfg = cfg
        self.enabled = bool(getattr(cfg.evaluation, "reachable_filter_enabled", True))
        self.input_regionizer = H3Regionizer.from_metadata(metadata["input_regionizer"])
        self.target_regionizers = [
            H3Regionizer.from_metadata(item)
            for item in metadata["target_regionizers"]
        ]
        self.input_centers = _cell_center_arrays(self.input_regionizer)
        self.target_centers = [
            _cell_center_arrays(regionizer)
            for regionizer in self.target_regionizers
        ]
        self._mask_cache: Dict[Tuple[int, int, int], Tuple[Optional[np.ndarray], Dict[str, Any]]] = {}

    def current_cell_from_region_id(self, region_id: int) -> str | None:
        return self.input_regionizer.decode_region_id(int(region_id))

    def _mask_for_region(
        self,
        *,
        current_region_id: int,
        horizon_idx: int,
        horizon_minutes: int,
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        cache_key = (int(current_region_id), int(horizon_idx), int(horizon_minutes))
        cached = self._mask_cache.get(cache_key)
        if cached is not None:
            mask, diagnostic = cached
            return mask, dict(diagnostic)

        if not self.enabled:
            diagnostic = {"enabled": False}
            self._mask_cache[cache_key] = (None, diagnostic)
            return None, dict(diagnostic)

        if horizon_idx < 0 or horizon_idx >= len(self.target_regionizers):
            diagnostic = {"enabled": False, "reason": "missing_target_regionizer"}
            self._mask_cache[cache_key] = (None, diagnostic)
            return None, dict(diagnostic)

        if current_region_id < 0 or current_region_id >= len(self.input_regionizer.region_tokens):
            diagnostic = {"enabled": False, "reason": "missing_current_cell_geometry"}
            self._mask_cache[cache_key] = (None, diagnostic)
            return None, dict(diagnostic)

        current_lat = float(self.input_centers[0][current_region_id])
        current_lon = float(self.input_centers[1][current_region_id])
        if not np.isfinite(current_lat) or not np.isfinite(current_lon):
            diagnostic = {"enabled": False, "reason": "missing_current_cell_geometry"}
            self._mask_cache[cache_key] = (None, diagnostic)
            return None, dict(diagnostic)

        target_regionizer = self.target_regionizers[horizon_idx]
        max_speed_kts = max(0.0, float(getattr(self.cfg.evaluation, "reachable_max_speed_kts", 750.0)))
        margin_km = max(0.0, float(getattr(self.cfg.evaluation, "reachable_margin_km", 250.0)))
        max_distance_km = max_speed_kts * KNOT_KMH * (float(horizon_minutes) / 60.0) + margin_km

        num_region_cells = len(target_regionizer.region_tokens)
        if num_region_cells == 0:
            diagnostic = {"enabled": False, "reason": "empty_regionizer"}
            self._mask_cache[cache_key] = (None, diagnostic)
            return None, dict(diagnostic)

        center_lat, center_lon = self.target_centers[horizon_idx]
        distances = _great_circle_km(current_lat, current_lon, center_lat, center_lon)
        reachable = np.isfinite(distances) & (distances <= max_distance_km)
        diagnostic = {
            "enabled": True,
            "max_speed_kts": float(max_speed_kts),
            "margin_km": float(margin_km),
            "max_distance_km": float(max_distance_km),
            "current_cell": self.current_cell_from_region_id(int(current_region_id)),
            "current_lat": current_lat,
            "current_lon": current_lon,
            "reachable_cell_count": int(reachable.sum()),
            "total_cell_count": int(num_region_cells),
        }
        self._mask_cache[cache_key] = (reachable, diagnostic)
        return reachable, dict(diagnostic)

    def apply(
        self,
        probs: np.ndarray,
        *,
        current_cell: str | None = None,
        current_region_id: int | None = None,
        horizon_idx: int,
        horizon_minutes: int,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if not self.enabled:
            return probs, {"enabled": False}

        if current_cell is None and current_region_id is not None:
            current_cell = self.current_cell_from_region_id(int(current_region_id))

        current_geometry = (
            self.input_regionizer.cell_geometry(str(current_cell))
            if current_cell is not None
            else None
        )
        if current_geometry is None:
            return probs, {"enabled": False, "reason": "missing_current_cell_geometry"}

        if horizon_idx < 0 or horizon_idx >= len(self.target_regionizers):
            return probs, {"enabled": False, "reason": "missing_target_regionizer"}

        target_regionizer = self.target_regionizers[horizon_idx]
        current_lat = float(current_geometry["center_lat"])
        current_lon = float(current_geometry["center_lon"])
        max_speed_kts = max(0.0, float(getattr(self.cfg.evaluation, "reachable_max_speed_kts", 750.0)))
        margin_km = max(0.0, float(getattr(self.cfg.evaluation, "reachable_margin_km", 250.0)))
        max_distance_km = max_speed_kts * KNOT_KMH * (float(horizon_minutes) / 60.0) + margin_km

        num_region_cells = len(target_regionizer.region_tokens)
        if num_region_cells == 0:
            return probs, {"enabled": False, "reason": "empty_regionizer"}

        center_lat, center_lon = self.target_centers[horizon_idx]
        distances = _great_circle_km(current_lat, current_lon, center_lat, center_lon)
        reachable = np.isfinite(distances) & (distances <= max_distance_km)

        filtered = probs.copy()
        cell_probs = filtered[:num_region_cells]
        removed_mass = float(cell_probs[~reachable].sum())
        remaining_cell_mass = float(cell_probs[reachable].sum())
        cell_probs[~reachable] = 0.0
        filtered[:num_region_cells] = cell_probs

        total_mass = float(filtered.sum())
        if total_mass <= 0.0:
            unknown_id = target_regionizer.unknown_region_id
            filtered = np.zeros_like(filtered)
            if 0 <= unknown_id < len(filtered):
                filtered[unknown_id] = 1.0
                total_mass = 1.0
            else:
                return probs, {
                    "enabled": False,
                    "reason": "all_probability_removed_without_unknown_class",
                    "max_distance_km": float(max_distance_km),
                }
        else:
            filtered = filtered / total_mass

        return filtered, {
            "enabled": True,
            "max_speed_kts": float(max_speed_kts),
            "margin_km": float(margin_km),
            "max_distance_km": float(max_distance_km),
            "current_cell": str(current_cell),
            "current_lat": current_lat,
            "current_lon": current_lon,
            "reachable_cell_count": int(reachable.sum()),
            "total_cell_count": int(num_region_cells),
            "removed_probability_mass": removed_mass,
            "remaining_cell_probability_mass_before_renorm": remaining_cell_mass,
        }

    def filter_probability_arrays(
        self,
        region_probabilities: Sequence[np.ndarray],
        *,
        current_region_ids: Sequence[int],
        horizons_minutes: Sequence[int],
        return_diagnostics: bool = True,
    ) -> Tuple[List[np.ndarray], List[List[Dict[str, Any]]]]:
        current_ids = np.asarray(current_region_ids, dtype=np.int64)
        filtered_arrays: List[np.ndarray] = []
        diagnostics: List[List[Dict[str, Any]]] = []

        for horizon_idx, probs in enumerate(region_probabilities):
            horizon_minutes = int(horizons_minutes[horizon_idx])
            target_regionizer = self.target_regionizers[horizon_idx]
            num_region_cells = len(target_regionizer.region_tokens)
            unknown_id = target_regionizer.unknown_region_id
            filtered = probs.copy()
            horizon_diagnostics: List[Dict[str, Any]] = [{} for _ in range(int(probs.shape[0]))] if return_diagnostics else []

            unique_ids, inverse = np.unique(current_ids, return_inverse=True)
            for unique_position, unique_region_id in enumerate(unique_ids.tolist()):
                mask = inverse == unique_position
                reachable, base_diagnostic = self._mask_for_region(
                    current_region_id=int(unique_region_id),
                    horizon_idx=horizon_idx,
                    horizon_minutes=horizon_minutes,
                )
                if reachable is None:
                    if return_diagnostics:
                        for sample_idx in np.flatnonzero(mask).tolist():
                            horizon_diagnostics[sample_idx] = dict(base_diagnostic)
                    continue

                group = filtered[mask]
                cell_probs = group[:, :num_region_cells]
                if return_diagnostics:
                    removed_mass = cell_probs[:, ~reachable].sum(axis=1)
                    remaining_mass = cell_probs[:, reachable].sum(axis=1)
                cell_probs[:, ~reachable] = 0.0
                group[:, :num_region_cells] = cell_probs

                totals = group.sum(axis=1)
                zero_mask = totals <= 0.0
                nonzero_mask = ~zero_mask
                if np.any(nonzero_mask):
                    group[nonzero_mask] = group[nonzero_mask] / totals[nonzero_mask, None]
                if np.any(zero_mask):
                    group[zero_mask] = 0.0
                    if 0 <= unknown_id < group.shape[1]:
                        group[zero_mask, unknown_id] = 1.0
                filtered[mask] = group

                if return_diagnostics:
                    sample_indices = np.flatnonzero(mask).tolist()
                    for local_idx, sample_idx in enumerate(sample_indices):
                        diagnostic = dict(base_diagnostic)
                        diagnostic["removed_probability_mass"] = float(removed_mass[local_idx])
                        diagnostic["remaining_cell_probability_mass_before_renorm"] = float(remaining_mass[local_idx])
                        horizon_diagnostics[sample_idx] = diagnostic

            filtered_arrays.append(filtered)
            if return_diagnostics:
                diagnostics.append(horizon_diagnostics)
            else:
                diagnostics.append([])

        return filtered_arrays, diagnostics
