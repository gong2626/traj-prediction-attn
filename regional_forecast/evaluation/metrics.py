from __future__ import annotations

from functools import lru_cache
from typing import Dict, Iterable, List, Sequence

import numpy as np

try:
    import h3
except ImportError:  # pragma: no cover - depends on local environment
    h3 = None


def _require_h3() -> None:
    if h3 is None:
        raise ImportError("The 'h3' package is required for neighbor-aware evaluation.")


def _ece(confidence: np.ndarray, correct: np.ndarray, n_bins: int) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    score = 0.0

    for left, right in zip(bins[:-1], bins[1:]):
        if right < 1.0:
            mask = (confidence >= left) & (confidence < right)
        else:
            mask = (confidence >= left) & (confidence <= right)

        if not np.any(mask):
            continue

        bin_conf = confidence[mask].mean()
        bin_acc = correct[mask].mean()
        score += mask.mean() * abs(bin_conf - bin_acc)

    return float(score)


def _update_ece_bins(
    confidence: np.ndarray,
    correct: np.ndarray,
    n_bins: int,
    bin_counts: np.ndarray,
    bin_conf_sums: np.ndarray,
    bin_correct_sums: np.ndarray,
) -> None:
    if confidence.size == 0:
        return
    bins = np.floor(np.clip(confidence, 0.0, 1.0) * n_bins).astype(np.int64)
    bins = np.clip(bins, 0, n_bins - 1)
    np.add.at(bin_counts, bins, 1.0)
    np.add.at(bin_conf_sums, bins, confidence.astype(np.float64, copy=False))
    np.add.at(bin_correct_sums, bins, correct.astype(np.float64, copy=False))


def _finalize_ece_bins(
    bin_counts: np.ndarray,
    bin_conf_sums: np.ndarray,
    bin_correct_sums: np.ndarray,
) -> float:
    total = float(bin_counts.sum())
    if total <= 0.0:
        return 0.0
    mask = bin_counts > 0.0
    if not np.any(mask):
        return 0.0
    conf_mean = np.zeros_like(bin_conf_sums, dtype=np.float64)
    correct_mean = np.zeros_like(bin_correct_sums, dtype=np.float64)
    conf_mean[mask] = bin_conf_sums[mask] / bin_counts[mask]
    correct_mean[mask] = bin_correct_sums[mask] / bin_counts[mask]
    return float(np.sum((bin_counts[mask] / total) * np.abs(conf_mean[mask] - correct_mean[mask])))


def _resolve_neighbor_rings(neighbor_hit_rings: Sequence[int], num_horizons: int) -> List[int]:
    if not neighbor_hit_rings:
        return [0] * num_horizons
    if len(neighbor_hit_rings) == 1:
        return [int(neighbor_hit_rings[0])] * num_horizons
    if len(neighbor_hit_rings) != num_horizons:
        raise ValueError(
            "evaluation.neighbor_hit_rings must have length 1 or match the number of horizons."
        )
    return [int(value) for value in neighbor_hit_rings]


@lru_cache(maxsize=200_000)
def _grid_disk(cell: str, ring: int) -> frozenset[str]:
    _require_h3()
    if ring <= 0:
        return frozenset([str(cell)])
    return frozenset(str(item) for item in h3.grid_disk(cell, ring))


@lru_cache(maxsize=200_000)
def _cell_area_km2(cell: str) -> float:
    _require_h3()
    return float(h3.cell_area(str(cell), unit="km^2"))


def _neighbor_topk_hit_rate(
    sorted_idx: np.ndarray,
    targets: np.ndarray,
    top_k: int,
    ring: int,
    region_tokens: Sequence[str],
) -> float:
    hits = 0
    sample_count = int(targets.shape[0])
    if sample_count == 0:
        return 0.0

    for sample_idx in range(sample_count):
        target_region = int(targets[sample_idx])
        if target_region < 0 or target_region >= len(region_tokens):
            continue
        target_cell = region_tokens[target_region]
        predicted_regions = sorted_idx[sample_idx, :top_k]
        matched = False
        for predicted_region in predicted_regions.tolist():
            predicted_region = int(predicted_region)
            if predicted_region < 0 or predicted_region >= len(region_tokens):
                continue
            predicted_cell = region_tokens[predicted_region]
            if target_cell in _grid_disk(predicted_cell, ring):
                matched = True
                break
        hits += int(matched)

    return float(hits / sample_count)


def _top_candidate_regions(
    probabilities: np.ndarray,
    cumulative_mass: float,
    max_cells: int,
    region_tokens: Sequence[str],
) -> List[List[str]]:
    order = np.argsort(-probabilities)
    selected: List[int] = []
    running_mass = 0.0

    for region_id in order.tolist():
        if region_id < 0 or region_id >= len(region_tokens):
            continue
        probability = float(probabilities[region_id])
        if probability <= 0.0:
            break
        selected.append(int(region_id))
        running_mass += probability
        if len(selected) >= max_cells or running_mass >= cumulative_mass:
            break

    if not selected:
        return []

    cell_scores = {
        str(region_tokens[region_id]): float(probabilities[region_id])
        for region_id in selected
    }
    remaining = set(cell_scores.keys())
    components: List[List[str]] = []

    while remaining:
        seed = remaining.pop()
        stack = [seed]
        component = [seed]

        while stack:
            current = stack.pop()
            adjacent = remaining.intersection(_grid_disk(current, 1))
            if not adjacent:
                continue
            for neighbor in adjacent:
                remaining.remove(neighbor)
                stack.append(neighbor)
                component.append(neighbor)

        components.append(sorted(component))

    components.sort(
        key=lambda cells: sum(cell_scores[cell] for cell in cells),
        reverse=True,
    )
    return components


def _top_valid_region_ids(
    probabilities: np.ndarray,
    max_cells: int,
    region_tokens: Sequence[str],
    cumulative_mass: float | None = None,
) -> List[int]:
    selected: List[int] = []
    running_mass = 0.0
    for region_id in np.argsort(-probabilities).tolist():
        region_id = int(region_id)
        if region_id < 0 or region_id >= len(region_tokens):
            continue
        probability = float(probabilities[region_id])
        if probability <= 0.0:
            break
        selected.append(region_id)
        running_mass += probability
        if len(selected) >= int(max_cells):
            break
        if cumulative_mass is not None and running_mass >= float(cumulative_mass):
            break
    return selected


def _area_for_region_ids(region_ids: Sequence[int], region_tokens: Sequence[str]) -> float:
    return float(sum(_cell_area_km2(str(region_tokens[int(region_id)])) for region_id in region_ids))


def _candidate_region_set_metrics(
    region_probs: np.ndarray,
    region_targets: np.ndarray,
    region_tokens: Sequence[str],
    top_ks: Iterable[int],
    cumulative_mass: float,
    max_cells: int,
) -> Dict[str, float]:
    requested_ks = sorted({int(k) for k in top_ks})
    cell_budgets = sorted({*requested_ks, int(max_cells)})
    if region_targets.shape[0] == 0:
        result = {f"candidate_top{k}_cover": 0.0 for k in requested_ks}
        for budget in cell_budgets:
            result[f"candidate_cover_at{budget}_cells"] = 0.0
            result[f"candidate_area_at{budget}_cells_km2_mean"] = 0.0
        result.update(
            {
                "candidate_union_cover": 0.0,
                "candidate_union_cell_count_mean": 0.0,
                "candidate_union_area_km2_mean": 0.0,
                "candidate_union_probability_mass_mean": 0.0,
                "candidate_cover_at_budget_cells": 0.0,
                "candidate_area_at_budget_cells_km2_mean": 0.0,
                "candidate_budget_cells": float(max_cells),
            }
        )
        return result

    cover_counts = {k: 0 for k in requested_ks}
    fixed_budget_hits = {budget: 0 for budget in cell_budgets}
    fixed_budget_area_sums = {budget: 0.0 for budget in cell_budgets}
    union_hits = 0
    union_cell_count_sum = 0.0
    union_area_sum = 0.0
    union_probability_mass_sum = 0.0

    for sample_idx in range(int(region_targets.shape[0])):
        target_region = int(region_targets[sample_idx])
        if target_region < 0 or target_region >= len(region_tokens):
            continue

        probabilities = region_probs[sample_idx]
        target_cell = str(region_tokens[target_region])
        candidate_regions = _top_candidate_regions(
            probabilities=probabilities,
            cumulative_mass=cumulative_mass,
            max_cells=max_cells,
            region_tokens=region_tokens,
        )

        union_cells: set[str] = set()
        for rank, cells in enumerate(candidate_regions, start=1):
            union_cells.update(cells)
            if rank in cover_counts and target_cell in union_cells:
                cover_counts[rank] += 1

        if target_cell in union_cells:
            union_hits += 1

        if candidate_regions:
            max_rank = len(candidate_regions)
            for k in requested_ks:
                if k > max_rank and target_cell in union_cells:
                    cover_counts[k] += 1

        selected_for_union = _top_valid_region_ids(
            probabilities=probabilities,
            max_cells=max_cells,
            region_tokens=region_tokens,
            cumulative_mass=cumulative_mass,
        )
        union_cell_count_sum += float(len(selected_for_union))
        union_area_sum += _area_for_region_ids(selected_for_union, region_tokens)
        union_probability_mass_sum += float(sum(float(probabilities[int(region_id)]) for region_id in selected_for_union))

        for budget in cell_budgets:
            selected_for_budget = _top_valid_region_ids(
                probabilities=probabilities,
                max_cells=budget,
                region_tokens=region_tokens,
                cumulative_mass=None,
            )
            fixed_budget_hits[budget] += int(target_region in selected_for_budget)
            fixed_budget_area_sums[budget] += _area_for_region_ids(selected_for_budget, region_tokens)

    sample_count = float(region_targets.shape[0])
    result = {
        f"candidate_top{k}_cover": float(cover_counts[k] / sample_count)
        for k in requested_ks
    }
    for budget in cell_budgets:
        result[f"candidate_cover_at{budget}_cells"] = float(fixed_budget_hits[budget] / sample_count)
        result[f"candidate_area_at{budget}_cells_km2_mean"] = float(fixed_budget_area_sums[budget] / sample_count)
    result.update(
        {
            "candidate_union_cover": float(union_hits / sample_count),
            "candidate_union_cell_count_mean": float(union_cell_count_sum / sample_count),
            "candidate_union_area_km2_mean": float(union_area_sum / sample_count),
            "candidate_union_probability_mass_mean": float(union_probability_mass_sum / sample_count),
            "candidate_cover_at_budget_cells": float(fixed_budget_hits[int(max_cells)] / sample_count),
            "candidate_area_at_budget_cells_km2_mean": float(fixed_budget_area_sums[int(max_cells)] / sample_count),
            "candidate_budget_cells": float(max_cells),
        }
    )
    return result


def _candidate_region_cover_rates(
    region_probs: np.ndarray,
    region_targets: np.ndarray,
    region_tokens: Sequence[str],
    top_ks: Iterable[int],
    cumulative_mass: float,
    max_cells: int,
) -> Dict[str, float]:
    return _candidate_region_set_metrics(
        region_probs=region_probs,
        region_targets=region_targets,
        region_tokens=region_tokens,
        top_ks=top_ks,
        cumulative_mass=cumulative_mass,
        max_cells=max_cells,
    )


def compute_multihorizon_metrics(
    region_probabilities: Sequence[np.ndarray],
    region_labels: np.ndarray,
    region_true_probabilities: Sequence[np.ndarray] | None,
    status_probabilities: np.ndarray,
    status_labels: np.ndarray,
    label_mask: np.ndarray,
    top_ks: Iterable[int],
    calibration_bins: int,
    horizons_minutes: List[int],
    region_tokens_per_horizon: Sequence[Sequence[str]],
    neighbor_hit_rings: Sequence[int],
    candidate_probability_mass: Sequence[float],
    candidate_top_cells: Sequence[int],
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    n_horizons = len(region_probabilities)
    resolved_neighbor_rings = _resolve_neighbor_rings(neighbor_hit_rings, n_horizons)

    for horizon_idx in range(n_horizons):
        valid = label_mask[:, horizon_idx].astype(bool)
        count = int(valid.sum())
        if count == 0:
            rows.append(
                {
                    "horizon_minutes": float(horizons_minutes[horizon_idx]),
                    "samples": 0.0,
                    "neighbor_ring": float(resolved_neighbor_rings[horizon_idx]),
                }
            )
            continue

        region_probs = region_probabilities[horizon_idx][valid, :]
        region_targets = region_labels[valid, horizon_idx]
        status_probs = status_probabilities[valid, horizon_idx, :]
        status_targets = status_labels[valid, horizon_idx]
        region_tokens = region_tokens_per_horizon[horizon_idx]

        if region_true_probabilities is not None:
            region_safe = np.clip(region_true_probabilities[horizon_idx][valid], 1e-12, 1.0)
        else:
            region_safe = np.clip(region_probs[np.arange(count), region_targets], 1e-12, 1.0)
        region_pred = region_probs.argmax(axis=-1)
        region_conf = region_probs.max(axis=-1)
        region_correct = region_pred == region_targets
        unknown_region_id = len(region_tokens)

        region_one_hot = np.eye(region_probs.shape[-1], dtype=np.float32)[region_targets]
        region_brier = np.mean(np.sum((region_probs - region_one_hot) ** 2, axis=-1))

        status_safe = np.clip(status_probs[np.arange(count), status_targets], 1e-12, 1.0)
        status_pred = status_probs.argmax(axis=-1)
        status_conf = status_probs.max(axis=-1)
        status_correct = status_pred == status_targets

        row: Dict[str, float] = {
            "horizon_minutes": float(horizons_minutes[horizon_idx]),
            "samples": float(count),
            "region_nll": float(-np.mean(np.log(region_safe))),
            "region_brier": float(region_brier),
            "region_ece": _ece(region_conf, region_correct.astype(np.float32), calibration_bins),
            "region_top1_unknown_rate": float((region_pred == unknown_region_id).mean()),
            "status_nll": float(-np.mean(np.log(status_safe))),
            "status_accuracy": float(status_correct.mean()),
            "status_ece": _ece(status_conf, status_correct.astype(np.float32), calibration_bins),
            "landed_rate": float((status_targets == 1).mean()),
            "neighbor_ring": float(resolved_neighbor_rings[horizon_idx]),
        }

        sorted_idx = np.argsort(-region_probs, axis=-1)
        ring = resolved_neighbor_rings[horizon_idx]
        for k in top_ks:
            topk = sorted_idx[:, :k]
            row[f"region_top{k}"] = float(np.any(topk == region_targets[:, None], axis=1).mean())
            row[f"region_top{k}_ring"] = _neighbor_topk_hit_rate(
                sorted_idx=sorted_idx,
                targets=region_targets,
                top_k=int(k),
                ring=ring,
                region_tokens=region_tokens,
            )

        row.update(
            _candidate_region_cover_rates(
                region_probs=region_probs,
                region_targets=region_targets,
                region_tokens=region_tokens,
                top_ks=top_ks,
                cumulative_mass=float(candidate_probability_mass[horizon_idx]),
                max_cells=int(candidate_top_cells[horizon_idx]),
            )
        )
        rows.append(row)

    return rows


class StreamingMultihorizonMetricAccumulator:
    def __init__(
        self,
        *,
        top_ks: Iterable[int],
        calibration_bins: int,
        horizons_minutes: List[int],
        region_tokens_per_horizon: Sequence[Sequence[str]],
        neighbor_hit_rings: Sequence[int],
        candidate_probability_mass: Sequence[float],
        candidate_top_cells: Sequence[int],
    ) -> None:
        self.top_ks = sorted({int(k) for k in top_ks})
        self.calibration_bins = int(calibration_bins)
        self.horizons_minutes = [int(value) for value in horizons_minutes]
        self.region_tokens_per_horizon = [list(tokens) for tokens in region_tokens_per_horizon]
        self.neighbor_rings = _resolve_neighbor_rings(neighbor_hit_rings, len(self.horizons_minutes))
        self.candidate_probability_mass = [float(value) for value in candidate_probability_mass]
        self.candidate_top_cells = [int(value) for value in candidate_top_cells]

        self._state: List[Dict[str, object]] = []
        for _ in self.horizons_minutes:
            self._state.append(
                {
                    "count": 0,
                    "region_nll_sum": 0.0,
                    "region_brier_sum": 0.0,
                    "status_nll_sum": 0.0,
                    "status_correct_sum": 0.0,
                    "landed_count": 0.0,
                    "region_top1_unknown_count": 0.0,
                    "topk_hit_counts": {int(k): 0.0 for k in self.top_ks},
                    "topk_ring_counts": {int(k): 0.0 for k in self.top_ks},
                    "candidate_cover_counts": {int(k): 0.0 for k in self.top_ks},
                    "candidate_union_hits": 0.0,
                    "candidate_metric_sums": {},
                    "region_bin_counts": np.zeros(self.calibration_bins, dtype=np.float64),
                    "region_bin_conf_sums": np.zeros(self.calibration_bins, dtype=np.float64),
                    "region_bin_correct_sums": np.zeros(self.calibration_bins, dtype=np.float64),
                    "status_bin_counts": np.zeros(self.calibration_bins, dtype=np.float64),
                    "status_bin_conf_sums": np.zeros(self.calibration_bins, dtype=np.float64),
                    "status_bin_correct_sums": np.zeros(self.calibration_bins, dtype=np.float64),
                }
            )

    def update(
        self,
        *,
        region_probabilities: Sequence[np.ndarray],
        region_labels: np.ndarray,
        status_probabilities: np.ndarray,
        status_labels: np.ndarray,
        label_mask: np.ndarray,
        region_true_probabilities: Sequence[np.ndarray] | None = None,
    ) -> None:
        for horizon_idx, state in enumerate(self._state):
            valid = label_mask[:, horizon_idx].astype(bool)
            count = int(valid.sum())
            if count == 0:
                continue

            region_probs = region_probabilities[horizon_idx][valid, :]
            region_targets = region_labels[valid, horizon_idx].astype(np.int64, copy=False)
            status_probs = status_probabilities[valid, horizon_idx, :]
            status_targets = status_labels[valid, horizon_idx].astype(np.int64, copy=False)
            region_tokens = self.region_tokens_per_horizon[horizon_idx]

            if region_true_probabilities is not None:
                region_safe = np.clip(region_true_probabilities[horizon_idx][valid], 1e-12, 1.0)
            else:
                region_safe = np.clip(region_probs[np.arange(count), region_targets], 1e-12, 1.0)
            region_pred = region_probs.argmax(axis=-1)
            region_conf = region_probs.max(axis=-1)
            region_correct = region_pred == region_targets
            status_safe = np.clip(status_probs[np.arange(count), status_targets], 1e-12, 1.0)
            status_pred = status_probs.argmax(axis=-1)
            status_conf = status_probs.max(axis=-1)
            status_correct = status_pred == status_targets
            sorted_idx = np.argsort(-region_probs, axis=-1)
            ring = int(self.neighbor_rings[horizon_idx])
            unknown_region_id = len(region_tokens)

            state["count"] = int(state["count"]) + count
            state["region_nll_sum"] = float(state["region_nll_sum"]) + float(-np.log(region_safe).sum())
            region_brier = np.sum(np.square(region_probs), axis=-1) - 2.0 * region_probs[np.arange(count), region_targets] + 1.0
            state["region_brier_sum"] = float(state["region_brier_sum"]) + float(region_brier.sum())
            state["status_nll_sum"] = float(state["status_nll_sum"]) + float(-np.log(status_safe).sum())
            state["status_correct_sum"] = float(state["status_correct_sum"]) + float(status_correct.sum())
            state["landed_count"] = float(state["landed_count"]) + float((status_targets == 1).sum())
            state["region_top1_unknown_count"] = float(state["region_top1_unknown_count"]) + float((region_pred == unknown_region_id).sum())

            _update_ece_bins(
                region_conf,
                region_correct.astype(np.float32, copy=False),
                self.calibration_bins,
                state["region_bin_counts"],  # type: ignore[arg-type]
                state["region_bin_conf_sums"],  # type: ignore[arg-type]
                state["region_bin_correct_sums"],  # type: ignore[arg-type]
            )
            _update_ece_bins(
                status_conf,
                status_correct.astype(np.float32, copy=False),
                self.calibration_bins,
                state["status_bin_counts"],  # type: ignore[arg-type]
                state["status_bin_conf_sums"],  # type: ignore[arg-type]
                state["status_bin_correct_sums"],  # type: ignore[arg-type]
            )

            for k in self.top_ks:
                topk = sorted_idx[:, :k]
                topk_hits = np.any(topk == region_targets[:, None], axis=1)
                state["topk_hit_counts"][int(k)] += float(topk_hits.sum())  # type: ignore[index]
                ring_rate = _neighbor_topk_hit_rate(
                    sorted_idx=sorted_idx,
                    targets=region_targets,
                    top_k=int(k),
                    ring=ring,
                    region_tokens=region_tokens,
                )
                state["topk_ring_counts"][int(k)] += float(ring_rate * count)  # type: ignore[index]

            candidate_cover = _candidate_region_cover_rates(
                region_probs=region_probs,
                region_targets=region_targets,
                region_tokens=region_tokens,
                top_ks=self.top_ks,
                cumulative_mass=float(self.candidate_probability_mass[horizon_idx]),
                max_cells=int(self.candidate_top_cells[horizon_idx]),
            )
            for k in self.top_ks:
                state["candidate_cover_counts"][int(k)] += float(candidate_cover[f"candidate_top{k}_cover"] * count)  # type: ignore[index]
            state["candidate_union_hits"] = float(state["candidate_union_hits"]) + float(candidate_cover["candidate_union_cover"] * count)
            candidate_metric_sums: Dict[str, float] = state["candidate_metric_sums"]  # type: ignore[assignment]
            old_candidate_fields = {
                "candidate_union_cover",
                *[f"candidate_top{k}_cover" for k in self.top_ks],
            }
            for key, value in candidate_cover.items():
                if key in old_candidate_fields:
                    continue
                candidate_metric_sums[key] = float(candidate_metric_sums.get(key, 0.0)) + float(value) * count

    def finalize(self) -> List[Dict[str, float]]:
        rows: List[Dict[str, float]] = []
        for horizon_idx, state in enumerate(self._state):
            count = int(state["count"])
            if count <= 0:
                rows.append(
                    {
                        "horizon_minutes": float(self.horizons_minutes[horizon_idx]),
                        "samples": 0.0,
                        "neighbor_ring": float(self.neighbor_rings[horizon_idx]),
                    }
                )
                continue

            row: Dict[str, float] = {
                "horizon_minutes": float(self.horizons_minutes[horizon_idx]),
                "samples": float(count),
                "region_nll": float(state["region_nll_sum"]) / count,
                "region_brier": float(state["region_brier_sum"]) / count,
                "region_ece": _finalize_ece_bins(
                    state["region_bin_counts"],  # type: ignore[arg-type]
                    state["region_bin_conf_sums"],  # type: ignore[arg-type]
                    state["region_bin_correct_sums"],  # type: ignore[arg-type]
                ),
                "region_top1_unknown_rate": float(state["region_top1_unknown_count"]) / count,
                "status_nll": float(state["status_nll_sum"]) / count,
                "status_accuracy": float(state["status_correct_sum"]) / count,
                "status_ece": _finalize_ece_bins(
                    state["status_bin_counts"],  # type: ignore[arg-type]
                    state["status_bin_conf_sums"],  # type: ignore[arg-type]
                    state["status_bin_correct_sums"],  # type: ignore[arg-type]
                ),
                "landed_rate": float(state["landed_count"]) / count,
                "neighbor_ring": float(self.neighbor_rings[horizon_idx]),
                "candidate_union_cover": float(state["candidate_union_hits"]) / count,
            }

            for k in self.top_ks:
                row[f"region_top{k}"] = float(state["topk_hit_counts"][int(k)]) / count  # type: ignore[index]
                row[f"region_top{k}_ring"] = float(state["topk_ring_counts"][int(k)]) / count  # type: ignore[index]
                row[f"candidate_top{k}_cover"] = float(state["candidate_cover_counts"][int(k)]) / count  # type: ignore[index]
            for key, value in sorted(dict(state["candidate_metric_sums"]).items()):  # type: ignore[arg-type]
                row[str(key)] = float(value) / count

            rows.append(row)
        return rows
