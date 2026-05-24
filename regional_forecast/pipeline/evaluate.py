from __future__ import annotations

import argparse
import json
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data._utils.collate import default_collate
from torch.utils.data import DataLoader

from regional_forecast.config import ProjectConfig, load_config
from regional_forecast.data.dataset import RegionalForecastDataset
from regional_forecast.data.regionizer import H3Regionizer
from regional_forecast.evaluation.metrics import (
    StreamingMultihorizonMetricAccumulator,
    compute_multihorizon_metrics,
)
from regional_forecast.evaluation.spatial_refinement import H3NeighborProbabilityRefiner
from regional_forecast.external_baselines import (
    AttentionTCNGRUHierarchicalH3RegionalForecaster,
    FlightBERTPPHierarchicalH3RegionalForecaster,
    TrAISformerHierarchicalH3RegionalForecaster,
)
from regional_forecast.models.baseline import RegionalMarkovBaseline
from regional_forecast.models.forecast_model import (
    AttentionHierarchicalH3RegionalForecaster,
    HierarchicalH3RegionalForecaster,
    HorizonDecoderHierarchicalH3RegionalForecaster,
    LongHorizonRegionalForecaster,
    MultiscaleLocalCausalTransformerH3RegionalForecaster,
    MultiscaleTransformerH3RegionalForecaster,
    TransformerHorizonDecoderHierarchicalH3RegionalForecaster,
)
from regional_forecast.reachability import ReachabilityFilter
from regional_forecast.registry import register_evaluation_run


def _resolve_device(cfg: ProjectConfig) -> torch.device:
    if cfg.training.device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _child_count_by_coarse_per_horizon(metadata: Dict[str, object]) -> List[List[int]]:
    if "child_count_by_coarse_per_horizon" in metadata:
        return [list(map(int, counts)) for counts in metadata["child_count_by_coarse_per_horizon"]]  # type: ignore[index]
    return [
        [max(1, len(children)) for children in horizon_meta["coarse_to_children"]] + [1]
        for horizon_meta in metadata["hierarchy_per_horizon"]  # type: ignore[index]
    ]


def _load_model(cfg: ProjectConfig, dataset: RegionalForecastDataset):
    checkpoint_run_name = cfg.evaluation.checkpoint_run_name or cfg.output.run_name
    checkpoint_path = cfg.checkpoint_dir / f"{checkpoint_run_name}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_type = checkpoint["model_type"]

    if model_type == "markov":
        model = RegionalMarkovBaseline.from_state_dict(checkpoint["model_state"])
        return model_type, model

    if model_type == "gru":
        model = LongHorizonRegionalForecaster(
            input_num_region_classes=dataset.metadata["input_num_region_classes"],
            num_region_classes_per_horizon=dataset.metadata["num_region_classes_per_horizon"],
            num_status_classes=dataset.metadata["num_status_classes"],
            num_numeric_features=dataset.metadata["num_numeric_features"],
            num_context_features=dataset.metadata["num_context_features"],
            config=cfg.model,
        )
    elif model_type == "hierarchical_gru":
        model = HierarchicalH3RegionalForecaster(
            input_num_region_classes=dataset.metadata["input_num_region_classes"],
            input_parent_num_region_classes=dataset.metadata["input_parent_num_region_classes"],
            num_coarse_region_classes_per_horizon=dataset.metadata["num_coarse_region_classes_per_horizon"],
            child_count_by_coarse_per_horizon=_child_count_by_coarse_per_horizon(dataset.metadata),
            max_children_per_horizon=dataset.metadata["max_children_per_horizon"],
            num_status_classes=dataset.metadata["num_status_classes"],
            num_numeric_features=dataset.metadata["num_numeric_features"],
            num_mask_features=dataset.metadata["num_mask_features"],
            num_context_features=dataset.metadata["num_context_features"],
            config=cfg.model,
        )
    elif model_type == "attention_hierarchical_gru":
        model = AttentionHierarchicalH3RegionalForecaster(
            input_num_region_classes=dataset.metadata["input_num_region_classes"],
            input_parent_num_region_classes=dataset.metadata["input_parent_num_region_classes"],
            num_coarse_region_classes_per_horizon=dataset.metadata["num_coarse_region_classes_per_horizon"],
            child_count_by_coarse_per_horizon=_child_count_by_coarse_per_horizon(dataset.metadata),
            max_children_per_horizon=dataset.metadata["max_children_per_horizon"],
            num_status_classes=dataset.metadata["num_status_classes"],
            num_numeric_features=dataset.metadata["num_numeric_features"],
            num_mask_features=dataset.metadata["num_mask_features"],
            num_context_features=dataset.metadata["num_context_features"],
            config=cfg.model,
        )
    elif model_type == "horizon_decoder_gru":
        model = HorizonDecoderHierarchicalH3RegionalForecaster(
            input_num_region_classes=dataset.metadata["input_num_region_classes"],
            input_parent_num_region_classes=dataset.metadata["input_parent_num_region_classes"],
            num_coarse_region_classes_per_horizon=dataset.metadata["num_coarse_region_classes_per_horizon"],
            child_count_by_coarse_per_horizon=_child_count_by_coarse_per_horizon(dataset.metadata),
            max_children_per_horizon=dataset.metadata["max_children_per_horizon"],
            num_status_classes=dataset.metadata["num_status_classes"],
            num_numeric_features=dataset.metadata["num_numeric_features"],
            num_mask_features=dataset.metadata["num_mask_features"],
            num_context_features=dataset.metadata["num_context_features"],
            config=cfg.model,
        )
    elif model_type == "transformer_horizon_decoder":
        model = TransformerHorizonDecoderHierarchicalH3RegionalForecaster(
            input_num_region_classes=dataset.metadata["input_num_region_classes"],
            input_parent_num_region_classes=dataset.metadata["input_parent_num_region_classes"],
            num_coarse_region_classes_per_horizon=dataset.metadata["num_coarse_region_classes_per_horizon"],
            child_count_by_coarse_per_horizon=_child_count_by_coarse_per_horizon(dataset.metadata),
            max_children_per_horizon=dataset.metadata["max_children_per_horizon"],
            num_status_classes=dataset.metadata["num_status_classes"],
            num_numeric_features=dataset.metadata["num_numeric_features"],
            num_mask_features=dataset.metadata["num_mask_features"],
            num_context_features=dataset.metadata["num_context_features"],
            max_history_steps=int(dataset.metadata["history_steps"]),
            config=cfg.model,
        )
    elif model_type == "multiscale_transformer_h3":
        model = MultiscaleTransformerH3RegionalForecaster(
            input_num_region_classes=dataset.metadata["input_num_region_classes"],
            input_parent_num_region_classes=dataset.metadata["input_parent_num_region_classes"],
            num_coarse_region_classes_per_horizon=dataset.metadata["num_coarse_region_classes_per_horizon"],
            child_count_by_coarse_per_horizon=_child_count_by_coarse_per_horizon(dataset.metadata),
            max_children_per_horizon=dataset.metadata["max_children_per_horizon"],
            num_status_classes=dataset.metadata["num_status_classes"],
            num_numeric_features=dataset.metadata["num_numeric_features"],
            num_mask_features=dataset.metadata["num_mask_features"],
            num_context_features=dataset.metadata["num_context_features"],
            max_history_steps=int(dataset.metadata["history_steps"]),
            config=cfg.model,
        )
    elif model_type == "multiscale_transformer_h3_local_causal":
        model = MultiscaleLocalCausalTransformerH3RegionalForecaster(
            input_num_region_classes=dataset.metadata["input_num_region_classes"],
            input_parent_num_region_classes=dataset.metadata["input_parent_num_region_classes"],
            num_coarse_region_classes_per_horizon=dataset.metadata["num_coarse_region_classes_per_horizon"],
            child_count_by_coarse_per_horizon=_child_count_by_coarse_per_horizon(dataset.metadata),
            max_children_per_horizon=dataset.metadata["max_children_per_horizon"],
            num_status_classes=dataset.metadata["num_status_classes"],
            num_numeric_features=dataset.metadata["num_numeric_features"],
            num_mask_features=dataset.metadata["num_mask_features"],
            num_context_features=dataset.metadata["num_context_features"],
            max_history_steps=int(dataset.metadata["history_steps"]),
            config=cfg.model,
        )
    elif model_type == "attention_tcn_gru_h3":
        model = AttentionTCNGRUHierarchicalH3RegionalForecaster(
            input_num_region_classes=dataset.metadata["input_num_region_classes"],
            input_parent_num_region_classes=dataset.metadata["input_parent_num_region_classes"],
            num_coarse_region_classes_per_horizon=dataset.metadata["num_coarse_region_classes_per_horizon"],
            child_count_by_coarse_per_horizon=_child_count_by_coarse_per_horizon(dataset.metadata),
            max_children_per_horizon=dataset.metadata["max_children_per_horizon"],
            num_status_classes=dataset.metadata["num_status_classes"],
            num_numeric_features=dataset.metadata["num_numeric_features"],
            num_mask_features=dataset.metadata["num_mask_features"],
            num_context_features=dataset.metadata["num_context_features"],
            config=cfg.model,
        )
    elif model_type == "traisformer_h3":
        model = TrAISformerHierarchicalH3RegionalForecaster(
            input_num_region_classes=dataset.metadata["input_num_region_classes"],
            input_parent_num_region_classes=dataset.metadata["input_parent_num_region_classes"],
            num_coarse_region_classes_per_horizon=dataset.metadata["num_coarse_region_classes_per_horizon"],
            child_count_by_coarse_per_horizon=_child_count_by_coarse_per_horizon(dataset.metadata),
            max_children_per_horizon=dataset.metadata["max_children_per_horizon"],
            num_status_classes=dataset.metadata["num_status_classes"],
            num_numeric_features=dataset.metadata["num_numeric_features"],
            num_mask_features=dataset.metadata["num_mask_features"],
            num_context_features=dataset.metadata["num_context_features"],
            max_history_steps=int(dataset.metadata["history_steps"]),
            config=cfg.model,
        )
    elif model_type == "flightbertpp_h3":
        model = FlightBERTPPHierarchicalH3RegionalForecaster(
            input_num_region_classes=dataset.metadata["input_num_region_classes"],
            input_parent_num_region_classes=dataset.metadata["input_parent_num_region_classes"],
            num_coarse_region_classes_per_horizon=dataset.metadata["num_coarse_region_classes_per_horizon"],
            child_count_by_coarse_per_horizon=_child_count_by_coarse_per_horizon(dataset.metadata),
            max_children_per_horizon=dataset.metadata["max_children_per_horizon"],
            num_status_classes=dataset.metadata["num_status_classes"],
            num_numeric_features=dataset.metadata["num_numeric_features"],
            num_mask_features=dataset.metadata["num_mask_features"],
            num_context_features=dataset.metadata["num_context_features"],
            max_history_steps=int(dataset.metadata["history_steps"]),
            config=cfg.model,
        )
    else:
        raise ValueError(f"Unsupported checkpoint model_type: {model_type}")

    try:
        model.load_state_dict(checkpoint["model_state"])
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint shape does not match the current model definition. "
            "Rebuild the dataset and retrain before running evaluation again."
        ) from exc
    model.eval()
    return model_type, model


def _move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=(device.type == "cuda")) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _stack_loader(dataset: RegionalForecastDataset, batch_size: int, num_workers: int, pin_memory: bool, persistent_workers: bool) -> DataLoader:
    use_workers = max(0, int(num_workers))

    def _collate(batch):
        if isinstance(batch, dict):
            return batch
        return default_collate(batch)

    loader_kwargs = {
        "dataset": dataset,
        "num_workers": use_workers,
        "pin_memory": bool(pin_memory and use_workers > 0),
        "persistent_workers": bool(persistent_workers and use_workers > 0),
        "collate_fn": _collate,
    }
    if use_workers > 0:
        loader_kwargs["prefetch_factor"] = 2

    if dataset.supports_chunk_batching:
        loader_kwargs["batch_sampler"] = dataset.build_batch_sampler(batch_size, shuffle=False, drop_last=False)
    else:
        loader_kwargs["batch_size"] = batch_size
        loader_kwargs["shuffle"] = False
    return DataLoader(**loader_kwargs)


def _iter_eval_batches(dataset: RegionalForecastDataset, batch_size: int) -> Iterable[tuple[List[int], Dict[str, torch.Tensor]]]:
    if dataset.supports_chunk_batching:
        sampler = dataset.build_batch_sampler(batch_size, shuffle=False, drop_last=False)
        for indices in sampler:
            batch_indices = [int(index) for index in indices]
            yield batch_indices, dataset.get_batch(batch_indices)
        return

    for start in range(0, len(dataset), batch_size):
        batch_indices = list(range(start, min(start + batch_size, len(dataset))))
        batch = default_collate([dataset[index] for index in batch_indices])
        yield batch_indices, batch


def _predict_probs_gru(cfg: ProjectConfig, dataset: RegionalForecastDataset, model) -> Dict[str, object]:
    device = _resolve_device(cfg)
    model = model.to(device)
    loader = _stack_loader(
        dataset,
        cfg.training.batch_size,
        cfg.training.num_workers,
        cfg.training.pin_memory,
        cfg.training.persistent_workers,
    )
    region_outputs: List[List[torch.Tensor]] | None = None
    status_outputs: List[torch.Tensor] = []

    with torch.no_grad():
        for raw_batch in loader:
            batch = _move_batch(raw_batch, device)
            outputs = model(batch["region_seq"], batch["numeric_seq"], batch["context"])
            batch_region_probs = [torch.softmax(logits, dim=-1).cpu() for logits in outputs["region_logits"]]
            if region_outputs is None:
                region_outputs = [[] for _ in range(len(batch_region_probs))]
            for horizon_idx, probs in enumerate(batch_region_probs):
                region_outputs[horizon_idx].append(probs)
            status_outputs.append(torch.softmax(outputs["status_logits"], dim=-1).cpu())

    if region_outputs is None:
        region_outputs = [[] for _ in range(len(dataset.metadata["horizons_minutes"]))]

    return {
        "region_probs": [torch.cat(chunks, dim=0) for chunks in region_outputs],
        "region_true_probs": None,
        "status_probs": torch.cat(status_outputs, dim=0),
        "coarse_probs": None,
    }


def _build_sparse_fine_probs(
    coarse_probs: torch.Tensor,
    top_parent_ids: torch.Tensor,
    child_probs_top: torch.Tensor,
    hierarchy: Dict[str, object],
    fine_class_count: int,
    unknown_fine_id: int,
    child_lookup_indices: torch.Tensor | None = None,
    child_lookup_valid: torch.Tensor | None = None,
) -> torch.Tensor:
    if child_lookup_indices is not None and child_lookup_valid is not None:
        device = coarse_probs.device
        lookup_indices = child_lookup_indices.to(device=device, non_blocking=True)
        lookup_valid = child_lookup_valid.to(device=device, non_blocking=True)
        max_children = min(int(child_probs_top.shape[-1]), int(lookup_indices.shape[-1]))

        selected_child_ids = lookup_indices[top_parent_ids][..., :max_children]
        selected_valid = lookup_valid[top_parent_ids][..., :max_children]
        parent_probs = coarse_probs.gather(1, top_parent_ids).unsqueeze(-1)
        local_probs = parent_probs * child_probs_top[..., :max_children]
        local_probs = local_probs * selected_valid.to(local_probs.dtype)

        batch_size = int(top_parent_ids.shape[0])
        fine_probs = torch.zeros((batch_size, int(fine_class_count)), device=device, dtype=torch.float32)
        fine_probs.scatter_add_(1, selected_child_ids.reshape(batch_size, -1), local_probs.reshape(batch_size, -1))
        used_mass = local_probs.sum(dim=(1, 2))
        fine_probs[:, int(unknown_fine_id)] = torch.clamp(1.0 - used_mass, min=0.0)
        return fine_probs.cpu()

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


def _build_child_lookup_tensors(
    hierarchy_per_horizon: Sequence[Dict[str, object]],
    *,
    device: torch.device,
    unknown_fine_ids: Sequence[int],
) -> List[tuple[torch.Tensor, torch.Tensor]]:
    lookups: List[tuple[torch.Tensor, torch.Tensor]] = []
    for horizon_idx, hierarchy in enumerate(hierarchy_per_horizon):
        coarse_to_children: Sequence[Sequence[int]] = hierarchy["coarse_to_children"]  # type: ignore[assignment]
        max_children = max(1, int(hierarchy.get("max_children", 1)))
        unknown_fine_id = int(unknown_fine_ids[horizon_idx])
        row_count = len(coarse_to_children) + 1
        child_indices = torch.full((row_count, max_children), unknown_fine_id, dtype=torch.long, device=device)
        valid_mask = torch.zeros((row_count, max_children), dtype=torch.bool, device=device)
        for coarse_id, children in enumerate(coarse_to_children):
            limit = min(len(children), max_children)
            if limit <= 0:
                continue
            child_indices[coarse_id, :limit] = torch.as_tensor(children[:limit], dtype=torch.long, device=device)
            valid_mask[coarse_id, :limit] = True
        lookups.append((child_indices, valid_mask))
    return lookups


def _temperature_scale_logits(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    temperature = float(temperature)
    if temperature <= 0.0:
        raise ValueError("Temperature must be positive.")
    if abs(temperature - 1.0) < 1.0e-8:
        return logits
    return logits / temperature


def _predict_probs_hierarchical(cfg: ProjectConfig, dataset: RegionalForecastDataset, model) -> Dict[str, object]:
    device = _resolve_device(cfg)
    model = model.to(device)
    loader = _stack_loader(
        dataset,
        cfg.training.batch_size,
        cfg.training.num_workers,
        cfg.training.pin_memory,
        cfg.training.persistent_workers,
    )
    num_horizons = len(dataset.metadata["horizons_minutes"])

    fine_outputs: List[List[torch.Tensor]] = [[] for _ in range(num_horizons)]
    true_prob_outputs: List[List[torch.Tensor]] = [[] for _ in range(num_horizons)]
    coarse_outputs: List[List[torch.Tensor]] = [[] for _ in range(num_horizons)]
    status_outputs: List[torch.Tensor] = []

    top_parent_k_per_horizon = [max(1, int(value)) for value in cfg.hierarchical_parent_top_k_per_horizon]
    coarse_temp_per_horizon = [max(1.0e-6, float(value)) for value in cfg.hierarchical_coarse_temperature_per_horizon]
    child_temp_per_horizon = [max(1.0e-6, float(value)) for value in cfg.hierarchical_child_temperature_per_horizon]
    fine_class_counts = dataset.metadata["num_region_classes_per_horizon"]
    unknown_fine_ids = dataset.metadata["unknown_region_ids_per_horizon"]
    hierarchy_per_horizon = dataset.metadata["hierarchy_per_horizon"]
    child_lookup_tables = _build_child_lookup_tensors(
        hierarchy_per_horizon,
        device=device,
        unknown_fine_ids=unknown_fine_ids,
    )

    with torch.no_grad():
        for raw_batch in loader:
            batch = _move_batch(raw_batch, device)
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
            status_outputs.append(torch.softmax(outputs["status_logits"], dim=-1).cpu())

            for horizon_idx, coarse_logits in enumerate(coarse_logits_list):
                coarse_logits = _temperature_scale_logits(coarse_logits, coarse_temp_per_horizon[horizon_idx])
                coarse_prob = torch.softmax(coarse_logits, dim=-1)
                coarse_outputs[horizon_idx].append(coarse_prob.cpu())

                current_top_k = min(top_parent_k_per_horizon[horizon_idx], int(coarse_prob.shape[-1]))
                _, top_parent_ids = torch.topk(coarse_prob, k=current_top_k, dim=-1)
                horizon_features = (
                    decoder_features[:, horizon_idx, :]
                    if decoder_features is not None
                    else features
                )
                child_probs_top = torch.softmax(
                    _temperature_scale_logits(
                        model.child_logits(horizon_features, top_parent_ids, horizon_idx),
                        child_temp_per_horizon[horizon_idx],
                    ),
                    dim=-1,
                )
                fine_probs = _build_sparse_fine_probs(
                    coarse_probs=coarse_prob,
                    top_parent_ids=top_parent_ids,
                    child_probs_top=child_probs_top,
                    hierarchy=hierarchy_per_horizon[horizon_idx],
                    fine_class_count=int(fine_class_counts[horizon_idx]),
                    unknown_fine_id=int(unknown_fine_ids[horizon_idx]),
                    child_lookup_indices=child_lookup_tables[horizon_idx][0],
                    child_lookup_valid=child_lookup_tables[horizon_idx][1],
                )
                fine_outputs[horizon_idx].append(fine_probs)

                true_coarse_ids = batch["coarse_region_labels"][:, horizon_idx]
                true_child_ids = batch["child_region_labels"][:, horizon_idx]
                true_child_probs = torch.softmax(
                    _temperature_scale_logits(
                        model.child_logits(horizon_features, true_coarse_ids, horizon_idx),
                        child_temp_per_horizon[horizon_idx],
                    ),
                    dim=-1,
                )
                batch_indices = torch.arange(true_coarse_ids.shape[0], device=device)
                exact_true_probs = coarse_prob[batch_indices, true_coarse_ids] * true_child_probs[
                    batch_indices,
                    true_child_ids,
                ]
                true_prob_outputs[horizon_idx].append(exact_true_probs.cpu())

    return {
        "region_probs": [torch.cat(chunks, dim=0) for chunks in fine_outputs],
        "region_true_probs": [torch.cat(chunks, dim=0).numpy() for chunks in true_prob_outputs],
        "status_probs": torch.cat(status_outputs, dim=0),
        "coarse_probs": [torch.cat(chunks, dim=0) for chunks in coarse_outputs],
    }


def _predict_probs(cfg: ProjectConfig, dataset: RegionalForecastDataset, model_type: str, model) -> Dict[str, object]:
    if model_type == "markov":
        predictions = model.predict(dataset.tensor("region_seq").long())
        return {
            "region_probs": predictions["region_probs"],
            "region_true_probs": None,
            "status_probs": predictions["status_probs"],
            "coarse_probs": None,
        }
    if model_type == "gru":
        return _predict_probs_gru(cfg, dataset, model)
    if model_type in {
        "hierarchical_gru",
        "attention_hierarchical_gru",
        "horizon_decoder_gru",
        "transformer_horizon_decoder",
        "multiscale_transformer_h3",
        "multiscale_transformer_h3_local_causal",
        "attention_tcn_gru_h3",
        "traisformer_h3",
        "flightbertpp_h3",
    }:
        return _predict_probs_hierarchical(cfg, dataset, model)
    raise ValueError(f"Unsupported model_type: {model_type}")


def _predict_batch_outputs(
    cfg: ProjectConfig,
    batch_cpu: Dict[str, torch.Tensor],
    model_type: str,
    model,
    device: torch.device,
    hierarchy_per_horizon: Sequence[Dict[str, object]] | None,
    fine_class_counts: Sequence[int] | None,
    unknown_fine_ids: Sequence[int] | None,
    child_lookup_tables: Sequence[tuple[torch.Tensor, torch.Tensor]] | None,
) -> tuple[List[np.ndarray], np.ndarray, List[np.ndarray] | None]:
    if model_type == "markov":
        predictions = model.predict(batch_cpu["region_seq"].long())
        return (
            [tensor.numpy() for tensor in predictions["region_probs"]],
            predictions["status_probs"].numpy(),
            None,
        )

    batch = _move_batch(batch_cpu, device)
    with torch.no_grad():
        if model_type == "gru":
            outputs = model(batch["region_seq"], batch["numeric_seq"], batch["context"])
            region_probs = [torch.softmax(logits, dim=-1).cpu().numpy() for logits in outputs["region_logits"]]
            status_probs = torch.softmax(outputs["status_logits"], dim=-1).cpu().numpy()
            return region_probs, status_probs, None

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
        status_probs = torch.softmax(outputs["status_logits"], dim=-1).cpu().numpy()

        if (
            hierarchy_per_horizon is None
            or fine_class_counts is None
            or unknown_fine_ids is None
            or child_lookup_tables is None
        ):
            raise ValueError("Hierarchical evaluation metadata is missing.")

        top_parent_k_per_horizon = [max(1, int(value)) for value in cfg.hierarchical_parent_top_k_per_horizon]
        coarse_temp_per_horizon = [max(1.0e-6, float(value)) for value in cfg.hierarchical_coarse_temperature_per_horizon]
        child_temp_per_horizon = [max(1.0e-6, float(value)) for value in cfg.hierarchical_child_temperature_per_horizon]
        fine_outputs: List[np.ndarray] = []
        coarse_outputs: List[np.ndarray] = []
        for horizon_idx, coarse_logits in enumerate(coarse_logits_list):
            coarse_logits = _temperature_scale_logits(coarse_logits, coarse_temp_per_horizon[horizon_idx])
            coarse_prob = torch.softmax(coarse_logits, dim=-1)
            coarse_outputs.append(coarse_prob.cpu().numpy())

            current_top_k = min(top_parent_k_per_horizon[horizon_idx], int(coarse_prob.shape[-1]))
            _, top_parent_ids = torch.topk(coarse_prob, k=current_top_k, dim=-1)
            horizon_features = (
                decoder_features[:, horizon_idx, :]
                if decoder_features is not None
                else features
            )
            child_probs_top = torch.softmax(
                _temperature_scale_logits(
                    model.child_logits(horizon_features, top_parent_ids, horizon_idx),
                    child_temp_per_horizon[horizon_idx],
                ),
                dim=-1,
            )
            fine_probs = _build_sparse_fine_probs(
                coarse_probs=coarse_prob,
                top_parent_ids=top_parent_ids,
                child_probs_top=child_probs_top,
                hierarchy=hierarchy_per_horizon[horizon_idx],
                fine_class_count=int(fine_class_counts[horizon_idx]),
                unknown_fine_id=int(unknown_fine_ids[horizon_idx]),
                child_lookup_indices=child_lookup_tables[horizon_idx][0],
                child_lookup_valid=child_lookup_tables[horizon_idx][1],
            )
            fine_outputs.append(fine_probs.numpy())

        return fine_outputs, status_probs, coarse_outputs


def _status_name(status_id: int) -> str:
    return "landed" if int(status_id) == 1 else "airborne"


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


def _build_prediction_examples(
    cfg: ProjectConfig,
    dataset: RegionalForecastDataset,
    region_probs: Sequence[torch.Tensor],
    status_probs: torch.Tensor,
    coarse_probs: Sequence[torch.Tensor] | None,
    region_seq: torch.Tensor,
    region_labels: torch.Tensor,
    status_labels: torch.Tensor,
    label_mask: torch.Tensor,
    aircraft_ids: Sequence[str],
    trajectory_ids: Sequence[str],
    timestamps: Sequence[str],
    raw_region_prob_arrays: Sequence[np.ndarray] | None = None,
    reachability_diagnostics: Sequence[Sequence[Dict[str, object]]] | None = None,
    example_limit: int | None = None,
) -> List[Dict[str, object]]:
    if cfg.evaluation.prediction_examples <= 0:
        return []

    input_regionizer = H3Regionizer.from_metadata(dataset.metadata["input_regionizer"])
    target_regionizers = [
        H3Regionizer.from_metadata(metadata)
        for metadata in dataset.metadata["target_regionizers"]
    ]
    coarse_regionizers = [
        H3Regionizer.from_metadata(metadata)
        for metadata in dataset.metadata.get("coarse_regionizers", [])
    ]

    available_examples = min(
        len(aircraft_ids),
        len(trajectory_ids),
        len(timestamps),
        int(region_seq.shape[0]),
        int(status_probs.shape[0]),
    )
    requested_examples = int(cfg.evaluation.prediction_examples) if example_limit is None else int(example_limit)
    max_examples = min(requested_examples, available_examples)
    examples: List[Dict[str, object]] = []
    region_prob_arrays = [tensor.numpy() for tensor in region_probs]
    model_region_prob_arrays = raw_region_prob_arrays if raw_region_prob_arrays is not None else region_prob_arrays
    status_prob_array = status_probs.numpy()
    coarse_prob_arrays = [tensor.numpy() for tensor in coarse_probs] if coarse_probs is not None else None
    top_k_export = max(int(k) for k in cfg.evaluation.top_ks) if cfg.evaluation.top_ks else 5
    candidate_probability_mass = cfg.candidate_probability_mass_per_horizon
    candidate_top_cells = cfg.candidate_top_cells_per_horizon
    parent_top_k_per_horizon = cfg.hierarchical_parent_top_k_per_horizon
    neighbor_rings = _resolve_neighbor_rings(
        cfg.evaluation.neighbor_hit_rings,
        len(dataset.metadata["horizons_minutes"]),
    )

    for index in range(max_examples):
        current_region_id = int(region_seq[index, -1].item())
        horizons: List[Dict[str, object]] = []

        for horizon_idx, horizon_minutes in enumerate(dataset.metadata["horizons_minutes"]):
            valid = bool(label_mask[index, horizon_idx].item())
            target_regionizer = target_regionizers[horizon_idx]
            probs = region_prob_arrays[horizon_idx][index]
            model_probs = model_region_prob_arrays[horizon_idx][index]
            top_region_ids = np.argsort(-probs)[:top_k_export].tolist()
            model_top_region_ids = np.argsort(-model_probs)[:top_k_export].tolist()
            candidates = target_regionizer.top_candidate_regions(
                probabilities=probs,
                cumulative_mass=float(candidate_probability_mass[horizon_idx]),
                max_cells=int(candidate_top_cells[horizon_idx]),
            )
            raw_top1_region_id = int(top_region_ids[0]) if top_region_ids else None
            raw_top1_cell = target_regionizer.decode_region_id(raw_top1_region_id) if raw_top1_region_id is not None else None
            raw_top1_probability = float(probs[raw_top1_region_id]) if raw_top1_region_id is not None else 0.0
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
            reachability = (
                dict(reachability_diagnostics[horizon_idx][index])
                if reachability_diagnostics is not None
                else {"enabled": False}
            )

            horizon_payload: Dict[str, object] = {
                "horizon_minutes": int(horizon_minutes),
                "status_probabilities": {
                    "airborne": float(status_prob_array[index, horizon_idx, 0]),
                    "landed": float(status_prob_array[index, horizon_idx, 1]),
                },
                "predicted_status": _status_name(int(status_prob_array[index, horizon_idx].argmax())),
                "neighbor_ring": int(neighbor_rings[horizon_idx]),
                "raw_top1_region_id": raw_top1_region_id,
                "raw_top1_cell": raw_top1_cell,
                "raw_top1_probability": raw_top1_probability,
                "raw_top1_is_unknown": bool(raw_top1_cell is None) if raw_top1_region_id is not None else False,
                "model_raw_top1_region_id": model_raw_top1_region_id,
                "model_raw_top1_cell": model_raw_top1_cell,
                "model_raw_top1_probability": model_raw_top1_probability,
                "model_raw_top1_is_unknown": (
                    bool(model_raw_top1_cell is None)
                    if model_raw_top1_region_id is not None
                    else False
                ),
                "reachability_filter": reachability,
                "top_region_predictions": [
                    {
                        "rank": rank,
                        "region_id": int(region_id),
                        "cell": target_regionizer.decode_region_id(int(region_id)),
                        "probability": float(probs[int(region_id)]),
                        "is_unknown": target_regionizer.decode_region_id(int(region_id)) is None,
                    }
                    for rank, region_id in enumerate(top_region_ids, start=1)
                ],
                "candidate_regions": candidates,
                "label_available": valid,
            }

            if coarse_prob_arrays is not None and horizon_idx < len(coarse_prob_arrays):
                coarse_regionizer = coarse_regionizers[horizon_idx]
                coarse_region_ids = np.argsort(-coarse_prob_arrays[horizon_idx][index])[
                    : int(parent_top_k_per_horizon[horizon_idx])
                ].tolist()
                horizon_payload["top_coarse_predictions"] = [
                    {
                        "rank": rank,
                        "region_id": int(region_id),
                        "cell": coarse_regionizer.decode_region_id(int(region_id)),
                        "probability": float(coarse_prob_arrays[horizon_idx][index][int(region_id)]),
                        "is_unknown": coarse_regionizer.decode_region_id(int(region_id)) is None,
                    }
                    for rank, region_id in enumerate(coarse_region_ids, start=1)
                ]

            if valid:
                true_region_id = int(region_labels[index, horizon_idx].item())
                true_status_id = int(status_labels[index, horizon_idx].item())
                horizon_payload["ground_truth"] = {
                    "region_id": true_region_id,
                    "cell": target_regionizer.decode_region_id(true_region_id),
                    "status": _status_name(true_status_id),
                }

            horizons.append(horizon_payload)

        examples.append(
            {
                "aircraft_id": aircraft_ids[index],
                "trajectory_id": trajectory_ids[index],
                "timestamp": timestamps[index],
                "current_region_id": current_region_id,
                "current_cell": input_regionizer.decode_region_id(current_region_id),
                "horizons": horizons,
            }
        )

    return examples


def run_evaluation(cfg: ProjectConfig) -> None:
    cfg.evaluation_dir.mkdir(parents=True, exist_ok=True)
    dataset = RegionalForecastDataset(str(cfg.dataset_dir / "test.pt"))
    model_type, model = _load_model(cfg, dataset)
    device = _resolve_device(cfg)
    if model_type != "markov":
        model = model.to(device)
        model.eval()

    reachability_filter = ReachabilityFilter(cfg, dataset.metadata)
    spatial_refiner = H3NeighborProbabilityRefiner(
        region_tokens_per_horizon=[
            metadata["region_tokens"]
            for metadata in dataset.metadata["target_regionizers"]
        ],
        enabled=bool(cfg.evaluation.h3_refinement_enabled),
        alpha_per_horizon=cfg.h3_refinement_alpha_per_horizon,
        ring_per_horizon=cfg.h3_refinement_ring_per_horizon,
        preserve_unknown=bool(cfg.evaluation.h3_refinement_preserve_unknown),
    )
    metric_accumulator = StreamingMultihorizonMetricAccumulator(
        top_ks=cfg.evaluation.top_ks,
        calibration_bins=cfg.evaluation.calibration_bins,
        horizons_minutes=dataset.metadata["horizons_minutes"],
        region_tokens_per_horizon=[metadata["region_tokens"] for metadata in dataset.metadata["target_regionizers"]],
        neighbor_hit_rings=cfg.evaluation.neighbor_hit_rings,
        candidate_probability_mass=cfg.candidate_probability_mass_per_horizon,
        candidate_top_cells=cfg.candidate_top_cells_per_horizon,
    )

    hierarchical_model = model_type in {
        "hierarchical_gru",
        "attention_hierarchical_gru",
        "horizon_decoder_gru",
        "transformer_horizon_decoder",
        "multiscale_transformer_h3",
        "multiscale_transformer_h3_local_causal",
        "attention_tcn_gru_h3",
        "traisformer_h3",
        "flightbertpp_h3",
    }
    hierarchy_per_horizon = dataset.metadata.get("hierarchy_per_horizon") if hierarchical_model else None
    fine_class_counts = dataset.metadata.get("num_region_classes_per_horizon") if hierarchical_model else None
    unknown_fine_ids = dataset.metadata.get("unknown_region_ids_per_horizon") if hierarchical_model else None
    child_lookup_tables = (
        _build_child_lookup_tensors(
            hierarchy_per_horizon,
            device=device,
            unknown_fine_ids=unknown_fine_ids,
        )
        if hierarchy_per_horizon is not None and hierarchical_model
        else None
    )

    examples: List[Dict[str, object]] = []
    example_budget = max(0, int(cfg.evaluation.prediction_examples))
    batch_size = max(1, int(cfg.training.batch_size))
    for batch_indices, batch_cpu in _iter_eval_batches(dataset, batch_size):
        raw_region_prob_arrays, status_probabilities, coarse_prob_arrays = _predict_batch_outputs(
            cfg,
            batch_cpu,
            model_type,
            model,
            device,
            hierarchy_per_horizon,
            fine_class_counts,
            unknown_fine_ids,
            child_lookup_tables,
        )
        refined_region_prob_arrays = spatial_refiner.refine_probability_arrays(raw_region_prob_arrays)
        filtered_region_prob_arrays, reachability_diagnostics = reachability_filter.filter_probability_arrays(
            refined_region_prob_arrays,
            current_region_ids=batch_cpu["region_seq"][:, -1].numpy(),
            horizons_minutes=dataset.metadata["horizons_minutes"],
            return_diagnostics=False,
        )

        metric_accumulator.update(
            region_probabilities=filtered_region_prob_arrays,
            region_labels=batch_cpu["region_labels"].numpy(),
            status_probabilities=status_probabilities,
            status_labels=batch_cpu["status_labels"].numpy(),
            label_mask=batch_cpu["label_mask"].numpy(),
            region_true_probabilities=None,
        )

        if len(examples) < example_budget:
            remaining = example_budget - len(examples)
            aircraft_ids = dataset.get_value_batch("aircraft_ids", batch_indices)
            trajectory_ids = dataset.get_value_batch("trajectory_ids", batch_indices)
            timestamps = dataset.get_value_batch("timestamps", batch_indices)
            example_raw_probs = [array[:remaining].copy() for array in raw_region_prob_arrays]
            example_refined_probs = spatial_refiner.refine_probability_arrays(example_raw_probs)
            example_filtered_probs, reachability_diagnostics = reachability_filter.filter_probability_arrays(
                example_refined_probs,
                current_region_ids=batch_cpu["region_seq"][:remaining, -1].numpy(),
                horizons_minutes=dataset.metadata["horizons_minutes"],
                return_diagnostics=True,
            )
            filtered_region_probs = [torch.from_numpy(array) for array in filtered_region_prob_arrays]
            coarse_probs = (
                [torch.from_numpy(array[:remaining].copy()) for array in coarse_prob_arrays]
                if coarse_prob_arrays is not None
                else None
            )
            examples.extend(
                _build_prediction_examples(
                    cfg,
                    dataset,
                    [torch.from_numpy(array) for array in example_filtered_probs],
                    torch.from_numpy(status_probabilities[:remaining].copy()),
                    coarse_probs,
                    region_seq=batch_cpu["region_seq"][:remaining],
                    region_labels=batch_cpu["region_labels"][:remaining],
                    status_labels=batch_cpu["status_labels"][:remaining],
                    label_mask=batch_cpu["label_mask"][:remaining],
                    aircraft_ids=aircraft_ids[:remaining],
                    trajectory_ids=trajectory_ids[:remaining],
                    timestamps=timestamps[:remaining],
                    raw_region_prob_arrays=example_raw_probs,
                    reachability_diagnostics=reachability_diagnostics,
                    example_limit=remaining,
                )
            )

    rows = metric_accumulator.finalize()

    df = pd.DataFrame(rows)
    csv_path = cfg.evaluation_dir / f"{cfg.output.run_name}_summary.csv"
    json_path = cfg.evaluation_dir / f"{cfg.output.run_name}_summary.json"
    examples_path = cfg.evaluation_dir / f"{cfg.output.run_name}_examples.json"

    df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)
    with open(examples_path, "w", encoding="utf-8") as handle:
        json.dump(examples, handle, ensure_ascii=False, indent=2)
    register_evaluation_run(
        cfg,
        metadata=dataset.metadata,
        summary_csv_path=csv_path,
        summary_json_path=json_path,
        examples_path=examples_path,
        rows=rows,
    )

    print(f"Evaluation done: {csv_path}")
    print(f"Examples saved: {examples_path}")
    print(f"Updated registry: {cfg.model_registry_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a long-horizon regional forecasting model.")
    parser.add_argument("--config", type=str, default="config/region_forecast.yaml")
    parser.add_argument("--scope", type=str, choices=["global", "type"], default=None)
    parser.add_argument("--name", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.scope is not None:
        cfg.dataset.scope = args.scope
    if args.name is not None:
        cfg.dataset.name = args.name

    print(f"Evaluation scope: {cfg.dataset_scope}")
    if cfg.dataset_name is not None:
        print(f"Evaluation name: {cfg.dataset_name}")
    print(f"Dataset dir: {cfg.dataset_dir}")
    print(f"Checkpoint dir: {cfg.checkpoint_dir}")
    print(f"Evaluation dir: {cfg.evaluation_dir}")
    run_evaluation(cfg)
