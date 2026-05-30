from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data._utils.collate import default_collate
from torch.utils.data import DataLoader

from regional_forecast.config import ProjectConfig, load_config
from regional_forecast.data.dataset import RegionalForecastDataset
from regional_forecast.evaluation.metrics import compute_multihorizon_metrics
from regional_forecast.external_baselines import (
    AttentionTCNGRUHierarchicalH3RegionalForecaster,
    FlightBERTPPHierarchicalH3RegionalForecaster,
    TrAISformerHierarchicalH3RegionalForecaster,
)
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
from regional_forecast.registry import register_training_run


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _resolve_device(cfg: ProjectConfig) -> torch.device:
    if cfg.training.device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if cfg.training.device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _collate_batch(batch):
    if isinstance(batch, dict):
        return batch
    return default_collate(batch)


def _initialize_from_checkpoint_if_requested(cfg: ProjectConfig, model: nn.Module) -> None:
    init_from = cfg.training.init_from_checkpoint
    if init_from is None:
        return

    init_path = cfg.resolve_path(init_from)
    if init_path is None or not init_path.exists():
        raise FileNotFoundError(f"Initialization checkpoint not found: {init_from}")

    checkpoint = torch.load(init_path, map_location="cpu", weights_only=False)
    source_state = checkpoint.get("model_state", checkpoint)
    if not isinstance(source_state, dict):
        raise ValueError(f"Initialization checkpoint does not contain a model_state dict: {init_path}")

    target_state = model.state_dict()
    compatible_state = {
        key: value
        for key, value in source_state.items()
        if key in target_state and isinstance(value, torch.Tensor) and target_state[key].shape == value.shape
    }
    missing_keys, unexpected_keys = model.load_state_dict(compatible_state, strict=False)
    frozen_parameter_count = 0
    if cfg.training.freeze_initialized_parameters:
        for name, parameter in model.named_parameters():
            if name in compatible_state:
                parameter.requires_grad_(False)
                frozen_parameter_count += parameter.numel()
    print(
        "Initialized model from checkpoint "
        f"{init_path} with {len(compatible_state)} matched tensors; "
        f"missing={len(missing_keys)}, unexpected={len(unexpected_keys)}, "
        f"frozen_params={frozen_parameter_count}."
    )


def _stack_loader(
    dataset: RegionalForecastDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
) -> DataLoader:
    use_workers = max(0, int(num_workers))

    loader_kwargs = {
        "dataset": dataset,
        "num_workers": use_workers,
        "pin_memory": bool(pin_memory and use_workers > 0),
        "persistent_workers": bool(persistent_workers and use_workers > 0),
        "collate_fn": _collate_batch,
    }
    if use_workers > 0:
        loader_kwargs["prefetch_factor"] = 2

    if dataset.supports_chunk_batching:
        loader_kwargs["batch_sampler"] = dataset.build_batch_sampler(batch_size, shuffle=shuffle, drop_last=False)
    else:
        loader_kwargs["batch_size"] = batch_size
        loader_kwargs["shuffle"] = shuffle
    return DataLoader(**loader_kwargs)


def _save_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _last_checkpoint_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_name(f"{checkpoint_path.stem}_last{checkpoint_path.suffix}")


def _capture_rng_state() -> Dict[str, object]:
    payload: Dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["cuda"] = torch.cuda.get_rng_state_all()
    return payload


def _restore_rng_state(payload: Dict[str, object] | None) -> None:
    if not payload:
        return
    python_state = payload.get("python")
    numpy_state = payload.get("numpy")
    torch_state = payload.get("torch")
    cuda_state = payload.get("cuda")
    if python_state is not None:
        random.setstate(python_state)
    if numpy_state is not None:
        np.random.set_state(numpy_state)
    if torch_state is not None:
        torch.set_rng_state(torch_state)
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


def _optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _build_training_checkpoint(
    *,
    model_type: str,
    model,
    metadata: Dict[str, object],
    cfg: ProjectConfig,
    epoch: int,
    best_epoch: int,
    best_val_loss: float,
    patience_ref_loss: float,
    epochs_without_improvement: int,
    history: List[Dict[str, float]],
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[ReduceLROnPlateau] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "model_type": model_type,
        "model_state": model.state_dict(),
        "metadata": metadata,
        "config": cfg.to_dict(),
        "epoch": int(epoch),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "patience_ref_loss": float(patience_ref_loss),
        "epochs_without_improvement": int(epochs_without_improvement),
        "history": history,
        "rng_state": _capture_rng_state(),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler_state"] = scaler.state_dict()
    return payload


def _resolve_resume_checkpoint(
    cfg: ProjectConfig,
    checkpoint_path: Path,
    *,
    resume: bool,
    resume_from: Optional[str],
    restart: bool,
) -> Optional[Path]:
    if restart:
        return None

    if resume_from is not None:
        resolved = cfg.resolve_path(resume_from)
        if resolved is None or not resolved.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_from}")
        return resolved

    if cfg.training.resume_from_checkpoint:
        resolved = cfg.resolve_path(cfg.training.resume_from_checkpoint)
        if resolved is None or not resolved.exists():
            raise FileNotFoundError(f"Configured resume checkpoint not found: {cfg.training.resume_from_checkpoint}")
        return resolved

    if not resume and not cfg.training.auto_resume:
        return None

    for candidate in (_last_checkpoint_path(checkpoint_path), checkpoint_path):
        if candidate.exists():
            return candidate

    if resume:
        raise FileNotFoundError(
            f"Resume requested, but neither {_last_checkpoint_path(checkpoint_path)} nor {checkpoint_path} exists."
        )
    return None


def _child_count_by_coarse_per_horizon(metadata: Dict[str, object]) -> List[List[int]]:
    if "child_count_by_coarse_per_horizon" in metadata:
        return [list(map(int, counts)) for counts in metadata["child_count_by_coarse_per_horizon"]]  # type: ignore[index]
    return [
        [max(1, len(children)) for children in horizon_meta["coarse_to_children"]] + [1]
        for horizon_meta in metadata["hierarchy_per_horizon"]  # type: ignore[index]
    ]


def _evaluate_baseline(model: RegionalMarkovBaseline, dataset: RegionalForecastDataset, cfg: ProjectConfig):
    predictions = model.predict(dataset.tensor("region_seq").long())
    return compute_multihorizon_metrics(
        region_probabilities=[tensor.numpy() for tensor in predictions["region_probs"]],
        region_labels=dataset.tensor("region_labels").numpy(),
        region_true_probabilities=None,
        status_probabilities=predictions["status_probs"].numpy(),
        status_labels=dataset.tensor("status_labels").numpy(),
        label_mask=dataset.tensor("label_mask").numpy(),
        top_ks=cfg.evaluation.top_ks,
        calibration_bins=cfg.evaluation.calibration_bins,
        horizons_minutes=dataset.metadata["horizons_minutes"],
        region_tokens_per_horizon=[meta["region_tokens"] for meta in dataset.metadata["target_regionizers"]],
        neighbor_hit_rings=cfg.evaluation.neighbor_hit_rings,
        candidate_probability_mass=cfg.candidate_probability_mass_per_horizon,
        candidate_top_cells=cfg.candidate_top_cells_per_horizon,
    )


def _masked_multitask_loss(
    outputs: Dict[str, object],
    region_labels: torch.Tensor,
    status_labels: torch.Tensor,
    label_mask: torch.Tensor,
    region_criterion: nn.Module,
    status_criterion: nn.Module,
) -> torch.Tensor:
    losses = []
    region_logits_list: List[torch.Tensor] = outputs["region_logits"]  # type: ignore[assignment]
    status_logits = outputs["status_logits"]  # type: ignore[assignment]

    for horizon_idx, region_logits_full in enumerate(region_logits_list):
        valid = label_mask[:, horizon_idx].bool()
        if not torch.any(valid):
            continue

        region_logits = region_logits_full[valid]
        status_logits_h = status_logits[:, horizon_idx, :][valid]
        region_target = region_labels[:, horizon_idx][valid]
        status_target = status_labels[:, horizon_idx][valid]

        losses.append(region_criterion(region_logits, region_target))
        losses.append(status_criterion(status_logits_h, status_target))

    if not losses:
        return region_logits_list[0].sum() * 0.0
    return torch.stack(losses).mean()


def _masked_hierarchical_loss(
    model,
    outputs: Dict[str, object],
    coarse_region_labels: torch.Tensor,
    child_region_labels: torch.Tensor,
    status_labels: torch.Tensor,
    label_mask: torch.Tensor,
    region_criterion: nn.Module,
    status_criterion: nn.Module,
    horizons_minutes: List[int],
    horizon_consistency_weight: float = 0.0,
    horizon_consistency_gap_scaled: bool = True,
) -> torch.Tensor:
    losses = []
    features: torch.Tensor = outputs["features"]  # type: ignore[assignment]
    decoder_features: torch.Tensor | None = outputs.get("decoder_features")  # type: ignore[assignment]
    coarse_logits_list: List[torch.Tensor] = outputs["coarse_logits"]  # type: ignore[assignment]
    status_logits: torch.Tensor = outputs["status_logits"]  # type: ignore[assignment]

    for horizon_idx, coarse_logits_full in enumerate(coarse_logits_list):
        valid = label_mask[:, horizon_idx].bool()
        if not torch.any(valid):
            continue

        coarse_target = coarse_region_labels[:, horizon_idx][valid]
        child_target = child_region_labels[:, horizon_idx][valid]
        status_target = status_labels[:, horizon_idx][valid]

        coarse_logits = coarse_logits_full[valid]
        horizon_features = (
            decoder_features[:, horizon_idx, :][valid]
            if decoder_features is not None
            else features[valid]
        )
        child_logits = model.child_logits(horizon_features, coarse_target, horizon_idx)
        status_logits_h = status_logits[:, horizon_idx, :][valid]
        valid_child = model.child_target_is_valid(coarse_target, child_target, horizon_idx)

        losses.append(region_criterion(coarse_logits, coarse_target))
        if torch.any(valid_child):
            losses.append(region_criterion(child_logits[valid_child], child_target[valid_child]))
        losses.append(status_criterion(status_logits_h, status_target))

    if not losses:
        return features.sum() * 0.0
    total_loss = torch.stack(losses).mean()
    total_loss = total_loss + _horizon_consistency_loss(
        decoder_features=decoder_features,
        label_mask=label_mask,
        horizons_minutes=horizons_minutes,
        weight=horizon_consistency_weight,
        gap_scaled=horizon_consistency_gap_scaled,
    )
    return total_loss


def _router_active_expert_mask(model, num_horizons: int, device: torch.device) -> torch.Tensor:
    active = torch.ones((num_horizons, 3), device=device, dtype=torch.float32)
    short_term_horizons = int(getattr(model, "short_term_branch_horizons", 0))
    if short_term_horizons < num_horizons:
        active[short_term_horizons:, 2] = 0.0
    return active


def _build_router_target_distribution(
    expert_losses: torch.Tensor,
    active_mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    safe_temperature = max(float(temperature), 1.0e-6)
    masked_losses = expert_losses.masked_fill(active_mask <= 0.0, float("inf"))
    logits = -masked_losses / safe_temperature
    logits = logits.masked_fill(active_mask <= 0.0, torch.finfo(logits.dtype).min)
    return torch.softmax(logits, dim=-1)


def _router_load_balancing_loss(
    router_weights: torch.Tensor,
    label_mask: torch.Tensor,
    active_mask: torch.Tensor,
) -> torch.Tensor:
    if router_weights.numel() == 0:
        return router_weights.new_zeros(())
    valid = label_mask.bool()
    if not torch.any(valid):
        return router_weights.new_zeros(())

    valid_weights = router_weights[valid]
    valid_active = active_mask.unsqueeze(0).expand(router_weights.shape[0], -1, -1)[valid]
    target = valid_active / valid_active.sum(dim=-1, keepdim=True).clamp_min(1.0)
    diff = valid_weights - target
    return torch.mean(torch.sum(diff * diff, dim=-1))


def _expert_auxiliary_loss(
    model,
    outputs: Dict[str, object],
    coarse_region_labels: torch.Tensor,
    child_region_labels: torch.Tensor,
    status_labels: torch.Tensor,
    label_mask: torch.Tensor,
    region_criterion: nn.Module,
    status_criterion: nn.Module,
    active_mask: torch.Tensor,
) -> torch.Tensor:
    if (
        "coarse_expert_features" not in outputs
        or "local_expert_features" not in outputs
        or "global_expert_features" not in outputs
    ):
        features: torch.Tensor = outputs["features"]  # type: ignore[assignment]
        return features.sum() * 0.0

    coarse_expert_features: torch.Tensor = outputs["coarse_expert_features"]  # type: ignore[assignment]
    local_expert_features: torch.Tensor = outputs["local_expert_features"]  # type: ignore[assignment]

    losses: List[torch.Tensor] = []
    for horizon_idx, coarse_head in enumerate(model.coarse_heads):
        valid = label_mask[:, horizon_idx].bool()
        if not torch.any(valid):
            continue

        expert_feature_map = {
            1: coarse_expert_features[:, horizon_idx, :][valid],
        }
        if bool(active_mask[horizon_idx, 2].item()):
            expert_feature_map[2] = local_expert_features[:, horizon_idx, :][valid]

        coarse_target = coarse_region_labels[:, horizon_idx][valid]
        child_target = child_region_labels[:, horizon_idx][valid]
        status_target = status_labels[:, horizon_idx][valid]
        status_head = model.status_heads[horizon_idx]

        for expert_idx, horizon_features in expert_feature_map.items():
            coarse_logits = coarse_head(horizon_features)
            losses.append(region_criterion(coarse_logits, coarse_target))

            child_logits = model.child_logits(horizon_features, coarse_target, horizon_idx)
            valid_child = model.child_target_is_valid(coarse_target, child_target, horizon_idx)
            if torch.any(valid_child):
                losses.append(region_criterion(child_logits[valid_child], child_target[valid_child]))

            status_logits_h = status_head(horizon_features)
            losses.append(status_criterion(status_logits_h, status_target))

    if not losses:
        return coarse_expert_features.sum() * 0.0
    return torch.stack(losses).mean()


def _temperature_scale_probabilities_tensor(probabilities: torch.Tensor, temperature: float) -> torch.Tensor:
    safe_temperature = max(float(temperature), 1.0e-6)
    if abs(safe_temperature - 1.0) < 1.0e-12:
        return probabilities
    scaled = torch.pow(probabilities.clamp_min(1.0e-12), 1.0 / safe_temperature)
    return scaled / scaled.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)


def _distillation_kl(student_probs: torch.Tensor, teacher_probs: torch.Tensor) -> torch.Tensor:
    student_log = torch.log(student_probs.clamp_min(1.0e-12))
    teacher = teacher_probs.clamp_min(1.0e-12)
    return torch.sum(teacher * (torch.log(teacher) - student_log), dim=-1).mean()


def _build_teacher_targets_for_batch(
    cfg: ProjectConfig,
    model,
    outputs: Dict[str, object],
    batch: Dict[str, torch.Tensor],
    coarse_region_labels: torch.Tensor,
    child_region_labels: torch.Tensor,
) -> Optional[Dict[str, List[torch.Tensor] | torch.Tensor]]:
    teacher_report_path = cfg.training.teacher_blend_report_path
    teacher_weight = float(cfg.training.teacher_distillation_weight)
    teacher_status_weight = float(cfg.training.teacher_status_distillation_weight)
    if (
        teacher_report_path is None
        or (teacher_weight <= 0.0 and teacher_status_weight <= 0.0)
    ):
        return None

    report_path = cfg.resolve_path(teacher_report_path)
    if report_path is None or not report_path.exists():
        raise FileNotFoundError(f"Teacher blend report not found: {teacher_report_path}")

    if not hasattr(model, "teacher_blend_runtime"):
        teacher_payload = json.loads(report_path.read_text(encoding="utf-8"))
        expert_cfgs: Dict[str, ProjectConfig] = {}
        for expert_name, config_path in teacher_payload["experts"].items():
            expert_cfg = load_config(str(cfg.resolve_path(config_path)))
            expert_cfg.training.device = cfg.training.device
            expert_cfgs[expert_name] = expert_cfg
        setattr(
            model,
            "teacher_blend_runtime",
            {
                "payload": teacher_payload,
                "expert_cfgs": expert_cfgs,
                "expert_models": {},
                "weight_map": {
                    int(item["horizon_minutes"]): item["weights"]
                    for item in teacher_payload.get("selected_weights_per_horizon", [])
                },
                "temperature_map": {
                    int(item["horizon_minutes"]): float(item["temperature"])
                    for item in teacher_payload.get("selected_temperatures_per_horizon", [])
                },
            },
        )

    runtime = getattr(model, "teacher_blend_runtime")
    expert_models: Dict[str, tuple[str, nn.Module]] = runtime["expert_models"]
    if not expert_models:
        train_ds = RegionalForecastDataset(str(cfg.dataset_dir / "train.pt"))
        for expert_name, expert_cfg in runtime["expert_cfgs"].items():
            expert_model_type, expert_model = _build_model(expert_cfg, train_ds)
            checkpoint_run_name = expert_cfg.evaluation.checkpoint_run_name or expert_cfg.output.run_name
            checkpoint_path = expert_cfg.checkpoint_dir / f"{checkpoint_run_name}.pt"
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            expert_model.load_state_dict(checkpoint["model_state"])
            expert_model = expert_model.to(batch["region_seq"].device)
            expert_model.eval()
            expert_models[expert_name] = (expert_model_type, expert_model)

    with torch.no_grad():
        expert_outputs: Dict[str, Dict[str, object]] = {}
        for expert_name, (expert_model_type, expert_model) in expert_models.items():
            if expert_model_type == "gru":
                continue
            expert_outputs[expert_name] = expert_model(
                batch["region_seq"],
                batch["parent_region_seq"],
                batch["numeric_seq"],
                batch["mask_seq"],
                batch["context"],
            )

    teacher_coarse_probs: List[torch.Tensor] = []
    teacher_child_probs: List[torch.Tensor] = []
    teacher_status_probs: List[torch.Tensor] = []
    horizons_minutes = [int(v) for v in cfg.dataset.horizons_minutes]
    for horizon_idx, horizon_minutes in enumerate(horizons_minutes):
        weight_map = runtime["weight_map"].get(horizon_minutes, {})
        temperature = float(runtime["temperature_map"].get(horizon_minutes, 1.0))
        blended_coarse = None
        blended_child = None
        blended_status = None
        used_weight = 0.0
        for expert_name, weight in weight_map.items():
            expert_weight = float(weight)
            if expert_weight <= 0.0 or expert_name not in expert_outputs or expert_name not in expert_models:
                continue
            expert_out = expert_outputs[expert_name]
            _, expert_model = expert_models[expert_name]
            expert_features: torch.Tensor = expert_out["features"]  # type: ignore[assignment]
            expert_decoder_features: torch.Tensor | None = expert_out.get("decoder_features")  # type: ignore[assignment]
            expert_horizon_features = (
                expert_decoder_features[:, horizon_idx, :]
                if expert_decoder_features is not None
                else expert_features
            )
            expert_coarse = torch.softmax(expert_out["coarse_logits"][horizon_idx], dim=-1)  # type: ignore[index]
            expert_child = torch.softmax(
                expert_model.child_logits(
                    expert_horizon_features,
                    coarse_region_labels[:, horizon_idx],
                    horizon_idx,
                ),
                dim=-1,
            )
            expert_status = torch.softmax(expert_out["status_logits"][:, horizon_idx, :], dim=-1)  # type: ignore[index]
            blended_coarse = (
                expert_weight * expert_coarse
                if blended_coarse is None
                else blended_coarse + expert_weight * expert_coarse
            )
            blended_child = (
                expert_weight * expert_child
                if blended_child is None
                else blended_child + expert_weight * expert_child
            )
            blended_status = (
                expert_weight * expert_status
                if blended_status is None
                else blended_status + expert_weight * expert_status
            )
            used_weight += expert_weight

        if blended_coarse is None or blended_child is None or blended_status is None or used_weight <= 0.0:
            return None
        blended_coarse = blended_coarse / used_weight
        blended_child = blended_child / used_weight
        blended_status = blended_status / used_weight
        teacher_coarse_probs.append(_temperature_scale_probabilities_tensor(blended_coarse, temperature))
        teacher_child_probs.append(_temperature_scale_probabilities_tensor(blended_child, temperature))
        teacher_status_probs.append(blended_status / blended_status.sum(dim=-1, keepdim=True).clamp_min(1.0e-12))

    return {
        "coarse_probs": teacher_coarse_probs,
        "child_probs": teacher_child_probs,
        "status_probs": torch.stack(teacher_status_probs, dim=1),
    }


def _teacher_distillation_loss(
    cfg: ProjectConfig,
    model,
    outputs: Dict[str, object],
    batch: Dict[str, torch.Tensor],
) -> torch.Tensor:
    teacher_weight = float(cfg.training.teacher_distillation_weight)
    teacher_status_weight = float(cfg.training.teacher_status_distillation_weight)
    if teacher_weight <= 0.0 and teacher_status_weight <= 0.0:
        return batch["region_labels"].new_zeros((), dtype=torch.float32)

    teacher_targets = _build_teacher_targets_for_batch(
        cfg,
        model,
        outputs,
        batch,
        batch["coarse_region_labels"],
        batch["child_region_labels"],
    )
    if teacher_targets is None:
        return batch["region_labels"].new_zeros((), dtype=torch.float32)

    features: torch.Tensor = outputs["features"]  # type: ignore[assignment]
    decoder_features: torch.Tensor | None = outputs.get("decoder_features")  # type: ignore[assignment]
    coarse_logits_list: List[torch.Tensor] = outputs["coarse_logits"]  # type: ignore[assignment]
    status_logits: torch.Tensor = outputs["status_logits"]  # type: ignore[assignment]

    losses: List[torch.Tensor] = []
    for horizon_idx, student_coarse_logits in enumerate(coarse_logits_list):
        valid = batch["label_mask"][:, horizon_idx].bool()
        if not torch.any(valid):
            continue
        student_coarse = torch.softmax(student_coarse_logits[valid], dim=-1)
        losses.append(
            _distillation_kl(
                student_coarse,
                teacher_targets["coarse_probs"][horizon_idx][valid],  # type: ignore[index]
            )
        )
        horizon_features = (
            decoder_features[:, horizon_idx, :][valid]
            if decoder_features is not None
            else features[valid]
        )
        coarse_target = batch["coarse_region_labels"][:, horizon_idx][valid]
        student_child = torch.softmax(model.child_logits(horizon_features, coarse_target, horizon_idx), dim=-1)
        valid_child = model.child_target_is_valid(
            coarse_target,
            batch["child_region_labels"][:, horizon_idx][valid],
            horizon_idx,
        )
        if torch.any(valid_child):
            losses.append(
                _distillation_kl(
                    student_child[valid_child],
                    teacher_targets["child_probs"][horizon_idx][valid][valid_child],  # type: ignore[index]
                )
            )

    total = batch["region_labels"].new_zeros((), dtype=torch.float32)
    if losses and teacher_weight > 0.0:
        total = total + torch.stack(losses).mean() * teacher_weight
    if teacher_status_weight > 0.0:
        total = total + _distillation_kl(
            torch.softmax(status_logits, dim=-1),
            teacher_targets["status_probs"],  # type: ignore[arg-type]
        ) * teacher_status_weight
    return total


def _should_compute_teacher_distillation(
    cfg: ProjectConfig,
    *,
    is_training: bool,
    global_step: int,
) -> bool:
    if not is_training:
        return False
    teacher_weight = float(cfg.training.teacher_distillation_weight)
    teacher_status_weight = float(cfg.training.teacher_status_distillation_weight)
    if teacher_weight <= 0.0 and teacher_status_weight <= 0.0:
        return False
    interval = max(1, int(cfg.training.teacher_distillation_interval_steps))
    step_index = max(0, int(global_step) - 1)
    return (step_index % interval) == 0


def _router_regularization_loss(
    cfg: ProjectConfig,
    model,
    outputs: Dict[str, object],
    batch: Dict[str, torch.Tensor],
    region_criterion: nn.Module,
    status_criterion: nn.Module,
) -> torch.Tensor:
    router_supervision_weight = float(cfg.training.router_supervision_weight)
    router_load_balancing_weight = float(cfg.training.router_load_balancing_weight)
    expert_auxiliary_weight = float(cfg.training.expert_auxiliary_weight)
    if (
        router_supervision_weight <= 0.0
        and router_load_balancing_weight <= 0.0
        and expert_auxiliary_weight <= 0.0
    ):
        return batch["region_labels"].new_zeros((), dtype=torch.float32)

    router_logits: torch.Tensor | None = outputs.get("router_logits")  # type: ignore[assignment]
    router_weights: torch.Tensor | None = outputs.get("router_weights")  # type: ignore[assignment]
    global_expert_features: torch.Tensor | None = outputs.get("global_expert_features")  # type: ignore[assignment]
    coarse_expert_features: torch.Tensor | None = outputs.get("coarse_expert_features")  # type: ignore[assignment]
    local_expert_features: torch.Tensor | None = outputs.get("local_expert_features")  # type: ignore[assignment]
    if (
        router_logits is None
        or router_weights is None
        or global_expert_features is None
        or coarse_expert_features is None
        or local_expert_features is None
    ):
        return batch["region_labels"].new_zeros((), dtype=torch.float32)

    num_horizons = int(router_weights.shape[1])
    active_mask = _router_active_expert_mask(model, num_horizons, router_weights.device)
    total = router_weights.new_zeros(())

    if router_supervision_weight > 0.0:
        expert_loss_rows: List[torch.Tensor] = []
        for horizon_idx, coarse_head in enumerate(model.coarse_heads):
            row = router_weights.new_zeros((router_weights.shape[0], 3))
            valid = batch["label_mask"][:, horizon_idx].bool()
            if torch.any(valid):
                coarse_target = batch["coarse_region_labels"][:, horizon_idx][valid]
                child_target = batch["child_region_labels"][:, horizon_idx][valid]
                status_target = batch["status_labels"][:, horizon_idx][valid]
                status_head = model.status_heads[horizon_idx]
                feature_map = {
                    0: global_expert_features[:, horizon_idx, :][valid],
                    1: coarse_expert_features[:, horizon_idx, :][valid],
                    2: local_expert_features[:, horizon_idx, :][valid],
                }
                for expert_idx, horizon_features in feature_map.items():
                    if not bool(active_mask[horizon_idx, expert_idx].item()):
                        row[valid, expert_idx] = 1.0e6
                        continue
                    coarse_logits = coarse_head(horizon_features)
                    child_logits = model.child_logits(horizon_features, coarse_target, horizon_idx)
                    status_logits_h = status_head(horizon_features)
                    valid_child = model.child_target_is_valid(coarse_target, child_target, horizon_idx)
                    expert_loss = region_criterion(coarse_logits, coarse_target)
                    if torch.any(valid_child):
                        expert_loss = expert_loss + region_criterion(child_logits[valid_child], child_target[valid_child])
                    expert_loss = expert_loss + status_criterion(status_logits_h, status_target)
                    row[valid, expert_idx] = expert_loss.detach()
            expert_loss_rows.append(row)

        full_active_mask = active_mask.unsqueeze(0).expand(router_weights.shape[0], -1, -1)
        router_target = _build_router_target_distribution(
            expert_losses=torch.stack(expert_loss_rows, dim=1).reshape(-1, 3),
            active_mask=full_active_mask.reshape(-1, 3),
            temperature=float(cfg.training.router_supervision_temperature),
        ).reshape_as(router_weights)
        valid = batch["label_mask"].reshape(-1)
        if torch.any(valid):
            student_log = torch.log(router_weights.reshape(-1, 3)[valid].clamp_min(1.0e-12))
            teacher_probs = router_target.reshape(-1, 3)[valid]
            total = total + F.kl_div(student_log, teacher_probs, reduction="batchmean") * router_supervision_weight

    if router_load_balancing_weight > 0.0:
        total = total + _router_load_balancing_loss(
            router_weights=router_weights,
            label_mask=batch["label_mask"],
            active_mask=active_mask,
        ) * router_load_balancing_weight

    if expert_auxiliary_weight > 0.0:
        total = total + _expert_auxiliary_loss(
            model=model,
            outputs=outputs,
            coarse_region_labels=batch["coarse_region_labels"],
            child_region_labels=batch["child_region_labels"],
            status_labels=batch["status_labels"],
            label_mask=batch["label_mask"],
            region_criterion=region_criterion,
            status_criterion=status_criterion,
            active_mask=active_mask,
        ) * expert_auxiliary_weight

    return total


def _horizon_consistency_loss(
    decoder_features: Optional[torch.Tensor],
    label_mask: torch.Tensor,
    horizons_minutes: List[int],
    weight: float,
    gap_scaled: bool = True,
) -> torch.Tensor:
    if decoder_features is None:
        return label_mask.new_zeros((), dtype=torch.float32)
    if weight <= 0.0 or decoder_features.shape[1] < 2:
        return decoder_features.new_zeros(())

    valid_pairs = label_mask[:, :-1].bool() & label_mask[:, 1:].bool()
    if not torch.any(valid_pairs):
        return decoder_features.new_zeros(())

    pair_losses = (decoder_features[:, 1:, :] - decoder_features[:, :-1, :]).pow(2).mean(dim=-1)
    if gap_scaled:
        gaps = [
            max(1.0, float(horizons_minutes[idx + 1] - horizons_minutes[idx]))
            for idx in range(len(horizons_minutes) - 1)
        ]
        gap_tensor = decoder_features.new_tensor(gaps)
        pair_weights = gap_tensor.min() / gap_tensor
        pair_losses = pair_losses * pair_weights.unsqueeze(0)

    return pair_losses[valid_pairs].mean() * float(weight)


def _build_model(cfg: ProjectConfig, train_ds: RegionalForecastDataset):
    model_type = str(cfg.model.model_type).lower()
    if model_type == "markov":
        return model_type, RegionalMarkovBaseline(
            input_num_region_classes=train_ds.metadata["input_num_region_classes"],
            num_region_classes_per_horizon=train_ds.metadata["num_region_classes_per_horizon"],
            num_status_classes=train_ds.metadata["num_status_classes"],
            num_horizons=len(train_ds.metadata["horizons_minutes"]),
        )
    if model_type == "gru":
        return model_type, LongHorizonRegionalForecaster(
            input_num_region_classes=train_ds.metadata["input_num_region_classes"],
            num_region_classes_per_horizon=train_ds.metadata["num_region_classes_per_horizon"],
            num_status_classes=train_ds.metadata["num_status_classes"],
            num_numeric_features=train_ds.metadata["num_numeric_features"],
            num_context_features=train_ds.metadata["num_context_features"],
            config=cfg.model,
        )
    if model_type == "hierarchical_gru":
        return model_type, HierarchicalH3RegionalForecaster(
            input_num_region_classes=train_ds.metadata["input_num_region_classes"],
            input_parent_num_region_classes=train_ds.metadata["input_parent_num_region_classes"],
            num_coarse_region_classes_per_horizon=train_ds.metadata["num_coarse_region_classes_per_horizon"],
            child_count_by_coarse_per_horizon=_child_count_by_coarse_per_horizon(train_ds.metadata),
            max_children_per_horizon=train_ds.metadata["max_children_per_horizon"],
            num_status_classes=train_ds.metadata["num_status_classes"],
            num_numeric_features=train_ds.metadata["num_numeric_features"],
            num_mask_features=train_ds.metadata["num_mask_features"],
            num_context_features=train_ds.metadata["num_context_features"],
            config=cfg.model,
        )
    if model_type == "attention_hierarchical_gru":
        return model_type, AttentionHierarchicalH3RegionalForecaster(
            input_num_region_classes=train_ds.metadata["input_num_region_classes"],
            input_parent_num_region_classes=train_ds.metadata["input_parent_num_region_classes"],
            num_coarse_region_classes_per_horizon=train_ds.metadata["num_coarse_region_classes_per_horizon"],
            child_count_by_coarse_per_horizon=_child_count_by_coarse_per_horizon(train_ds.metadata),
            max_children_per_horizon=train_ds.metadata["max_children_per_horizon"],
            num_status_classes=train_ds.metadata["num_status_classes"],
            num_numeric_features=train_ds.metadata["num_numeric_features"],
            num_mask_features=train_ds.metadata["num_mask_features"],
            num_context_features=train_ds.metadata["num_context_features"],
            config=cfg.model,
        )
    if model_type == "horizon_decoder_gru":
        return model_type, HorizonDecoderHierarchicalH3RegionalForecaster(
            input_num_region_classes=train_ds.metadata["input_num_region_classes"],
            input_parent_num_region_classes=train_ds.metadata["input_parent_num_region_classes"],
            num_coarse_region_classes_per_horizon=train_ds.metadata["num_coarse_region_classes_per_horizon"],
            child_count_by_coarse_per_horizon=_child_count_by_coarse_per_horizon(train_ds.metadata),
            max_children_per_horizon=train_ds.metadata["max_children_per_horizon"],
            num_status_classes=train_ds.metadata["num_status_classes"],
            num_numeric_features=train_ds.metadata["num_numeric_features"],
            num_mask_features=train_ds.metadata["num_mask_features"],
            num_context_features=train_ds.metadata["num_context_features"],
            config=cfg.model,
        )
    if model_type == "transformer_horizon_decoder":
        return model_type, TransformerHorizonDecoderHierarchicalH3RegionalForecaster(
            input_num_region_classes=train_ds.metadata["input_num_region_classes"],
            input_parent_num_region_classes=train_ds.metadata["input_parent_num_region_classes"],
            num_coarse_region_classes_per_horizon=train_ds.metadata["num_coarse_region_classes_per_horizon"],
            child_count_by_coarse_per_horizon=_child_count_by_coarse_per_horizon(train_ds.metadata),
            max_children_per_horizon=train_ds.metadata["max_children_per_horizon"],
            num_status_classes=train_ds.metadata["num_status_classes"],
            num_numeric_features=train_ds.metadata["num_numeric_features"],
            num_mask_features=train_ds.metadata["num_mask_features"],
            num_context_features=train_ds.metadata["num_context_features"],
            max_history_steps=int(train_ds.metadata["history_steps"]),
            config=cfg.model,
        )
    if model_type == "multiscale_transformer_h3":
        return model_type, MultiscaleTransformerH3RegionalForecaster(
            input_num_region_classes=train_ds.metadata["input_num_region_classes"],
            input_parent_num_region_classes=train_ds.metadata["input_parent_num_region_classes"],
            num_coarse_region_classes_per_horizon=train_ds.metadata["num_coarse_region_classes_per_horizon"],
            child_count_by_coarse_per_horizon=_child_count_by_coarse_per_horizon(train_ds.metadata),
            max_children_per_horizon=train_ds.metadata["max_children_per_horizon"],
            num_status_classes=train_ds.metadata["num_status_classes"],
            num_numeric_features=train_ds.metadata["num_numeric_features"],
            num_mask_features=train_ds.metadata["num_mask_features"],
            num_context_features=train_ds.metadata["num_context_features"],
            max_history_steps=int(train_ds.metadata["history_steps"]),
            config=cfg.model,
        )
    if model_type == "multiscale_transformer_h3_local_causal":
        return model_type, MultiscaleLocalCausalTransformerH3RegionalForecaster(
            input_num_region_classes=train_ds.metadata["input_num_region_classes"],
            input_parent_num_region_classes=train_ds.metadata["input_parent_num_region_classes"],
            num_coarse_region_classes_per_horizon=train_ds.metadata["num_coarse_region_classes_per_horizon"],
            child_count_by_coarse_per_horizon=_child_count_by_coarse_per_horizon(train_ds.metadata),
            max_children_per_horizon=train_ds.metadata["max_children_per_horizon"],
            num_status_classes=train_ds.metadata["num_status_classes"],
            num_numeric_features=train_ds.metadata["num_numeric_features"],
            num_mask_features=train_ds.metadata["num_mask_features"],
            num_context_features=train_ds.metadata["num_context_features"],
            max_history_steps=int(train_ds.metadata["history_steps"]),
            config=cfg.model,
        )
    if model_type == "multiscale_transformer_h3_gated_local":
        return model_type, MultiscaleGatedLocalTransformerH3RegionalForecaster(
            input_num_region_classes=train_ds.metadata["input_num_region_classes"],
            input_parent_num_region_classes=train_ds.metadata["input_parent_num_region_classes"],
            num_coarse_region_classes_per_horizon=train_ds.metadata["num_coarse_region_classes_per_horizon"],
            child_count_by_coarse_per_horizon=_child_count_by_coarse_per_horizon(train_ds.metadata),
            max_children_per_horizon=train_ds.metadata["max_children_per_horizon"],
            num_status_classes=train_ds.metadata["num_status_classes"],
            num_numeric_features=train_ds.metadata["num_numeric_features"],
            num_mask_features=train_ds.metadata["num_mask_features"],
            num_context_features=train_ds.metadata["num_context_features"],
            max_history_steps=int(train_ds.metadata["history_steps"]),
            config=cfg.model,
        )
    if model_type == "horizon_router_transformer_h3":
        return model_type, HorizonRouterTransformerH3RegionalForecaster(
            input_num_region_classes=train_ds.metadata["input_num_region_classes"],
            input_parent_num_region_classes=train_ds.metadata["input_parent_num_region_classes"],
            num_coarse_region_classes_per_horizon=train_ds.metadata["num_coarse_region_classes_per_horizon"],
            child_count_by_coarse_per_horizon=_child_count_by_coarse_per_horizon(train_ds.metadata),
            max_children_per_horizon=train_ds.metadata["max_children_per_horizon"],
            num_status_classes=train_ds.metadata["num_status_classes"],
            num_numeric_features=train_ds.metadata["num_numeric_features"],
            num_mask_features=train_ds.metadata["num_mask_features"],
            num_context_features=train_ds.metadata["num_context_features"],
            max_history_steps=int(train_ds.metadata["history_steps"]),
            config=cfg.model,
        )
    if model_type == "attention_tcn_gru_h3":
        return model_type, AttentionTCNGRUHierarchicalH3RegionalForecaster(
            input_num_region_classes=train_ds.metadata["input_num_region_classes"],
            input_parent_num_region_classes=train_ds.metadata["input_parent_num_region_classes"],
            num_coarse_region_classes_per_horizon=train_ds.metadata["num_coarse_region_classes_per_horizon"],
            child_count_by_coarse_per_horizon=_child_count_by_coarse_per_horizon(train_ds.metadata),
            max_children_per_horizon=train_ds.metadata["max_children_per_horizon"],
            num_status_classes=train_ds.metadata["num_status_classes"],
            num_numeric_features=train_ds.metadata["num_numeric_features"],
            num_mask_features=train_ds.metadata["num_mask_features"],
            num_context_features=train_ds.metadata["num_context_features"],
            config=cfg.model,
        )
    if model_type == "traisformer_h3":
        return model_type, TrAISformerHierarchicalH3RegionalForecaster(
            input_num_region_classes=train_ds.metadata["input_num_region_classes"],
            input_parent_num_region_classes=train_ds.metadata["input_parent_num_region_classes"],
            num_coarse_region_classes_per_horizon=train_ds.metadata["num_coarse_region_classes_per_horizon"],
            child_count_by_coarse_per_horizon=_child_count_by_coarse_per_horizon(train_ds.metadata),
            max_children_per_horizon=train_ds.metadata["max_children_per_horizon"],
            num_status_classes=train_ds.metadata["num_status_classes"],
            num_numeric_features=train_ds.metadata["num_numeric_features"],
            num_mask_features=train_ds.metadata["num_mask_features"],
            num_context_features=train_ds.metadata["num_context_features"],
            max_history_steps=int(train_ds.metadata["history_steps"]),
            config=cfg.model,
        )
    if model_type == "flightbertpp_h3":
        return model_type, FlightBERTPPHierarchicalH3RegionalForecaster(
            input_num_region_classes=train_ds.metadata["input_num_region_classes"],
            input_parent_num_region_classes=train_ds.metadata["input_parent_num_region_classes"],
            num_coarse_region_classes_per_horizon=train_ds.metadata["num_coarse_region_classes_per_horizon"],
            child_count_by_coarse_per_horizon=_child_count_by_coarse_per_horizon(train_ds.metadata),
            max_children_per_horizon=train_ds.metadata["max_children_per_horizon"],
            num_status_classes=train_ds.metadata["num_status_classes"],
            num_numeric_features=train_ds.metadata["num_numeric_features"],
            num_mask_features=train_ds.metadata["num_mask_features"],
            num_context_features=train_ds.metadata["num_context_features"],
            max_history_steps=int(train_ds.metadata["history_steps"]),
            config=cfg.model,
        )
    raise ValueError(f"Unsupported model_type: {cfg.model.model_type}")


def _compute_loss_for_batch(
    cfg: ProjectConfig,
    model_type: str,
    model,
    batch: Dict[str, torch.Tensor],
    region_criterion: nn.Module,
    status_criterion: nn.Module,
    *,
    is_training: bool,
    global_step: int,
):
    if model_type == "gru":
        outputs = model(
            batch["region_seq"],
            batch["numeric_seq"],
            batch["context"],
        )
        loss = _masked_multitask_loss(
            outputs=outputs,
            region_labels=batch["region_labels"],
            status_labels=batch["status_labels"],
            label_mask=batch["label_mask"],
            region_criterion=region_criterion,
            status_criterion=status_criterion,
        )
        return outputs, loss

    outputs = model(
        batch["region_seq"],
        batch["parent_region_seq"],
        batch["numeric_seq"],
        batch["mask_seq"],
        batch["context"],
    )
    loss = _masked_hierarchical_loss(
        model=model,
        outputs=outputs,
        coarse_region_labels=batch["coarse_region_labels"],
        child_region_labels=batch["child_region_labels"],
        status_labels=batch["status_labels"],
        label_mask=batch["label_mask"],
        region_criterion=region_criterion,
        status_criterion=status_criterion,
        horizons_minutes=cfg.dataset.horizons_minutes,
        horizon_consistency_weight=cfg.training.horizon_consistency_weight,
        horizon_consistency_gap_scaled=cfg.training.horizon_consistency_gap_scaled,
    )
    if is_training:
        if _should_compute_teacher_distillation(cfg, is_training=is_training, global_step=global_step):
            loss = loss + _teacher_distillation_loss(
                cfg=cfg,
                model=model,
                outputs=outputs,
                batch=batch,
            )
        loss = loss + _router_regularization_loss(
            cfg=cfg,
            model=model,
            outputs=outputs,
            batch=batch,
            region_criterion=region_criterion,
            status_criterion=status_criterion,
        )
    return outputs, loss


def _move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=(device.type == "cuda")) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _accumulate_weighted_loss(
    total: Optional[torch.Tensor],
    loss: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    contribution = loss.detach() * int(batch_size)
    if total is None:
        return contribution
    return total + contribution


def run_training(
    cfg: ProjectConfig,
    *,
    resume: bool = False,
    resume_from: Optional[str] = None,
    restart: bool = False,
) -> None:
    _set_seed(cfg.training.seed)
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_ds = RegionalForecastDataset(str(cfg.dataset_dir / "train.pt"))
    val_ds = RegionalForecastDataset(str(cfg.dataset_dir / "val.pt"))
    sample_counts = {
        "train": len(train_ds),
        "val": len(val_ds),
    }

    checkpoint_path = cfg.checkpoint_dir / f"{cfg.output.run_name}.pt"
    summary_path = cfg.checkpoint_dir / f"{cfg.output.run_name}_train_summary.json"

    model_type, model = _build_model(cfg, train_ds)
    _initialize_from_checkpoint_if_requested(cfg, model)

    if model_type == "markov":
        model.fit(
            train_ds.tensor("region_seq").long(),
            train_ds.tensor("region_labels").long(),
            train_ds.tensor("status_labels").long(),
            train_ds.tensor("label_mask").bool(),
        )
        torch.save(
            {
                "model_type": "markov",
                "model_state": model.state_dict(),
                "metadata": train_ds.metadata,
                "config": cfg.to_dict(),
            },
            checkpoint_path,
        )
        metrics = _evaluate_baseline(model, val_ds, cfg)
        _save_json(summary_path, {"mode": "markov", "validation_metrics": metrics})
        register_training_run(
            cfg,
            model_type="markov",
            metadata=train_ds.metadata,
            checkpoint_path=checkpoint_path,
            summary_path=summary_path,
            best_val_loss=0.0,
            sample_counts=sample_counts,
        )
        print(f"Saved baseline checkpoint: {checkpoint_path}")
        print(f"Registered model: {cfg.model_registry_path}")
        return

    device = _resolve_device(cfg)
    use_amp = bool(cfg.training.amp and device.type == "cuda")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg.training.lr_scheduler_factor,
        patience=cfg.training.lr_scheduler_patience,
        threshold=cfg.training.lr_scheduler_threshold,
        threshold_mode=str(cfg.training.lr_scheduler_threshold_mode),
    )
    region_criterion = nn.CrossEntropyLoss()
    status_criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_val_loss = float("inf")
    best_epoch = 0
    patience_ref_loss = float("inf")
    epochs_without_improvement = 0
    history: List[Dict[str, float]] = []
    start_epoch = 1

    resume_checkpoint = _resolve_resume_checkpoint(
        cfg,
        checkpoint_path,
        resume=resume,
        resume_from=resume_from,
        restart=restart,
    )
    if resume_checkpoint is not None:
        checkpoint = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
        checkpoint_model_type = str(checkpoint.get("model_type", "")).lower()
        if checkpoint_model_type != model_type:
            raise ValueError(
                f"Resume checkpoint model_type {checkpoint_model_type!r} does not match current model_type {model_type!r}."
            )
        model.load_state_dict(checkpoint["model_state"])
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            _optimizer_to_device(optimizer, device)
        if "scheduler_state" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        if "scaler_state" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state"])

        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_epoch = int(checkpoint.get("best_epoch", 0))
        best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        patience_ref_loss = float(checkpoint.get("patience_ref_loss", best_val_loss))
        epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", 0))
        history = list(checkpoint.get("history", []))
        _restore_rng_state(checkpoint.get("rng_state"))
        print(
            f"Resuming from checkpoint: {resume_checkpoint} "
            f"(completed_epoch={start_epoch - 1}, best_val_loss={best_val_loss:.4f})"
        )

        if start_epoch > cfg.training.epochs:
            print(
                f"Checkpoint already reached configured epochs ({cfg.training.epochs}). "
                "Nothing to resume."
            )
            return

    train_loader = _stack_loader(
        train_ds,
        cfg.training.batch_size,
        True,
        cfg.training.num_workers,
        cfg.training.pin_memory,
        cfg.training.persistent_workers,
    )
    val_loader = _stack_loader(
        val_ds,
        cfg.training.batch_size,
        False,
        cfg.training.num_workers,
        cfg.training.pin_memory,
        cfg.training.persistent_workers,
    )

    completed_epoch = start_epoch - 1
    last_checkpoint_path = _last_checkpoint_path(checkpoint_path)
    global_step = 0
    for epoch in range(start_epoch, cfg.training.epochs + 1):
        model.train()
        train_loss_total: Optional[torch.Tensor] = None
        train_count = 0

        for raw_batch in train_loader:
            batch = _move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            with torch.amp.autocast("cuda", enabled=use_amp):
                _, loss = _compute_loss_for_batch(
                    cfg=cfg,
                    model_type=model_type,
                    model=model,
                    batch=batch,
                    region_criterion=region_criterion,
                    status_criterion=status_criterion,
                    is_training=True,
                    global_step=global_step,
                )

            if not torch.isfinite(loss):
                print("Skipping non-finite training batch loss.")
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            batch_size = batch["region_labels"].shape[0]
            train_loss_total = _accumulate_weighted_loss(train_loss_total, loss, int(batch_size))
            train_count += batch_size

        model.eval()
        val_loss_total: Optional[torch.Tensor] = None
        val_count = 0
        val_has_non_finite = False
        with torch.no_grad():
            for raw_batch in val_loader:
                batch = _move_batch(raw_batch, device)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    _, loss = _compute_loss_for_batch(
                        cfg=cfg,
                        model_type=model_type,
                        model=model,
                        batch=batch,
                        region_criterion=region_criterion,
                        status_criterion=status_criterion,
                        is_training=False,
                        global_step=global_step,
                    )

                if not torch.isfinite(loss):
                    val_has_non_finite = True
                    print("Encountered non-finite validation batch loss.")
                    break

                batch_size = batch["region_labels"].shape[0]
                val_loss_total = _accumulate_weighted_loss(val_loss_total, loss, int(batch_size))
                val_count += batch_size

        train_epoch_loss = (
            0.0
            if train_loss_total is None
            else float((train_loss_total / max(train_count, 1)).item())
        )
        if val_has_non_finite:
            val_epoch_loss = float("inf")
        else:
            val_epoch_loss = (
                0.0
                if val_loss_total is None
                else float((val_loss_total / max(val_count, 1)).item())
            )
        scheduler.step(val_epoch_loss)
        current_lr = float(optimizer.param_groups[0]["lr"])
        history.append({"epoch": epoch, "train_loss": train_epoch_loss, "val_loss": val_epoch_loss})
        print(f"epoch={epoch:02d} train_loss={train_epoch_loss:.4f} val_loss={val_epoch_loss:.4f} lr={current_lr:.6g}")

        previous_best = best_val_loss
        is_best = val_epoch_loss < previous_best
        is_significant = val_epoch_loss < (patience_ref_loss - cfg.training.early_stopping_min_delta)
        if is_best:
            best_val_loss = val_epoch_loss
            best_epoch = epoch
            torch.save(
                _build_training_checkpoint(
                    model_type=model_type,
                    model=model,
                    metadata=train_ds.metadata,
                    cfg=cfg,
                    epoch=epoch,
                    best_epoch=best_epoch,
                    best_val_loss=best_val_loss,
                    patience_ref_loss=patience_ref_loss,
                    epochs_without_improvement=epochs_without_improvement,
                    history=history,
                ),
                checkpoint_path,
            )

        if is_significant:
            patience_ref_loss = val_epoch_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        completed_epoch = epoch
        if cfg.training.save_last_checkpoint:
            torch.save(
                _build_training_checkpoint(
                    model_type=model_type,
                    model=model,
                    metadata=train_ds.metadata,
                    cfg=cfg,
                    epoch=epoch,
                    best_epoch=best_epoch,
                    best_val_loss=best_val_loss,
                    patience_ref_loss=patience_ref_loss,
                    epochs_without_improvement=epochs_without_improvement,
                    history=history,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                ),
                last_checkpoint_path,
            )

        if epoch >= cfg.training.min_epochs and epochs_without_improvement >= cfg.training.early_stopping_patience:
            print(
                "Early stopping triggered "
                f"(no validation improvement for {epochs_without_improvement} epochs)."
            )
            break

    _save_json(
        summary_path,
        {
            "mode": model_type,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "completed_epochs": completed_epoch,
            "resumed_from": None if resume_checkpoint is None else str(resume_checkpoint),
            "history": history,
        },
    )
    register_training_run(
        cfg,
        model_type=model_type,
        metadata=train_ds.metadata,
        checkpoint_path=checkpoint_path,
        summary_path=summary_path,
        best_val_loss=best_val_loss,
        sample_counts=sample_counts,
    )
    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Registered model: {cfg.model_registry_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a long-horizon regional forecasting model.")
    parser.add_argument("--config", type=str, default="config/region_forecast.yaml")
    parser.add_argument("--scope", type=str, choices=["global", "type"], default=None)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--resume", action="store_true", help="Resume from the latest saved training checkpoint.")
    parser.add_argument("--resume-from", type=str, default=None, help="Resume from a specific checkpoint path.")
    parser.add_argument("--restart", action="store_true", help="Ignore any auto-resume setting and start from scratch.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.scope is not None:
        cfg.dataset.scope = args.scope
    if args.name is not None:
        cfg.dataset.name = args.name

    print(f"Training scope: {cfg.dataset_scope}")
    if cfg.dataset_name is not None:
        print(f"Training name: {cfg.dataset_name}")
    print(f"Dataset dir: {cfg.dataset_dir}")
    print(f"Checkpoint dir: {cfg.checkpoint_dir}")
    run_training(cfg, resume=args.resume, resume_from=args.resume_from, restart=args.restart)
