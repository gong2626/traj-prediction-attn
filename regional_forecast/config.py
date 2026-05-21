from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import ceil
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence, Type, TypeVar

import yaml


T = TypeVar("T")
CONFIG_DIR_NAMES = {"config", "configs"}


@dataclass
class ColumnConfig:
    aircraft_id: str = "target_id"
    trajectory_id: str = "target_id"
    time: str = "timestamp"
    lat: str = "latitude"
    lon: str = "longitude"
    altitude: str = "altitude"
    speed: str = "speed"
    heading: str = "rotation"
    vertical_rate: str = "vrate"
    aircraft_type: str = "aircraft_type"
    nationality: str = "nationality"
    callsign: str = "callsign"


@dataclass
class AOIConfig:
    enabled: bool = False
    name: str = "global"
    mode: str = "global"
    lat_min: float = -90.0
    lat_max: float = 90.0
    lon_min: float = -180.0
    lon_max: float = 180.0
    polygon: List[List[float]] = field(default_factory=list)
    keep_strategy: str = "clip"
    context_points: int = 0
    min_points_in_aoi: int = 1


@dataclass
class GridConfig:
    kind: str = "h3"
    h3_resolution: int = 3
    input_parent_resolution: Optional[int] = None
    target_h3_resolutions: List[int] = field(default_factory=list)
    coarse_h3_resolutions: List[int] = field(default_factory=list)


@dataclass
class DatasetConfig:
    input_path: str = "data/csv"
    input_format: str = "csv"
    file_pattern: str = "*.csv"
    aircraft_type_map_path: Optional[str] = None
    scope: str = "global"
    name: Optional[str] = None
    default_aircraft_type: str = "unknown_type"
    sample_interval_minutes: int = 10
    history_minutes: int = 120
    horizons_minutes: List[int] = field(default_factory=lambda: [30, 60, 120, 300])
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    max_trajectories: Optional[int] = None
    mission_gap_minutes: int = 90
    max_interpolate_gap_minutes: int = 30
    ground_altitude_m: float = 300.0
    ground_speed_threshold: float = 30.0
    ground_vrate_threshold: float = 5.0
    ground_stop_minutes: int = 30
    min_segment_points: int = 12
    split_unit: str = "aircraft"


@dataclass
class ModelConfig:
    model_type: str = "hierarchical_gru"
    embedding_dim: int = 128
    parent_embedding_dim: int = 64
    numeric_projection_dim: int = 64
    mask_projection_dim: int = 16
    context_projection_dim: int = 32
    hidden_dim: int = 256
    num_layers: int = 3
    dropout: float = 0.2
    child_context_dim: int = 64
    enable_retrieval: bool = False
    retrieval_top_k: int = 8
    temporal_attention_dim: int = 128
    temporal_attention_dropout: float = 0.0
    horizon_embedding_dim: int = 32
    transformer_num_layers: int = 2
    transformer_num_heads: int = 4
    transformer_ffn_dim: int = 512
    transformer_dropout: float = 0.1
    multiscale_stride: int = 2
    multiscale_num_layers: int = 1


@dataclass
class TrainingConfig:
    batch_size: int = 256
    epochs: int = 60
    min_epochs: int = 15
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    num_workers: int = 8
    device: str = "cuda"
    amp: bool = True
    pin_memory: bool = True
    persistent_workers: bool = True
    lr_scheduler_patience: int = 4
    lr_scheduler_factor: float = 0.5
    lr_scheduler_threshold: float = 1e-4
    lr_scheduler_threshold_mode: str = "rel"
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.0
    horizon_consistency_weight: float = 0.0
    horizon_consistency_gap_scaled: bool = True
    auto_resume: bool = False
    resume_from_checkpoint: Optional[str] = None
    save_last_checkpoint: bool = True
    seed: int = 42


@dataclass
class EvaluationConfig:
    top_ks: List[int] = field(default_factory=lambda: [1, 3, 5])
    calibration_bins: int = 10
    neighbor_hit_rings: List[int] = field(default_factory=lambda: [0, 1, 2, 3])
    candidate_probability_mass: float | List[float] = 0.8
    candidate_top_cells: int | List[int] = 12
    prediction_examples: int = 20
    hierarchical_parent_top_k: int | List[int] = 8
    reachable_filter_enabled: bool = True
    reachable_max_speed_kts: float = 750.0
    reachable_margin_km: float = 250.0
    h3_refinement_enabled: bool = False
    h3_refinement_alpha: float | List[float] = 0.0
    h3_refinement_ring: int | List[int] = 1
    h3_refinement_preserve_unknown: bool = True
    checkpoint_run_name: Optional[str] = None


@dataclass
class CatalogConfig:
    include_global: bool = True
    evaluate_after_training: bool = True
    min_rows_per_type: int = 1000
    min_aircraft_per_type: Optional[int] = None
    max_types: Optional[int] = None
    skip_existing: bool = False


@dataclass
class OutputConfig:
    root_dir: str = "artifacts"
    run_name: str = "global_h3_movement_gru"


@dataclass
class ProjectConfig:
    columns: ColumnConfig = field(default_factory=ColumnConfig)
    aoi: AOIConfig = field(default_factory=AOIConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    catalog: CatalogConfig = field(default_factory=CatalogConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @property
    def history_steps(self) -> int:
        return max(1, ceil(self.dataset.history_minutes / self.dataset.sample_interval_minutes))

    @property
    def horizon_steps(self) -> List[int]:
        return [
            max(1, ceil(horizon / self.dataset.sample_interval_minutes))
            for horizon in self.dataset.horizons_minutes
        ]

    @property
    def target_h3_resolutions(self) -> List[int]:
        if not self.grid.target_h3_resolutions:
            return [self.grid.h3_resolution] * len(self.dataset.horizons_minutes)
        if len(self.grid.target_h3_resolutions) == 1:
            return [int(self.grid.target_h3_resolutions[0])] * len(self.dataset.horizons_minutes)
        if len(self.grid.target_h3_resolutions) != len(self.dataset.horizons_minutes):
            raise ValueError(
                "grid.target_h3_resolutions must have length 1 or match dataset.horizons_minutes."
            )
        return [int(value) for value in self.grid.target_h3_resolutions]

    @property
    def input_parent_resolution(self) -> int:
        if self.grid.input_parent_resolution is not None:
            return max(0, int(self.grid.input_parent_resolution))
        return max(0, int(self.grid.h3_resolution) - 1)

    @property
    def coarse_h3_resolutions(self) -> List[int]:
        if not self.grid.coarse_h3_resolutions:
            return [max(0, int(value) - 1) for value in self.target_h3_resolutions]
        if len(self.grid.coarse_h3_resolutions) == 1:
            return [int(self.grid.coarse_h3_resolutions[0])] * len(self.dataset.horizons_minutes)
        if len(self.grid.coarse_h3_resolutions) != len(self.dataset.horizons_minutes):
            raise ValueError(
                "grid.coarse_h3_resolutions must have length 1 or match dataset.horizons_minutes."
            )
        return [max(0, int(value)) for value in self.grid.coarse_h3_resolutions]

    def _resolve_eval_sequence(
        self,
        raw: float | int | Sequence[float] | Sequence[int],
        cast_type,
    ) -> List[float] | List[int]:
        if isinstance(raw, (list, tuple)):
            values = list(raw)
        else:
            values = [raw]

        if len(values) == 1:
            return [cast_type(values[0])] * len(self.dataset.horizons_minutes)
        if len(values) != len(self.dataset.horizons_minutes):
            raise ValueError(
                "Evaluation config values must have length 1 or match dataset.horizons_minutes."
            )
        return [cast_type(value) for value in values]

    @property
    def candidate_probability_mass_per_horizon(self) -> List[float]:
        return self._resolve_eval_sequence(self.evaluation.candidate_probability_mass, float)

    @property
    def candidate_top_cells_per_horizon(self) -> List[int]:
        return self._resolve_eval_sequence(self.evaluation.candidate_top_cells, int)

    @property
    def hierarchical_parent_top_k_per_horizon(self) -> List[int]:
        return self._resolve_eval_sequence(self.evaluation.hierarchical_parent_top_k, int)

    @property
    def h3_refinement_alpha_per_horizon(self) -> List[float]:
        return self._resolve_eval_sequence(self.evaluation.h3_refinement_alpha, float)

    @property
    def h3_refinement_ring_per_horizon(self) -> List[int]:
        return self._resolve_eval_sequence(self.evaluation.h3_refinement_ring, int)

    @property
    def dataset_scope(self) -> str:
        scope = str(self.dataset.scope or "global").strip().lower()
        if scope not in {"global", "type"}:
            raise ValueError("dataset.scope must be one of: global, type")
        return scope

    @property
    def dataset_name(self) -> Optional[str]:
        if self.dataset.name is None:
            return None
        value = str(self.dataset.name).strip()
        return value or None

    @property
    def dataset_name_slug(self) -> str:
        if self.dataset_scope == "global":
            return "global"
        if self.dataset_name is None:
            raise ValueError("dataset.name is required when dataset.scope is 'type'.")
        slug = re.sub(r"[^0-9A-Za-z]+", "_", self.dataset_name).strip("_").lower()
        return slug or "unknown_type"

    @property
    def base_dir(self) -> Path:
        source_path = getattr(self, "_config_path", None)
        if source_path is None:
            return Path.cwd()

        config_path = Path(str(source_path)).resolve()
        parent = config_path.parent
        if parent.name.lower() in CONFIG_DIR_NAMES and parent.parent != parent:
            return parent.parent
        return parent

    def resolve_path(self, raw_path: str | Path | None) -> Path | None:
        if raw_path is None:
            return None

        path = Path(str(raw_path))
        if path.is_absolute():
            return path

        config_relative = (self.base_dir / path).resolve()
        cwd_relative = path.resolve()

        if config_relative.exists():
            return config_relative
        if cwd_relative.exists():
            return cwd_relative
        return config_relative

    @property
    def artifact_root(self) -> Path:
        resolved = self.resolve_path(self.output.root_dir)
        if resolved is None:
            raise ValueError("output.root_dir cannot be null.")
        return resolved

    @property
    def dataset_dir(self) -> Path:
        if self.dataset_scope == "type":
            return self.artifact_root / "datasets" / "by_type" / self.dataset_name_slug
        return self.artifact_root / "datasets"

    @property
    def checkpoint_dir(self) -> Path:
        if self.dataset_scope == "type":
            return self.artifact_root / "checkpoints" / "by_type" / self.dataset_name_slug
        return self.artifact_root / "checkpoints"

    @property
    def evaluation_dir(self) -> Path:
        if self.dataset_scope == "type":
            return self.artifact_root / "evaluations" / "by_type" / self.dataset_name_slug
        return self.artifact_root / "evaluations"

    @property
    def catalog_dir(self) -> Path:
        return self.artifact_root / "catalog"

    @property
    def model_registry_path(self) -> Path:
        return self.artifact_root / "model_registry.json"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _load_dataclass(cls: Type[T], raw: Optional[Dict[str, Any]]) -> T:
    raw = raw or {}
    valid = {name: raw[name] for name in cls.__dataclass_fields__ if name in raw}
    return cls(**valid)


def load_config_dict(raw: Dict[str, Any], source_path: str | Path | None = None) -> ProjectConfig:
    cfg = ProjectConfig(
        columns=_load_dataclass(ColumnConfig, raw.get("columns")),
        aoi=_load_dataclass(AOIConfig, raw.get("aoi")),
        grid=_load_dataclass(GridConfig, raw.get("grid")),
        dataset=_load_dataclass(DatasetConfig, raw.get("dataset")),
        model=_load_dataclass(ModelConfig, raw.get("model")),
        training=_load_dataclass(TrainingConfig, raw.get("training")),
        evaluation=_load_dataclass(EvaluationConfig, raw.get("evaluation")),
        catalog=_load_dataclass(CatalogConfig, raw.get("catalog")),
        output=_load_dataclass(OutputConfig, raw.get("output")),
    )
    if source_path is not None:
        setattr(cfg, "_config_path", str(Path(source_path).resolve()))
    return cfg


def load_config(path: str) -> ProjectConfig:
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return load_config_dict(raw, source_path=path)
