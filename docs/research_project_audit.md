# Global Aviation Multi-Horizon Regional Forecasting Project Audit

## Audit Scope

This audit covers the current repository state, code structure, data pipeline, model structure, training/evaluation/prediction flow, artifact layout, and the current baseline assets already present in the workspace.

The goal of this document is to establish a stable engineering and research baseline before any model changes.

## Repository State

- Audit date: 2026-05-18 UTC
- Branch: `main`
- Git status at audit start: local branch is ahead of `origin/main` by 1 commit
- Latest local commit at audit start: `57a1a34 feat: 项目初始化，将进入正式改造环节`

There is no immediate git conflict, but future experiments should use isolated `run_name` values and experiment-specific configs to avoid contaminating the existing baseline assets.

## Top-Level Structure

Core directories and files:

- `config/`
  - runtime YAML configs
- `regional_forecast/`
  - core library code
  - `data/`: preprocessing, H3 regionization, dataset loading
  - `models/`: baseline and neural forecasters
  - `pipeline/`: build/train/evaluate/predict entry logic
  - `evaluation/`: metrics
- `tools/`
  - ADRR preparation, visualization, data utilities
- `artifacts/`
  - existing datasets, checkpoints, evaluation outputs, registry
- `data/`
  - ADRR raw month data and prepared shard directory

Thin entrypoints:

- `build_dataset.py`
- `training.py`
- `evaluation.py`
- `predict.py`
- `train_catalog.py`

These files only delegate into `regional_forecast.pipeline.*`.

## Important Audit Finding: README vs Code Drift

The repository README is useful, but it is not a perfect source of truth for the current ADRR setup.

Notable mismatches:

- README still describes the project in terms of `30/60/120/300` minute horizons.
- The active ADRR config uses `10/30/60/120` minute horizons.
- README contains stale Windows-style file links such as `/e:/traj2/...`, which are not valid in the current Linux workspace.

For the research workflow, the code and generated artifacts should be treated as authoritative over README prose.

## Active ADRR Configuration

Primary config for the current large-scale run:

- file: `config/region_forecast_adrr.yaml`
- input: `data/adrr_202403_shards`
- format: Parquet shards
- sample interval: 5 minutes
- history window: 30 minutes
- horizons: `[10, 30, 60, 120]`
- grid:
  - input H3 resolution: 3
  - input parent resolution: 2
  - target H3 resolutions: `[3, 3, 2, 2]`
  - coarse H3 resolutions: `[2, 2, 1, 1]`
- model type: `hierarchical_gru`
- batch size: `1024`
- epochs: `400`
- auto resume: enabled
- output root: `artifacts`
- run name: `global_h3_movement_gru`

## Data Pipeline

### 1. ADRR Raw Month Preparation

ADRR preprocessing starts in `tools/prepare_adrr_month.py`.

Purpose:

- ingest one month of ADRR source tables
- normalize them into a standard trajectory schema
- partition them into deterministic shard files

Existing acceleration hooks:

- `--workers`
- `--num-shards`
- chunked CSV reading
- optional `--max-flights` smoke-test truncation

Research implication:

- a small debug dataset can be produced without hand-preparing data, using `--max-flights`

### 2. Dataset Build

Entry:

- `build_dataset.py` -> `regional_forecast.pipeline.build_dataset.main()`

Core implementation:

- `regional_forecast.data.preprocess.build_dataset_artifacts()`
- `regional_forecast.data.adrr_sharded.build_artifacts_from_sharded_dir()`

There are two dataset-build paths:

1. generic path for CSV/Parquet/JSON trajectory inputs
2. optimized ADRR sharded path for `data/adrr_202403_shards`

The current ADRR flow uses the optimized sharded path.

### 3. Mission Splitting

Raw trajectories are split into mission segments by:

- time gap threshold: `mission_gap_minutes`
- long ground-stop boundary detection

This is implemented in `regional_forecast.data.preprocess.split_missions()`.

### 4. Resampling

Each mission is resampled onto a fixed time grid:

- frequency: `sample_interval_minutes`
- interpolation for continuous signals
- `observed_flag` and `gap_flag` are generated

This is implemented in `regional_forecast.data.preprocess._resample_single_segment()`.

### 5. Regionization and Feature Construction

`regional_forecast.data.preprocess.add_regions_and_features()` adds:

- input H3 cell / region id
- parent H3 region id
- target fine region ids per horizon
- target coarse region ids per horizon
- hierarchy metadata from coarse to fine
- numeric / mask / context features

### 6. Split Strategy

Train/val/test split is performed at aircraft id level, not per sample:

- `train_ratio`
- `val_ratio`
- `test_ratio`
- split unit recorded as `aircraft`

This is good research hygiene because it reduces leakage across samples from the same aircraft.

### 7. Sample Construction

Each sample contains:

- history sequence of input regions
- history sequence of parent regions
- history numeric features
- history mask features
- current context features
- future region labels for all horizons
- future coarse labels for all horizons
- future child labels within the coarse branch
- future status labels
- label mask

Samples are only created from airborne end states and are kept if at least one future horizon is valid.

## Feature Inventory

### Numeric Features: 18

From existing artifact metadata:

- `lat_sin`
- `lat_cos`
- `lon_sin`
- `lon_cos`
- `delta_north_km_norm`
- `delta_east_km_norm`
- `step_distance_norm`
- `track_bearing_sin`
- `track_bearing_cos`
- `region_changed`
- `parent_region_changed`
- `altitude_km`
- `speed_norm`
- `heading_sin`
- `heading_cos`
- `vrate_norm`
- `delta_minutes_norm`
- `ground_flag`

### Mask Features: 2

- `observed_flag`
- `gap_flag`

### Context Features: 9

- `hour_sin`
- `hour_cos`
- `weekday_sin`
- `weekday_cos`
- `current_lat_sin`
- `current_lat_cos`
- `current_lon_sin`
- `current_lon_cos`
- `target_hash`

## Dataset Artifact Format

Current dataset artifacts under `artifacts/datasets/` are stored in `chunked_npy_v1` format.

Observed sample counts:

- train: `14,505,637`
- val: `3,108,933`
- test: `3,098,032`

Observed chunk count per split:

- train: `128`
- val: `128`
- test: `128`

This is important because:

- the project already uses an IO-efficient chunked dataset format
- DataLoader batching is chunk-aware
- future debug/subset experiments should reuse this pattern instead of creating giant monolithic tensors

## Model Inventory

The codebase currently supports three model types:

1. `markov`
2. `gru`
3. `hierarchical_gru`

The active ADRR baseline uses `hierarchical_gru`.

### `gru`

`LongHorizonRegionalForecaster`:

- region embedding
- numeric projection
- context projection
- GRU encoder
- uses only the final hidden state
- separate linear region heads per horizon
- separate linear status heads per horizon

### `hierarchical_gru`

`HierarchicalH3RegionalForecaster`:

- fine region embedding
- parent region embedding
- numeric projection
- mask projection
- context projection
- GRU encoder
- uses only the final hidden state
- separate coarse heads per horizon
- separate status heads per horizon
- per-horizon child branch prediction conditioned on chosen coarse region

This model is the main research baseline.

### Current Architectural Limitation

The current hierarchical model discards the full GRU hidden-state sequence and keeps only the final hidden state as the summary representation.

This directly motivates the first planned research direction:

- temporal attention over encoder states

## Current Forward Prediction Logic

For `hierarchical_gru`:

1. encode history into one feature vector
2. predict coarse region logits for each horizon
3. choose top coarse parents
4. predict child logits conditioned on parent ids
5. reconstruct sparse fine-region probabilities
6. predict airborne/landed status probabilities
7. optionally apply reachability filtering

This is a clean place to insert:

- temporal attention pooling
- horizon embeddings / horizon-aware decoder logic
- lightweight hierarchy-aware or neighbor-aware refinement

## Losses

### `gru`

Masked multitask loss:

- cross-entropy over region class per valid horizon
- cross-entropy over status class per valid horizon
- average across valid horizon losses

### `hierarchical_gru`

Masked hierarchical loss:

- coarse region cross-entropy
- child index cross-entropy within valid children
- status cross-entropy
- averaged over valid horizons

The baseline currently does not include:

- path consistency loss
- calibration-aware loss
- auxiliary neighbor or graph regularization

## Evaluation and Metrics

Evaluation entry:

- `evaluation.py` -> `regional_forecast.pipeline.evaluate.run_evaluation()`

Current metrics already implemented per horizon:

- `region_top1`
- `region_top3`
- `region_top5`
- `region_top{k}_ring`
- `candidate_top{k}_cover`
- `candidate_union_cover`
- `region_nll`
- `region_brier`
- `region_ece`
- `region_top1_unknown_rate`
- `status_nll`
- `status_accuracy`
- `status_ece`
- `landed_rate`

This is stronger than the initial README suggests: NLL, Brier, and ECE are already available.

### Reachability

Evaluation applies a `ReachabilityFilter` to predicted probability arrays before reporting the main metrics.

Research implication:

- reachability-filter ablation should be added explicitly in the research plan
- current default metrics are post-filter metrics, not raw model metrics

### Current Missing Evaluation Outputs

The code does not yet explicitly report:

- candidate cell count summary
- candidate area summary
- inference latency summary
- automatic global vs type comparison table
- automatic reachability on/off paired report

These should be added during the research workflow.

## Prediction Flow

Prediction entry:

- `predict.py` -> `regional_forecast.pipeline.predict.main()`

Current capabilities:

- direct scope selection: global or type
- aircraft-type based routing via registry fallback
- resampling and feature construction for inference
- coarse/fine hierarchical probability reconstruction
- geometry and GeoJSON export
- candidate-region export for visualization

Compatibility constraint:

- any new model type must remain loadable through the same train/evaluate/predict path
- registry behavior must preserve loading of existing `hierarchical_gru` checkpoints

## Registry and Reproducibility

Registry file:

- `artifacts/model_registry.json`

Current registry tracks:

- scope / name / run_name
- checkpoint path
- dataset dir
- training summary path
- evaluation output paths
- reduced metric summary

This is enough to support reproducible experiment bookkeeping, but not enough by itself for a paper-ready experiment log.

Additional research logging should go to:

- `docs/experiments/*.md`
- `docs/ablation.md`
- experiment-specific YAML configs under `configs/`

## Existing Baseline Assets Already Present

The workspace already contains a trained and evaluated global baseline:

- checkpoint: `artifacts/checkpoints/global_h3_movement_gru.pt`
- dataset manifests: `artifacts/datasets/{train,val,test}.pt`
- evaluation summary:
  - `artifacts/evaluations/global_h3_movement_gru_summary.csv`
  - `artifacts/evaluations/global_h3_movement_gru_summary.json`
  - `artifacts/evaluations/global_h3_movement_gru_examples.json`

These are useful audit references, but they are not a substitute for the planned stage-1 debug/subset reproduction.

## Existing Full Baseline Metrics

From `artifacts/evaluations/global_h3_movement_gru_summary.json`:

| Horizon | Top1 | Top3 | Top5 | Candidate Union Cover | Region ECE | Region NLL | Region Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10m | 0.7786 | 0.9586 | 0.9877 | 0.9153 | 0.0148 | 0.6932 | 0.3207 |
| 30m | 0.6384 | 0.8966 | 0.9548 | 0.8950 | 0.0390 | 1.1284 | 0.4981 |
| 60m | 0.7046 | 0.9374 | 0.9817 | 0.8960 | 0.0040 | 0.8589 | 0.4080 |
| 120m | 0.5375 | 0.8378 | 0.9280 | 0.8662 | 0.0143 | 1.4050 | 0.6045 |

Additional observations from the same summary:

- `status_accuracy = 1.0` at all horizons
- `status_ece` is essentially zero
- `landed_rate = 0.0` at all horizons

This strongly suggests the status task is currently trivial or degenerate on the present test distribution.

## Baseline Weaknesses Identified from Existing Assets

### 1. 120-minute horizon is the weakest

This is the clearest accuracy bottleneck:

- lowest `top1`
- lowest `top3`
- lowest `top5`
- lowest candidate union cover
- highest NLL and Brier

### 2. 30-minute horizon is unexpectedly weaker than 60-minute horizon

This is non-monotonic and should be investigated during stage 1.

Possible causes:

- class distribution differences by horizon
- coarse/fine hierarchy behavior
- reachability filter interaction
- sample availability pattern

### 3. Status prediction is not currently informative

Because landed rate is zero for all reported horizons:

- status accuracy is not a meaningful discriminator right now
- future paper claims should not rely on the status head unless data balance changes

### 4. Calibration is already measurable

The project already computes:

- `region_ece`
- `status_ece`
- `region_nll`
- `region_brier`

This is good for research. It means future model improvements can be evaluated beyond top-k accuracy.

## Code Quality and Extensibility Assessment

### Strengths

- clear pipeline separation
- config-driven paths and runtime behavior
- chunked dataset format for large-scale training
- explicit metadata saved into dataset artifacts and checkpoints
- registry-based model resolution
- evaluation already supports calibration and candidate-region metrics
- existing GPU/AMP/DataLoader worker hooks are present

### Weaknesses

- model definitions are currently simple and horizon heads are mostly independent
- temporal sequence information is compressed into only the final hidden state
- README is partially outdated
- evaluation lacks some paper-facing summaries
- no built-in experiment report documents under `docs/`

## Immediate Research Implications

### What should be preserved

- `hierarchical_gru` must remain untouched as the baseline
- registry compatibility must be preserved
- new model types must be config-selectable
- old checkpoints must remain loadable

### What can be improved first with low risk

Recommended first-round model change:

- add temporal attention pooling over encoder states
- add horizon embedding or horizon-aware head conditioning

Why this is the safest first step:

- it fits naturally into the existing GRU encoder architecture
- it does not require changing dataset artifacts
- it can be introduced behind a new `model_type`
- it supports clean ablation against the current baseline

### What should not be attempted first

- full-graph GNN over all H3 cells
- heavy external trajectory foundation models
- any change that requires rebuilding the full-month dataset repeatedly

## Stage-1 Execution Implications

Stage 1 should not use the full ADRR month for iteration.

Instead, it should create:

1. a smoke/debug preparation path using:
   - `tools/prepare_adrr_month.py --max-flights ...`
2. a debug/subset config using:
   - `dataset.max_trajectories`
   - smaller batch size if needed
   - shorter training schedule
   - unique `run_name`
3. a baseline reproduction record under:
   - `docs/experiments/baseline.md`

## Audit Conclusion

The repository already contains a functional large-scale hierarchical baseline with:

- ADRR month-scale data preparation
- chunked dataset artifacts
- GRU-based hierarchical H3 forecasting
- calibration-aware evaluation
- registry-based model resolution

The codebase is in a good state for incremental research iteration.

The most promising low-risk research entry point is:

- `hierarchical_gru` -> `attention_hierarchical_gru`

with the first ablation focusing on:

- temporal attention pooling
- horizon-aware head conditioning

This can be done without breaking the existing artifact format or baseline compatibility.
