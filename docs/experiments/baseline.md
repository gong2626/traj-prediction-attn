# Baseline Reproduction Log

## Scope

This document tracks the reproducible baseline workflow for the current ADRR-based global hierarchical baseline.

It distinguishes between:

- existing full-scale baseline assets already present under `artifacts/`
- new research-stage debug/subset reproductions created during this project

## Environment

- system: Ubuntu 22.04
- Python: 3.12
- PyTorch: 2.8.0
- CUDA: 12.8
- GPU: NVIDIA GeForce RTX 4080 SUPER, 32760 MiB
- CPU: 16 vCPU
- RAM: 80 GB

## Existing Full Baseline Assets Found During Audit

Existing run:

- run name: `global_h3_movement_gru`
- dataset root: `artifacts/datasets`
- checkpoint: `artifacts/checkpoints/global_h3_movement_gru.pt`
- evaluation summary:
  - `artifacts/evaluations/global_h3_movement_gru_summary.csv`
  - `artifacts/evaluations/global_h3_movement_gru_summary.json`

Observed full baseline metrics:

| Horizon | Top1 | Top3 | Top5 | Candidate Union Cover | Region ECE | Region NLL | Region Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10m | 0.7786 | 0.9586 | 0.9877 | 0.9153 | 0.0148 | 0.6932 | 0.3207 |
| 30m | 0.6384 | 0.8966 | 0.9548 | 0.8950 | 0.0390 | 1.1284 | 0.4981 |
| 60m | 0.7046 | 0.9374 | 0.9817 | 0.8960 | 0.0040 | 0.8589 | 0.4080 |
| 120m | 0.5375 | 0.8378 | 0.9280 | 0.8662 | 0.0143 | 1.4050 | 0.6045 |

Current weaknesses from the full baseline:

- weakest horizon: `120m`
- non-monotonic issue: `30m` is weaker than `60m`
- status task appears degenerate in the current distribution

## New Debug Reproduction

### Purpose

Before any research change, verify that:

- build dataset
- train
- evaluate

can all run under an isolated debug config without touching the full baseline artifacts.

### Debug Config

File:

- `configs/region_forecast_adrr_smoke.yaml`

Key settings:

- `dataset.max_trajectories: 32`
- `model.model_type: hierarchical_gru`
- `training.batch_size: 256`
- `training.epochs: 2`
- `training.device: cuda`
- `training.amp: true`
- `training.num_workers: 8`
- `output.root_dir: artifacts/research_smoke`
- `output.run_name: adrr_hierarchical_gru_smoke`

### Build Command

```bash
python build_dataset.py --config configs/region_forecast_adrr_smoke.yaml --workers 8
```

Observed output:

- train: `464`
- val: `129`
- test: `102`

### Training Command

```bash
python training.py --config configs/region_forecast_adrr_smoke.yaml
```

Acceleration settings used:

- workers: `8`
- batch size: `256`
- device: `cuda`
- amp: `true`
- pin memory: `true`
- persistent workers: `true`

Observed training log:

- epoch 1: train loss `2.0149`, val loss `1.8697`
- epoch 2: train loss `1.7959`, val loss `1.9469`
- wall time: about `5s`

Produced artifacts:

- `artifacts/research_smoke/checkpoints/adrr_hierarchical_gru_smoke.pt`
- `artifacts/research_smoke/checkpoints/adrr_hierarchical_gru_smoke_last.pt`
- `artifacts/research_smoke/checkpoints/adrr_hierarchical_gru_smoke_train_summary.json`

### Evaluation Command

```bash
python evaluation.py --config configs/region_forecast_adrr_smoke.yaml
```

## Debugging Finding During Baseline Reproduction

### Problem

The first smoke evaluation attempt failed on GPU with:

- `CUDA error: device-side assert triggered`
- failure site: `regional_forecast/pipeline/evaluate.py`
- failing expression: child lookup indexing inside `_build_sparse_fine_probs()`

### Root Cause

The fast-path child lookup tables built in evaluation did not include a row for the `unknown coarse id`.

In small-subset runs, `top_parent_ids` can contain the unknown coarse class.
The lookup tensors were built with row count `len(coarse_to_children)` instead of `len(coarse_to_children) + 1`, causing an out-of-bounds index.

### Fix

Minimal compatibility-preserving fix:

- update evaluation-side lookup tensor construction to allocate one extra row
- fill the extra row with `unknown_fine_id`
- keep the validity mask false for that row so probability mass falls back to unknown fine

Changed file:

- `regional_forecast/pipeline/evaluate.py`

Regression test added:

- `tests/test_evaluate_sparse_probs.py`

Verification:

```bash
python -m unittest tests.test_evaluate_sparse_probs -v
python evaluation.py --config configs/region_forecast_adrr_smoke.yaml
```

Both passed after the fix.

### Git Diff Summary for This Fix

At this point, the meaningful code diff is:

- `regional_forecast/pipeline/evaluate.py`: fix unknown coarse lookup overflow in hierarchical evaluation

## Smoke Evaluation Results

File:

- `artifacts/research_smoke/evaluations/adrr_hierarchical_gru_smoke_summary.json`

Metrics:

| Horizon | Top1 | Top3 | Top5 | Candidate Union Cover | Region ECE | Region NLL | Region Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10m | 0.0000 | 0.0784 | 0.1569 | 0.6863 | 0.8806 | 12.0320 | 1.7695 |
| 30m | 0.0000 | 0.0256 | 0.1026 | 0.2564 | 0.7995 | 15.4234 | 1.6401 |
| 60m | 0.0000 | 0.0000 | 0.0000 | 0.3000 | 0.1090 | 4.1191 | 1.0132 |
| 120m | 0.0000 | 0.0000 | 0.0000 | 0.7419 | 0.7042 | 10.5975 | 1.4890 |

Interpretation:

- the smoke run is only a correctness run, not a performance baseline
- two epochs on 32 trajectories are far too small for meaningful quality claims
- the purpose of this run was to validate the end-to-end pipeline and expose hidden failures

## Subset Baseline Reproduction

### Purpose

Build a small but non-trivial comparison baseline for model ablations without touching the existing full baseline artifacts.

### Subset Baseline Config

File:

- `configs/region_forecast_adrr_subset_baseline.yaml`

Key settings:

- `dataset.max_trajectories: 256`
- `model.model_type: hierarchical_gru`
- `training.batch_size: 512`
- `training.epochs: 8`
- `training.device: cuda`
- `training.amp: true`
- `training.num_workers: 8`
- `output.root_dir: artifacts/research_subset_baseline`
- `output.run_name: adrr_hierarchical_gru_subset_baseline`

### Build Command

```bash
python build_dataset.py --config configs/region_forecast_adrr_subset_baseline.yaml --workers 8
```

Observed output:

- train: `4875`
- val: `1122`
- test: `1262`

### Training Command

```bash
python training.py --config configs/region_forecast_adrr_subset_baseline.yaml
```

Acceleration settings used:

- workers: `8`
- batch size: `512`
- device: `cuda`
- amp: `true`
- pin memory: `true`
- persistent workers: `true`

Observed training log:

- epoch 1: train loss `2.4895`, val loss `2.3759`
- epoch 2: train loss `2.2850`, val loss `2.4476`
- epoch 3: train loss `2.2544`, val loss `2.3599`
- epoch 4: train loss `2.2151`, val loss `2.3754`
- epoch 5: train loss `2.1372`, val loss `2.3501`
- epoch 6: train loss `2.0638`, val loss `2.2802`
- epoch 7: train loss `1.9848`, val loss `2.2803`
- epoch 8: train loss `1.8964`, val loss `2.2546`
- wall time: about `14s`

Produced artifacts:

- `artifacts/research_subset_baseline/checkpoints/adrr_hierarchical_gru_subset_baseline.pt`
- `artifacts/research_subset_baseline/checkpoints/adrr_hierarchical_gru_subset_baseline_last.pt`
- `artifacts/research_subset_baseline/checkpoints/adrr_hierarchical_gru_subset_baseline_train_summary.json`

### Evaluation Command

```bash
python evaluation.py --config configs/region_forecast_adrr_subset_baseline.yaml
```

Observed wall time:

- about `3s`

Produced artifacts:

- `artifacts/research_subset_baseline/evaluations/adrr_hierarchical_gru_subset_baseline_summary.csv`
- `artifacts/research_subset_baseline/evaluations/adrr_hierarchical_gru_subset_baseline_summary.json`
- `artifacts/research_subset_baseline/evaluations/adrr_hierarchical_gru_subset_baseline_examples.json`

### Subset Baseline Metrics

| Horizon | Top1 | Top3 | Top5 | Candidate Union Cover | Region ECE | Region NLL | Region Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10m | 0.0016 | 0.0594 | 0.1197 | 0.4002 | 0.7460 | 15.2750 | 1.6257 |
| 30m | 0.0000 | 0.0289 | 0.0596 | 0.1626 | 0.7104 | 18.3048 | 1.5751 |
| 60m | 0.0132 | 0.0661 | 0.1156 | 0.3370 | 0.4258 | 12.5854 | 1.3133 |
| 120m | 0.0000 | 0.0271 | 0.0468 | 0.2624 | 0.5667 | 19.7117 | 1.4143 |

Additional observations:

- `status_accuracy = 1.0` at all horizons
- `status_ece` remains near zero
- `landed_rate = 0.0` at all horizons

### Relative Interpretation

This subset baseline is intentionally much weaker than the full baseline because:

- only 256 trajectories are used
- the run is short
- the goal is controlled research iteration, not production quality

However, it is already useful as an ablation anchor because:

- it is reproducible
- it exercises the same hierarchical path as the real baseline
- it is small enough for rapid model iteration

### Current Weakest Horizon on the Subset

The subset run keeps the same pattern as the full baseline:

- `120m` is still among the hardest horizons
- `30m` is also unexpectedly weak

This supports keeping the first model change focused on representation quality and horizon conditioning.

## Baseline Reproduction Status

### Completed

- stage-0 audit document written
- isolated smoke config created
- smoke dataset build passed
- smoke training passed
- smoke evaluation passed after fixing a real evaluation bug
- subset baseline config created
- subset baseline build/train/eval completed
- reproducible subset metrics recorded

### Stage 1 Outcome

Stage 1 now has:

- an audited full baseline reference
- a validated smoke pipeline
- a reproducible subset baseline for fast ablation work

## Current Problem Analysis

1. The codebase baseline path is functional after the evaluation bug fix.
2. The current status head remains low-value for research claims unless the data distribution changes.
3. The full baseline suggests the main model bottleneck is longer-horizon region prediction, especially `120m`.
4. The first model change should therefore target representation quality rather than rebuilding the entire pipeline.

## Next Step

Use the subset baseline as the comparison point for the first research model:

- temporal attention over encoder states
- horizon-aware conditioning
