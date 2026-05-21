# Experiment: Multiscale Transformer Tuning and Horizon Consistency Round 5

## Goal

Round 4 established that `multiscale_transformer_h3` is the strongest structural branch so far, but it still had two clear issues:

1. best validation loss appeared at epoch `1`, which strongly suggested an optimization problem rather than an architecture limit
2. `120m` top-k improved, but `NLL` remained worse than desired

This round tests two narrower changes on top of the same architecture:

1. a tuned optimization recipe
2. a lightweight cross-horizon consistency regularizer over adjacent decoder features

The intent was to separate:

- gains from better training dynamics
- gains from the new regularizer

## Code and Config Changes

Files changed in this round:

- `regional_forecast/config.py`
- `regional_forecast/pipeline/train.py`
- `tests/test_training_losses.py`
- `configs/region_forecast_adrr_multiscale_transformer_tuned_subset.yaml`
- `configs/region_forecast_adrr_multiscale_transformer_consistency_smoke.yaml`
- `configs/region_forecast_adrr_multiscale_transformer_consistency_subset.yaml`

New training config fields:

- `training.horizon_consistency_weight`
- `training.horizon_consistency_gap_scaled`

Regularizer definition:

1. take adjacent per-horizon `decoder_features`
2. compute mean squared difference between adjacent horizon features
3. mask out invalid adjacent horizon pairs
4. down-weight wider horizon gaps using `min_gap / gap`
5. multiply by `horizon_consistency_weight`
6. add the result to the standard hierarchical loss

Current implementation scope:

- only active when `decoder_features` exist
- fully disabled by default with `horizon_consistency_weight: 0.0`
- does not modify model loading or checkpoint compatibility

## Verification

```bash
OMP_NUM_THREADS=1 python -m unittest tests.test_attention_hierarchical_model tests.test_evaluate_sparse_probs tests.test_training_losses -v
python -m py_compile regional_forecast/config.py regional_forecast/models/forecast_model.py regional_forecast/models/__init__.py regional_forecast/pipeline/train.py regional_forecast/pipeline/evaluate.py regional_forecast/pipeline/predict.py tests/test_training_losses.py
```

Both passed.

## Smoke Validation

Config:

- `configs/region_forecast_adrr_multiscale_transformer_consistency_smoke.yaml`

Commands:

```bash
OMP_NUM_THREADS=1 python build_dataset.py --config configs/region_forecast_adrr_multiscale_transformer_consistency_smoke.yaml --workers 8
OMP_NUM_THREADS=1 python training.py --config configs/region_forecast_adrr_multiscale_transformer_consistency_smoke.yaml
OMP_NUM_THREADS=1 python evaluation.py --config configs/region_forecast_adrr_multiscale_transformer_consistency_smoke.yaml
```

Runtime:

- build: `5.08s`
- train: `5.58s`
- eval: `2.60s`

Smoke training summary:

- best val loss: `1.8396`
- best epoch: `1`
- completed epochs: `4`

The smoke run confirmed that the new loss path is numerically stable and does not break train/eval.

## Subset Experiments

### E5a. Tuned Multiscale Transformer

Config:

- `configs/region_forecast_adrr_multiscale_transformer_tuned_subset.yaml`

Changes relative to round 4:

- `dropout: 0.15` instead of `0.2`
- `transformer_dropout: 0.05` instead of `0.1`
- `learning_rate: 1.5e-4` instead of `3e-4`
- `epochs: 12`
- `lr_scheduler_patience: 3`
- `early_stopping_patience: 5`
- `horizon_consistency_weight: 0.0`

Commands:

```bash
OMP_NUM_THREADS=1 python build_dataset.py --config configs/region_forecast_adrr_multiscale_transformer_tuned_subset.yaml --workers 8
OMP_NUM_THREADS=1 python training.py --config configs/region_forecast_adrr_multiscale_transformer_tuned_subset.yaml
OMP_NUM_THREADS=1 python evaluation.py --config configs/region_forecast_adrr_multiscale_transformer_tuned_subset.yaml
```

Runtime:

- build: `6.49s`
- train: `24.96s`
- eval: `3.33s`

Training summary:

- best val loss: `2.3383`
- best epoch: `10`
- completed epochs: `12`

### E5b. Tuned Multiscale Transformer + Consistency

Config:

- `configs/region_forecast_adrr_multiscale_transformer_consistency_subset.yaml`

Changes relative to E5a:

- `horizon_consistency_weight: 0.05`
- `horizon_consistency_gap_scaled: true`

Commands:

```bash
OMP_NUM_THREADS=1 python build_dataset.py --config configs/region_forecast_adrr_multiscale_transformer_consistency_subset.yaml --workers 8
OMP_NUM_THREADS=1 python training.py --config configs/region_forecast_adrr_multiscale_transformer_consistency_subset.yaml
OMP_NUM_THREADS=1 python evaluation.py --config configs/region_forecast_adrr_multiscale_transformer_consistency_subset.yaml
```

Runtime:

- build: `6.58s`
- train: `23.35s`
- eval: `3.27s`

Training summary:

- best val loss: `2.3381`
- best epoch: `6`
- completed epochs: `11`

## Metrics

### Reference Round 4 Multiscale

Run:

- `adrr_multiscale_transformer_h3_subset`

| Horizon | Top3 | Top5 | Candidate Cover | ECE | NLL | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10m | 0.0232 | 0.0502 | 0.3063 | 0.9592 | 19.4738 | 1.9201 |
| 30m | 0.0198 | 0.0297 | 0.1035 | 0.9258 | 21.2589 | 1.8613 |
| 60m | 0.0368 | 0.0788 | 0.2405 | 0.7386 | 15.4574 | 1.5665 |
| 120m | 0.0363 | 0.0466 | 0.1623 | 0.8315 | 22.2338 | 1.7000 |

### E5a. Tuned Multiscale

Run:

- `adrr_multiscale_transformer_h3_tuned_subset`

| Horizon | Top1 | Top3 | Top5 | Candidate Cover | ECE | NLL | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10m | 0.0048 | 0.0744 | 0.1150 | 0.3614 | 0.7152 | 16.6594 | 1.6060 |
| 30m | 0.0000 | 0.0363 | 0.0661 | 0.1927 | 0.6951 | 19.6638 | 1.5776 |
| 60m | 0.0526 | 0.1222 | 0.1695 | 0.3167 | 0.4445 | 14.1401 | 1.3352 |
| 120m | 0.0363 | 0.0967 | 0.1382 | 0.2902 | 0.5189 | 18.7927 | 1.3829 |

### E5b. Tuned Multiscale + Consistency

Run:

- `adrr_multiscale_transformer_h3_consistency_subset`

| Horizon | Top1 | Top3 | Top5 | Candidate Cover | ECE | NLL | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10m | 0.0000 | 0.0415 | 0.0928 | 0.3285 | 0.8457 | 18.1457 | 1.7490 |
| 30m | 0.0000 | 0.0297 | 0.0441 | 0.1498 | 0.8130 | 20.4820 | 1.6995 |
| 60m | 0.0145 | 0.0670 | 0.1248 | 0.3298 | 0.5066 | 14.9441 | 1.3876 |
| 120m | 0.0017 | 0.0501 | 0.0881 | 0.2867 | 0.6381 | 18.9963 | 1.4728 |

## Delta vs Round 4

### E5a. Tuned Multiscale

| Horizon | Top3 Delta | Top5 Delta | Cover Delta | ECE Delta | NLL Delta | Brier Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10m | +0.0512 | +0.0647 | +0.0551 | -0.2439 | -2.8144 | -0.3140 |
| 30m | +0.0165 | +0.0363 | +0.0892 | -0.2307 | -1.5951 | -0.2836 |
| 60m | +0.0854 | +0.0907 | +0.0762 | -0.2942 | -1.3173 | -0.2314 |
| 120m | +0.0604 | +0.0915 | +0.1278 | -0.3126 | -3.4411 | -0.3172 |

### E5b. Consistency vs Round 4

| Horizon | Top3 Delta | Top5 Delta | Cover Delta | ECE Delta | NLL Delta | Brier Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10m | +0.0184 | +0.0425 | +0.0222 | -0.1135 | -1.3281 | -0.1711 |
| 30m | +0.0099 | +0.0143 | +0.0463 | -0.1129 | -0.7769 | -0.1617 |
| 60m | +0.0302 | +0.0460 | +0.0894 | -0.2320 | -0.5134 | -0.1789 |
| 120m | +0.0138 | +0.0415 | +0.1244 | -0.1934 | -3.2374 | -0.2273 |

## Delta vs Tuned Multiscale

This isolates the effect of the new regularizer:

| Horizon | Top3 Delta | Top5 Delta | Cover Delta | ECE Delta | NLL Delta | Brier Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10m | -0.0329 | -0.0222 | -0.0329 | +0.1305 | +1.4863 | +0.1430 |
| 30m | -0.0066 | -0.0220 | -0.0430 | +0.1179 | +0.8182 | +0.1219 |
| 60m | -0.0552 | -0.0447 | +0.0131 | +0.0621 | +0.8040 | +0.0524 |
| 120m | -0.0466 | -0.0501 | -0.0035 | +0.1192 | +0.2036 | +0.0899 |

## Inference Benchmark

Dataset:

- history-60 subset test split, `1035` samples

Setup:

- device: `cuda`
- 1 warmup pass
- mean of 3 timed forward-only passes

| Model | Mean Time (s) | Std (s) | ms / sample | samples / sec |
| --- | ---: | ---: | ---: | ---: |
| Round 4 multiscale | 0.1322 | 0.0123 | 0.1277 | 7829.55 |
| E5a tuned multiscale | 0.1151 | 0.0066 | 0.1112 | 8992.76 |
| E5b consistency multiscale | 0.1227 | 0.0102 | 0.1185 | 8435.70 |

The tuned branch is not slower than round 4 in practice. The regularizer adds no inference cost because it is train-only.

## Main Findings

1. The dominant round-4 bottleneck was optimization, not architecture.
   - round 4 peaked at epoch `1`
   - E5a peaks at epoch `10`
   - almost every tracked forecasting metric improves substantially

2. The tuned multiscale model is the new lead branch.
   - it beats the round-4 multiscale model on all tracked horizons for `Top3`, `Top5`, `ECE`, `NLL`, and `Brier`
   - it also exceeds the fair history-60 baseline on all tracked `Top3` and `Top5` metrics, with large calibration gains

3. The current consistency regularizer is not helping.
   - `horizon_consistency_weight=0.05` is too strong, or the regularizer itself is not aligned with the forecasting objective
   - E5b still beats round 4, but it is consistently worse than E5a

4. The `120m` problem from round 4 is largely fixed by tuning alone.
   - round 4 `120m NLL`: `22.2338`
   - tuned `120m NLL`: `18.7927`
   - round 4 `120m Top5`: `0.0466`
   - tuned `120m Top5`: `0.1382`

## Problem Analysis

What this round suggests:

- the multiscale architecture was under-trained rather than fundamentally flawed
- lower LR, lighter dropout, and more patience were enough to unlock much stronger behavior
- a naive smoothness penalty across adjacent horizon decoder features can suppress useful horizon specialization

Remaining issues:

- `status_accuracy` is saturated at `1.0`, so status prediction remains uninformative on this subset
- evaluator outputs still do not expose candidate cell count / candidate area, so coverage remains the main candidate proxy for now
- no reachability filter ablation was run in this round

## Next Step

Recommended next move:

1. keep `adrr_multiscale_transformer_h3_tuned_subset` as the mainline research model
2. do **not** keep the current consistency regularizer as part of the main method
3. move to a lightweight H3-aware refinement on top of the tuned model:
   - coarse-logit neighbor smoothing
   - or trainable H3 neighbor bias / refinement

Optional follow-up ablation:

- retry consistency with a much smaller weight such as `0.005` or `0.01`
- or move the regularizer from decoder features to probability-space KL only for compatible horizon pairs
