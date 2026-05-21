# Experiment: H3-Aware Candidate Refinement

## Goal

The tuned multiscale model has stronger top-k, NLL, and fixed-budget candidate coverage than the GRU devsplit baseline, but its probability-mass candidate sets are much sharper. This lowers `candidate_union_cover` because the `0.8` mass threshold is reached with fewer cells.

This experiment tests a lightweight H3-aware postprocessor:

1. diffuse each known H3 cell's probability mass to its ring-1 H3 disk
2. preserve the unknown-class probability
3. apply reachability filtering after diffusion
4. evaluate without retraining the model

## Code Changes

Files changed:

- `regional_forecast/evaluation/spatial_refinement.py`
- `regional_forecast/config.py`
- `regional_forecast/pipeline/evaluate.py`
- `regional_forecast/pipeline/predict.py`
- `tests/test_spatial_refinement.py`

New config fields under `evaluation`:

- `h3_refinement_enabled`
- `h3_refinement_alpha`
- `h3_refinement_ring`
- `h3_refinement_preserve_unknown`
- `checkpoint_run_name`

`checkpoint_run_name` lets evaluation write a distinct ablation run while loading an existing checkpoint.

## Configs

- `configs/region_forecast_adrr_devsplit_multiscale_h3_refine_a005.yaml`
- `configs/region_forecast_adrr_devsplit_multiscale_h3_refine_a010.yaml`
- `configs/region_forecast_adrr_devsplit_multiscale_h3_refine_a020.yaml`

All load checkpoint:

- `adrr_devsplit_multiscale_transformer_h3_tuned`

## Verification

```bash
python -m py_compile regional_forecast/evaluation/spatial_refinement.py regional_forecast/config.py regional_forecast/pipeline/evaluate.py regional_forecast/pipeline/predict.py
OMP_NUM_THREADS=1 python -m unittest tests.test_spatial_refinement tests.test_candidate_metrics tests.test_evaluate_sparse_probs -v
```

Results:

- `7` tests passed
- py_compile passed

## Evaluation Commands

No training was run.

```bash
OMP_NUM_THREADS=1 python evaluation.py --config configs/region_forecast_adrr_devsplit_multiscale_h3_refine_a005.yaml
OMP_NUM_THREADS=1 python evaluation.py --config configs/region_forecast_adrr_devsplit_multiscale_h3_refine_a010.yaml
OMP_NUM_THREADS=1 python evaluation.py --config configs/region_forecast_adrr_devsplit_multiscale_h3_refine_a020.yaml
```

Runtime:

- alpha `0.05`: `14.14s`
- alpha `0.10`: `14.29s`
- alpha `0.20`: `14.16s`

Evaluation settings:

- batch size: `1024`
- device: `cuda`
- reachability filter: enabled
- H3 refinement: CPU postprocess, ring `1`

## Results Versus Tuned Multiscale

Macro mean over 10/30/60/120 minute horizons:

| Run | Top5 | NLL | Brier | ECE | Union Cover | Union Cells | Union Area km2 | Cover at Budget |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tuned | 0.7235 | 2.4497 | 0.7783 | 0.0562 | 0.8063 | 7.1653 | 323031 | 0.9469 |
| a005 | 0.7244 | 2.3116 | 0.7790 | 0.0642 | 0.8201 | 7.5102 | 340941 | 0.9593 |
| a010 | 0.7251 | 2.2956 | 0.7803 | 0.0751 | 0.8328 | 7.8822 | 361078 | 0.9615 |
| a020 | 0.7283 | 2.2906 | 0.7846 | 0.0974 | 0.8608 | 8.6843 | 405082 | 0.9639 |

Delta versus tuned multiscale:

| Run | dTop5 | dNLL | dECE | dUnion Cover | dUnion Cells | dCover at Budget |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| a005 | +0.0009 | -0.1381 | +0.0081 | +0.0137 | +0.3449 | +0.0124 |
| a010 | +0.0016 | -0.1542 | +0.0189 | +0.0265 | +0.7169 | +0.0146 |
| a020 | +0.0048 | -0.1591 | +0.0413 | +0.0545 | +1.5190 | +0.0170 |

## Horizon Details for alpha 0.20

| Horizon | Top5 | Union Cover | Union Cells | Union Area km2 | Cover at Budget | NLL | ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10m | 0.7525 | 0.8795 | 8.15 | 100973 | 0.9895 | 2.2054 | 0.1465 |
| 30m | 0.6322 | 0.8401 | 10.88 | 135099 | 0.9273 | 2.6608 | 0.0945 |
| 60m | 0.8204 | 0.8842 | 6.57 | 575270 | 0.9839 | 1.8725 | 0.0996 |
| 120m | 0.7083 | 0.8396 | 9.14 | 808987 | 0.9548 | 2.4237 | 0.0491 |

## Interpretation

H3 refinement is useful as a candidate-quality postprocessor:

- probability-mass candidate coverage improves monotonically with alpha
- fixed-budget coverage improves for every tested alpha
- top-5 and NLL are not degraded; both slightly improve
- candidate area increases, but alpha `0.20` still stays below the GRU baseline area on every horizon

The tradeoff is calibration:

- ECE worsens as alpha increases
- alpha `0.05` is the safest setting if calibration is prioritized
- alpha `0.20` is the best setting if candidate recall is prioritized

Current recommendation:

- keep H3 refinement configurable and default-off
- use alpha `0.10` as the balanced ablation point
- report alpha `0.05/0.10/0.20` as a candidate-refinement sensitivity study

## Next Step

The next method improvement should make the refinement train-aware or confidence-aware:

1. only refine low-margin predictions
2. tune alpha per horizon
3. evaluate calibration repair with temperature scaling after refinement
4. test whether graph/H3 message passing during training can produce the same coverage gain with less ECE cost
