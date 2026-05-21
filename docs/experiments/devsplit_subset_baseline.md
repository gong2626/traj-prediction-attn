# Experiment: Split-Preserving Dev Subset

## Why This Was Needed

The earlier `max_trajectories=256` subsets were not valid proxies for the full ADRR task.

They were built by truncating aircraft before dataset construction. That produced severe train/test label mismatch:

| Dataset | Horizon | Test Unseen Label Rate |
| --- | ---: | ---: |
| full baseline dataset | 120m | 0.000016 |
| old `subset30` | 120m | 0.401207 |
| old `subset60` | 120m | 0.436960 |

This explained the misleadingly low old subset metrics. The tuned round-5 model had `120m top5=0.1382`, but that was on a broken proxy task where the model predicted the unknown class for most examples.

## New Subset Construction

New script:

- `tools/build_dev_subset_from_artifacts.py`

Core strategy:

1. use already-built full dataset artifacts:
   - `artifacts/datasets/train.pt`
   - `artifacts/datasets/val.pt`
   - `artifacts/datasets/test.pt`
2. preserve the original full-data split and vocabulary
3. sample within each split instead of rebuilding from raw shards
4. filter val/test samples so all valid horizon labels have sufficient support in the full train split
5. force the sampled train subset to cover all labels selected in sampled val/test

Command:

```bash
OMP_NUM_THREADS=1 python tools/build_dev_subset_from_artifacts.py \
  --source-dataset-dir artifacts/datasets \
  --output-root artifacts/research_devsplit \
  --train-size 30000 \
  --val-size 2000 \
  --test-size 2000 \
  --min-train-support 10 \
  --train-examples-per-label 4 \
  --seed 42
```

Runtime:

- `24.05s`

Generated subset:

| Split | Samples |
| --- | ---: |
| train | 30000 |
| val | 2000 |
| test | 2000 |

Health check:

| Split | Horizon | Valid Samples | Unique Labels | Unseen Rate vs Dev Train |
| --- | ---: | ---: | ---: | ---: |
| val | 10m | 2000 | 1175 | 0.0 |
| val | 30m | 1703 | 1088 | 0.0 |
| val | 60m | 1368 | 536 | 0.0 |
| val | 120m | 964 | 490 | 0.0 |
| test | 10m | 2000 | 1206 | 0.0 |
| test | 30m | 1732 | 1127 | 0.0 |
| test | 60m | 1364 | 560 | 0.0 |
| test | 120m | 929 | 490 | 0.0 |

This subset is still much smaller than full training, but it no longer changes the task into a high-unseen-label problem.

## Baseline on Devsplit

Config:

- `configs/region_forecast_adrr_devsplit_baseline.yaml`

Command:

```bash
OMP_NUM_THREADS=1 python training.py --config configs/region_forecast_adrr_devsplit_baseline.yaml
OMP_NUM_THREADS=1 python evaluation.py --config configs/region_forecast_adrr_devsplit_baseline.yaml
```

Runtime:

- train: `58.44s`
- eval: `7.52s`

Training:

- best val loss: `0.8876`
- best epoch: `60`

Metrics:

| Horizon | Top1 | Top3 | Top5 | Candidate Cover | Unknown Top1 Rate | ECE | NLL | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10m | 0.2855 | 0.5640 | 0.6990 | 0.9335 | 0.0965 | 0.1109 | 2.6818 | 0.8692 |
| 30m | 0.2344 | 0.4648 | 0.5941 | 0.8666 | 0.1212 | 0.0807 | 3.0962 | 0.8947 |
| 60m | 0.4289 | 0.6965 | 0.8138 | 0.8937 | 0.0022 | 0.1203 | 2.0214 | 0.7461 |
| 120m | 0.3240 | 0.5705 | 0.6706 | 0.8654 | 0.1302 | 0.0682 | 3.1982 | 0.8242 |

Compared with the old broken subset, this is much closer to a useful proxy. It still does not reproduce full baseline top-k because the training sample is only `30k`, but candidate coverage and label support are now healthy.

## Multiscale Transformer on Devsplit

Config:

- `configs/region_forecast_adrr_devsplit_multiscale_tuned.yaml`

Command:

```bash
OMP_NUM_THREADS=1 python training.py --config configs/region_forecast_adrr_devsplit_multiscale_tuned.yaml
OMP_NUM_THREADS=1 python evaluation.py --config configs/region_forecast_adrr_devsplit_multiscale_tuned.yaml
```

Runtime:

- train: `68.49s`
- eval: `7.39s`

Training:

- best val loss: `0.7688`
- best epoch: `59`

Metrics:

| Horizon | Top1 | Top3 | Top5 | Candidate Cover | Unknown Top1 Rate | ECE | NLL | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10m | 0.3805 | 0.6255 | 0.7520 | 0.8320 | 0.0045 | 0.1047 | 2.3232 | 0.7866 |
| 30m | 0.2968 | 0.5173 | 0.6299 | 0.7950 | 0.0069 | 0.0642 | 2.8200 | 0.8445 |
| 60m | 0.4619 | 0.7067 | 0.8072 | 0.8189 | 0.0007 | 0.0377 | 1.9506 | 0.6922 |
| 120m | 0.3498 | 0.5834 | 0.7051 | 0.7793 | 0.0054 | 0.0179 | 2.7052 | 0.7899 |

Delta vs devsplit GRU baseline:

| Horizon | Top1 | Top3 | Top5 | Candidate Cover | Unknown Top1 Rate | ECE | NLL | Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10m | +0.0950 | +0.0615 | +0.0530 | -0.1015 | -0.0920 | -0.0062 | -0.3586 | -0.0826 |
| 30m | +0.0624 | +0.0525 | +0.0358 | -0.0716 | -0.1143 | -0.0165 | -0.2763 | -0.0502 |
| 60m | +0.0330 | +0.0103 | -0.0066 | -0.0748 | -0.0015 | -0.0826 | -0.0708 | -0.0539 |
| 120m | +0.0258 | +0.0129 | +0.0344 | -0.0861 | -0.1249 | -0.0503 | -0.4930 | -0.0344 |

## Interpretation

The old subset conclusion should be downgraded:

- it was useful only for code smoke tests
- it was not a valid performance proxy

The new devsplit is a better research proxy:

- label unseen rate is fixed
- vocabulary and split are inherited from full artifacts
- baseline no longer collapses to tiny top-k values

Current model conclusion on the healthier devsplit:

- `multiscale_transformer_h3_tuned` improves top1/top3, most top5, NLL, Brier, ECE, and unknown prediction rate
- it hurts candidate coverage
- therefore the next method step should focus on spatial/candidate-region refinement rather than more temporal encoder changes

After upgrading candidate metrics, the candidate regression is more nuanced:

- the multiscale model reaches the probability-mass threshold with fewer cells and much smaller area
- under fixed budget coverage, it improves `30m` and `120m`, is tied at `10m`, and is slightly lower at `60m`
- the next refinement should target probability-mass coverage at comparable area, not simply expand all candidate sets

## Next Step

Recommended next experiment:

1. keep this devsplit as the fast iteration dataset
2. add lightweight H3-aware candidate refinement on top of `multiscale_transformer_h3_tuned`
3. optimize specifically for candidate coverage without losing the top-k and NLL gains

Do not compare old `max_trajectories` subset metrics against the full baseline anymore.
