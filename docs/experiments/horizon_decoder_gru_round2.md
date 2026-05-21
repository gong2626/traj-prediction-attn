# Experiment: Horizon Decoder GRU Round 2

## Goal

Evaluate the second research modification over both prior references:

- subset baseline: `hierarchical_gru`
- round 1 attention model: `attention_hierarchical_gru`

This round replaces the shared temporal pooling summary with per-horizon query attention over GRU encoder states while preserving the existing coarse-to-fine H3 hierarchy.

## Configs

Smoke:

- `configs/region_forecast_adrr_horizon_decoder_smoke.yaml`

Subset:

- `configs/region_forecast_adrr_horizon_decoder_subset.yaml`

Reference runs:

- baseline: `configs/region_forecast_adrr_subset_baseline.yaml`
- round 1 attention: `configs/region_forecast_adrr_attention_subset.yaml`

## Model Change

New model type:

- `horizon_decoder_gru`

Key changes:

1. keep the same GRU encoder and hierarchical coarse-to-fine decoding structure
2. keep a shared base summary from the final GRU hidden state plus context
3. add learned horizon embeddings
4. build one learned query per horizon
5. attend from each horizon query to the full encoder sequence
6. use the resulting horizon-specific decoder features for coarse, status, and child prediction

Files changed in this round:

- `regional_forecast/models/forecast_model.py`
- `regional_forecast/models/__init__.py`
- `regional_forecast/pipeline/train.py`
- `regional_forecast/pipeline/evaluate.py`
- `regional_forecast/pipeline/predict.py`
- `tests/test_attention_hierarchical_model.py`
- `configs/region_forecast_adrr_horizon_decoder_smoke.yaml`
- `configs/region_forecast_adrr_horizon_decoder_subset.yaml`

## Git Diff Summary

Round 2 extends the round 1 work by adding:

- `HorizonDecoderHierarchicalH3RegionalForecaster`
- config-selectable `model_type == "horizon_decoder_gru"`
- train/evaluate/predict checkpoint loading for the new model
- regression coverage for horizon-specific decoder outputs

Working tree diff summary at the end of this round:

```text
regional_forecast/config.py
regional_forecast/models/__init__.py
regional_forecast/models/forecast_model.py
regional_forecast/pipeline/evaluate.py
regional_forecast/pipeline/predict.py
regional_forecast/pipeline/train.py
```

## Smoke Validation

### Commands

```bash
python build_dataset.py --config configs/region_forecast_adrr_horizon_decoder_smoke.yaml
python training.py --config configs/region_forecast_adrr_horizon_decoder_smoke.yaml
python evaluation.py --config configs/region_forecast_adrr_horizon_decoder_smoke.yaml
```

### Verification

```bash
python -m unittest tests.test_attention_hierarchical_model tests.test_evaluate_sparse_probs -v
python -m py_compile regional_forecast/models/forecast_model.py regional_forecast/models/__init__.py regional_forecast/pipeline/train.py regional_forecast/pipeline/evaluate.py regional_forecast/pipeline/predict.py
```

Both passed before the smoke run.

### Acceleration Settings

- dataset build command: config-driven dataset pipeline
- build observed workers: `1`
- training batch size: `256`
- device: `cuda`
- amp: `true`
- dataloader workers: `8`
- pin memory: `true`
- persistent workers: `true`

### Notes

The smoke pipeline ran cleanly end to end.

One operational note remains:

- `build_dataset.py` reported `workers: 1` even though the config keeps `training.num_workers=8`

This does not block the current experiments, but it is a clear future performance cleanup item.

## Subset Experiment

### Commands

```bash
python build_dataset.py --config configs/region_forecast_adrr_horizon_decoder_subset.yaml
python training.py --config configs/region_forecast_adrr_horizon_decoder_subset.yaml
python evaluation.py --config configs/region_forecast_adrr_horizon_decoder_subset.yaml
```

### Acceleration Settings

- dataset build command: config-driven dataset pipeline
- build observed workers: `1`
- training batch size: `512`
- device: `cuda`
- amp: `true`
- dataloader workers: `8`
- pin memory: `true`
- persistent workers: `true`

### Runtime

Observed subset wall times:

- dataset build: about `28s`
- training: about `17s`
- evaluation: about `3s`

Training summary:

- best val loss: `2.2960`
- best epoch: `5`
- completed epochs: `8`
- early stopping triggered after epoch `8`

## Metrics

### Baseline Reference

Run:

- `adrr_hierarchical_gru_subset_baseline`

| Horizon | Top1 | Top3 | Top5 | Candidate Union Cover | Region ECE | Region NLL | Region Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10m | 0.0016 | 0.0594 | 0.1197 | 0.4002 | 0.7460 | 15.2750 | 1.6257 |
| 30m | 0.0000 | 0.0289 | 0.0596 | 0.1626 | 0.7104 | 18.3048 | 1.5751 |
| 60m | 0.0132 | 0.0661 | 0.1156 | 0.3370 | 0.4258 | 12.5854 | 1.3133 |
| 120m | 0.0000 | 0.0271 | 0.0468 | 0.2624 | 0.5667 | 19.7117 | 1.4143 |

### Round 1 Attention Reference

Run:

- `adrr_attention_hierarchical_gru_subset`

| Horizon | Top1 | Top3 | Top5 | Candidate Union Cover | Region ECE | Region NLL | Region Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10m | 0.0000 | 0.0269 | 0.0531 | 0.3344 | 0.9496 | 17.7149 | 1.9026 |
| 30m | 0.0000 | 0.0172 | 0.0298 | 0.1111 | 0.9008 | 20.2471 | 1.8192 |
| 60m | 0.0000 | 0.0485 | 0.0727 | 0.2632 | 0.6841 | 13.2177 | 1.5017 |
| 120m | 0.0000 | 0.0317 | 0.0709 | 0.2323 | 0.7921 | 20.1004 | 1.6417 |

### Round 2 Horizon Decoder

Run:

- `adrr_horizon_decoder_gru_subset`

| Horizon | Top1 | Top3 | Top5 | Candidate Union Cover | Region ECE | Region NLL | Region Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10m | 0.0000 | 0.0602 | 0.0895 | 0.3487 | 0.8268 | 16.8390 | 1.7332 |
| 30m | 0.0000 | 0.0253 | 0.0443 | 0.1210 | 0.7779 | 19.1925 | 1.6672 |
| 60m | 0.0011 | 0.0297 | 0.0881 | 0.2852 | 0.4667 | 13.5325 | 1.3573 |
| 120m | 0.0000 | 0.0377 | 0.0603 | 0.2609 | 0.6145 | 19.3746 | 1.4691 |

### Delta vs Baseline

| Horizon | Top3 Delta | Top5 Delta | Candidate Union Delta | Region NLL Delta | Region Brier Delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| 10m | +0.0008 | -0.0301 | -0.0515 | +1.5640 | +0.1075 |
| 30m | -0.0036 | -0.0154 | -0.0416 | +0.8876 | +0.0921 |
| 60m | -0.0363 | -0.0275 | -0.0518 | +0.9471 | +0.0440 |
| 120m | +0.0106 | +0.0136 | -0.0015 | -0.3371 | +0.0548 |

### Delta vs Round 1 Attention

| Horizon | Top3 Delta | Top5 Delta | Candidate Union Delta | Region NLL Delta | Region Brier Delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| 10m | +0.0333 | +0.0365 | +0.0143 | -0.8760 | -0.1694 |
| 30m | +0.0081 | +0.0145 | +0.0099 | -1.0546 | -0.1520 |
| 60m | -0.0187 | +0.0154 | +0.0220 | +0.3148 | -0.1444 |
| 120m | +0.0060 | -0.0106 | +0.0287 | -0.7258 | -0.1726 |

## Inference Benchmark

Benchmark setup:

- dataset: subset test split, `1262` samples
- device: `cuda`
- measurement scope: model forward + hierarchical fine-probability reconstruction + reachability filter
- method: one warmup pass, then mean of `3` timed passes

| Model | Mean Time (s) | Std (s) | ms / sample | samples / sec |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.2797 | 0.0101 | 0.2217 | 4511.46 |
| attention_hierarchical_gru | 0.2796 | 0.0006 | 0.2215 | 4513.94 |
| horizon_decoder_gru | 0.2917 | 0.0047 | 0.2312 | 4325.73 |

Interpretation:

- the round 2 decoder adds only a small inference cost over baseline on the subset benchmark
- it is materially more efficient than running a much heavier decoder family would be

## Result Summary

Round 2 is a **clear improvement over round 1 attention**, but it is **not yet a full replacement for the subset baseline**.

Observed pattern:

- it restores most of the damage introduced by shared temporal attention
- it slightly improves `10m top3`
- it improves `120m top3`, `120m top5`, and `120m region_nll`
- it still regresses on most other likelihood, calibration, and coverage metrics relative to baseline

## Interpretation

What this result means:

1. the core hypothesis was directionally correct:
   per-horizon querying is better than one shared temporal attention summary
2. the current decoder is still not strong enough to dominate the simple baseline on the small subset
3. the most promising gain is at the longest horizon, where horizon-conditioned sequence selection matters more
4. short- and medium-horizon performance likely still needs stronger regularization or a better coupling between coarse and child prediction

## Problems Observed

1. `status_accuracy` remains `1.0` across all runs, which confirms the status task is still label-degenerate on this subset
2. candidate union coverage still drops relative to baseline
3. `30m` remains difficult and unstable
4. dataset build parallelism still appears underutilized in the current pipeline

## Next Step Recommendation

Do not stop at round 2.

Recommended next move:

1. keep `horizon_decoder_gru` as the current best research branch after round 1
2. add a lightweight structural bias rather than a heavy graph model:
   - coarse-logit neighbor smoothing
   - learned H3 parent/neighbor embedding refinement
3. consider an auxiliary consistency objective between adjacent horizons or between coarse and fine predictions
4. separate the status task from the main research claim unless the label distribution becomes more informative
