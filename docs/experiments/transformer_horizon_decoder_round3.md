# Experiment: Transformer Horizon Decoder Round 3

## Goal

Evaluate a less conservative third-round model change:

- replace the GRU sequence encoder with a Transformer encoder
- keep the hierarchical H3 coarse-to-fine output stack
- keep the round-2 horizon-query decoder logic

This round also introduces a fairer comparison setup by increasing the history window from `30` to `60` minutes for all compared models.

## Configs

Smoke:

- `configs/region_forecast_adrr_history60_baseline_smoke.yaml`
- `configs/region_forecast_adrr_transformer_horizon_smoke.yaml`

Subset:

- fair baseline: `configs/region_forecast_adrr_history60_baseline_subset.yaml`
- GRU horizon decoder control: `configs/region_forecast_adrr_horizon_decoder_history60_subset.yaml`
- Transformer variant: `configs/region_forecast_adrr_transformer_horizon_subset.yaml`

## Model Change

New model type:

- `transformer_horizon_decoder`

Core changes:

1. replace GRU history encoding with a Transformer encoder
2. add learned positional embeddings
3. use temporal masks for:
   - Transformer padding
   - masked summary pooling
   - horizon-query attention
4. retain the round-2 horizon-query decoder and hierarchical H3 heads

Files changed in this round:

- `regional_forecast/config.py`
- `regional_forecast/models/forecast_model.py`
- `regional_forecast/models/__init__.py`
- `regional_forecast/pipeline/train.py`
- `regional_forecast/pipeline/evaluate.py`
- `regional_forecast/pipeline/predict.py`
- `tests/test_attention_hierarchical_model.py`
- `configs/region_forecast_adrr_history60_baseline_smoke.yaml`
- `configs/region_forecast_adrr_history60_baseline_subset.yaml`
- `configs/region_forecast_adrr_horizon_decoder_history60_subset.yaml`
- `configs/region_forecast_adrr_transformer_horizon_smoke.yaml`
- `configs/region_forecast_adrr_transformer_horizon_subset.yaml`

## Git Diff Summary

Current working tree diff summary after rounds 0-3:

```text
regional_forecast/config.py
regional_forecast/models/__init__.py
regional_forecast/models/forecast_model.py
regional_forecast/pipeline/evaluate.py
regional_forecast/pipeline/predict.py
regional_forecast/pipeline/train.py
```

Round-3 specific additions:

- Transformer encoder model class
- transformer config fields
- history-60 comparison configs
- additional shape coverage in model tests

## Verification

### Tests

```bash
python -m unittest tests.test_attention_hierarchical_model tests.test_evaluate_sparse_probs -v
python -m py_compile regional_forecast/config.py regional_forecast/models/forecast_model.py regional_forecast/models/__init__.py regional_forecast/pipeline/train.py regional_forecast/pipeline/evaluate.py regional_forecast/pipeline/predict.py
```

Both passed before the full subset runs.

## Smoke Validation

### Commands

```bash
OMP_NUM_THREADS=1 python build_dataset.py --config configs/region_forecast_adrr_history60_baseline_smoke.yaml --workers 8
OMP_NUM_THREADS=1 python training.py --config configs/region_forecast_adrr_history60_baseline_smoke.yaml
OMP_NUM_THREADS=1 python evaluation.py --config configs/region_forecast_adrr_history60_baseline_smoke.yaml

OMP_NUM_THREADS=1 python build_dataset.py --config configs/region_forecast_adrr_transformer_horizon_smoke.yaml --workers 8
OMP_NUM_THREADS=1 python training.py --config configs/region_forecast_adrr_transformer_horizon_smoke.yaml
OMP_NUM_THREADS=1 python evaluation.py --config configs/region_forecast_adrr_transformer_horizon_smoke.yaml
```

### Acceleration Settings

- dataset build workers: `8`
- smoke batch size: `256`
- device: `cuda`
- amp: `true`
- dataloader workers: `8`
- pin memory: `true`
- persistent workers: `true`

### Sample Counts

History-60 smoke dataset sizes:

- train: `343`
- val: `99`
- test: `66`

Compared with history-30 smoke:

- train: `464`
- val: `129`
- test: `102`

Interpretation:

- increasing history length reduced the number of usable windows
- this must be kept in mind when comparing to earlier history-30 experiments

## Subset Experiment

### Commands

```bash
OMP_NUM_THREADS=1 python build_dataset.py --config configs/region_forecast_adrr_history60_baseline_subset.yaml --workers 8
OMP_NUM_THREADS=1 python training.py --config configs/region_forecast_adrr_history60_baseline_subset.yaml
OMP_NUM_THREADS=1 python evaluation.py --config configs/region_forecast_adrr_history60_baseline_subset.yaml

OMP_NUM_THREADS=1 python build_dataset.py --config configs/region_forecast_adrr_horizon_decoder_history60_subset.yaml --workers 8
OMP_NUM_THREADS=1 python training.py --config configs/region_forecast_adrr_horizon_decoder_history60_subset.yaml
OMP_NUM_THREADS=1 python evaluation.py --config configs/region_forecast_adrr_horizon_decoder_history60_subset.yaml

OMP_NUM_THREADS=1 python build_dataset.py --config configs/region_forecast_adrr_transformer_horizon_subset.yaml --workers 8
OMP_NUM_THREADS=1 python training.py --config configs/region_forecast_adrr_transformer_horizon_subset.yaml
OMP_NUM_THREADS=1 python evaluation.py --config configs/region_forecast_adrr_transformer_horizon_subset.yaml
```

### Acceleration Settings

- dataset build workers: `8`
- subset batch size: `512`
- device: `cuda`
- amp: `true`
- dataloader workers: `8`
- pin memory: `true`
- persistent workers: `true`

### Sample Counts

History-60 subset dataset sizes:

- train: `3909`
- val: `907`
- test: `1035`

Compared with history-30 subset baseline:

- train: `4875`
- val: `1122`
- test: `1262`

This means history-60 experiments are not directly comparable to history-30 runs without adjustment.

### Runtime

Observed wall times:

- baseline60 build: about `6.5s`
- baseline60 train: about `8.9s`
- baseline60 eval: about `3.2s`
- horizon_decoder60 build: about `6.5s`
- horizon_decoder60 train: about `9.6s`
- horizon_decoder60 eval: about `3.3s`
- transformer60 build: about `6.6s`
- transformer60 train: about `9.8s`
- transformer60 eval: about `3.2s`

Training summaries:

- all three history-60 models found their best validation loss at epoch `1`
- all three early-stopped after epoch `4`

Interpretation:

- longer history increased task difficulty enough that current optimization settings may now be under-tuned

## Metrics

### Fair History-60 Baseline

Run:

- `adrr_hierarchical_gru_history60_subset`

| Horizon | Top1 | Top3 | Top5 | Candidate Union Cover | Region ECE | Region NLL | Region Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10m | 0.0000 | 0.0184 | 0.0464 | 0.2792 | 0.9595 | 19.1417 | 1.9210 |
| 30m | 0.0000 | 0.0176 | 0.0352 | 0.1112 | 0.9374 | 21.7821 | 1.8812 |
| 60m | 0.0000 | 0.0131 | 0.0631 | 0.2930 | 0.7441 | 14.6522 | 1.5742 |
| 120m | 0.0000 | 0.0173 | 0.0242 | 0.1952 | 0.8650 | 21.3835 | 1.7555 |

### GRU Horizon Decoder Control

Run:

- `adrr_horizon_decoder_gru_history60_subset`

| Horizon | Top1 | Top3 | Top5 | Candidate Union Cover | Region ECE | Region NLL | Region Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10m | 0.0000 | 0.0280 | 0.0502 | 0.2850 | 0.9725 | 19.7754 | 1.9453 |
| 30m | 0.0000 | 0.0220 | 0.0297 | 0.1167 | 0.9517 | 21.6405 | 1.9073 |
| 60m | 0.0000 | 0.0197 | 0.0329 | 0.2378 | 0.8274 | 16.0004 | 1.6944 |
| 120m | 0.0000 | 0.0276 | 0.0276 | 0.1606 | 0.8823 | 22.1466 | 1.7829 |

### Transformer Horizon Decoder

Run:

- `adrr_transformer_horizon_decoder_subset`

| Horizon | Top1 | Top3 | Top5 | Candidate Union Cover | Region ECE | Region NLL | Region Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10m | 0.0000 | 0.0193 | 0.0415 | 0.2763 | 0.9674 | 19.5851 | 1.9357 |
| 30m | 0.0000 | 0.0198 | 0.0286 | 0.1068 | 0.9413 | 21.6196 | 1.8886 |
| 60m | 0.0000 | 0.0342 | 0.0683 | 0.2405 | 0.7943 | 15.5879 | 1.6442 |
| 120m | 0.0000 | 0.0155 | 0.0276 | 0.1744 | 0.8634 | 21.2631 | 1.7510 |

### Delta vs Fair History-60 Baseline

| Horizon | Top3 Delta | Top5 Delta | Candidate Union Delta | Region NLL Delta | Region Brier Delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| 10m | +0.0010 | -0.0048 | -0.0029 | +0.4434 | +0.0147 |
| 30m | +0.0022 | -0.0066 | -0.0044 | -0.1625 | +0.0074 |
| 60m | +0.0210 | +0.0053 | -0.0526 | +0.9357 | +0.0700 |
| 120m | -0.0017 | +0.0035 | -0.0207 | -0.1204 | -0.0045 |

### Delta vs GRU Horizon Decoder Control

| Horizon | Top3 Delta | Top5 Delta | Candidate Union Delta | Region NLL Delta | Region Brier Delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| 10m | -0.0087 | -0.0087 | -0.0087 | -0.1902 | -0.0097 |
| 30m | -0.0022 | -0.0011 | -0.0099 | -0.0210 | -0.0187 |
| 60m | +0.0145 | +0.0355 | +0.0026 | -0.4125 | -0.0502 |
| 120m | -0.0121 | +0.0000 | +0.0138 | -0.8835 | -0.0319 |

### Context vs Round-2 History-30 GRU Horizon Decoder

Run:

- `adrr_horizon_decoder_gru_subset`

This comparison is not apples-to-apples, but it is useful context:

- history-30 GRU horizon decoder was materially better than all history-60 runs on the current subset
- this suggests the longer window reduced effective sample support and made optimization harder under the current training recipe

## Inference Benchmark

Benchmark setup:

- dataset: history-60 subset test split, `1035` samples
- device: `cuda`
- measurement scope: model forward + hierarchical fine reconstruction + reachability filter
- method: one warmup pass, then mean of `3` timed passes

| Model | Mean Time (s) | Std (s) | ms / sample | samples / sec |
| --- | ---: | ---: | ---: | ---: |
| baseline60 | 0.2389 | 0.0075 | 0.2308 | 4332.48 |
| horizon_decoder60 | 0.2386 | 0.0006 | 0.2305 | 4337.66 |
| transformer60 | 0.2519 | 0.0008 | 0.2434 | 4108.52 |

Interpretation:

- the Transformer encoder adds only a small inference cost over the GRU variants
- runtime is not the main blocker here; optimization and sample efficiency are

## Result Summary

Round 3 does **not** produce a dominant new best model, but it does produce a clearer research signal.

The strongest findings are:

1. moving from history-30 to history-60 made the subset task materially harder for all tested models
2. within the history-60 setting, the Transformer encoder is generally better than the GRU horizon decoder on probabilistic quality
3. the Transformer gives the clearest ranking gain at `60m`
4. the Transformer improves `120m` likelihood and calibration relative to the GRU horizon decoder, even when top-k gains are limited

## Interpretation

What this round tells us:

1. a stronger sequence encoder helps, but not enough by itself
2. simply increasing history length can hurt when the training recipe and data volume are not adjusted with it
3. the current subset is now exposing a need for either:
   - multiscale temporal modeling
   - better optimization for longer windows
4. the encoder swap is still worth keeping as a foundation because it improves the more paper-relevant medium/long horizon behavior against the GRU horizon decoder control

## Problems Observed

1. all history-60 models hit their best validation loss at epoch `1`
2. status prediction remains degenerate with `status_accuracy = 1.0`
3. candidate union coverage still drops when model complexity increases
4. the current training recipe appears tuned for history-30, not history-60

## Next Step Recommendation

The best next move is **not** a heavy H3 graph model yet.

Recommended round 4:

1. keep `transformer_horizon_decoder` as the main branch
2. add multiscale temporal context:
   - short recent window for local maneuver cues
   - longer coarse window for strategic route intent
3. retune optimization for longer context:
   - lower learning rate
   - more patience
   - possibly weaker dropout

Only after that should we decide whether H3 graph refinement is still needed.
