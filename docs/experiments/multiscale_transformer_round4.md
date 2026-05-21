# Experiment: Multiscale Transformer H3 Round 4

## Goal

Build the next research step on top of round 3:

- keep the Transformer horizon-query decoder branch
- add an explicit second temporal scale
- test whether multiscale context can recover some of the performance lost when moving from `history=30` to `history=60`

## External References Consulted

This round was informed by targeted external references rather than only internal iteration.

Primary references consulted:

- Pyraformer (paper + local code)
  - paper: `https://openreview.net/forum?id=0EXmFzUn5I`
  - local repo snapshot: `.research_refs/Pyraformer`
- HiVT
  - paper PDF: `https://openaccess.thecvf.com/content/CVPR2022/papers/Zhou_HiVT_Hierarchical_Vector_Transformer_for_Multi-Agent_Motion_Prediction_CVPR_2022_paper.pdf`
- Crossformer
  - official repo: `https://github.com/thinklab-sjtu/crossformer`
- Wayformer
  - paper: `https://arxiv.org/abs/2207.05844`

What was borrowed conceptually:

1. Pyraformer:
   multiresolution temporal representation is useful for long-range dependency capture
2. HiVT / Wayformer:
   hierarchical attention and scale-specific representations are effective in trajectory-style prediction
3. Crossformer:
   long-window sequence models benefit from changing the representation granularity rather than only deepening one flat encoder

What was **not** copied:

- no direct Pyraformer sparse attention graph
- no direct Wayformer scene-token stack
- no direct HiVT multi-agent interaction graph

Instead, this round keeps the current project’s H3 hierarchical output stack and adds the lightest multiscale temporal encoder that fits the codebase cleanly.

## Configs

Smoke:

- `configs/region_forecast_adrr_multiscale_transformer_smoke.yaml`

Subset:

- `configs/region_forecast_adrr_multiscale_transformer_subset.yaml`

Reference runs:

- fair history-60 baseline:
  - `configs/region_forecast_adrr_history60_baseline_subset.yaml`
- single-scale Transformer:
  - `configs/region_forecast_adrr_transformer_horizon_subset.yaml`

## Model Change

New model type:

- `multiscale_transformer_h3`

Core changes:

1. fine-scale Transformer encoder over the full history sequence
2. stride-based temporal pooling from the fine sequence into a coarse sequence
3. a shallow coarse-scale Transformer encoder
4. horizon queries attend to both:
   - fine sequence
   - coarse sequence
5. decoder features fuse:
   - shared base summary
   - fine attended representation
   - coarse attended representation
   - horizon embedding

Current configuration:

- `history_minutes: 60`
- `multiscale_stride: 2`
- `multiscale_num_layers: 1`

## Files Changed

- `regional_forecast/config.py`
- `regional_forecast/models/forecast_model.py`
- `regional_forecast/models/__init__.py`
- `regional_forecast/pipeline/train.py`
- `regional_forecast/pipeline/evaluate.py`
- `regional_forecast/pipeline/predict.py`
- `tests/test_attention_hierarchical_model.py`
- `configs/region_forecast_adrr_multiscale_transformer_smoke.yaml`
- `configs/region_forecast_adrr_multiscale_transformer_subset.yaml`

## Verification

```bash
OMP_NUM_THREADS=1 python -m unittest tests.test_attention_hierarchical_model tests.test_evaluate_sparse_probs -v
python -m py_compile regional_forecast/config.py regional_forecast/models/forecast_model.py regional_forecast/models/__init__.py regional_forecast/pipeline/train.py regional_forecast/pipeline/evaluate.py regional_forecast/pipeline/predict.py
```

Both passed.

## Smoke Validation

### Commands

```bash
OMP_NUM_THREADS=1 python build_dataset.py --config configs/region_forecast_adrr_multiscale_transformer_smoke.yaml --workers 8
OMP_NUM_THREADS=1 python training.py --config configs/region_forecast_adrr_multiscale_transformer_smoke.yaml
OMP_NUM_THREADS=1 python evaluation.py --config configs/region_forecast_adrr_multiscale_transformer_smoke.yaml
```

### Runtime

- build: about `5.1s`
- train: about `4.7s`
- eval: about `2.6s`

Smoke train summary:

- epoch 1 val loss: `1.8334`
- epoch 2 val loss: `1.9354`

This was slightly cleaner than the single-scale Transformer smoke run and justified moving to the subset experiment.

## Subset Experiment

### Commands

```bash
OMP_NUM_THREADS=1 python build_dataset.py --config configs/region_forecast_adrr_multiscale_transformer_subset.yaml --workers 8
OMP_NUM_THREADS=1 python training.py --config configs/region_forecast_adrr_multiscale_transformer_subset.yaml
OMP_NUM_THREADS=1 python evaluation.py --config configs/region_forecast_adrr_multiscale_transformer_subset.yaml
```

### Runtime

- build: about `6.6s`
- train: about `10.9s`
- eval: about `3.4s`

Training summary:

- best val loss: `2.4486`
- best epoch: `1`
- completed epochs: `4`

## Metrics

### Fair History-60 Baseline

Run:

- `adrr_hierarchical_gru_history60_subset`

| Horizon | Top3 | Top5 | Candidate Union Cover | Region ECE | Region NLL | Region Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10m | 0.0184 | 0.0464 | 0.2792 | 0.9595 | 19.1417 | 1.9210 |
| 30m | 0.0176 | 0.0352 | 0.1112 | 0.9374 | 21.7821 | 1.8812 |
| 60m | 0.0131 | 0.0631 | 0.2930 | 0.7441 | 14.6522 | 1.5742 |
| 120m | 0.0173 | 0.0242 | 0.1952 | 0.8650 | 21.3835 | 1.7555 |

### Single-Scale Transformer

Run:

- `adrr_transformer_horizon_decoder_subset`

| Horizon | Top3 | Top5 | Candidate Union Cover | Region ECE | Region NLL | Region Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10m | 0.0193 | 0.0415 | 0.2763 | 0.9674 | 19.5851 | 1.9357 |
| 30m | 0.0198 | 0.0286 | 0.1068 | 0.9413 | 21.6196 | 1.8886 |
| 60m | 0.0342 | 0.0683 | 0.2405 | 0.7943 | 15.5879 | 1.6442 |
| 120m | 0.0155 | 0.0276 | 0.1744 | 0.8634 | 21.2631 | 1.7510 |

### Multiscale Transformer

Run:

- `adrr_multiscale_transformer_h3_subset`

| Horizon | Top3 | Top5 | Candidate Union Cover | Region ECE | Region NLL | Region Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10m | 0.0232 | 0.0502 | 0.3063 | 0.9592 | 19.4738 | 1.9201 |
| 30m | 0.0198 | 0.0297 | 0.1035 | 0.9258 | 21.2589 | 1.8613 |
| 60m | 0.0368 | 0.0788 | 0.2405 | 0.7386 | 15.4574 | 1.5665 |
| 120m | 0.0363 | 0.0466 | 0.1623 | 0.8315 | 22.2338 | 1.7000 |

### Delta vs Fair History-60 Baseline

| Horizon | Top3 Delta | Top5 Delta | Candidate Union Delta | Region NLL Delta | Region Brier Delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| 10m | +0.0048 | +0.0039 | +0.0271 | +0.3321 | -0.0009 |
| 30m | +0.0022 | -0.0055 | -0.0077 | -0.5231 | -0.0199 |
| 60m | +0.0237 | +0.0158 | -0.0526 | +0.8052 | -0.0076 |
| 120m | +0.0190 | +0.0225 | -0.0328 | +0.8502 | -0.0555 |

### Delta vs Single-Scale Transformer

| Horizon | Top3 Delta | Top5 Delta | Candidate Union Delta | Region NLL Delta | Region Brier Delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| 10m | +0.0039 | +0.0087 | +0.0300 | -0.1113 | -0.0156 |
| 30m | +0.0000 | +0.0011 | -0.0033 | -0.3607 | -0.0273 |
| 60m | +0.0026 | +0.0105 | +0.0000 | -0.1305 | -0.0777 |
| 120m | +0.0207 | +0.0190 | -0.0121 | +0.9706 | -0.0510 |

## Inference Benchmark

Benchmark setup:

- dataset: history-60 subset test split, `1035` samples
- device: `cuda`
- method: one warmup pass, then mean of `3` timed passes

| Model | Mean Time (s) | Std (s) | ms / sample | samples / sec |
| --- | ---: | ---: | ---: | ---: |
| baseline60 | 0.2318 | 0.0002 | 0.2240 | 4464.14 |
| transformer60 | 0.2476 | 0.0052 | 0.2392 | 4180.48 |
| multiscale60 | 0.2638 | 0.0008 | 0.2549 | 3923.44 |

Interpretation:

- multiscale modeling adds a modest inference cost over the single-scale Transformer
- the added cost is still small relative to the total pipeline

## Result Summary

Round 4 is the first new branch that looks genuinely promising rather than merely interesting.

Most important findings:

1. the multiscale model is better than the single-scale Transformer on:
   - `10m top3/top5`
   - `30m NLL/Brier`
   - `60m top3/top5`, `ECE`, and `Brier`
   - `120m top3/top5`, `ECE`, and `Brier`
2. it materially improves `120m top3/top5` over both history-60 baseline and single-scale Transformer
3. it still shows a calibration/likelihood tradeoff at `120m`:
   - `top-k` and `Brier` improve
   - `NLL` worsens relative to the single-scale Transformer

## Interpretation

This round supports the core hypothesis from the literature refresh:

- longer-horizon sequence modeling benefits from explicit multiscale temporal structure

More specifically:

1. the single-scale Transformer was not enough
2. the extra coarse temporal path helps recover useful medium- and long-range cues
3. a simple stride-pooled coarse branch is already enough to move the curve in the right direction

## Problems Observed

1. best validation loss is still at epoch `1`
2. candidate union coverage remains weak at the longest horizons
3. `120m NLL` worsened even though `120m top-k` improved
4. status prediction remains degenerate

## Recommended Next Step

Keep `multiscale_transformer_h3` as the current lead research branch.

Recommended round 5:

1. retune optimization for history-60 multiscale training:
   - lower learning rate
   - more patience
   - milder dropout
2. add a consistency regularizer across adjacent horizons
3. only after that decide whether H3 graph refinement is still necessary
