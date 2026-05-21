# Experiment: Attention Hierarchical GRU Round 1

## Goal

Evaluate the first research modification over the subset baseline:

- temporal attention pooling over GRU encoder states
- learned horizon embeddings for horizon-aware decoding

This experiment preserves:

- the same ADRR subset pipeline
- the same coarse-to-fine hierarchical prediction logic
- the same evaluation stack

## Configs

Smoke:

- `configs/region_forecast_adrr_attention_smoke.yaml`

Subset:

- `configs/region_forecast_adrr_attention_subset.yaml`

Baseline reference:

- `configs/region_forecast_adrr_subset_baseline.yaml`

## Model Change

New model type:

- `attention_hierarchical_gru`

Key changes:

1. expose full GRU encoder sequence
2. apply learned temporal attention pooling instead of using only the final hidden state
3. add learned horizon embeddings
4. build per-horizon decoder features before coarse/status/child predictions

Files changed:

- `regional_forecast/config.py`
- `regional_forecast/models/forecast_model.py`
- `regional_forecast/models/__init__.py`
- `regional_forecast/pipeline/train.py`
- `regional_forecast/pipeline/evaluate.py`
- `regional_forecast/pipeline/predict.py`

Additional regression tests:

- `tests/test_attention_hierarchical_model.py`
- `tests/test_evaluate_sparse_probs.py`

## Smoke Validation

### Commands

```bash
python build_dataset.py --config configs/region_forecast_adrr_attention_smoke.yaml --workers 8
python training.py --config configs/region_forecast_adrr_attention_smoke.yaml
python evaluation.py --config configs/region_forecast_adrr_attention_smoke.yaml
```

### Acceleration Settings

- build workers: `8`
- training batch size: `256`
- device: `cuda`
- amp: `true`
- dataloader workers: `8`
- pin memory: `true`
- persistent workers: `true`

### Notes

Two implementation issues were found and fixed during smoke:

1. hierarchical evaluation unknown coarse lookup overflow
2. temporal attention mask fill overflow under AMP/fp16

Both now have regression coverage.

## Subset Experiment

### Commands

```bash
python build_dataset.py --config configs/region_forecast_adrr_attention_subset.yaml --workers 8
python training.py --config configs/region_forecast_adrr_attention_subset.yaml
python evaluation.py --config configs/region_forecast_adrr_attention_subset.yaml
```

### Acceleration Settings

- build workers: `8`
- training batch size: `512`
- device: `cuda`
- amp: `true`
- dataloader workers: `8`
- pin memory: `true`
- persistent workers: `true`

### Runtime

Observed subset training:

- early stopped at epoch 4
- wall time: about `10s`

Observed subset evaluation:

- wall time: about `4s`

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

### Attention Round 1

Run:

- `adrr_attention_hierarchical_gru_subset`

| Horizon | Top1 | Top3 | Top5 | Candidate Union Cover | Region ECE | Region NLL | Region Brier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10m | 0.0000 | 0.0269 | 0.0531 | 0.3344 | 0.9496 | 17.7149 | 1.9026 |
| 30m | 0.0000 | 0.0172 | 0.0298 | 0.1111 | 0.9008 | 20.2471 | 1.8192 |
| 60m | 0.0000 | 0.0485 | 0.0727 | 0.2632 | 0.6841 | 13.2177 | 1.5017 |
| 120m | 0.0000 | 0.0317 | 0.0709 | 0.2323 | 0.7921 | 20.1004 | 1.6417 |

### Delta vs Baseline

| Horizon | Top3 Delta | Top5 Delta | Candidate Union Delta | Region NLL Delta | Region Brier Delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| 10m | -0.0325 | -0.0666 | -0.0658 | +2.4399 | +0.2769 |
| 30m | -0.0117 | -0.0298 | -0.0515 | +1.9422 | +0.2441 |
| 60m | -0.0176 | -0.0429 | -0.0738 | +0.6323 | +0.1884 |
| 120m | +0.0045 | +0.0241 | -0.0302 | +0.3887 | +0.2274 |

## Result Summary

This first attention version is **not better than the subset baseline overall**.

Observed pattern:

- `10m`, `30m`, `60m` all regress on top-k and candidate coverage
- `120m` gets a small gain on `top3/top5`, but calibration and likelihood metrics worsen
- `region_nll`, `region_brier`, and `region_ece` worsen across all horizons

## Interpretation

Likely reasons:

1. the shared attention pooling is too unconstrained on the small subset
2. a single pooled attention summary may blur short-horizon and long-horizon needs
3. horizon embeddings alone are probably too weak to produce useful horizon-specific querying
4. the model may need per-horizon temporal attention rather than shared pooling

## Recommendation

Do not keep this version as the preferred model.

Instead, move to a second-round refinement:

1. replace shared temporal attention pooling with per-horizon query attention over encoder states
2. keep the rest of the hierarchical coarse-to-fine pipeline unchanged
3. compare:
   - baseline
   - shared attention pooling
   - horizon-query attention

## Smoke / Verification Commands Run

```bash
python -m unittest tests.test_attention_hierarchical_model tests.test_evaluate_sparse_probs -v
python -m py_compile regional_forecast/models/forecast_model.py regional_forecast/pipeline/train.py regional_forecast/pipeline/evaluate.py regional_forecast/pipeline/predict.py tests/test_attention_hierarchical_model.py tests/test_evaluate_sparse_probs.py
```

Both passed before the experiment runs.
