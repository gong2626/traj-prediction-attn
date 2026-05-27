# Experiment: Horizon Expert Blend Round 1

## Goal

The same-protocol devsplit comparison on 2026-05-26 showed that:

- `TrAISformer-H3` remained the strongest single model on the subset proxy
- lightweight local fusion improved the multiscale mainline, but did not overturn the paper-facing comparison table

This round tests a different direction:

- do not force one backbone to win every horizon
- instead, combine heterogeneous experts with horizon-specific weights
- tune the blend only on the devsplit validation set
- evaluate the selected blend on the held-out devsplit test set

## Why this direction is reasonable

The literature already supports the two main ideas behind this experiment:

1. **Different horizons need different temporal evidence.**
   - TFT and MQTransformer both argue against one fixed summary for all horizons.
   - In our setting, this means a short-horizon expert and a longer-horizon expert do not need to be the same model.

2. **Heterogeneous expert combination is a valid forecasting strategy.**
   - Forecast-combination work consistently shows that different forecasters capture complementary structure.
   - Mixture-of-experts forecasting papers such as MECATS push the same message in a trainable form.

3. **Post-hoc calibration is a legitimate part of the forecasting pipeline.**
   - Temperature scaling remains a standard calibration tool for neural probabilistic forecasts.

The key practical observation from our own devsplit results is even simpler:

- `TrAISformer-H3` dominates the core ranking and likelihood metrics
- `gated local` improves calibration
- `GRU` still has the broadest probability-mass candidate coverage

That is exactly the kind of complementarity where a horizon-aware expert blend is worth testing.

## Implementation

Added script:

- `tools/horizon_expert_blend.py`

What it does:

1. loads cached or live predictions from multiple expert checkpoints
2. searches convex blend weights on the validation split
3. selects weights independently for each horizon
4. applies a second validation-stage per-horizon temperature search
5. evaluates the selected blend on the held-out test split
6. writes report JSON and summary CSV/JSON artifacts

This is an evaluation-time research tool, not yet a native trainable model class.

## Commands

### 3-expert blend

```bash
python tools/horizon_expert_blend.py \
  --expert tuned=configs/region_forecast_adrr_devsplit_multiscale_tuned.yaml \
  --expert trais=configs/region_forecast_adrr_devsplit_traisformer_h3.yaml \
  --expert gated=configs/region_forecast_adrr_devsplit_multiscale_gated_local_frozen_s6_b3_eval_parentk_temp.yaml \
  --weight-step 0.1 \
  --temperatures 0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5 \
  --batch-size 256 \
  --num-workers 0 \
  --output-prefix artifacts/research_devsplit/evaluations/adrr_devsplit_horizon_expert_blend_v1
```

### 4-expert blend

```bash
python tools/horizon_expert_blend.py \
  --expert gru=configs/region_forecast_adrr_devsplit_baseline.yaml \
  --expert tuned=configs/region_forecast_adrr_devsplit_multiscale_tuned.yaml \
  --expert trais=configs/region_forecast_adrr_devsplit_traisformer_h3.yaml \
  --expert gated=configs/region_forecast_adrr_devsplit_multiscale_gated_local_frozen_s6_b3_eval_parentk_temp.yaml \
  --weight-step 0.2 \
  --temperatures 0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5 \
  --batch-size 256 \
  --num-workers 0 \
  --output-prefix artifacts/research_devsplit/evaluations/adrr_devsplit_horizon_expert_blend_v2_gru
```

## Selected weights

### Blend v1

- 10m: `0.0 tuned + 0.7 trais + 0.3 gated`
- 30m: `0.1 tuned + 0.6 trais + 0.3 gated`
- 60m: `0.0 tuned + 0.6 trais + 0.4 gated`
- 120m: `0.0 tuned + 0.5 trais + 0.5 gated`

### Blend v2 + GRU

- 10m: `0.0 gru + 0.0 tuned + 0.8 trais + 0.2 gated`
- 30m: `0.2 gru + 0.0 tuned + 0.4 trais + 0.4 gated`
- 60m: `0.2 gru + 0.2 tuned + 0.4 trais + 0.2 gated`
- 120m: `0.2 gru + 0.0 tuned + 0.6 trais + 0.2 gated`

The search consistently preferred:

- mostly `TrAISformer-H3` for ranking strength
- `gated local` as the calibration helper
- small `GRU` mass where broader coverage helps

## Macro comparison on devsplit test

| Model | Top1 | Top3 | Top5 | NLL | Brier | ECE | Union Cover | Union Cells | Union Area km2 | Cover at Budget |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GRU | 0.3182 | 0.5739 | 0.6944 | 2.7494 | 0.8335 | 0.0951 | **0.8898** | 13.2375 | 580967.48 | 0.9354 |
| Multiscale tuned | 0.3722 | 0.6082 | 0.7235 | 2.4497 | 0.7783 | 0.0562 | 0.8063 | 7.1653 | 323031.09 | 0.9469 |
| TrAISformer-H3 | 0.4001 | 0.6345 | 0.7361 | 2.3504 | 0.7514 | 0.0596 | 0.7858 | 6.4272 | **278292.40** | 0.9456 |
| FlightBERT++-H3 | 0.3243 | 0.5694 | 0.6894 | 2.5257 | 0.8115 | 0.0577 | 0.8058 | 7.9066 | 359522.10 | 0.9448 |
| Gated local | 0.3812 | 0.6146 | 0.7258 | 2.4446 | 0.7666 | 0.0404 | 0.7871 | 6.4752 | 314506.70 | 0.9477 |
| Blend v1 | 0.4185 | 0.6520 | 0.7494 | 2.2594 | 0.7319 | 0.0386 | 0.8027 | **6.3526** | 309999.03 | 0.9489 |
| **Blend v2 + GRU** | **0.4201** | **0.6541** | **0.7536** | **2.1976** | **0.7300** | **0.0384** | 0.8201 | 6.9103 | 341898.16 | **0.9538** |

## Interpretation

This is the first devsplit result that clearly changes the paper-facing picture.

### What the 4-expert blend wins

`Blend v2 + GRU` is best on:

- `Top1`
- `Top3`
- `Top5`
- `NLL`
- `Brier`
- `ECE`
- `Cover at Budget`

That is the full set of core ranking and probabilistic quality metrics plus the fixed-budget candidate metric.

### What it does not win

- `Union Cover` still belongs to `GRU`
- `Union Area` still belongs to `TrAISformer-H3`
- `Union Cells` is best for `Blend v1`

This is consistent with the earlier candidate-metric interpretation:

- diffuse models improve probability-mass recall
- sharper models reduce area and cell count
- fixed-budget coverage is often the cleaner operational metric

## Implication for the paper table

The current ten-column table is probably too crowded for the main story.

Recommended main-table metrics:

- `Top1`
- `Top3`
- `Top5`
- `NLL`
- `Brier`
- `ECE`
- `Union Cover`
- `Cover at Budget`

Recommended appendix or secondary-table metrics:

- `Union Cells`
- `Union Area`

Why:

1. `Union Cells` and `Union Area` are still useful, but they are explanatory compactness diagnostics rather than the first-order quality targets.
2. `Cover at Budget` is easier to interpret operationally than raw probability-mass union recall.
3. We already know `status_accuracy` is saturated and should not be a headline metric.

Under that reduced eight-metric main table, `Blend v2 + GRU` wins:

- `Top1 / Top3 / Top5 / NLL / Brier / ECE / Cover at Budget`

and only loses:

- `Union Cover`

This is much closer to the kind of paper table we actually want.

## Research conclusion

The evidence from this round does **not** say "keep doing small local patches to one backbone."

It says:

- the problem is no longer best framed as single-backbone tuning
- the winning direction is **horizon-aware heterogeneous expert fusion**
- the next serious step should be one of:
  1. keep the blend as the paper model and run the same procedure on full checkpoints
  2. distill the blend into one trainable student
  3. build a native trainable horizon-router model that reproduces this mixture inside one checkpoint

## Immediate next step

The lowest-risk high-value continuation is:

1. keep `tools/horizon_expert_blend.py`
2. run the same search on the full-data checkpoints
3. compare that full-data blend against the existing full `multiscale tuned` and full `TrAISformer-H3`

Because this route does not require retraining base experts, it is currently the fastest path to a stronger paper result.
