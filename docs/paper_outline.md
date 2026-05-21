# Paper Outline Draft

## Working Title

Global Multi-Horizon Regional Aircraft Trajectory Forecasting with Hierarchical H3 Prediction and Multiscale Horizon-Query Transformers

## Core Claim

Large-scale open aviation trajectories can be forecast as calibrated multi-horizon regional probability distributions by combining:

1. hierarchical H3 coarse-to-fine region prediction
2. per-horizon query decoding
3. explicit multiscale temporal Transformer encoding

The final full-data model is:

- `global_multiscale_transformer_h3_tuned`

H3 refinement is not the default final method. It is an optional recall-oriented postprocessor and should be reported as a trade-off ablation.

## Contributions

1. A global multi-horizon H3 regional aircraft trajectory forecasting formulation on month-scale ADRR data.
2. A hierarchical coarse-to-fine probabilistic H3 decoder for scalable global regional prediction.
3. A multiscale horizon-query Transformer that improves top-k accuracy, likelihood, Brier score, and calibration over a hierarchical GRU baseline.
4. A full-scale comparison against a classical Markov regional transition baseline, a hierarchical GRU baseline, and a Transformer structural ablation.
5. A candidate-region evaluation protocol that jointly reports recall, candidate cell count, candidate area, and calibration.
6. A full-scale ablation showing that H3 neighbor refinement improves candidate recall but trades off calibration.

## 1. Introduction

- motivation for global open aviation trajectory forecasting
- difficulty of long-horizon, probabilistic, region-level prediction
- gap between open large-scale aviation data and scalable multi-horizon forecasting methods

## 2. Related Work

- aircraft trajectory prediction
- temporal attention and multi-horizon forecasting
- graph / spatial attention
- hierarchical spatial discretization and H3-style region forecasting

## 3. Problem Formulation

- input history trajectory
- target future H3 region distributions at multiple horizons
- coarse-to-fine hierarchy
- probabilistic and calibration-aware evaluation

## 4. Baseline Method

- H3 regionization
- GRU encoding
- coarse-to-fine hierarchical decoding
- reachability filtering

## 5. Proposed Method

- hierarchical H3 coarse-to-fine prediction
- multiscale Transformer history encoding
- per-horizon query attention over encoder states
- horizon-aware decoder features
- reachability filtering as a plausibility safeguard
- optional H3 neighbor refinement as an ablation, not the main method

## 6. Experimental Setup

- ADRR month-scale dataset
- subset baseline for fast ablations
- full-scale reference baseline
- classical Markov regional transition baseline
- metrics: top-k, candidate coverage, NLL, Brier, ECE, runtime

## 7. Results

### Main Full-Scale Result

Macro mean over 10/30/60/120 minute horizons:

| Run | Top1 | Top3 | Top5 | NLL | Brier | ECE | Union Cover | Union Cells | Union Area km2 | Cover at Budget |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full Markov baseline | 0.3184 | 0.6166 | 0.7587 | 2.2788 | 0.8111 | 0.0323 | 0.8749 | 9.9542 | 545174 | 0.9699 |
| full GRU baseline | 0.6648 | 0.9076 | 0.9631 | 1.0214 | 0.4578 | 0.0180 | 0.8931 | 2.4189 | 128965 | 0.9980 |
| transformer horizon ablation | 0.6744 | 0.9129 | 0.9656 | 0.9946 | 0.4447 | 0.0123 | 0.8920 | 2.2721 | 122553 | 0.9981 |
| full final multiscale | 0.6827 | 0.9174 | 0.9685 | 0.9641 | 0.4338 | 0.0093 | 0.8922 | 2.1829 | 118810 | 0.9983 |

Interpretation:

- the Markov baseline gives a classical non-neural reference and is far below all learned models
- the final model improves top-k, NLL, Brier, and ECE on every horizon
- the final model produces smaller candidate sets and smaller candidate area
- probability-mass candidate coverage is essentially tied because the final model is sharper

### Horizon-Wise Main Result

Delta: final multiscale minus full GRU baseline.

| Horizon | dTop1 | dTop3 | dTop5 | dNLL | dBrier | dECE | dUnion Cover | dUnion Cells | dUnion Area km2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10m | +0.0199 | +0.0059 | +0.0031 | -0.0516 | -0.0281 | -0.0085 | -0.0004 | -0.1864 | -2316 |
| 30m | +0.0214 | +0.0129 | +0.0076 | -0.0759 | -0.0303 | -0.0210 | -0.0052 | -0.3683 | -4567 |
| 60m | +0.0115 | +0.0054 | +0.0018 | -0.0340 | -0.0158 | -0.0016 | +0.0023 | -0.0908 | -7705 |
| 120m | +0.0190 | +0.0153 | +0.0092 | -0.0674 | -0.0220 | -0.0040 | -0.0005 | -0.2988 | -26031 |

### Training Summary

| Run | Best Epoch | Completed Epochs | Best Val Loss |
| --- | ---: | ---: | ---: |
| final multiscale | 182 | 183 | 0.3055 |
| transformer horizon ablation | 182 | 182 | 0.3177 |

## 8. Ablation Study

### Multiscale Branch

Macro mean:

| Run | Top1 | Top3 | Top5 | NLL | Brier | ECE | Union Cover | Union Cells | Union Area km2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| transformer horizon ablation | 0.6744 | 0.9129 | 0.9656 | 0.9946 | 0.4447 | 0.0123 | 0.8920 | 2.2721 | 122553 |
| multiscale final | 0.6827 | 0.9174 | 0.9685 | 0.9641 | 0.4338 | 0.0093 | 0.8922 | 2.1829 | 118810 |

Conclusion:

- the multiscale branch improves validation loss and all key macro probabilistic metrics
- this supports multiscale temporal encoding as a main contribution

### H3 Neighbor Refinement

Macro mean:

| Run | Top5 | NLL | Brier | ECE | Union Cover | Union Cells | Union Area km2 | Cover at Budget |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| final | 0.9685 | 0.9641 | 0.4338 | 0.0093 | 0.8922 | 2.1829 | 118810 | 0.9983 |
| refine a005 | 0.9677 | 0.9593 | 0.4344 | 0.0310 | 0.9052 | 2.3315 | 127062 | 0.9987 |
| refine a010 | 0.9673 | 0.9796 | 0.4372 | 0.0585 | 0.9192 | 2.5168 | 137220 | 0.9987 |
| refine a020 | 0.9662 | 1.0340 | 0.4493 | 0.1135 | 0.9496 | 3.0980 | 167889 | 0.9987 |

Conclusion:

- H3 refinement improves probability-mass candidate coverage
- it worsens calibration substantially on full data
- it should be framed as a recall-vs-calibration trade-off

### Reachability Filtering

Conclusion:

- reachability on/off has only a tiny aggregate effect in the full experiment
- it should be described as a practical physical-plausibility guard

## 9. Discussion

- why long horizons remain difficult
- effect of regional discretization
- candidate coverage must be interpreted together with candidate area and calibration
- H3 refinement exposes a recall-vs-calibration trade-off
- status prediction is saturated under the current label distribution

## 10. Conclusion

- full-scale experiments support the proposed multiscale horizon-query H3 forecaster
- learned hierarchical forecasting is decisively stronger than the Markov regional transition baseline
- final model improves accuracy, likelihood, Brier score, and calibration over the hierarchical GRU baseline
- multiscale temporal encoding is validated by a full structural ablation
- H3 refinement is useful as an optional candidate recall mechanism but not the default calibrated predictor
