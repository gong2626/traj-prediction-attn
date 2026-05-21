# Experiment: Full-Scale Results Round 6

## Goal

Validate whether the devsplit research trend holds on the full ADRR-derived dataset and determine whether the current method stack is strong enough to support paper writing.

Full runs included:

- full Markov classical baseline
- existing full GRU baseline
- full tuned multiscale final model
- full structural ablation: transformer horizon decoder without multiscale branch
- full H3 refinement sensitivity
- full reachability ablation

## Full Training Summary

### Final Model

- run: `global_multiscale_transformer_h3_tuned`
- best epoch: `182`
- completed epochs: `183`
- best val loss: `0.3055236`

### Structural Ablation

- run: `global_transformer_horizon_decoder_ablation`
- best epoch: `182`
- completed epochs: `182`
- best val loss: `0.3176994`

The multiscale final model converged to a clearly lower validation loss than the non-multiscale Transformer horizon decoder ablation.

## Main Result: Final Model vs Full Baselines

### Macro Mean Over 10/30/60/120 Minutes

| Run | Top1 | Top3 | Top5 | NLL | Brier | ECE | Union Cover | Union Cells | Union Area km2 | Cover at Budget |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full Markov baseline | 0.3184 | 0.6166 | 0.7587 | 2.2788 | 0.8111 | 0.0323 | 0.8749 | 9.9542 | 545174 | 0.9699 |
| full GRU baseline | 0.6648 | 0.9076 | 0.9631 | 1.0214 | 0.4578 | 0.0180 | 0.8931 | 2.4189 | 128965 | 0.9980 |
| transformer horizon ablation | 0.6744 | 0.9129 | 0.9656 | 0.9946 | 0.4447 | 0.0123 | 0.8920 | 2.2721 | 122553 | 0.9981 |
| full final multiscale | 0.6827 | 0.9174 | 0.9685 | 0.9641 | 0.4338 | 0.0093 | 0.8922 | 2.1829 | 118810 | 0.9983 |

### Delta: Final - Markov Baseline

Macro mean gains are large:

- Top1: `+0.3643`
- Top3: `+0.3009`
- Top5: `+0.2097`
- NLL: `-1.3146`
- Brier: `-0.3773`
- ECE: `-0.0230`
- union candidate cells: `-7.7713`
- union candidate area: `-426364 km2`
- cover at budget: `+0.0283`

The Markov baseline is therefore useful as a classical lower-bound reference, but it is not competitive with the learned hierarchical models on the full global task.

### Delta: Final - GRU Baseline

| Horizon | dTop1 | dTop3 | dTop5 | dNLL | dBrier | dECE | dUnion Cover | dUnion Cells | dUnion Area km2 | dCover at Budget |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10m | +0.0199 | +0.0059 | +0.0031 | -0.0516 | -0.0281 | -0.0085 | -0.0004 | -0.1864 | -2316 | +0.0001 |
| 30m | +0.0214 | +0.0129 | +0.0076 | -0.0759 | -0.0303 | -0.0210 | -0.0052 | -0.3683 | -4567 | +0.0006 |
| 60m | +0.0115 | +0.0054 | +0.0018 | -0.0340 | -0.0158 | -0.0016 | +0.0023 | -0.0908 | -7705 | +0.0001 |
| 120m | +0.0190 | +0.0153 | +0.0092 | -0.0674 | -0.0220 | -0.0040 | -0.0005 | -0.2988 | -26031 | +0.0004 |

## Interpretation

The full-data result is strong and publication-usable:

- the learned models decisively outperform the classical Markov baseline
- top-k improves on every horizon
- NLL improves on every horizon
- Brier improves on every horizon
- ECE improves on every horizon
- candidate area shrinks on every horizon
- fixed-budget candidate coverage is at least preserved and slightly improved

The only metric that does not strictly improve is probability-mass `candidate_union_cover`, which stays essentially tied overall and is slightly lower on 10m/30m/120m. This matches the earlier devsplit interpretation: the final model is sharper and uses fewer H3 cells to represent its candidate mass.

## Structural Ablation: Final vs Transformer Horizon Decoder

### Macro Mean

| Run | Top1 | Top3 | Top5 | NLL | Brier | ECE | Union Cover | Union Cells | Union Area km2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| transformer horizon ablation | 0.6744 | 0.9129 | 0.9656 | 0.9946 | 0.4447 | 0.0123 | 0.8920 | 2.2721 | 122553 |
| multiscale final | 0.6827 | 0.9174 | 0.9685 | 0.9641 | 0.4338 | 0.0093 | 0.8922 | 2.1829 | 118810 |

## Interpretation

This confirms the multiscale branch is not cosmetic:

- full final validation loss is better: `0.3055` vs `0.3177`
- macro top-k is better
- macro NLL, Brier, and ECE are better
- candidate sets are smaller

So the paper can legitimately claim that explicit multiscale temporal modeling adds value beyond a single-scale Transformer horizon decoder.

## H3 Refinement on Full Data

### Macro Mean

| Run | Top5 | NLL | Brier | ECE | Union Cover | Union Cells | Union Area km2 | Cover at Budget |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| final | 0.9685 | 0.9641 | 0.4338 | 0.0093 | 0.8922 | 2.1829 | 118810 | 0.9983 |
| refine a005 | 0.9677 | 0.9593 | 0.4344 | 0.0310 | 0.9052 | 2.3315 | 127062 | 0.9987 |
| refine a010 | 0.9673 | 0.9796 | 0.4372 | 0.0585 | 0.9192 | 2.5168 | 137220 | 0.9987 |
| refine a020 | 0.9662 | 1.0340 | 0.4493 | 0.1135 | 0.9496 | 3.0980 | 167889 | 0.9987 |

## Interpretation

The full-data trend is different from the devsplit optimism:

- H3 refinement still increases `candidate_union_cover`
- but on full data it harms calibration strongly
- and at moderate-to-large alpha it also harms top-5 and probabilistic accuracy

So on the full dataset:

- `candidate_union_cover` can be improved with H3 refinement
- but the best overall paper model should remain the **unrefined final model**
- H3 refinement should be reported as an ablation/trade-off, not as the default final inference path

This is still a useful paper result because it shows:

1. the base model is already strong
2. spatial postprocessing can trade calibration for candidate recall
3. candidate metrics must be reported jointly with probabilistic calibration

## Reachability Ablation

### Final vs No Reachability

Macro differences are tiny:

- dTop5: `-0.00007`
- dNLL: `-0.00042`
- dECE: `-0.00013`
- dUnion Cover: `+0.00010`

## Interpretation

Under the current full setup, reachability filtering has only a very small aggregate effect on the reported metrics. That means:

- it is not driving the main performance gains
- it can be framed as a practical plausibility safeguard rather than the main methodological contribution

## Paper-Ready Takeaway

The current paper-worthy method should be framed as:

1. hierarchical H3 regional forecasting
2. horizon-query Transformer decoder
3. explicit multiscale temporal encoding
4. calibrated probabilistic regional prediction at global scale

Recommended final method for the paper main table:

- `global_multiscale_transformer_h3_tuned`

Recommended ablations to report:

- full Markov baseline
- full GRU baseline
- full transformer horizon decoder ablation
- H3 refinement `alpha = 0.05 / 0.10 / 0.20`
- reachability on/off

## Next Step

The experiment side is now strong enough to begin drafting the paper seriously.

The immediate next work should be:

1. write the main results table into `docs/paper_outline.md`
2. update `docs/method.md` to mark unrefined multiscale as the final method
3. write `docs/ablation.md` with the full-data conclusions
4. optionally add one calibration figure and one candidate-area-vs-coverage figure
