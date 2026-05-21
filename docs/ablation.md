# Ablation Notes

## Available Ablations

### A0. Baseline

- model type: `hierarchical_gru`
- run: `adrr_hierarchical_gru_subset_baseline`

### A1. Shared Temporal Attention + Horizon Embedding

- model type: `attention_hierarchical_gru`
- run: `adrr_attention_hierarchical_gru_subset`

### A2. Per-Horizon Query Decoder

- model type: `horizon_decoder_gru`
- run: `adrr_horizon_decoder_gru_subset`

### A2b. Per-Horizon Query Decoder with 60-Minute History

- model type: `horizon_decoder_gru`
- run: `adrr_horizon_decoder_gru_history60_subset`

### A3. Transformer Horizon Decoder with 60-Minute History

- model type: `transformer_horizon_decoder`
- run: `adrr_transformer_horizon_decoder_subset`

### A4. Multiscale Transformer H3 with 60-Minute History

- model type: `multiscale_transformer_h3`
- run: `adrr_multiscale_transformer_h3_subset`

### A5. Tuned Multiscale Transformer H3

- model type: `multiscale_transformer_h3`
- run: `adrr_multiscale_transformer_h3_tuned_subset`

### A6. Tuned Multiscale Transformer H3 + Horizon Consistency Regularizer

- model type: `multiscale_transformer_h3`
- run: `adrr_multiscale_transformer_h3_consistency_subset`

### A7. Tuned Multiscale Transformer H3 + H3 Candidate Refinement

- model type: `multiscale_transformer_h3`
- runs:
  - `adrr_devsplit_multiscale_transformer_h3_refine_a005`
  - `adrr_devsplit_multiscale_transformer_h3_refine_a010`
  - `adrr_devsplit_multiscale_transformer_h3_refine_a020`

## Current Observation

A1 does not outperform A0 overall on the 256-trajectory subset baseline.

Weak conclusion:

- simple shared attention pooling is not enough

Round 2 result:

- A2 is clearly better than A1
- A2 gives the best new-model result at `120m`
- A2 still does not consistently exceed A0 on the subset baseline

Round 3 result:

- moving from `30` to `60` minutes of history hurts all tested models on the current subset
- A3 is still better than A2b on most probabilistic metrics
- A3 gives the clearest gain at `60m`, but it does not dominate every horizon

Round 4 result:

- A4 is better than A3 on most top-k, ECE, and Brier metrics
- A4 is the current best-performing research branch overall
- A4 still has a tradeoff at `120m`, where `top-k` improves but `NLL` worsens

Round 5 result:

- A5 is the new strongest branch by a wide margin
- most of the gain came from optimization tuning rather than a new structural block
- A6 confirms that the current decoder-feature consistency penalty is not helping at weight `0.05`

Devsplit correction:

- earlier `max_trajectories` subset metrics are no longer treated as performance evidence
- the valid fast proxy is now the split-preserving subset under `artifacts/research_devsplit`
- on this healthier subset, `multiscale_transformer_h3_tuned` improves top-k, NLL, Brier, ECE, and unknown prediction rate versus the devsplit GRU baseline, but candidate coverage regresses

Current strongest hypothesis:

- per-horizon querying over encoder states is more appropriate than one shared temporal attention summary
- explicit multiscale temporal structure is more valuable right now than immediate H3 graph complexity
- light H3-aware refinement improves candidate coverage, but should be paired with calibration repair or confidence-aware gating

## Full-Scale Conclusion

Full-data experiments now provide the decisive ranking.

### Classical Baseline

- run: `global_markov_baseline`

Conclusion:

- the Markov regional transition baseline is much weaker than all learned models
- macro Top5 is `0.7587`, compared with `0.9631` for GRU and `0.9685` for the final multiscale model
- macro NLL is `2.2788`, compared with `1.0214` for GRU and `0.9641` for the final multiscale model
- the Markov baseline needs larger candidate sets and larger candidate area while still giving lower coverage-at-budget
- it is useful as a classical reference baseline, not as a competitive method for the final task

### Final Main Model

- run: `global_multiscale_transformer_h3_tuned`
- this is the final paper model

Why:

- better top-k than the full GRU baseline on every horizon
- better NLL, Brier, and ECE on every horizon
- smaller candidate area and smaller candidate cell count
- fixed-budget candidate coverage is preserved or slightly improved

### Structural Ablation

- run: `global_transformer_horizon_decoder_ablation`

Conclusion:

- the single-scale Transformer horizon decoder is already stronger than GRU
- but the multiscale branch gives a further full-data gain
- therefore the multiscale temporal branch is a real contribution, not training noise

### H3 Refinement Ablation

- runs:
  - `global_multiscale_transformer_h3_tuned_h3_refine_a005`
  - `global_multiscale_transformer_h3_tuned_h3_refine_a010`
  - `global_multiscale_transformer_h3_tuned_h3_refine_a020`

Conclusion:

- H3 refinement consistently increases probability-mass candidate coverage
- but on the full dataset it also increases ECE substantially
- larger alpha also degrades top-5 and probabilistic quality
- therefore H3 refinement should stay in the paper as a trade-off ablation, not as the default final model

### Reachability Ablation

- runs:
  - `global_multiscale_transformer_h3_tuned_no_reachability`
  - `global_multiscale_transformer_h3_tuned_h3_refine_a010_no_reachability`

Conclusion:

- under the current full setup, reachability filtering has only a small aggregate metric effect
- it should be described as a practical plausibility constraint rather than the primary source of gains

## Recommended Paper Ablations

The paper should report at least:

1. full Markov baseline
2. full GRU baseline
3. full Transformer horizon decoder ablation
4. full tuned multiscale final model
5. H3 refinement alpha sweep
6. reachability on/off

## Final Message

The full-scale evidence is now strong enough to stop method hunting for this round.

The model story is coherent:

- hierarchical H3 forecasting is the task setup
- Transformer horizon querying is better than GRU encoding
- multiscale temporal encoding adds clear value
- H3 spatial refinement is useful but introduces a recall-vs-calibration trade-off
