# Transformer Horizon Decoder Design

## Goal

Implement a third-round research model that upgrades the sequence encoder from GRU to Transformer while preserving:

- hierarchical H3 coarse-to-fine decoding
- horizon-query decoder logic introduced in round 2
- train/evaluate/predict compatibility
- config-driven model selection
- reproducible smoke and subset experiments

## Scope

This round changes only the sequence encoder and comparison setup.

In scope:

- add a new model type: `transformer_horizon_decoder`
- retain the current horizon-query decoder and hierarchical fine reconstruction
- create a fair 60-minute-history subset baseline for comparison
- run smoke and subset experiments
- record results in experiment and method docs

Out of scope for this round:

- H3 graph message passing
- multiscale dual-encoder architectures
- new losses
- full-month training

## Why 60-Minute History

Current subset experiments use `history_minutes: 30`, which is short for attention-based sequence modeling.

For round 3:

- use `history_minutes: 60`
- keep the same horizons `[10, 30, 60, 120]`
- keep the same subset size and data source

This increases available temporal context without immediately pushing the subset into a much slower or sparser regime. If the model shows signal at 60 minutes, a later extension can test 120-minute history.

## Model Design

### High-Level Structure

1. embed input H3 region ids and parent H3 region ids
2. project numeric and mask features
3. concatenate feature channels into a sequence tensor
4. add positional encoding
5. run a Transformer encoder over the history sequence
6. compute a masked mean pooled summary plus context projection
7. build one learned query per horizon from the shared base summary and horizon embedding
8. attend each horizon query over the Transformer encoder outputs
9. decode coarse logits, status logits, and child logits from horizon-specific decoder features

### Decoder Reuse

The round-2 decoder path remains the same:

- horizon embeddings
- per-horizon query projection
- attention over encoder states
- per-horizon decoder feature projection
- existing coarse/status/child heads

This keeps the third-round ablation focused on the encoder change.

### Sequence Masking

The model must respect the same temporal mask used in round 1 and round 2:

- valid observed history points remain attendable
- padded unknown points inserted by history padding are masked out

The same mask will be used for:

- Transformer self-attention key padding
- masked summary pooling
- horizon-query attention over encoder outputs

## Config Additions

Add transformer-specific model config fields:

- `transformer_num_layers`
- `transformer_num_heads`
- `transformer_ffn_dim`
- `transformer_dropout`

Defaults should be lightweight enough for the subset runs on a 32 GB GPU.

## Fair Comparison Setup

Round 3 should compare:

1. `hierarchical_gru` with `history_minutes: 60`
2. `transformer_horizon_decoder` with `history_minutes: 60`

Both should use:

- the same ADRR shard source
- the same `max_trajectories`
- the same horizons
- the same training/evaluation settings when practical

This avoids attributing improvements to a longer history window alone.

## Files Expected To Change

- `regional_forecast/config.py`
  - transformer config fields
- `regional_forecast/models/forecast_model.py`
  - new Transformer encoder model
- `regional_forecast/models/__init__.py`
  - export new class
- `regional_forecast/pipeline/train.py`
  - `model_type` factory support
- `regional_forecast/pipeline/evaluate.py`
  - checkpoint loading and hierarchical dispatch support
- `regional_forecast/pipeline/predict.py`
  - prediction-time loading support
- `tests/test_attention_hierarchical_model.py`
  - shape and interface tests for the new model
- `configs/`
  - smoke and subset configs for:
    - transformer model
    - fair 60-minute baseline
- `docs/experiments/`
  - round-3 experiment log
- `docs/method.md`
- `docs/ablation.md`
- `docs/paper_outline.md`

## Risks

1. the subset may still be too small for the Transformer to clearly dominate
2. longer history may reduce usable windows and change subset sample counts
3. if the Transformer is too large, it can overfit or slow down enough to blur the comparison

## Success Criteria

Round 3 is successful if at least one of these happens against the fair 60-minute baseline:

- better `region_top3` or `region_top5` on at least two horizons
- lower `region_nll` on at least two horizons
- better `120m` metrics without large regression at short horizons

Even if it does not fully win, the round is still valuable if it clearly outperforms the round-2 GRU horizon decoder and gives a cleaner foundation for a later multiscale model.
