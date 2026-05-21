# Research Plan for Global Aviation Multi-Horizon Regional Forecasting

## Goal

Evolve the current `hierarchical_gru` baseline into a stronger, paper-ready family of models while preserving:

- baseline reproducibility
- checkpoint compatibility for existing runs
- config-driven switching
- train/evaluate/predict support across all model variants

## Current Baseline Position

Current strongest reference points:

- full baseline already exists in `artifacts/`
- reproducible subset baseline now exists in `artifacts/research_subset_baseline/`
- evaluation bug in hierarchical lookup reconstruction has already been fixed and regression-tested

Current baseline pain points:

- only final GRU hidden state is used
- horizon heads are mostly independent
- long-horizon quality is weak, especially `120m`
- `30m` is unexpectedly weaker than `60m`
- no lightweight structure-aware H3 refinement yet

## Recommended Priority Order

1. `attention_hierarchical_gru`
2. `horizon_decoder_gru`
3. `transformer_horizon_decoder`
4. `multiscale_transformer_h3`
5. `graph_attention_h3` or lighter H3-neighbor refinement

This order is recommended because it matches both:

- the codebase shape
- the reviewed literature

## Direction 1: Temporal Attention

### Recommended Priority

- highest

### Core Idea

Instead of using only the final GRU hidden state, expose the full encoder output sequence and learn attention weights over time.

Two useful variants:

1. shared temporal attention pooling
2. horizon-conditioned temporal attention pooling

### Why This Fits the Current Code

- no dataset rebuild is required
- existing GRU encoder remains intact
- new behavior stays inside model forward logic
- easiest ablation against `hierarchical_gru`

### Proposed Model Name

- `attention_hierarchical_gru`

### Expected File Changes

- `regional_forecast/config.py`
  - add model config fields for attention pooling
- `regional_forecast/models/forecast_model.py`
  - add new model class or extend model factory support
- `regional_forecast/pipeline/train.py`
  - add new `model_type` handling
- `regional_forecast/pipeline/evaluate.py`
  - add new `model_type` loading path
- `regional_forecast/pipeline/predict.py`
  - add new `model_type` loading path
- `configs/`
  - add experiment configs
- `docs/experiments/`
  - add experiment record

### Minimal Parameter Set

- `temporal_attention_enabled`
- `temporal_attention_dim`
- `temporal_attention_dropout`
- `temporal_attention_mode`:
  - `shared`
  - `horizon_query`

### Expected Experiments

1. baseline subset vs attention pooling
2. shared attention vs horizon-conditioned attention
3. with and without reachability filter

### Success Standard

A successful first step should achieve at least one of:

- better `region_top3` or `region_top5` on at least 2 horizons
- lower `region_nll` on at least 2 horizons
- lower `region_ece` without major top-k regression

### Risk

- low to medium
- main risk is overfitting the small subset or collapsing to noisy attention weights

## Direction 2: Horizon-Aware Decoder

### Recommended Priority

- second

### Core Idea

Replace fully independent per-horizon heads with a shared decoder structure that uses learned horizon embeddings or horizon queries.

Two concrete versions:

1. shared head + horizon embedding
2. query-based decoder over encoder outputs

### Why This Fits the Current Code

- current project already predicts multiple horizons from one history sequence
- literature strongly supports context-specific horizon decoding
- can be layered on top of temporal attention

### Proposed Model Names

- `attention_hierarchical_gru` for the first lightweight version
- `horizon_decoder_gru` for the stronger query-decoder version

### Expected File Changes

- `regional_forecast/config.py`
  - add horizon embedding / decoder config
- `regional_forecast/models/forecast_model.py`
  - implement shared horizon-aware decoder
- `regional_forecast/pipeline/train.py`
- `regional_forecast/pipeline/evaluate.py`
- `regional_forecast/pipeline/predict.py`
- `configs/`

### Minimal Parameter Set

- `horizon_embedding_dim`
- `horizon_decoder_hidden_dim`
- `horizon_decoder_layers`
- `horizon_query_attention_heads`
- `share_status_head`

### Expected Experiments

1. temporal attention only
2. temporal attention + horizon embedding
3. temporal attention + query decoder

### Success Standard

- consistent improvement on `30m` and `120m`
- lower `region_nll` and `region_brier`
- no breakage in hierarchical fine reconstruction

### Risk

- medium
- decoder complexity can exceed the data scale quickly
- if too aggressive, gains on long horizons may come with short-horizon regression

## Direction 3: Graph / H3 Spatial Attention

### Recommended Priority

- third

### Core Idea

Use H3 structure more explicitly, but start light.

Recommended progression:

1. evaluation-time neighbor-aware smoothing
2. trainable H3 embedding refinement
3. one-step message passing over coarse logits or H3 neighbors

Avoid:

- full dense GNN over all cells
- large graph transformer over the entire global grid

### Why This Fits the Current Code

- H3 hierarchy already exists in metadata
- neighbor rings are already used during evaluation
- H3 APIs support parent-child and neighbor traversal directly

### Proposed Model Name

- `graph_attention_h3`

### Expected File Changes

- `regional_forecast/models/forecast_model.py`
  - optional spatial refinement layer
- `regional_forecast/data/regionizer.py` or helper module
  - if neighbor adjacency helpers are needed
- `regional_forecast/pipeline/evaluate.py`
  - if an evaluation-only smoothing baseline is added
- `configs/`
- possibly a new utility module under `regional_forecast/models/`

### Expected Experiments

1. no spatial refinement
2. evaluation-only neighbor smoothing
3. trainable neighbor refinement

### Success Standard

- better candidate coverage or ring-aware top-k
- improved long-horizon calibration
- limited compute overhead

### Risk

- medium
- easy to add complexity without clear gains
- adjacency construction must remain efficient at global scale

## Status Update

Completed so far:

1. `attention_hierarchical_gru`
2. `horizon_decoder_gru`
3. `transformer_horizon_decoder`
4. `multiscale_transformer_h3`
5. tuned multiscale optimization ablation
6. horizon-consistency regularizer ablation

Current best research takeaway:

- horizon-query decoding is better than shared temporal pooling
- a Transformer encoder is directionally helpful for medium/long horizons
- explicit multiscale temporal modeling improves the single-scale Transformer branch
- the tuned multiscale training recipe is currently the strongest result
- the current decoder-feature consistency regularizer is not helpful at `weight=0.05`
- old `max_trajectories` subsets were invalid performance proxies because they created high train/test label mismatch
- the current fast proxy is the split-preserving `artifacts/research_devsplit` subset
- upgraded candidate metrics show the multiscale branch uses much smaller probability-mass candidate sets; fixed-budget coverage is not generally worse

## Recommended Next Implementation

The next structural implementation should be:

- lightweight H3-aware refinement on top of the tuned multiscale branch, evaluated with fixed-budget and area-aware candidate metrics

Recommended order:

1. evaluation-time coarse-logit neighbor smoothing baseline
2. trainable H3 neighbor bias / refinement layer
3. only after that, consider heavier graph attention

Why this is next:

- optimization tuning already unlocked large gains
- the consistency regularizer did not become the next win
- the healthier devsplit shows the multiscale model improves top-k and calibration but loses candidate coverage
- the upgraded metrics show this loss is largely caused by smaller candidate sets under the probability-mass stopping rule
- the remaining opportunity is now better spatial structure use rather than more temporal smoothness
3. learned horizon embeddings
4. otherwise unchanged coarse-to-fine hierarchical prediction

This is the most stable first move because:

- it stays close to the baseline
- it matches the strongest actionable literature signals
- it preserves existing data artifacts
- it creates clean ablations

## Detailed First-Round Experiment Plan

### E0: Already Completed

- audit existing full baseline
- build subset baseline
- fix evaluation bug

### E1: Completed

- implement `attention_hierarchical_gru`
- run smoke and subset experiments
- document regression relative to baseline

### E2: Completed

- implement `horizon_decoder_gru`
- run smoke and subset experiments
- document improvement over round-1 attention and mixed results vs baseline

### E3: Completed

- implement `transformer_horizon_decoder`
- build fair `history_minutes: 60` comparison runs
- compare:
  - `hierarchical_gru`
  - `horizon_decoder_gru`
  - `transformer_horizon_decoder`

### E4: Completed

- implement `multiscale_transformer_h3`
- consult external references including Pyraformer / HiVT / Wayformer / Crossformer
- run smoke and subset experiments
- identify multiscale temporal modeling as the strongest current direction

### Recommended Next Experiment

The next experiment should be:

- `multiscale_transformer_h3_tuned`

with:

1. lower learning rate
2. longer patience
3. lighter dropout
4. optional cross-horizon consistency regularization

### E1: Attention Pooling Only

Model:

- `attention_hierarchical_gru`

Change:

- replace last-hidden-state summary with shared temporal attention pooling

Compare against:

- subset baseline `adrr_hierarchical_gru_subset_baseline`

### E2: Attention Pooling + Horizon Embedding

Model:

- `attention_hierarchical_gru`

Change:

- add horizon embeddings to coarse/status/child decoding

Compare against:

- E1
- subset baseline

### E3: Reachability Ablation

For the best of E1/E2:

- evaluate with reachability on
- evaluate with reachability off

### E4: Optional Query Decoder

If E2 is promising:

- promote to `horizon_decoder_gru`

## Success Criteria by Stage

### Stage 4 Success

The first improved model is worth keeping if it shows:

- measurable gain on subset baseline
- especially on `30m` and/or `120m`
- no pipeline breakage
- no checkpoint loading issues

### Stage 5 Success

Second-round work is justified if:

- first-round gains are stable
- the remaining weakness clearly points to horizon-specific decoding or spatial structure

### Paper-Readiness Indicators

The work starts looking publishable when we have:

- stable gains on multiple horizons
- clear ablations for temporal attention and horizon conditioning
- calibration metrics, not only top-k metrics
- reachability filter ablation
- strong qualitative examples and candidate-region visualizations

## Concrete File Plan

### Likely Files to Modify in Round 1

- `regional_forecast/config.py`
- `regional_forecast/models/forecast_model.py`
- `regional_forecast/pipeline/train.py`
- `regional_forecast/pipeline/evaluate.py`
- `regional_forecast/pipeline/predict.py`
- `configs/region_forecast_adrr_attention_*.yaml`
- `docs/method.md`
- `docs/experiments/*.md`

### Files to Avoid Touching Unless Necessary

- existing full baseline artifacts under `artifacts/`
- dataset build format
- registry semantics beyond compatibility additions

## Main Risks

1. overfitting on the subset baseline
2. adding too much decoder complexity too early
3. confusing raw-model improvements with reachability-filter improvements
4. making architecture changes that require dataset rebuilds unnecessarily
5. breaking predict/evaluate compatibility for old checkpoints

## Mitigations

1. keep the first change narrow
2. use new `model_type` values instead of mutating the baseline class in place
3. record raw config and exact run names for every experiment
4. run smoke test and subset test before any larger experiment
5. keep regression tests around evaluation and model loading

## Immediate Next Step

Proceed to the first implementation:

- add `attention_hierarchical_gru`
- run smoke
- run subset experiment
- compare against `adrr_hierarchical_gru_subset_baseline`
