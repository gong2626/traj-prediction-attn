# Literature Review for Global Aviation Multi-Horizon Regional Forecasting

## Review Goal

This review focuses on papers and technical resources that can directly inform the current project:

- temporal attention over history
- horizon-aware multi-step forecasting
- graph / spatial attention
- aircraft / ADS-B trajectory prediction
- hierarchical spatial indexing with H3
- probabilistic forecasting and calibration

Priority is given to ideas that can be added to the existing `hierarchical_gru` baseline without rewriting the whole system.

## Selection Summary

The most actionable themes for this codebase are:

1. attention over encoder states instead of only using the final hidden state
2. query- or embedding-based horizon conditioning for multi-horizon decoding
3. lightweight spatial refinement using H3 hierarchy or neighbor structure

The least actionable near-term themes are:

- heavy multi-agent interaction stacks
- large end-to-end transformer rewrites
- models that depend on weather cubes, airport maps, or rich social context not present in the current dataset artifacts

## Reviewed Sources

### 1. [Attention-LSTM based prediction model for aircraft 4-D trajectory](https://www.nature.com/articles/s41598-022-19794-1)

- Type: aircraft trajectory prediction paper
- Method core:
  - LSTM-based trajectory forecasting with an attention mechanism to emphasize informative historical states
- Relation to this project:
  - directly supports replacing last-hidden-state summarization with attention pooling
  - same high-level family as the current GRU baseline
- Can it land in the current codebase?
  - yes, very directly
  - map to `hierarchical_gru` by exposing GRU sequence outputs and adding temporal attention pooling
- Implementation risk:
  - low
  - little artifact or pipeline impact

### 2. [A deep learning framework for predicting aircraft trajectories from sparse satellite observations](https://www.nature.com/articles/s41598-025-27064-z)

- Type: aircraft trajectory prediction paper
- Method core:
  - hybrid CNN + LSTM + Transformer with temporal attention over encoder outputs
  - explicitly emphasizes multi-scale temporal modeling
- Relation to this project:
  - supports the idea that temporal attention is useful even when recurrence remains the backbone
  - reinforces a staged design: keep GRU backbone, add richer temporal summarization
- Can it land in the current codebase?
  - partially
  - the temporal attention part is highly reusable
  - the full CNN/LSTM/Transformer hybrid is too large for first iteration
- Implementation risk:
  - medium
  - full architecture would be overkill now
  - attention-over-history is low risk and relevant

### 3. [Data-Driven 4D Trajectory Prediction Model Using Attention-TCN-GRU](https://www.mdpi.com/2226-4310/11/4/313)

- Type: aircraft trajectory prediction paper
- Method core:
  - TCN encoder + GRU decoder + attention
  - designed to better capture long trajectory patterns and emphasize important sequence segments
- Relation to this project:
  - confirms that attention-augmented recurrent aviation models are practical and competitive
  - suggests a future path if plain temporal attention saturates: temporal convolution before GRU
- Can it land in the current codebase?
  - partially
  - direct first-step landing: attention over GRU states
  - second-step landing: optional temporal-conv frontend before GRU
- Implementation risk:
  - medium
  - adding TCN changes model complexity and tuning burden

### 4. [Noise robust aircraft trajectory prediction via autoregressive transformers with hybrid positional encoding](https://www.nature.com/articles/s41598-025-96512-7)

- Type: aircraft trajectory prediction paper
- Method core:
  - autoregressive transformer
  - hybrid positional encoding
  - robustness to noisy ADS-B observations
- Relation to this project:
  - useful for thinking about sequence encoding and robustness
  - less aligned with the current discrete regional multi-horizon setup than attention-augmented GRU is
- Can it land in the current codebase?
  - partially
  - positional / temporal encoding ideas can inspire horizon embeddings or future sequence encoders
  - full autoregressive transformer is not the first move
- Implementation risk:
  - medium to high
  - larger architecture shift
  - may complicate compatibility with current coarse-to-fine heads

### 5. [A Generalized Approach to Aircraft Trajectory Prediction](https://ntrs.nasa.gov/api/citations/20220002176/downloads/Generalizing_Trajectory_Prediction_FORMATTED.pdf?attachment=true)

- Type: NASA technical report
- Method core:
  - evaluates multiple recurrent and hybrid designs
  - explicitly argues for generalization across routes, datasets, and model variants
  - discusses GRU, self-attention, and modular hybrid-recurrent designs
- Relation to this project:
  - highly aligned with the current goal of avoiding route-specific overfitting and building a generalizable global predictor
  - supports conservative modular evolution instead of one-shot redesign
- Can it land in the current codebase?
  - yes, conceptually
  - especially useful for justifying modular GRU-plus-attention evolution
- Implementation risk:
  - low
  - mostly design guidance rather than a direct code template

### 6. [Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting](https://arxiv.org/abs/1912.09363)

- Type: multi-horizon forecasting paper
- Method core:
  - recurrent local processing plus attention for long-range dependencies
  - explicit multi-horizon forecasting architecture
  - interpretable variable and temporal selection
- Relation to this project:
  - one of the strongest conceptual references for multi-horizon design
  - supports shared backbone + horizon-specific conditioning instead of completely separate heads
- Can it land in the current codebase?
  - partially
  - full TFT is too broad
  - horizon conditioning, shared decoder structure, and attention over temporal features are very actionable
- Implementation risk:
  - medium
  - must be simplified to fit the current project

### 7. [MQTransformer: Multi-Horizon Forecasts with Context Dependent and Feedback-Aware Attention](https://arxiv.org/abs/2009.14799)

- Type: multi-horizon forecasting paper
- Method core:
  - context-dependent attention for multi-horizon outputs
  - decoder-encoder attention aligned to the target context
- Relation to this project:
  - directly relevant to horizon-aware decoding
  - suggests that each horizon should query history differently instead of sharing one pooled summary
- Can it land in the current codebase?
  - yes, in simplified form
  - a horizon embedding or learned horizon query can be added on top of GRU encoder states
- Implementation risk:
  - medium
  - requires careful design to avoid overcomplicating the baseline

### 8. [Spatio-Temporal Graph Transformer Networks for Pedestrian Trajectory Prediction (STAR)](https://arxiv.org/abs/2005.08514)

- Type: graph / trajectory prediction paper
- Method core:
  - attention-based spatio-temporal graph modeling
  - uses attention for both temporal and interaction structure
- Relation to this project:
  - relevant for the graph / spatial attention direction
  - useful as a design reference for lightweight structure-aware refinement
- Can it land in the current codebase?
  - partially
  - not as a full social graph model
  - but its message is actionable: attention can model structured spatial dependencies without heavy recurrent graph stacks
- Implementation risk:
  - medium
  - direct port would mismatch the current single-trajectory regional framing

### 9. [GATraj: A Graph- and Attention-based Multi-Agent Trajectory Prediction Model](https://arxiv.org/abs/2209.07857)

- Type: graph / attention trajectory paper
- Method core:
  - graph-based modeling with attention
  - emphasizes efficient interaction modeling and ablations
- Relation to this project:
  - relevant for lightweight spatial relation modeling
  - useful for thinking about sparse neighbor attention rather than full dense graph reasoning
- Can it land in the current codebase?
  - partially
  - full multi-agent interaction graph is not the current problem definition
  - but graph-attention-style local refinement over H3 neighbors is feasible
- Implementation risk:
  - medium
  - easy to overshoot and build something much heavier than needed

### 10. [Predicting Like A Pilot: Dataset and Method to Predict Socially-Aware Aircraft Trajectories in Non-Towered Terminal Airspace](https://arxiv.org/abs/2109.15158)

- Type: aircraft trajectory prediction paper
- Method core:
  - aircraft trajectory prediction with agent-agent interaction and environmental context
  - introduces TrajAir dataset and TrajAirNet baseline
- Relation to this project:
  - relevant mainly for aircraft-domain interaction modeling and aviation-specific trajectory priors
  - less aligned with current global en-route regional forecasting than terminal-airspace modeling
- Can it land in the current codebase?
  - partially
  - not directly as a multi-agent model
  - but useful for later thinking about traffic-density context or interaction-aware extensions
- Implementation risk:
  - medium
  - current dataset artifacts do not expose explicit neighboring-aircraft state

### 11. [ASCENT: Transformer-Based Aircraft Trajectory Prediction in Non-Towered Terminal Airspace](https://arxiv.org/abs/2603.16550)

- Type: recent aircraft trajectory prediction paper
- Method core:
  - lightweight transformer encoder
  - query-based decoder
  - parameterized multi-modal output design
- Relation to this project:
  - highly relevant to the planned horizon-aware decoder direction
  - especially useful as evidence that query-based decoding can stay lightweight
- Can it land in the current codebase?
  - yes, partially
  - the best extractable idea is the query-based decoder pattern, not the full model
- Implementation risk:
  - medium
  - introducing a full transformer rewrite would be too much
  - horizon queries on top of GRU encoder states are tractable

### 12. [H3: Uber’s Hexagonal Hierarchical Spatial Index](https://www.uber.com/en-US/blog/h3/) and [uber/h3 GitHub repository](https://github.com/uber/h3)

- Type: core spatial indexing technical resource
- Method core:
  - hierarchical hexagonal indexing
  - neighbor traversal
  - compact/uncompact hierarchy operations
  - directed edge representation
- Relation to this project:
  - foundational
  - the current code already depends on H3 for spatial discretization and hierarchical coarse-to-fine mapping
- Can it land in the current codebase?
  - already landed
  - further use is possible through neighbor-aware smoothing, parent-child structure priors, and edge-based movement modeling
- Implementation risk:
  - low for neighbor smoothing
  - medium for trainable graph operations

### 13. [OpenSky Report 2025: Improving Crowdsourced Flight Trajectories with ADS-C Data](https://arxiv.org/abs/2505.06254)

- Type: open aviation data resource paper
- Method core:
  - expands open flight trajectory coverage using ADS-C
  - emphasizes broader open-trajectory availability
- Relation to this project:
  - useful for future data scaling and data quality discussion
  - supports the paper narrative around open large-scale flight trajectory research
- Can it land in the current codebase?
  - not immediately as a model component
  - yes as future data source or paper discussion
- Implementation risk:
  - low technically
  - high scope-wise if treated as a data migration now

### 14. [castacks/trajairnet GitHub repository](https://github.com/castacks/trajairnet)

- Type: open-source aircraft trajectory prediction project
- Method core:
  - open implementation of TrajAirNet and aircraft social-trajectory workflow
- Relation to this project:
  - useful as an aircraft-domain open-source reference
  - especially for how aviation trajectory projects package datasets, prediction outputs, and domain assumptions
- Can it land in the current codebase?
  - selectively
  - code is not directly portable, but design patterns and aviation-specific assumptions are useful
- Implementation risk:
  - medium
  - problem setting mismatch with current global regional forecasting task

## Synthesis

## High-Confidence Near-Term Ideas

These are the best fits for the current codebase:

1. temporal attention pooling over GRU encoder states
2. learned horizon embeddings
3. horizon-query decoding over encoder states
4. lightweight H3-neighbor-aware refinement or smoothing

## Medium-Term Ideas

These are promising but should wait until after the first ablation:

1. TCN or convolutional frontend before GRU
2. lightweight transformer block on top of GRU outputs
3. graph-style refinement over coarse logits or H3 neighbors

## Low-Priority or High-Risk Ideas

These should not be the first implementation:

1. full transformer rewrite
2. multi-agent social interaction model
3. weather-cube architectures requiring new data modalities
4. large generative or multi-modal trajectory systems

## Review Conclusion

The literature supports a conservative but publishable progression:

1. keep the current hierarchical H3 regional formulation
2. improve temporal summarization with attention
3. make decoding horizon-aware
4. only then add lightweight spatial refinement using H3 neighborhood structure

This aligns well with the current repository and minimizes experimental risk.

## Round 4 Addendum

During the multiscale redesign after round 3, these references were revisited as direct architectural inputs:

1. Pyraformer
   - main idea reused:
     use a multiresolution temporal representation instead of forcing one flat sequence encoder to carry all long-range structure
2. HiVT
   - main idea reused:
     hierarchical representation spaces can improve trajectory-style prediction even when the downstream task is not a full multi-agent scene forecast
3. Wayformer
   - main idea reused:
     keep the core architecture simple and let attention operate over well-structured tokens rather than adding large bespoke modules too early
4. Crossformer
   - main idea reused:
     changing the representation granularity can be more valuable than only increasing model depth at one scale

These references directly motivated the `multiscale_transformer_h3` branch implemented after round 3.
