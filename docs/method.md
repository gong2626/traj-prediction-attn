# Method Notes

## Baseline

The baseline method is a hierarchical H3 regional forecaster:

1. map historical trajectory points to H3 regions
2. encode history with a GRU
3. use the final hidden state plus context features as the summary
4. predict coarse H3 region logits for each horizon
5. predict child logits conditioned on coarse region
6. reconstruct sparse fine-region probabilities
7. predict airborne / landed status
8. optionally apply reachability filtering during evaluation and prediction

## Round 1 Extension: Attention Hierarchical GRU

The first extension keeps the hierarchical coarse-to-fine prediction structure and changes the sequence summary and decoding process:

1. use the full GRU encoder output sequence
2. apply temporal attention pooling across encoder states
3. fuse pooled representation with context
4. inject learned horizon embeddings
5. build horizon-specific decoder features
6. predict coarse logits, status logits, and child logits from these horizon-aware features

This version is implemented as:

- `attention_hierarchical_gru`

## Round 2 Extension: Horizon Decoder GRU

The second extension keeps the same hierarchical output structure but changes the way horizon-specific features are built:

1. keep the GRU encoder
2. keep a shared base summary from the final GRU hidden state plus context
3. add a learned embedding for each forecast horizon
4. build one learned query per horizon
5. use each horizon query to attend over the full encoder state sequence
6. concatenate base summary, horizon-conditioned attended sequence, and horizon embedding
7. project this into per-horizon decoder features
8. predict coarse logits, status logits, and child logits from these decoder features

This version is implemented as:

- `horizon_decoder_gru`

## Round 3 Extension: Transformer Horizon Decoder

The third extension replaces the recurrent encoder with a Transformer encoder while keeping the round-2 horizon-query decoder and hierarchical H3 output structure:

1. embed region, parent-region, numeric, and mask channels
2. concatenate them into a sequence feature tensor
3. project into a Transformer model dimension
4. add learned positional embeddings
5. run masked Transformer self-attention over the observed history
6. compute a masked summary representation
7. fuse this summary with context features
8. build one learned query per horizon
9. attend each horizon query over the Transformer encoder states
10. decode coarse logits, status logits, and child logits from per-horizon decoder features

This version is implemented as:

- `transformer_horizon_decoder`

## Round 4 Extension: Multiscale Transformer H3

The fourth extension adds an explicit second temporal scale on top of the Transformer branch:

1. encode the full fine-grained history with a Transformer
2. pool the fine sequence into a coarser temporal sequence with a fixed stride
3. run a second, shallower Transformer on the coarse sequence
4. let each horizon query attend to both:
   - the fine sequence
   - the coarse sequence
5. fuse fine-scale and coarse-scale attended features before the hierarchical H3 output heads

This version is implemented as:

- `multiscale_transformer_h3`

## Round 5 Extension: Tuned Multiscale Training and Horizon Consistency

The fifth round keeps the same multiscale architecture and changes the training regime:

1. lower learning rate
2. lighter dropout
3. longer scheduler and early-stopping patience
4. optional adjacent-horizon consistency penalty over per-horizon decoder features

The consistency term is:

- train-only
- masked by valid adjacent horizons
- gap-scaled so closer horizons are regularized more strongly than wider pairs

This round produced two important practical outcomes:

1. the tuned optimization recipe unlocked much stronger performance than round 4
2. the current consistency penalty was not beneficial at `weight=0.05`

## Current Status

Round 1 is functional and reproducible, but it does not outperform the subset baseline overall.

Round 2 is the current preferred research branch among the new model variants because:

- it is clearly better than round 1 attention
- it improves the longest-horizon metrics most relevant to the paper direction
- it adds only a small inference cost over baseline

However, round 2 still does not consistently beat the subset baseline across all horizons and calibration metrics.

Round 3 shows that the encoder upgrade is directionally useful, but it also exposed an important interaction:

- increasing history from `30` to `60` minutes reduced the number of usable subset windows and made optimization harder for all models

Within the fair history-60 comparison:

- the Transformer variant improves medium/long-horizon behavior relative to the GRU horizon decoder control
- it does not yet become a clear global winner over all top-k, coverage, and likelihood metrics

Round 4 established the right architecture direction, but round 5 showed that the bigger issue was training dynamics.

Current strongest model:

- `multiscale_transformer_h3` with the tuned round-5 training recipe

## Round 6 Extension: H3-Aware Candidate Refinement

The sixth round keeps the tuned multiscale checkpoint fixed and adds a configurable spatial postprocessor:

1. take per-horizon fine H3 probabilities from the model
2. diffuse known-cell probability mass over each cell's local H3 disk
3. preserve the unknown class
4. apply reachability filtering after refinement
5. build candidate regions from the refined reachable distribution

This is implemented as evaluation/prediction-time refinement, controlled by:

- `evaluation.h3_refinement_enabled`
- `evaluation.h3_refinement_alpha`
- `evaluation.h3_refinement_ring`
- `evaluation.h3_refinement_preserve_unknown`

Full-data conclusion:

- H3 refinement improves probability-mass candidate recall
- but on full data it degrades calibration and eventually hurts top-k / NLL as alpha increases
- therefore it is best treated as an ablation or optional deployment-time trade-off, not the default final method

Current takeaway:

- multiscale horizon-query Transformer encoding is the best current backbone
- careful optimization matters more right now than adding a generic smoothness regularizer
- the final paper method should remain the unrefined tuned multiscale model
- H3-aware refinement is still worth reporting as a recall-vs-calibration trade-off study
