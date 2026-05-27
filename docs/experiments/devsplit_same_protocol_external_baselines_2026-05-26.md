# Devsplit Same-Protocol External Baselines (2026-05-26)

## Goal

The earlier `research_devsplit` iteration compared new model variants mostly against the current multiscale mainline. For paper-facing model selection, that is not enough. This round adds same-protocol external baselines on the split-preserving `artifacts/research_devsplit` proxy so that future subset optimization can be judged against the models that matter in the final comparison table.

## Protocol

- Dataset root: `artifacts/research_devsplit`
- Split policy: existing split-preserving proxy, not the old `max_trajectories=256` subset
- Horizons: `10 / 30 / 60 / 120 min`
- Grid: H3 hierarchical decode, same evaluation metrics as the mainline
- New baseline configs:
  - `configs/region_forecast_adrr_devsplit_traisformer_h3.yaml`
  - `configs/region_forecast_adrr_devsplit_flightbertpp_h3.yaml`

## Training notes

- `TrAISformer-H3` devsplit run:
  - checkpoint: `artifacts/research_devsplit/checkpoints/adrr_devsplit_traisformer_h3.pt`
  - eval: `artifacts/research_devsplit/evaluations/adrr_devsplit_traisformer_h3_summary.json`
  - best val loss: `0.7449`
- `FlightBERT++-H3` devsplit run:
  - checkpoint: `artifacts/research_devsplit/checkpoints/adrr_devsplit_flightbertpp_h3.pt`
  - eval: `artifacts/research_devsplit/evaluations/adrr_devsplit_flightbertpp_h3_summary.json`
  - best val loss: `0.8039`

## Macro comparison

Macro means below are unweighted averages over the four forecast horizons.

| Model | Top1 | Top3 | Top5 | NLL | Brier | ECE | Union Cover | Union Cells | Union Area km² | Cover at Budget |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GRU | 0.3182 | 0.5739 | 0.6944 | 2.7494 | 0.8335 | 0.0951 | **0.8898** | 13.2375 | 580967.48 | 0.9354 |
| Multiscale tuned | 0.3722 | 0.6082 | 0.7235 | 2.4497 | 0.7783 | 0.0562 | 0.8063 | 7.1653 | 323031.09 | 0.9469 |
| TrAISformer-H3 | **0.4001** | **0.6345** | **0.7361** | **2.3504** | **0.7514** | 0.0596 | 0.7858 | **6.4272** | **278292.40** | 0.9456 |
| FlightBERT++-H3 | 0.3243 | 0.5694 | 0.6894 | 2.5257 | 0.8115 | 0.0577 | 0.8058 | 7.9066 | 359522.10 | 0.9448 |
| Gated local challenger | 0.3812 | 0.6146 | 0.7258 | 2.4446 | 0.7666 | **0.0404** | 0.7871 | 6.4752 | 314506.70 | **0.9477** |

## What changed relative to the earlier self-comparison

- The gated-local challenger still improves over the devsplit tuned mainline on:
  - `Top1 / Top3 / Top5`
  - `NLL`
  - `Brier`
  - `ECE`
  - `Cover at Budget`
- But once the external baselines are added, the paper-facing picture is still unfavorable:
  - `TrAISformer-H3` takes `Top1 / Top3 / Top5 / NLL / Brier / Union Cells / Union Area`
  - the gated-local challenger only takes `ECE` and `Cover at Budget`
  - `GRU` still has the highest `Union Cover`, mostly because it spreads probability mass more broadly

## Short-horizon view

Average over `10 / 30 min`:

| Model | Top1 | Top3 | Top5 | NLL | Brier | ECE | Union Cover |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Multiscale tuned | 0.3386 | 0.5714 | 0.6910 | 2.5716 | 0.8156 | 0.0845 | 0.8135 |
| TrAISformer-H3 | **0.3854** | **0.6033** | **0.7081** | **2.3925** | **0.7706** | 0.0553 | **0.8052** |
| Gated local challenger | 0.3565 | 0.5842 | 0.6955 | 2.5614 | 0.7921 | **0.0530** | 0.7752 |

This means the gated-local route helps the current multiscale backbone, but it does not yet overturn the short-horizon advantage of `TrAISformer-H3` on the same proxy.

## Current decision

- Do **not** promote the gated-local challenger to a full run yet.
- Treat this table as the new subset gate for future iterations.
- Future optimization must be compared against the same-protocol `TrAISformer-H3`, not only against the current multiscale mainline.

## Caveat

These same-protocol devsplit runs are already useful for direction finding, but they are not perfectly training-budget matched:

- the existing devsplit multiscale tuned mainline was produced with a longer schedule and a larger batch size
- the new external-baseline runs were intentionally kept lighter to keep iteration cost reasonable on the local CPU-only environment

Even with that caveat, `TrAISformer-H3` remains strong enough here that any claim like "the new route already fixes the paper table optics" would be premature.
