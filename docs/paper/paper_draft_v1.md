# Large-Scale Multi-Horizon Regional Aircraft Trajectory Forecasting with Hierarchical H3 Prediction and Multiscale Horizon-Query Transformers

## Abstract

Open aviation trajectory data are now large enough to support learning-based forecasting at network scale, yet most existing aircraft trajectory studies still target short-horizon coordinate regression or terminal-area motion forecasting. These settings do not fully match the needs or the data quality of open flight archives, where long-horizon uncertainty, irregular coverage, and scalable evaluation are central concerns. We study multi-horizon regional aircraft trajectory forecasting on one month of open flight trajectories derived from EUROCONTROL's Aviation Data Repository for Research (ADRR) \cite{eurocontrol_adrr_2025}. Given 30 minutes of flight history resampled at 5-minute intervals, the task is to predict future H3 region distributions at 10, 30, 60, and 120 minutes.

We formulate the problem as hierarchical coarse-to-fine H3 forecasting and propose a multiscale horizon-query Transformer that combines fine-scale temporal encoding, coarse pooled temporal encoding, and horizon-specific query attention over encoder states. The model outputs probabilistic regional forecasts together with candidate-region sets that can be evaluated by both accuracy and calibration. We evaluate on 20,712,602 forecast windows constructed from commercial flights operating in and over Europe, split by aircraft to reduce leakage across train, validation, and test sets.

Compared with a hierarchical GRU baseline, the proposed model improves macro top-1/top-3/top-5 accuracy from 0.6648/0.9076/0.9631 to 0.6827/0.9174/0.9685, reduces negative log-likelihood from 1.0214 to 0.9641, reduces Brier score from 0.4578 to 0.4338, and reduces expected calibration error from 0.0180 to 0.0093. Relative to a classical Markov regional transition baseline, gains are much larger, while candidate regions also become substantially smaller. An H3-neighbor refinement postprocessor improves candidate recall but degrades calibration, making it better suited as an ablation than as the default inference path. These results support regional probabilistic forecasting as a scalable alternative to long-horizon point prediction on open aviation trajectory data.

## 1. Introduction

Open air traffic data have enabled a wide range of research in aviation surveillance, environmental analysis, and operational modeling. Community-driven infrastructures such as the OpenSky Network have shown that large-scale aircraft trajectory data can support reproducible research at unprecedented scale \cite{opensky2014,strohmeier2021crowdsourced}. EUROCONTROL's Aviation Data Repository for Research (ADRR) extends this opportunity with large, structured trajectory and flight-information archives for commercial flights operating in and over Europe \cite{eurocontrol_adrr_2025}. These data sources make it realistic to revisit trajectory forecasting from the perspective of open, large-scale, and reproducible modeling rather than narrowly curated benchmark scenarios.

However, the forecasting problem supported by such data is not identical to the problem emphasized in much of the aircraft trajectory prediction literature. Many prior works focus on short-term continuous trajectory extrapolation, four-dimensional point prediction, or terminal-area interactions \cite{jia2022attentionlstm,ma2024attentiontcngru,li2025noiserobust,shan2025sparse}. Those formulations are valuable, but they are less well aligned with long-horizon forecasting from open operational data, where observations are noisy, route structure is highly multimodal, and exact pointwise forecasts become difficult to interpret or calibrate. For strategic traffic awareness, surveillance resilience, and high-level flow reasoning, a probabilistic regional forecast can be more appropriate than a single predicted coordinate sequence.

This paper therefore studies **multi-horizon regional aircraft trajectory forecasting**. Instead of predicting one future track, we predict distributions over future H3 regions at multiple lead times. This formulation offers three advantages. First, it respects the uncertainty that naturally grows with forecast horizon. Second, it aligns with large-scale open trajectory archives whose spatial quality and operational diversity make exact long-range point regression brittle. Third, it enables calibration-aware evaluation through negative log-likelihood, Brier score, expected calibration error, and candidate-region recall under explicit area budgets.

Our starting point is an existing hierarchical H3 regional forecaster built around a GRU encoder and coarse-to-fine decoding. We keep the hierarchical spatial formulation, because it already provides a scalable way to model regional uncertainty. We then ask a narrower research question: **what temporal architecture best supports multi-horizon probabilistic regional forecasting on large-scale open aviation trajectories?** Based on a staged empirical study, we find that the strongest improvement comes from replacing the single summary representation with horizon-specific query decoding over Transformer encoder states, and further strengthening this with an explicit multiscale temporal branch.

The resulting model, a **multiscale horizon-query Transformer**, uses one encoder over fine-scale history, one encoder over pooled coarse-scale history, and one learned query per forecast horizon. The hierarchical H3 decoder remains compatible with the original baseline, which makes the architectural changes easy to isolate through ablations. On the full dataset, the final model consistently outperforms a classical Markov baseline, the original hierarchical GRU baseline, and a single-scale Transformer horizon decoder.

This paper makes the following contributions:

1. We formulate large-scale multi-horizon regional aircraft trajectory forecasting on open ADRR-derived flight trajectories as a hierarchical H3 probabilistic prediction problem.
2. We propose a multiscale horizon-query Transformer that improves regional accuracy, likelihood, Brier score, and calibration over a hierarchical GRU baseline while preserving the same coarse-to-fine output structure.
3. We introduce a candidate-region evaluation view that jointly reports recall, candidate cell count, candidate area, and calibration rather than relying on top-k accuracy alone.
4. We provide full-scale comparisons against a classical Markov regional transition baseline and a Transformer structural ablation, and show that lightweight H3-neighbor refinement creates a recall-versus-calibration trade-off rather than a free gain.

## 2. Related Work

### 2.1 Aircraft trajectory prediction

Learning-based aircraft trajectory prediction has primarily focused on continuous trajectory estimation. Representative examples include attention-enhanced recurrent models for 4D trajectory prediction \cite{jia2022attentionlstm}, TCN-GRU hybrids with attention \cite{ma2024attentiontcngru}, robust autoregressive Transformers for noisy ADS-B observations \cite{li2025noiserobust}, and hybrid deep learning frameworks for sparse satellite observations \cite{shan2025sparse}. These works show that temporal attention and richer sequence encoders are useful in aviation forecasting. At the same time, most of them evaluate pointwise trajectory prediction rather than long-horizon probabilistic regional forecasting.

Our work differs in two ways. First, the output is a distribution over future H3 regions instead of a coordinate sequence. Second, we focus on multi-horizon forecasting over a large operational flight archive rather than on a narrowly defined local airspace scenario. This makes calibration and candidate-region efficiency central evaluation targets.

### 2.2 Multi-horizon forecasting and horizon-aware decoding

The broader forecasting literature has shown that multi-horizon prediction benefits from architectures that do not force all horizons to share one fixed summary vector. Temporal Fusion Transformers explicitly combine local sequential processing with attention for multi-horizon prediction \cite{lim2021tft}. MQTransformer further argues for context-dependent horizon decoding via decoder-encoder attention rather than a single compressed representation \cite{mqtransformer2020}. These ideas are especially relevant for our setting because short-, medium-, and long-horizon aircraft forecasts depend on different portions of trajectory history.

Our model adopts this principle directly: each horizon is represented by a learned embedding and query, and each query attends over the encoded history. Rather than porting a generic time-series forecaster end-to-end, we integrate horizon-aware decoding into the existing hierarchical H3 aviation forecaster.

### 2.3 Calibration and candidate-region forecasting

For safety-relevant or operational forecasting tasks, accuracy alone is insufficient; predictive probabilities should also be calibrated. Guo et al. showed that modern neural networks can be substantially miscalibrated even when they are accurate \cite{guo2017calibration}. This issue is especially important for trajectory forecasting, where a candidate region set may look operationally useful while being probabilistically overconfident or spatially bloated.

Our evaluation therefore reports not only top-k accuracy, but also negative log-likelihood (NLL), Brier score, and expected calibration error (ECE). In addition, we evaluate candidate-region quality using fixed-budget coverage, candidate area, and secondary probability-mass union diagnostics. This lets us distinguish sharper and better-calibrated models from models that achieve high recall only by spreading mass over larger regions.

### 2.4 Open aviation trajectory data and hierarchical spatial indexing

The OpenSky Network has established open flight surveillance as a realistic research substrate \cite{opensky2014,strohmeier2021crowdsourced}. ADRR complements this landscape by providing structured access to detailed commercial flight trajectories and associated network information for research and development \cite{eurocontrol_adrr_2025}. On the spatial side, H3 offers a hierarchical hexagonal indexing system that supports coarse-to-fine reasoning and neighbor traversal at global scale \cite{h3docs2025}. These properties make H3 attractive for probabilistic regional forecasting, where the model should reason over large spatial support while retaining structured locality.

## 3. Problem Formulation

Let a flight history be a sequence of observations

\[
X = \{x_1, x_2, \dots, x_T\},
\]

where each \(x_t\) contains latitude, longitude, altitude, speed, heading, vertical rate, and derived temporal and mask features. In our experiments, the history spans 30 minutes and is resampled at 5-minute intervals, producing a short but regular input sequence with explicit observation and gap indicators.

For each forecast horizon \(h \in \{10, 30, 60, 120\}\) minutes, the goal is to predict:

1. a probability distribution over target H3 cells \(p_h(r \mid X)\),
2. a coarse parent distribution used for hierarchical decoding, and
3. an airborne/landed status distribution.

The target is not a single coordinate but a future **region distribution**. This is motivated by the fact that long-horizon future movement is multimodal even under clean observations, and the uncertainty is amplified further in open operational data.

We use a hierarchical output space. For each horizon, the fine target H3 resolution is paired with a coarser parent resolution. The model first predicts the coarse parent region and then predicts the child cell conditional on that parent. This reduces output complexity and matches the hierarchical structure of H3.

Given predicted probabilities, we evaluate both classification quality and candidate-region quality. The main probabilistic metrics are NLL, Brier score, and ECE. The headline candidate-region metrics are:

- **cover at budget**: recall under a fixed per-horizon cell budget,
- **candidate union area**: the mean physical area covered by the candidate set.

We also retain two secondary diagnostics:

- **candidate union cover**: whether the ground-truth target is contained in the constructed probability-mass candidate set,
- **candidate union cell count**: the mean number of cells in that set.

This evaluation design reflects the real deployment question more directly than top-k accuracy alone: a useful regional predictor should be accurate, calibrated, and spatially efficient, and broad candidate recall should not be rewarded unless it remains compact.

## 4. Method

### 4.1 Data preprocessing and feature construction

The data pipeline starts from one month of ADRR-derived trajectories. Following repository preprocessing, raw tracks are segmented into missions using time-gap rules, resampled to a 5-minute grid, and enriched with both numeric motion features and mask features. The final model consumes:

- 18 numeric features,
- 2 mask features (`observed_flag`, `gap_flag`),
- 9 context features,
- input H3 region tokens and parent H3 region tokens.

The use of explicit mask channels is important even though the current paper does not yet introduce a dedicated uncertainty-aware missing-data model. It preserves information about which sequence steps are directly observed and which arise from gap handling.

### 4.2 Hierarchical H3 forecasting baseline

The baseline model is a hierarchical GRU forecaster. The input sequence is embedded through region embeddings, parent-region embeddings, numeric projections, and mask projections. A GRU encoder produces a sequence summary, which is fused with context features. For each horizon, the model predicts:

1. coarse parent-region logits,
2. airborne/landed status logits,
3. child logits conditioned on the predicted coarse region.

This design already offers a strong and scalable spatial prior, but it compresses all horizons into one shared temporal summary. We found this to be the main temporal bottleneck.

### 4.3 Horizon-query Transformer decoder

To address that bottleneck, we replace the recurrent sequence summarizer with a horizon-aware query decoder. The core idea is that different forecast horizons should attend to different parts of the observed history.

The single-scale Transformer variant works as follows:

1. Project the concatenated sequence features into a Transformer hidden dimension.
2. Add learned positional embeddings and run masked self-attention over the history.
3. Compute a masked mean summary of encoder states and fuse it with context features.
4. Introduce one learned horizon embedding and one learned query per forecast horizon.
5. Use each horizon query to attend over the encoded history sequence.
6. Fuse the shared summary, the horizon-attended representation, and the horizon embedding into a horizon-specific decoder feature.
7. Feed that decoder feature into the same hierarchical H3 output heads.

This design keeps the spatial output interface unchanged while letting each horizon retrieve its own temporal evidence.

### 4.4 Proposed multiscale horizon-query Transformer

Our final model adds an explicit second temporal scale on top of the Transformer horizon decoder. The motivation is simple: short local maneuvers and longer movement trends should not compete for the same attention granularity.

The proposed architecture contains two temporal branches:

1. **Fine-scale branch.** A Transformer encoder processes the original resampled sequence.
2. **Coarse-scale branch.** The fine sequence is pooled with a fixed stride and passed through a shallower Transformer encoder.

For each horizon query \(q_h\), we compute attention over both the fine sequence and the coarse sequence:

\[
\alpha_h^{(f)} = \mathrm{softmax}(q_h K_f^\top), \qquad
\alpha_h^{(c)} = \mathrm{softmax}(q_h K_c^\top),
\]

and obtain attended representations

\[
z_h^{(f)} = \alpha_h^{(f)} V_f, \qquad
z_h^{(c)} = \alpha_h^{(c)} V_c.
\]

These are fused with the shared context feature and the horizon embedding:

\[
d_h = \phi([b; z_h^{(f)}; z_h^{(c)}; e_h]),
\]

where \(b\) is the shared base feature, \(e_h\) is the horizon embedding, and \(\phi(\cdot)\) is a small projection block. The resulting \(d_h\) is then used for coarse-region prediction, status prediction, and coarse-conditioned child prediction.

This multiscale design preserves the main advantage of the single-scale horizon-query decoder while improving access to longer-term temporal structure.

### 4.5 Training objective

The main loss is the average of:

- cross-entropy over coarse H3 classes,
- cross-entropy over valid child classes,
- cross-entropy over airborne/landed status.

We also implemented an optional adjacent-horizon consistency penalty over decoder features, but in our experiments its contribution was not beneficial enough to retain in the final model. Therefore, the final submitted variant uses the tuned multiscale architecture without consistency regularization.

### 4.6 Optional H3-neighbor refinement

We further evaluate an inference-time spatial postprocessor that diffuses a fraction of fine-cell probability mass to local H3 neighbors. This is not part of the final model. Its purpose is to test whether lightweight spatial smoothing can improve candidate-region recall. As shown later, it increases candidate coverage but noticeably harms calibration, so we keep it as an ablation.

## 5. Experimental Setup

### 5.1 Dataset

We evaluate on one month of ADRR-derived commercial flight trajectories from March 2024. According to EUROCONTROL's ADRR description, the repository contains detailed data for commercial flights operating in and over Europe \cite{eurocontrol_adrr_2025}. After mission segmentation, resampling, feature construction, and validity filtering, the resulting dataset contains:

- 14,505,637 training windows,
- 3,108,933 validation windows,
- 3,098,032 test windows,

for a total of **20,712,602** forecast windows. We split by aircraft rather than by window, which reduces leakage from repeated trajectories of the same aircraft across train, validation, and test sets.

### 5.2 Spatial and temporal setup

The input history covers 30 minutes at a 5-minute interval. Forecast horizons are 10, 30, 60, and 120 minutes. We use H3-based discretization with target resolutions \([3,3,2,2]\) and corresponding coarse resolutions \([2,2,1,1]\). This choice keeps shorter horizons spatially finer and longer horizons spatially broader, which reflects the increase in uncertainty with forecast lead time.

### 5.3 Baselines

We compare four representative models:

1. **Markov baseline.** A classical regional transition model trained from empirical transition counts.
2. **Hierarchical GRU baseline.** The original coarse-to-fine H3 forecaster.
3. **Transformer horizon decoder ablation.** A single-scale horizon-query Transformer without the multiscale branch.
4. **Proposed multiscale model.** The final multiscale horizon-query Transformer with the same hierarchical output interface.

This baseline stack is deliberately aligned with the problem setting. Many prior aviation trajectory models target short-term point prediction or terminal-area motion and are therefore not directly comparable to long-horizon H3-region forecasting on large-scale open traffic archives.

### 5.4 Metrics

We report eight headline metrics in the main tables:

- region top-1 / top-3 / top-5 accuracy,
- region NLL,
- region Brier score,
- region ECE with 10 calibration bins,
- cover at fixed cell budget,
- candidate union area.

We additionally retain candidate union cover and candidate union cell count as secondary diagnostics. In practice, status accuracy saturates at 1.0 in all full-scale runs and is therefore not a discriminative metric for model ranking. The main analysis focuses on regional quality, calibration, and budgeted candidate-set utility.

### 5.5 Implementation details

The final model uses a hidden dimension of 256, two Transformer layers in the fine-scale branch, one Transformer layer in the coarse branch, four attention heads, and horizon embeddings of dimension 32. Training uses Adam-style optimization with mixed precision on GPU, batch size 1024, weight decay \(10^{-4}\), early stopping, and a learning-rate scheduler. The final tuned run converges at epoch 182 with the best validation loss of 0.3055.

## 6. Results

### 6.1 Main full-scale comparison

Table 1 reports macro means across the four forecast horizons using the eight headline metrics.

| Model | Top1 | Top3 | Top5 | NLL | Brier | ECE | Cover at Budget | Union Area km^2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Markov baseline | 0.3184 | 0.6166 | 0.7587 | 2.2788 | 0.8111 | 0.0323 | 0.9699 | 545174 |
| Hierarchical GRU | 0.6648 | 0.9076 | 0.9631 | 1.0214 | 0.4578 | 0.0180 | 0.9980 | 128965 |
| Transformer horizon decoder | 0.6744 | 0.9129 | 0.9656 | 0.9946 | 0.4447 | 0.0123 | 0.9981 | 122553 |
| **Proposed multiscale model** | **0.6827** | **0.9174** | **0.9685** | **0.9641** | **0.4338** | **0.0093** | **0.9983** | **118810** |

Two observations stand out. First, all learned hierarchical models strongly outperform the classical Markov baseline. Second, the proposed multiscale model is not merely more accurate; it is also better calibrated and more spatially efficient than the GRU baseline. Fixed-budget coverage is preserved, while candidate area shrinks substantially. The secondary union diagnostics show that the model achieves this with a sharper predictive distribution rather than a looser candidate construction.

### 6.2 Improvements over the hierarchical GRU baseline

Against the hierarchical GRU baseline, the proposed model improves all top-k metrics and all probabilistic metrics:

- top-1: \(0.6648 \rightarrow 0.6827\),
- top-3: \(0.9076 \rightarrow 0.9174\),
- top-5: \(0.9631 \rightarrow 0.9685\),
- NLL: \(1.0214 \rightarrow 0.9641\),
- Brier: \(0.4578 \rightarrow 0.4338\),
- ECE: \(0.0180 \rightarrow 0.0093\).

The gain is consistent across horizons. The 120-minute horizon remains the hardest, but even there the model improves top-1, top-3, top-5, NLL, Brier, and ECE relative to GRU.

### 6.3 Structural ablation: is the multiscale branch necessary?

To isolate the value of explicit multiscale temporal encoding, we compare the final model with a single-scale Transformer horizon decoder that already has horizon-specific queries but no multiscale branch.

| Model | Top1 | Top3 | Top5 | NLL | Brier | ECE | Cover at Budget | Union Area km^2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Transformer horizon decoder | 0.6744 | 0.9129 | 0.9656 | 0.9946 | 0.4447 | 0.0123 | 0.9981 | 122553 |
| **Proposed multiscale model** | **0.6827** | **0.9174** | **0.9685** | **0.9641** | **0.4338** | **0.0093** | **0.9983** | **118810** |

The multiscale branch improves every headline macro metric and also reaches a lower validation loss (0.3055 vs. 0.3177). This supports the claim that explicit dual-scale temporal modeling adds value beyond simply replacing the GRU with a Transformer.

### 6.4 H3-neighbor refinement is a recall-versus-calibration trade-off

We next evaluate H3-neighbor refinement with three diffusion strengths. The refinement improves probability-mass candidate recall, but it is not a free gain.

| Model | Top5 | NLL | Brier | ECE | Cover at Budget | Union Area km^2 | Union Cover |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Final model | 0.9685 | 0.9641 | 0.4338 | 0.0093 | 0.9983 | 118810 | 0.8922 |
| + H3 refine, \(\alpha=0.05\) | 0.9677 | 0.9593 | 0.4344 | 0.0310 | 0.9987 | 127062 | 0.9052 |
| + H3 refine, \(\alpha=0.10\) | 0.9673 | 0.9796 | 0.4372 | 0.0585 | 0.9987 | 137220 | 0.9192 |
| + H3 refine, \(\alpha=0.20\) | 0.9662 | 1.0340 | 0.4493 | 0.1135 | 0.9987 | 167889 | 0.9496 |

The refinement increases the secondary union-cover diagnostic, but it quickly degrades calibration and eventually worsens top-5 and NLL. Therefore, it should not be presented as the final model. Its value lies in exposing a useful operational trade-off: local spatial smoothing can improve recall, but only by sacrificing confidence quality and spatial compactness.

### 6.5 Reachability filtering has a small aggregate effect

Turning reachability filtering off produces only very small aggregate metric changes. This suggests that, in the present setup, reachability filtering acts more as a plausibility safeguard than as the primary source of predictive gains.

## 7. Discussion

The main result of this study is not simply that a Transformer beats a GRU. The more important finding is that **multi-horizon regional aircraft forecasting benefits from horizon-specific temporal retrieval and explicit multiscale temporal structure, while still requiring calibration-aware evaluation**. A model can improve candidate recall by diffusing probability mass more broadly, but such gains may come at the cost of confidence reliability. This is exactly what the H3 refinement ablation reveals.

The regional formulation is also important. On open operational data, especially at 60- to 120-minute horizons, pointwise long-range trajectory regression can create a false sense of precision. The proposed H3 formulation makes uncertainty explicit, supports hierarchical reasoning, and allows downstream users to control the trade-off between region count, covered area, and recall.

At the same time, the current paper has clear limitations.

First, although the overall framework is spatially general, the present experiments use Europe-wide commercial traffic from ADRR rather than a truly global dataset. The title and claims of the final paper should reflect this empirical scope precisely.

Second, the current model does not yet incorporate dedicated missing-data uncertainty modeling. The preprocessing pipeline preserves gap indicators, but the forecasting model itself is not yet explicitly uncertainty-aware in the sense of GRU-D, continuous-time latent dynamics, or conformal regional prediction. This is an important next step rather than a hidden contribution.

Third, external aviation baselines remain limited by task mismatch. Much of the existing literature addresses short-term coordinate forecasting, terminal-area interaction, or dense contextual inputs unavailable in ADRR-derived artifacts. We therefore use a representative hierarchy of baselines that are faithful to the current task: Markov, hierarchical GRU, Transformer horizon decoder, and the proposed multiscale model.

These limitations do not undermine the current contribution. Instead, they define a clear next-stage research agenda: gap-aware uncertainty modeling, calibration-aware post-processing, and broader cross-dataset validation.

## 8. Conclusion

We presented a large-scale study of multi-horizon regional aircraft trajectory forecasting on ADRR-derived open aviation data. Starting from a hierarchical H3 GRU baseline, we introduced a horizon-query Transformer decoder and extended it with an explicit multiscale temporal branch. The final model improves top-k accuracy, NLL, Brier score, and calibration relative to both the GRU baseline and a single-scale Transformer ablation, while also producing smaller candidate region sets. A classical Markov baseline is far less competitive, which confirms that the task requires richer temporal representation learning than simple empirical transition modeling can provide.

The H3 refinement study further shows that candidate recall and calibration should be reported together. A model that expands candidate regions aggressively can appear stronger under recall-only analysis while becoming less trustworthy probabilistically. We therefore argue that large-scale regional aircraft forecasting should be evaluated as a joint problem of accuracy, uncertainty quality, and spatial efficiency.

Overall, the results support hierarchical H3 forecasting with multiscale horizon-query temporal encoding as a strong baseline for long-horizon regional aviation prediction on open trajectory archives. They also provide a stable foundation for the next methodological step: explicitly uncertainty-aware forecasting under missing and irregular flight observations.

## Data Availability, Code Availability, and Reproducibility Statement

The experiments use one month of flight trajectories derived from EUROCONTROL's Aviation Data Repository for Research (ADRR) \cite{eurocontrol_adrr_2025}. ADRR access is provided by EUROCONTROL for research and development use under its terms of use, but the raw data cannot be publicly redistributed by us. To support reproducibility, the project repository contains:

- the end-to-end preprocessing script used to standardize ADRR data,
- configuration files for all reported runs,
- model implementations for all compared methods,
- experiment logs and evaluation summaries,
- scripts for rebuilding development subsets and rerunning full experiments.

In a public release, we recommend publishing code, configuration files, model checkpoints where permitted, and exact preprocessing instructions, while directing readers to EUROCONTROL for raw-data access.

## Notes for the Next Revision

This draft is intentionally evidence-bounded. The following items should be added in the next revision:

1. A system figure showing the multiscale horizon-query H3 architecture.
2. A calibration or reliability figure comparing GRU, Transformer ablation, and the final model.
3. A candidate-area versus candidate-cover trade-off figure for the H3 refinement sweep.
4. A compact appendix table listing per-horizon results for all main runs.
5. A venue-specific rewrite once the target conference or journal is fixed.
