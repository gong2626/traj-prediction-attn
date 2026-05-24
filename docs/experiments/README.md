# Experiments Index

本文档用于区分论文主线结果、消融结果、快速开发子集结果和早期探索记录，避免后续写作和答辩时把过程性材料与最终结论混在一起。

## 1. 论文主线结果

以下文档和结果文件构成当前论文主线：

- [baseline.md](/root/autodl-tmp/docs/experiments/baseline.md)
- [full_scale_results_round6.md](/root/autodl-tmp/docs/experiments/full_scale_results_round6.md)
- [final_result_manifest.md](/root/autodl-tmp/docs/experiments/final_result_manifest.md)

对应的核心 full-scale 评估结果：

- `artifacts/evaluations/global_markov_baseline_summary.csv`
- `artifacts/evaluations/global_h3_movement_gru_summary.csv`
- `artifacts/evaluations/global_transformer_horizon_decoder_ablation_summary.csv`
- `artifacts/evaluations/global_multiscale_transformer_h3_tuned_summary.csv`
- `artifacts/evaluations/global_traisformer_h3_baseline_summary.csv`
- `artifacts/evaluations/global_flightbertpp_h3_baseline_summary.csv`

## 2. 论文消融与分析

这些结果用于支撑后处理、候选区域和结构差异分析，不作为主模型替代品：

- [candidate_metrics_upgrade.md](/root/autodl-tmp/docs/experiments/candidate_metrics_upgrade.md)
- [h3_candidate_refinement.md](/root/autodl-tmp/docs/experiments/h3_candidate_refinement.md)
- [eval_only_decode_sweep.md](/root/autodl-tmp/docs/experiments/eval_only_decode_sweep.md)

对应的关键评估产物：

- `artifacts/evaluations/global_multiscale_transformer_h3_tuned_no_reachability_summary.csv`
- `artifacts/evaluations/global_multiscale_transformer_h3_tuned_h3_refine_a005_summary.csv`
- `artifacts/evaluations/global_multiscale_transformer_h3_tuned_h3_refine_a010_summary.csv`
- `artifacts/evaluations/global_multiscale_transformer_h3_tuned_h3_refine_a020_summary.csv`
- `artifacts/evaluations/global_multiscale_transformer_h3_tuned_h3_refine_a010_no_reachability_summary.csv`

## 3. 开发子集与快速代理实验

这些结果用于低成本验证趋势和快速调试，不应直接替代 full-scale 结论：

- [devsplit_subset_baseline.md](/root/autodl-tmp/docs/experiments/devsplit_subset_baseline.md)

相关目录：

- `artifacts/research_devsplit/`

## 4. 早期结构探索记录

以下文档主要保存研究演化轨迹，保留以便回溯，但通常不需要在论文正文中逐项引用：

- [attention_hierarchical_gru_round1.md](/root/autodl-tmp/docs/experiments/attention_hierarchical_gru_round1.md)
- [horizon_decoder_gru_round2.md](/root/autodl-tmp/docs/experiments/horizon_decoder_gru_round2.md)
- [transformer_horizon_decoder_round3.md](/root/autodl-tmp/docs/experiments/transformer_horizon_decoder_round3.md)
- [multiscale_transformer_round4.md](/root/autodl-tmp/docs/experiments/multiscale_transformer_round4.md)
- [multiscale_transformer_tuning_round5.md](/root/autodl-tmp/docs/experiments/multiscale_transformer_tuning_round5.md)

## 5. 当前建议

若目标是继续论文收尾，优先顺序应为：

1. 只引用 `final_result_manifest.md` 中列出的正式结果；
2. 结构消融只保留 single-scale vs multiscale；
3. H3 refinement 只保留为召回与校准权衡分析；
4. eval-only decode sweep 仅在其确实优于现有正式主结果时，才升级为论文结果。
