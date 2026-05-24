# Final Result Manifest

本清单定义当前项目中应被视为“正式论文结果”的实验集合。后续论文表格、图和结论应默认从这些文件中取数。

## 主结果模型

### 1. Markov baseline

- Run name: `global_markov_baseline`
- Summary: [global_markov_baseline_summary.csv](/root/autodl-tmp/artifacts/evaluations/global_markov_baseline_summary.csv)

### 2. Hierarchical GRU baseline

- Run name: `global_h3_movement_gru`
- Summary: [global_h3_movement_gru_summary.csv](/root/autodl-tmp/artifacts/evaluations/global_h3_movement_gru_summary.csv)

### 3. Transformer horizon decoder ablation

- Run name: `global_transformer_horizon_decoder_ablation`
- Summary: [global_transformer_horizon_decoder_ablation_summary.csv](/root/autodl-tmp/artifacts/evaluations/global_transformer_horizon_decoder_ablation_summary.csv)

### 4. Proposed final model

- Run name: `global_multiscale_transformer_h3_tuned`
- Summary: [global_multiscale_transformer_h3_tuned_summary.csv](/root/autodl-tmp/artifacts/evaluations/global_multiscale_transformer_h3_tuned_summary.csv)

### 5. External adapted baseline: TrAISformer-H3

- Run name: `global_traisformer_h3_baseline`
- Summary: [global_traisformer_h3_baseline_summary.csv](/root/autodl-tmp/artifacts/evaluations/global_traisformer_h3_baseline_summary.csv)

### 6. External adapted baseline: FlightBERT++-H3

- Run name: `global_flightbertpp_h3_baseline`
- Summary: [global_flightbertpp_h3_baseline_summary.csv](/root/autodl-tmp/artifacts/evaluations/global_flightbertpp_h3_baseline_summary.csv)

## 消融与后处理分析

### Reachability ablation

- [global_multiscale_transformer_h3_tuned_no_reachability_summary.csv](/root/autodl-tmp/artifacts/evaluations/global_multiscale_transformer_h3_tuned_no_reachability_summary.csv)

### H3 refinement sweep

- [global_multiscale_transformer_h3_tuned_h3_refine_a005_summary.csv](/root/autodl-tmp/artifacts/evaluations/global_multiscale_transformer_h3_tuned_h3_refine_a005_summary.csv)
- [global_multiscale_transformer_h3_tuned_h3_refine_a010_summary.csv](/root/autodl-tmp/artifacts/evaluations/global_multiscale_transformer_h3_tuned_h3_refine_a010_summary.csv)
- [global_multiscale_transformer_h3_tuned_h3_refine_a020_summary.csv](/root/autodl-tmp/artifacts/evaluations/global_multiscale_transformer_h3_tuned_h3_refine_a020_summary.csv)
- [global_multiscale_transformer_h3_tuned_h3_refine_a010_no_reachability_summary.csv](/root/autodl-tmp/artifacts/evaluations/global_multiscale_transformer_h3_tuned_h3_refine_a010_no_reachability_summary.csv)

## 当前不纳入正式结果的内容

以下内容保留作为研究记录，但默认不进入论文主表：

- early rounds (`round1` 到 `round5`) 的过程性文档
- devsplit 子集上的绝对指标
- evaluation-only decode sweep 结果

它们的用途分别是：

- 说明研究演化路径；
- 进行快速趋势验证；
- 判断是否值得进一步 full-scale 重评估。
