# Evaluation-Only Decode Sweep

## 目的

这一轮实验不重新训练模型，而是在固定 checkpoint `global_multiscale_transformer_h3_tuned` 的前提下，仅调整层级解码阶段的两个因素：

1. `hierarchical_parent_top_k`
2. 评估期 `coarse` / `child` temperature scaling

目标是检查：在不改变训练过程的前提下，是否能把最终模型在 top-5、NLL 或校准指标上再向强基线靠近一些。

## 实现

相关代码与脚本：

- [regional_forecast/config.py](/root/autodl-tmp/regional_forecast/config.py)
- [regional_forecast/pipeline/evaluate.py](/root/autodl-tmp/regional_forecast/pipeline/evaluate.py)
- [tools/run_multiscale_eval_sweeps.py](/root/autodl-tmp/tools/run_multiscale_eval_sweeps.py)
- [tools/run_multiscale_eval_sweeps.sh](/root/autodl-tmp/tools/run_multiscale_eval_sweeps.sh)

新增的评估配置项：

- `hierarchical_coarse_temperature`
- `hierarchical_child_temperature`

两者都只在评估期 softmax 前作用于 logits，不影响训练 checkpoint。

## Devsplit sweep

结果汇总：

- [devsplit_multiscale_eval_sweep_results.md](/root/autodl-tmp/artifacts/research_devsplit/evaluations/sweeps/devsplit_multiscale_eval_sweep_results.md)
- [devsplit_multiscale_eval_sweep_results.json](/root/autodl-tmp/artifacts/research_devsplit/evaluations/sweeps/devsplit_multiscale_eval_sweep_results.json)

测试的主要组合包括：

- base
- `parentk_short_up`
- `parentk_all_up`
- `temp_sharp_short`
- `temp_smooth_short`
- `parentk_up_temp_sharp`

## 当前结论

### 1. 增大 parent top-k 是最稳妥的方向

`parentk_all_up = [96,72,64,16]` 在 devsplit 上带来：

- 略高的 top-5
- 更低的 NLL
- 更低的 Brier
- 更低的 ECE
- 更小的候选面积

同时几乎不损失 union cover。

### 2. 短时域 sharpening temperature 更像权衡工具

将短时域 temperature 调到 0.95 后：

- Brier 和 ECE 会改善
- 候选集合会更紧
- 但 union cover 会下降

因此，这一类设置更适合作为“校准与召回之间的权衡控制”，而不是默认主结果设置。

## 对 full-scale 的建议

若要继续 full-scale 无重训验证，优先级建议为：

1. `base`
2. `parentk_all_up`

不建议一开始就扫完整 temperature 组合，因为从 devsplit 看，它们不像 parent top-k 那样稳定收益明确。
