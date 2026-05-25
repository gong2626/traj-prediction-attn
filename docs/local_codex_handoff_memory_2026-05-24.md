# Local Codex Handoff Memory

更新时间：2026-05-24 UTC

这个文件用于把当前航空轨迹多时间尺度区域预测项目，从 autoDL 环境平移到本地 Windows + Codex 新会话时，作为高信号项目记忆直接读取。

## 1. 当前项目主线

当前不要从零开始分析，也不要优先写论文、整理表格或继续做 decode-only sweep。

当前最重要的事是：

- 继续改进模型；
- 重点解决短时域 `10/30 min` 的排序能力；
- 尽量不破坏当前模型在 `120 min` 上的优势；
- 所有新改动都必须 config 控制；
- 保留旧 baseline、旧主模型和旧 checkpoint 兼容性。

## 2. 当前任务定义

- 输入：`30 min` 历史轨迹
- 采样间隔：`5 min`
- 输入步数：`6`
- 输出：未来 `10 / 30 / 60 / 120 min` 的 H3 区域概率

当前正式空间协议：

- `input h3_resolution = 3`
- `input_parent_resolution = 2`
- `target_h3_resolutions = [3, 3, 2, 2]`
- `coarse_h3_resolutions = [2, 2, 1, 1]`

当前不建议贸然改更细网格，比如升到 `h3_resolution = 4`。

## 3. 数据与运行背景

autoDL 环境信息：

- 项目目录：`/root/autodl-tmp`
- Ubuntu 22.04
- Python 3.12
- PyTorch 2.8.0
- CUDA 12.8
- 1 张 32GB GPU
- 16 核 CPU
- 80GB RAM

数据情况：

- 原始数据：`ADRR 2024-03` 一个月全量
- 已构建 full 数据集
- 当前无需从原始 parquet 重新开始

耗时量级：

- full 训练一次通常十几个小时
- full 评估一次约 `2.5 ~ 3` 小时

## 4. 正式结果与主结果来源

正式结果以：

- [docs/experiments/final_result_manifest.md](/root/autodl-tmp/docs/experiments/final_result_manifest.md)

为准。

核心 full run：

- `global_markov_baseline`
- `global_h3_movement_gru`
- `global_transformer_horizon_decoder_ablation`
- `global_multiscale_transformer_h3_tuned`
- `global_traisformer_h3_baseline`
- `global_flightbertpp_h3_baseline`

对应 full 评估文件位于：

- `artifacts/evaluations/*.csv`

## 5. 当前正式主模型

当前正式主模型：

- `run_name: global_multiscale_transformer_h3_tuned`
- `model_type: multiscale_transformer_h3`

核心代码：

- [regional_forecast/models/forecast_model.py](/root/autodl-tmp/regional_forecast/models/forecast_model.py)
- [regional_forecast/pipeline/train.py](/root/autodl-tmp/regional_forecast/pipeline/train.py)
- [regional_forecast/pipeline/evaluate.py](/root/autodl-tmp/regional_forecast/pipeline/evaluate.py)
- [regional_forecast/pipeline/predict.py](/root/autodl-tmp/regional_forecast/pipeline/predict.py)

结构要点：

- 统一 hierarchical coarse-to-fine H3 输出头
- 单尺度 horizon-query Transformer 主干
- 额外 coarse temporal branch
- 推理时可选 reachability filter
- H3 neighbor refinement 只作为消融，不是最终方法

## 6. 当前正式主结果

`global_multiscale_transformer_h3_tuned` 的 full macro mean：

- Top1: `0.6827`
- Top3: `0.9174`
- Top5: `0.9685`
- NLL: `0.9641`
- Brier: `0.4338`
- ECE: `0.0093`
- Union Cover: `0.8922`
- Union Cells: `2.1829`
- Union Area: `118810 km²`

相对项目内部 GRU baseline `global_h3_movement_gru`：

- Top1: `0.6648 -> 0.6827`
- Top3: `0.9076 -> 0.9174`
- Top5: `0.9631 -> 0.9685`
- NLL: `1.0214 -> 0.9641`
- Brier: `0.4578 -> 0.4338`
- ECE: `0.0180 -> 0.0093`

所以主模型相对内部 baseline 是稳定提升的。

## 7. 当前最强 external baseline

当前最强 external baseline：

- `global_traisformer_h3_baseline`

full macro mean：

- Top1: `0.6803`
- Top3: `0.9178`
- Top5: `0.9697`
- NLL: `0.9556`
- Brier: `0.4361`
- ECE: `0.0094`
- Union Cover: `0.8939`
- Union Cells: `2.2297`
- Union Area: `122810 km²`

当前局面：

- TrAISformer-H3 在 `Top3 / Top5 / NLL / Union Cover` 上略强
- 当前主模型在 `Top1 / Brier / 候选集合紧致性` 上更好
- 当前主模型在 `120 min` 上优于 TrAISformer-H3

主矛盾不是“当前模型完全不行”，而是：

- 表一里 TrAISformer-H3 看起来更像赢家
- 但当前主模型其实在长时域与概率质量上更均衡

## 8. 当前真正的研究主矛盾

当前主模型的问题不是长时域，而是：

- `10 / 30 min` 的短时域排序指标略输给 TrAISformer-H3
- 特别是 `Top3 / Top5 / NLL`
- 当前多尺度结构在长时域有价值
- 但短时局部顺序建模还不够强

所以当前下一轮模型改进必须优先打这个点。

## 9. 已做过且不建议继续纠缠的方向

### 9.1 H3 refinement

已经在 full 上做过：

- `a005`
- `a010`
- `a020`

结论：

- 会提升 coverage
- 但明显破坏 calibration
- 不应作为最终方法

### 9.2 no-reachability ablation

已经做过。

结论：

- 它是分析性实验
- 不是主要提升来源

### 9.3 decode-only sweep

已经做过：

- parent top-k
- temperature scaling

结果文件：

- `global_multiscale_eval_parentk_temp_base_base_summary.csv`
- `global_multiscale_eval_parentk_temp_base_parentk_short_up_summary.csv`
- `global_multiscale_eval_parentk_temp_base_temp_sharp_short_summary.csv`
- `global_multiscale_eval_parentk_temp_base_parentk_up_temp_sharp_summary.csv`

结论：

- `parentk_short_up` 只有极微弱正向收益
- temperature sharpening 更像候选更紧但 coverage 更差
- decode-only tuning 不足以翻盘

不要继续在这上面花太多时间。

## 10. 关键文档

建议优先读：

- [docs/research_project_audit.md](/root/autodl-tmp/docs/research_project_audit.md)
- [docs/research_plan.md](/root/autodl-tmp/docs/research_plan.md)
- [docs/experiments/final_result_manifest.md](/root/autodl-tmp/docs/experiments/final_result_manifest.md)
- [docs/experiments/README.md](/root/autodl-tmp/docs/experiments/README.md)

## 11. 当前代码结构上最关键的对照对象

主模型：

- [regional_forecast/models/forecast_model.py](/root/autodl-tmp/regional_forecast/models/forecast_model.py)
  - `MultiscaleTransformerH3RegionalForecaster`

TrAISformer-H3 适配 baseline：

- [regional_forecast/external_baselines/traisformer_h3.py](/root/autodl-tmp/regional_forecast/external_baselines/traisformer_h3.py)

公共外部 baseline 组件：

- [regional_forecast/external_baselines/common.py](/root/autodl-tmp/regional_forecast/external_baselines/common.py)

## 12. 最近一轮真正实现过的结构改造

已经实现过一个新的实验变体：

- `model_type: multiscale_transformer_h3_local_causal`

核心类：

- [regional_forecast/models/forecast_model.py](/root/autodl-tmp/regional_forecast/models/forecast_model.py:970)

它做的事：

- 保留原 `multiscale_transformer_h3` 的 fine branch
- 保留 coarse multiscale branch
- 额外增加一个 recent local causal branch
- 默认只看最近 `4` 个历史步
- 默认只作用到前 `2` 个 horizon，也就是 `10 / 30 min`
- 再把 `base + fine + coarse + local + horizon_embed` 融合成 decoder feature

新配置字段：

- [regional_forecast/config.py](/root/autodl-tmp/regional_forecast/config.py:105)

相关配置文件：

- [configs/region_forecast_adrr_devsplit_multiscale_local_causal_smoke.yaml](/root/autodl-tmp/configs/region_forecast_adrr_devsplit_multiscale_local_causal_smoke.yaml)
- [configs/region_forecast_adrr_devsplit_multiscale_local_causal.yaml](/root/autodl-tmp/configs/region_forecast_adrr_devsplit_multiscale_local_causal.yaml)
- [configs/region_forecast_adrr_full_multiscale_local_causal.yaml](/root/autodl-tmp/configs/region_forecast_adrr_full_multiscale_local_causal.yaml)

训练/评估/预测入口已经支持这个 model_type：

- [regional_forecast/pipeline/train.py](/root/autodl-tmp/regional_forecast/pipeline/train.py)
- [regional_forecast/pipeline/evaluate.py](/root/autodl-tmp/regional_forecast/pipeline/evaluate.py)
- [regional_forecast/pipeline/predict.py](/root/autodl-tmp/regional_forecast/pipeline/predict.py)

## 13. 最近一轮实验结果：结论是负向

已经跑完：

1. `python training.py --config configs/region_forecast_adrr_devsplit_multiscale_local_causal_smoke.yaml`
2. `python evaluation.py --config configs/region_forecast_adrr_devsplit_multiscale_local_causal_smoke.yaml`
3. `python training.py --config configs/region_forecast_adrr_devsplit_multiscale_local_causal.yaml`
4. `python evaluation.py --config configs/region_forecast_adrr_devsplit_multiscale_local_causal.yaml`

主要对照：

- 新模型：`adrr_devsplit_multiscale_transformer_h3_local_causal`
- 旧主线：`adrr_devsplit_multiscale_transformer_h3_tuned`

对应结果：

- [adrr_devsplit_multiscale_transformer_h3_local_causal_summary.json](/root/autodl-tmp/artifacts/research_devsplit/evaluations/adrr_devsplit_multiscale_transformer_h3_local_causal_summary.json)
- [adrr_devsplit_multiscale_transformer_h3_tuned_summary.json](/root/autodl-tmp/artifacts/research_devsplit/evaluations/adrr_devsplit_multiscale_transformer_h3_tuned_summary.json)

### 13.1 devsplit 逐时域差值，`new - base`

#### 10 min

- Top1: `-0.0260`
- Top3: `-0.0280`
- Top5: `-0.0350`
- NLL: `+0.1564`

#### 30 min

- Top1: `-0.0069`
- Top3: `-0.0069`
- Top5: `+0.0006`
- NLL: `+0.1021`

#### 60 min

- Top1: `-0.0110`
- Top3: `+0.0044`
- Top5: `-0.0015`
- NLL: `-0.0213`

#### 120 min

- Top1: `-0.0183`
- Top3: `-0.0065`
- Top5: `-0.0022`
- NLL: `-0.0518`

### 13.2 关键短时平均结果

短时平均，只看 `10 / 30 min`：

- Top3: `-0.0175`
- Top5: `-0.0172`
- NLL: `+0.1293`

这是核心结论。

也就是说，这版改造没有解决短时排序弱的问题，反而把最关键的 `10 / 30 min` 排序能力拉低了。

### 13.3 devsplit macro mean 对比

旧主线 `adrr_devsplit_multiscale_transformer_h3_tuned`：

- Top1: `0.372246`
- Top3: `0.608247`
- Top5: `0.723538`
- NLL: `2.449719`
- Brier: `0.778300`
- ECE: `0.056151`
- Union Cover: `0.806321`
- Union Cells: `7.165287`
- Union Area: `323031.091925`

新模型 `adrr_devsplit_multiscale_transformer_h3_local_causal`：

- Top1: `0.356689`
- Top3: `0.599000`
- Top5: `0.714027`
- NLL: `2.496075`
- Brier: `0.786561`
- ECE: `0.057099`
- Union Cover: `0.806398`
- Union Cells: `7.376979`
- Union Area: `339494.121532`

结论：

- macro 基本全面退步
- `candidate_union_cover` 几乎没变
- 但 `union cells` 和 `union area` 变大
- 说明这版不是“更保守换更高 recall”
- 而是“更松，但排序更差”

### 13.4 训练侧信号

训练摘要：

- [adrr_devsplit_multiscale_transformer_h3_local_causal_train_summary.json](/root/autodl-tmp/artifacts/research_devsplit/checkpoints/adrr_devsplit_multiscale_transformer_h3_local_causal_train_summary.json)
- [adrr_devsplit_multiscale_transformer_h3_tuned_train_summary.json](/root/autodl-tmp/artifacts/research_devsplit/checkpoints/adrr_devsplit_multiscale_transformer_h3_tuned_train_summary.json)

best val loss：

- tuned: `0.7688`
- local_causal: `0.7849`

训练侧也没有给出正向信号。

### 13.5 对这轮改造的结论

这版 `multiscale_transformer_h3_local_causal`：

- 可以保留为负结果研究记录
- 不建议继续调参
- 不建议上 full

## 14. 当前最推荐的下一步

当前仍然建议围绕“短时局部模式建模增强”继续做，但不要复制上一轮的硬拼接做法。

最推荐的下一步是更窄、更稳的单点改造：

### 方案 A：gated residual local fusion

思路：

- 仍保留 local causal branch
- 但不要直接 concat 后强行混入
- 只对 `10 / 30 min` 的 decoder path 做 gated residual 注入
- gate 初始偏向关闭，让模型先退化回当前 tuned 主模型，再学习是否打开 local 信息

为什么这个方向比上一轮更合理：

- 上一轮问题很可能不是“local 信息无价值”
- 而是“注入方式太硬，先破坏了已学到的 fine/coarse 排序”
- gated residual 更适合作为保守增量改造

### 方案 B：short-horizon auxiliary decoder

思路：

- 保持主 decoder 不动
- 为 `10 / 30 min` 加一个更轻量的 local short-term auxiliary branch
- 只在训练时提供额外监督或额外 residual

但相比方案 A，当前更推荐先做 A。

## 15. 当前明确不建议优先做的事

不要优先做：

- 更细 H3 网格
- 继续 decode-only sweep
- 大量后处理堆叠
- 先整理论文结果或润色论文

## 16. 新会话接手时建议直接做的事

新会话中，建议新 Codex 直接按下面顺序接手：

1. 先阅读本文件；
2. 阅读：
   - `docs/research_project_audit.md`
   - `docs/research_plan.md`
   - `docs/experiments/final_result_manifest.md`
3. 阅读主模型和 TrAISformer-H3 代码；
4. 明确上一轮 `multiscale_transformer_h3_local_causal` 是负结果；
5. 基于当前 tuned 主模型，优先实现一个更保守的 short-term gated residual 改造；
6. 仍然遵守：
   - 保留旧模型加载兼容；
   - 新模型必须 config 控制；
   - 先 smoke/devsplit，再决定是否 full；
   - 每次只引入一个明确改造点；
   - 所有新增实验必须独立 run_name。

## 17. 一句话交接结论

当前项目不是缺一个“更大更新”，而是缺一个真正能改善 `10 / 30 min` 排序、同时不破坏 `120 min` 优势的保守结构增量；最近实现的 `multiscale_transformer_h3_local_causal` 已经在 devsplit 上验证为负结果，不要继续往 full 推。
