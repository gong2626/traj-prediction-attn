# Q3 完整规模路由训练消融方案

## 目标

本方案用于补齐中文主稿 Q3：在完整规模 ADRR 数据、相同初始化和相同评价协议下，判断 `horizon-wise distillation` 与 `loss-aware routing regularization` 是否分别改善 Horizon-Router。

主文暂不使用开发子集诊断结果。三组实验完成前，`sections_zh/06_results.tex` 的 Q3 保持占位，不插入表格和数值结论。

## 共同控制条件

- 数据：`data/adrr_202403_shards`，`aircraft` 级 train/val/test 划分。
- 任务协议：30 分钟历史，预测 10/30/60/120 分钟，目标 H3 分辨率 `[3, 3, 2, 2]`，粗层分辨率 `[2, 2, 1, 1]`。
- 初始化：全部从 `artifacts/checkpoints/global_multiscale_transformer_h3_tuned.pt` warm start。
- 模型结构：全部使用 `horizon_router_transformer_h3`，保持三 expert、router 初始化、输出头和 reachability filter 一致。
- 训练预算：batch size 4096，学习率 `2e-4`，weight decay `1e-4`，最多 1000 epoch，最少 8 epoch，early stopping patience 8，seed 42。
- 主表指标：只报告 `Top1 / Top5 / NLL / Brier / ECE` 的四 horizon 宏平均。

## 三组实验

### A. Router Only

- 配置：`configs/region_forecast_adrr_full_horizon_router_only.yaml`
- run name：`global_horizon_router_transformer_h3_router_only`
- 目的：只测试三 expert + horizon-level router 结构本身，不加入 teacher distillation 和路由正则。
- 关键开关：
  - `teacher_blend_report_path: null`
  - `teacher_distillation_weight: 0.0`
  - `teacher_status_distillation_weight: 0.0`
  - `router_supervision_weight: 0.0`
  - `router_load_balancing_weight: 0.0`
  - `expert_auxiliary_weight: 0.0`

### B. Router + Horizon-Wise Distillation

- 配置：`configs/region_forecast_adrr_full_horizon_router_distill.yaml`
- run name：`global_horizon_router_transformer_h3_distill`
- 目的：在 A 的基础上只加入教师分布蒸馏，隔离蒸馏对概率质量和排序指标的贡献。
- 关键开关：
  - `teacher_blend_report_path: "artifacts/evaluations/global_horizon_expert_blend_full_v1_report.json"`
  - `teacher_distillation_weight: 0.08`
  - `teacher_status_distillation_weight: 0.01`
  - `router_supervision_weight: 0.0`
  - `router_load_balancing_weight: 0.0`
  - `expert_auxiliary_weight: 0.0`

### C. Router + Distillation + Loss-Aware Routing Regularization

- 配置：`configs/region_forecast_adrr_full_horizon_router_distill_routerreg.yaml`
- run name：`global_horizon_router_transformer_h3_distill_routerreg`
- 目的：最终模型，测试在蒸馏基础上加入 loss-aware router supervision、load balancing 和 expert auxiliary loss 后是否继续改善。
- 关键开关：
  - `teacher_distillation_weight: 0.08`
  - `teacher_status_distillation_weight: 0.01`
  - `router_supervision_weight: 0.08`
  - `router_supervision_temperature: 0.7`
  - `router_load_balancing_weight: 0.02`
  - `expert_auxiliary_weight: 0.05`

如果现有 C 组结果确实来自当前配置、当前数据划分和当前评估脚本，可以直接复用；否则建议三组按同一代码版本重新跑一遍。

## 建议运行命令

在 AutoDL 上进入项目根目录后，先确认数据集 artifact 和教师报告存在：

```bash
ls artifacts/datasets/train.pt artifacts/datasets/val.pt artifacts/datasets/test.pt
ls artifacts/checkpoints/global_multiscale_transformer_h3_tuned.pt
ls artifacts/evaluations/global_horizon_expert_blend_full_v1_report.json
```

如果 `artifacts/datasets/*.pt` 不存在，先构建一次数据集：

```bash
python build_dataset.py --config configs/region_forecast_adrr_full_horizon_router_distill_routerreg.yaml --workers 10
```

然后依次运行三组实验：

```bash
python training.py --config configs/region_forecast_adrr_full_horizon_router_only.yaml
python evaluation.py --config configs/region_forecast_adrr_full_horizon_router_only.yaml

python training.py --config configs/region_forecast_adrr_full_horizon_router_distill.yaml
python evaluation.py --config configs/region_forecast_adrr_full_horizon_router_distill.yaml

python training.py --config configs/region_forecast_adrr_full_horizon_router_distill_routerreg.yaml
python evaluation.py --config configs/region_forecast_adrr_full_horizon_router_distill_routerreg.yaml
```

如果某组已经有旧 checkpoint，但需要从相同初始化重新开始，训练命令加 `--restart`。

## 需要回收的输出

每组至少回收以下文件：

- `artifacts/checkpoints/<run_name>_train_summary.json`
- `artifacts/evaluations/<run_name>_summary.csv`
- `artifacts/evaluations/<run_name>_summary.json`

补写 Q3 时使用 `summary.csv` 中的宏平均 `Top1 / Top5 / NLL / Brier / ECE`。建议 Q3 表格只保留三行，不再报告开发子集、训练 epoch 或吞吐细节。

## Q3 写作判据

- A 到 B 的变化用于判断 `horizon-wise distillation` 是否带来稳定收益。
- B 到 C 的变化用于判断 `loss-aware routing regularization` 是否在蒸馏之外继续改善。
- 若 C 只改善部分指标，正文应写成权衡而不是单向提升。
- 若 A/B/C 差异很小，主文应把训练机制降级为稳定训练设计，避免写成强因果贡献。
