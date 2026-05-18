# 区域级飞机动向预测使用说明

## 1. 项目用途
本项目用于基于历史 ADS-B 轨迹做**区域级飞机动向预测**，输出未来 `30/60/120/300` 分钟的：

- 候选出现区域
- 每个区域的置信度
- `airborne / landed` 状态概率

当前主模型是 **H3 分层区域预测模型**，支持：

- 全局模型训练
- 按机型训练专属模型
- 批量按机型训练
- 预测时按机型路由模型，找不到时回退到全局模型

顶层入口脚本：

- [build_dataset.py](/e:/traj2/build_dataset.py:1)
- [training.py](/e:/traj2/training.py:1)
- [evaluation.py](/e:/traj2/evaluation.py:1)
- [train_catalog.py](/e:/traj2/train_catalog.py:1)
- [predict.py](/e:/traj2/predict.py:1)

## 2. 环境准备
```bash
pip install -r requirements.txt
```

如果需要运行 `tools/render_*` 之类的可视化脚本，再执行：

```bash
pip install -r requirements-tools.txt
```

Windows 到华为云容器的完整操作步骤见 [docs/windows_deployment_guide.md](docs/windows_deployment_guide.md)。

核心依赖：

- `torch`
- `pandas`
- `numpy`
- `h3`
- `pyyaml`

可视化工具依赖：

- `folium`（仅在线地图模式需要；离线模式不需要）

## 3. 配置文件
常用配置：

- 真实/工程数据：[config/region_forecast.yaml](/e:/traj2/config/region_forecast.yaml:1)
- 模拟数据：[config/region_forecast_synthetic.yaml](/e:/traj2/config/region_forecast_synthetic.yaml:1)

最需要确认的字段：

```yaml
columns:
  aircraft_id: "target_id"
  trajectory_id: "target_id"
  time: "timestamp"
  lat: "latitude"
  lon: "longitude"
  altitude: "altitude"
  speed: "speed"
  heading: "rotation"
  vertical_rate: "vrate"
  aircraft_type: "aircraft_type"

dataset:
  input_path: "data/sim_catalog"
  aircraft_type_map_path: "data/aircraft_type_map.csv"
  sample_interval_minutes: 10
  history_minutes: 60
  horizons_minutes: [30, 60, 120, 300]

output:
  root_dir: "artifacts"
  run_name: "synthetic_h3_movement_gru"
```

说明：

- `columns.aircraft_id`：飞机唯一标识字段，默认用 `target_id`
- `dataset.aircraft_type_map_path`：可选，飞机唯一标识到机型的映射表
- `output.root_dir`：所有数据集、checkpoint、评估、注册表、预测结果的根目录

## 4. 输入数据格式

### 4.1 轨迹 CSV
最少需要这些列：

- `target_id`
- `timestamp`
- `longitude`
- `latitude`

推荐完整列：

```csv
target_id,timestamp,longitude,latitude,altitude,speed,rotation,vrate,aircraft_type
ea_0001,2026-01-03 10:00:00,139.78,35.55,0,0,0,0,P8A
ea_0001,2026-01-03 10:10:00,140.10,35.20,1200,180,145,1200,P8A
ea_0001,2026-01-03 10:20:00,141.00,34.60,5200,340,150,1800,P8A
ea_0001,2026-01-03 10:30:00,142.10,33.90,9800,430,154,900,P8A
```

规则：

- 一个文件可以是一架飞机的一段长期历史，也可以是单次在线探测片段
- 时间字段会转成 UTC 时间戳
- 如果数据里没有 `aircraft_type` 列，可以用映射表补

### 4.2 机型映射表 CSV
如果原始轨迹没有 `aircraft_type` 列，推荐维护一份映射表。

示例：

```csv
aircraft_id,aircraft_type,source_batch
ea_0000,SIM_BATCH1_TYPE,sim_batch1
ea_0001,SIM_BATCH1_TYPE,sim_batch1
em_0000,SIM_BATCH2_TYPE,sim_batch2
em_0001,SIM_BATCH2_TYPE,sim_batch2
```

说明：

- `aircraft_id` 要和 `columns.aircraft_id` 对应，默认就是 `target_id`
- `aircraft_type` 是训练和预测路由使用的机型名
- 额外列会被忽略，可以自由增加备注字段

## 5. 典型流程

### 5.1 全局模型
```bash
python build_dataset.py --config config/region_forecast.yaml
python training.py --config config/region_forecast.yaml
python evaluation.py --config config/region_forecast.yaml
```

### 5.2 单机型模型
```bash
python build_dataset.py --config config/region_forecast.yaml --scope type --name A320
python training.py --config config/region_forecast.yaml --scope type --name A320
python evaluation.py --config config/region_forecast.yaml --scope type --name A320
```

### 5.3 批量按机型训练
```bash
python train_catalog.py --config config/region_forecast.yaml
```

常用参数：

```bash
python train_catalog.py --config config/region_forecast.yaml --type A320 --type B737  
python train_catalog.py --config config/region_forecast.yaml --max-types 10  
python train_catalog.py --config config/region_forecast.yaml --skip-global  
python train_catalog.py --config config/region_forecast.yaml --skip-existing  #作用：跳过已在注册表里有记录的机型，适合增量训练新机型
```

批量训练逻辑：

- 先训练全局模型
- 扫描所有机型
- 样本量达阈值的机型训练专属模型
- 样本量不足的机型登记为 `skipped_fallback`
- 预测时这类机型自动回退到全局模型

### 5.4 单总表拆分后再清洗

如果原始数据是“一个总 CSV，`mbbh` 表示一次飞行”，可以先按 `mbbh` 拆成多个文件，再接清洗流程：

```bash
python tools/split_and_clean_mbbh_csv.py \
  --config config/region_forecast.yaml \
  --input data/raw/air_pos-0416-0512.csv \
  --split-output data/split_csv \
  --clean-output data/cleaned_csv \
  --workers 4
```

如果只想先拆分不清洗，加 `--split-only`。  
如果 `target_type` 想作为机型字段写入清洗结果，加 `--copy-target-type`。

## 6. 预测

### 6.1 用全局模型预测
```bash
python predict.py \
  --config config/region_forecast.yaml \
  --input data/probe/probe_0001.csv \
  --scope global \
  --name global \
  --output artifacts/predictions/probe_0001.json
```

### 6.2 用机型模型预测
```bash
python predict.py \
  --config config/region_forecast.yaml \
  --input data/probe/probe_0001.csv \
  --scope type \
  --name A320 \
  --output artifacts/predictions/probe_0001.json
```

### 6.3 按机型自动路由
```bash
python predict.py \
  --config config/region_forecast.yaml \
  --input data/probe/probe_0001.csv \
  --aircraft-type A320 \
  --include-geometry \
  --include-geojson \
  --geojson-output artifacts/predictions/probe_0001_all.geojson \
  --geojson-output-dir artifacts/predictions/probe_0001_by_horizon \
  --output artifacts/predictions/probe_0001.json
```

说明：

- `--scope/--name`：强制指定模型
- `--aircraft-type`：先找对应机型模型，找不到则回退全局模型
- 不传 `--output` 时只会打印 JSON 到终端，不会自动保存文件

### 6.4 作为 Python 库调用
```python
import pandas as pd
from regional_forecast.api import predict_from_dataframe

pos_pd = pd.read_csv("data/probe/probe_0001.csv")

res = predict_from_dataframe(
    pos_pd,
    config_path="config/region_forecast.yaml",
    traj_name="probe_0001",
    aircraft_type="A320",      # 可选
    sort_by_timestamp=True,
    include_dataframe=True,
    include_geometry=True,
    include_geojson=True,
    geojson_output_path="artifacts/predictions/probe_0001_all.geojson",
    geojson_output_dir="artifacts/predictions/probe_0001_by_horizon",
)

print(res["summary"])
print(res["records"])
prediction = res["prediction"]
print(prediction["horizons"][0]["candidate_regions"])
print(prediction["geojson"]["by_horizon"][0]["feature_collection"])
```

返回值主键：

- `res["prediction"]`：完整预测结果，结构与 `predict.py` 输出 JSON 一致
- `res["summary"]`：简要摘要，适合日志和接口返回
- `res["records"]`：按 horizon 摘要后的记录列表
- `res["dataframe"]`：原始输入 DataFrame 的副本，仅在 `include_dataframe=True` 时返回
- `include_geometry=True` 时，预测结果里会额外返回 H3 网格顶点坐标，坐标顺序为 `[[lon, lat], ...]`
- `include_geojson=True` 时，预测结果里会额外返回候选区域网格的 GeoJSON：
  - `prediction["horizons"][i]["candidate_regions"][j]["cells_geojson"]`
  - `prediction["horizons"][i]["candidate_regions_geojson"]`
  - `prediction["geojson"]["by_horizon"]`
  - `prediction["geojson"]["all_horizons"]`
- 如需直接给 QGIS 导入，不要导入完整预测结果 JSON；请导入：
  - `geojson_output_path` / `output_path` 生成的纯 `.geojson`
  - 或 `geojson_output_dir` / `output_dir` 下按 horizon 拆分的 `.geojson`
- Python 交互终端里直接回显的是字典对象表示，通常会显示单引号，这不是标准 JSON；落盘文件会用标准 JSON 双引号格式保存

如需直接拿“所有 horizon 的 GeoJSON 汇总”，可用专门接口：

```python
import pandas as pd
from regional_forecast.api import predict_geojson_from_dataframe

pos_pd = pd.read_csv("data/probe/probe_0001.csv")

geo = predict_geojson_from_dataframe(
    pos_pd,
    config_path="config/region_forecast.yaml",
    traj_name="probe_0001",
    aircraft_type="A320",
    output_path="artifacts/predictions/probe_0001_all.geojson",
    output_dir="artifacts/predictions/probe_0001_by_horizon",
)

print(geo["summary"])
print(geo["geojson"]["by_horizon"][0]["feature_collection"])
print(geo["geojson"]["all_horizons"])
```

历史不足时的行为：

- 当前模型需要固定长度历史窗口
- 如果输入历史不足，系统会自动左侧补齐
- 输出里会标记：
  - `history_insufficient`
  - `history_steps_required`
  - `history_steps_observed`
  - `padding_steps`

### 6.5 评估样例离线可视化

默认 Folium 地图会加载外网 CDN 和在线瓦片。内网环境需要生成无外链 HTML：

```bash
python tools/render_example_images.py \
  --input artifacts/evaluations/global_h3_movement_gru_examples.json \
  --output artifacts/example_maps/examples_offline.html \
  --visual-mode candidate \
  --num-examples 0 \
  --offline
```

离线模式使用内联 SVG，不加载在线底图；可以在内网浏览器直接打开。

## 7. 输出目录
默认都在 `output.root_dir` 下。

### 7.1 数据集
```text
artifacts/
  datasets/
    train.pt
    val.pt
    test.pt
    metadata.json
    by_type/
      a320/
      b737/
```

### 7.2 模型
```text
artifacts/
  checkpoints/
    <run_name>.pt
    <run_name>_train_summary.json
    by_type/
      a320/
      b737/
```

### 7.3 评估
```text
artifacts/
  evaluations/
    <run_name>_summary.csv
    <run_name>_summary.json
    <run_name>_examples.json
    by_type/
      a320/
      b737/
```

### 7.4 批量训练摘要
```text
artifacts/
  catalog/
    <run_name>_catalog_summary.json
```

### 7.5 注册表
```text
artifacts/
  model_registry.json
```

## 8. 结果示例

### 8.1 评估摘要示例
`*_summary.json / csv` 中每行对应一个 horizon：

```json
[
  {
    "horizon_minutes": 30,
    "samples": 3694,
    "region_top1": 0.4031,
    "region_top3": 0.7566,
    "candidate_union_cover": 0.9413,
    "status_accuracy": 0.9878
  }
]
```

关键指标：

- `region_top1/top3/top5`：真实区域命中率
- `candidate_union_cover`：候选区域并集覆盖率
- `status_accuracy`：空中/已落地状态分类准确率
- `region_top1_unknown_rate`：预测 top1 落到 unknown 的比例

### 8.2 预测结果示例
`predict.py` 输出 JSON 的核心结构：

```json
{
  "model": {
    "scope": "type",
    "name": "SIM_BATCH1_TYPE",
    "fallback_used": false
  },
  "input": {
    "trajectory_id": "ea_0001",
    "aircraft_type": "SIM_BATCH1_TYPE",
    "history_insufficient": false,
    "history_steps_required": 6,
    "history_steps_observed": 6
  },
  "horizons": [
    {
      "horizon_minutes": 30,
      "predicted_status": "airborne",
      "status_probabilities": {
        "airborne": 0.99,
        "landed": 0.01
      },
      "candidate_regions": [
        {
          "center_cell": "831f1bfffffffff",
          "center_lat": 25.42,
          "center_lon": 122.15,
          "confidence": 0.56,
          "center_cell_geometry": {
            "cell": "831f1bfffffffff",
            "resolution": 3,
            "center_lat": 25.42,
            "center_lon": 122.15,
            "boundary_lonlat": [
              [121.65, 25.89],
              [122.58, 26.11],
              [123.09, 25.48],
              [122.67, 24.63],
              [121.73, 24.41],
              [121.22, 25.04],
              [121.65, 25.89]
            ]
          },
          "cells": [
            "831f1bfffffffff",
            "831f18fffffffff"
          ]
        }
      ]
    }
  ]
}
```

## 9. 交接注意事项

- 如果原始数据没有 `aircraft_type`，请优先维护 `aircraft_type_map.csv`
- `train_catalog.py` 依赖机型字段或映射表才能真正按机型训练
- 预测时推荐显式传 `--output`
- 如果部署环境用的是单独目录，比如 `artifacts/temp`，只要 `output.root_dir` 和注册表路径一致即可
- 评估和预测都依赖训练时产出的 `metadata` 与 `checkpoint`，不要只拷贝模型文件不拷贝目录结构

## 10. 推荐最小交付命令

### 批量训练
```bash
python train_catalog.py --config config/region_forecast.yaml
```

### 预测
```bash
python predict.py \
  --config config/region_forecast.yaml \
  --input data/probe/probe_0001.csv \
  --aircraft-type A320 \
  --output artifacts/predictions/probe_0001.json
```
