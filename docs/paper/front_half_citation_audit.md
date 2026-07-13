# 前半部分引用核查记录

本文档记录中文版前半部分（摘要、引言、相关工作）中主要参考文献的核查方式。目的不是把所有文献都写成逐页笔记，而是明确每条引用在正文中承担什么论断，避免把标题、摘要或官方页面没有支撑的内容写进论文。

## 已核查并用于正文论断的文献

| 文献 | 核查方式 | 在正文中承担的作用 |
|---|---|---|
| FAA TBO 页面 | 直接访问官方页面 | 支撑 TBO、日运行轨迹集合、未来需求和 traffic management actions 之间的背景关系 |
| ICAO GANP 门户 | 直接访问官方门户；PDF 下载被 403 拦截 | 支撑“轨迹化运行/全球空管现代化”这一官方背景，不使用细节性技术结论 |
| Mercer et al. 2013 | 下载 NASA/ARC 本地 PDF；Crossref DOI 核验 BibTeX | 支撑“轨迹预测不确定性会影响管制员表现和工具可接受性” |
| Reisman et al. 2007 | Crossref DOI 核验 BibTeX | 支撑“轨迹精度会传递到 TFM 操作效果” |
| Tobaruela et al. 2014 | Crossref DOI 核验 BibTeX | 支撑 air traffic predictability framework 与交通可预测性背景 |
| Chen et al. 2017 | Crossref DOI 核验 BibTeX | 支撑 ATFM under uncertainty / chance-constrained optimization 背景 |
| Nunez-Portillo et al. 2024 | Crossref DOI 核验 BibTeX；MDPI PDF 下载被 403 拦截 | 支撑“不利天气下交通拥堵预测需要处理不确定性” |
| Chatterji et al. 2003 | Crossref DOI 核验 BibTeX | 支撑“运行数据质量会影响 TFM 决策分析” |
| OpenSky 2014 | DOI/元数据核验 | 支撑 OpenSky 是面向研究的大规模 ADS-B 传感网络 |
| OpenSky ESSD 2021 | DOI/ESSD 元数据核验 | 支撑 OpenSky 2019--2020 数据集作为开放历史航班数据资源 |
| Strohmeier 2020 | Crossref DOI 核验 BibTeX | 支撑 crowdsourced air traffic data 的研究使用背景 |
| ADRR 官方页面与 metadata PDF | 直接访问官方页面；下载 ADRR metadata PDF | 支撑 ADRR 的研究定位、商业航班运行档案和结构化数据来源 |
| Attention-LSTM 2022 | Crossref DOI 与 Scientific Reports 页面核验 | 支撑 attention + LSTM 四维轨迹预测代表路线 |
| Attention-TCN-GRU 2024 | Crossref DOI 与 MDPI 页面核验 | 支撑 TCN/GRU/attention 组合用于四维轨迹预测 |
| WTFTP 2023 | Crossref DOI 与 Nature Communications 页面核验 | 支撑时频小波变换用于航空轨迹预测 |
| WTFTP+ 2025/2026 | Crossref DOI 与 Nature Communications 页面核验 | 支撑 WTFTP 路线扩展到 multi-horizon flight trajectory prediction |
| FlightBERT++ 2024 | DOI/AAAI 元数据核验；此前已下载 PDF 阅读 | 支撑 non-autoregressive multi-horizon、horizon-aware context generator 等方法定位 |
| Li et al. 2025 | Crossref DOI 与 Scientific Reports 页面核验 | 支撑噪声 ADS-B 观测下的 Transformer 轨迹预测研究 |
| Shan et al. 2025 | Crossref DOI 与 Scientific Reports 页面核验 | 支撑稀疏卫星观测下的深度学习轨迹预测框架 |
| Zeng et al. 2022 | Crossref DOI 核验 BibTeX；MDPI PDF 下载被 403 拦截 | 支撑民航四维轨迹预测综述背景 |
| Kwak & Lee 2025 | Crossref DOI 核验 BibTeX；MDPI PDF 下载被 403 拦截 | 支撑深度学习航空轨迹预测综述背景 |
| Krozel 2000 | Crossref DOI 核验 BibTeX | 支撑 aircraft intent inference 早期研究方向 |
| Foo et al. 2007 | Crossref DOI 核验 BibTeX | 支撑 intent inference 与 surveillance/conformance monitoring 的联系 |
| Yang et al. 2015 | Crossref DOI 核验 BibTeX；网页下载被 403 拦截 | 支撑终端区基于在线轨迹聚类的 intent inference 方向 |
| Lee & Lee 2022 | Crossref DOI 核验 BibTeX | 支撑从记录 ADS-B 轨迹中提取 flight plans 的研究方向 |
| TFT 2021 | Crossref DOI 核验 BibTeX | 支撑多时域时间序列预测中的 attention/gating 背景 |
| MQTransformer 2020 | 下载 arXiv PDF 到本地 | 支撑 multi-horizon 中不同 horizon 读取不同历史上下文的动机 |
| TrAISformer 2021 | 下载 arXiv PDF 到本地 | 支撑离散 token + Transformer 轨迹建模路线 |
| DGGS 2003 | Crossref DOI 核验 BibTeX | 支撑离散全球网格系统的多分辨率空间背景 |
| H3 docs | 官方文档页面 | 支撑 H3 的层级索引、父子映射和邻域操作 |
| Gneiting & Raftery 2007 | DOI/元数据核验；此前已下载 PDF 阅读 | 支撑严格适定评分规则和概率预测评价 |
| Brier 1950 | Crossref DOI 核验 BibTeX | 支撑 Brier 分数来源 |
| Guo et al. 2017 | 下载 PMLR PDF 到本地 | 支撑现代神经网络误校准与 ECE 背景 |
| Jacobs et al. 1991 | Crossref DOI 核验 BibTeX | 支撑 adaptive mixture of local experts 的经典来源 |
| Shazeer et al. 2017 | 下载 arXiv PDF 到本地 | 支撑 sparsely-gated mixture-of-experts 路由思想 |
| Switch Transformer 2022 | 下载 JMLR PDF 到本地 | 支撑大规模专家路由与 sparse expert 方法谱系 |
| Bucila et al. 2006 | DOI/Crossref/ACM 元数据核验；本地 PDF 缓存；Crossref BibTeX 存在字符乱码，已按 DOI 与 ACM/DBLP 元数据清理 | 支撑 model compression / 用压缩模型逼近大模型或集成预测行为 |
| Hinton et al. 2015 | arXiv API 与本地 PDF 缓存核验 | 支撑 softened targets 与神经网络知识蒸馏背景 |
| Furlanello et al. 2018 | PMLR 官方页面与本地 PDF 缓存核验 | 支撑同结构教师/学生的 Born Again Neural Networks 蒸馏背景 |

## 使用边界

1. 正文把 TBO/ATFM 文献用于说明“未来空域需求、轨迹不确定性和流量管理”背景，不声称本文已经实现完整 ATFM 优化或扇区容量控制。
2. 意图推断、计划提取和 conformance monitoring 文献只用于相关工作定位，正文不把本文方法写成航路意图恢复模型。
3. 航空点轨迹预测文献只用于说明已有时间建模与多时域预测路线，正文不把它们硬说成区域概率预测方法。
4. 对 PDF 下载失败但 DOI/官方页面可核验的文献，只使用标题、摘要或元数据明确支持的高层论断。
5. 前半部分引用量目前约为 39 个唯一 citation key；引用集中在 Related Work，引言只保留直接支撑背景链条和方法动机的文献。

## 本地缓存

已下载到 `docs/paper/literature_cache/` 的关键材料包括：

- `mercer_2013_trajectory_prediction_uncertainty.pdf`
- `eurocontrol_adrr_metadata_2025.pdf`
- `mqtransformer_2020.pdf`
- `traisformer_2021.pdf`
- `shazeer_2017_sparsely_gated_moe.pdf`
- `fedus_2022_switch_transformers.pdf`
- `guo_2017_calibration.pdf`
- `bucila_2006_model_compression.pdf`
- `hinton_2015_distillation.pdf`
- `furlanello_2018_born_again.pdf`
