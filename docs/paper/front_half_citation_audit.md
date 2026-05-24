# 前半部分引用核查记录

本文档记录中文版前半部分（摘要、引言、相关工作）中主要参考文献的核查方式。目的不是把所有文献都做成逐页笔记，而是明确：哪些引用基于本地全文或网页全文，哪些只用于标题层面的研究归类，避免把摘要或标题没有支持的内容写进正文。

## 已核查并用于正文论断的文献

| 文献 | 核查方式 | 在正文中承担的作用 |
|---|---|---|
| OpenSky 2014 | 检查论文标题与检索页元数据 | 支撑“OpenSky 是面向研究的大规模 ADS-B 传感网络”这一背景事实 |
| OpenSky ESSD 2021 | 检查 ESSD 论文元数据与引文页 | 支撑“OpenSky 后续发布了面向研究的数据集” |
| ADRR 页面 | 直接访问官方页面 | 支撑 ADRR 的数据来源与研究定位 |
| Attention-LSTM 2022 | 阅读 Nature 网页正文与摘要 | 支撑“该工作是 4D 轨迹点预测中的 attention+LSTM 代表方法之一” |
| Attention-TCN-GRU 2024 | 阅读 MDPI 检索页摘要与方法概览 | 支撑“该工作在终端区 4D 轨迹预测中采用 seq2seq + attention + TCN + GRU” |
| WTFTP 2023 | 阅读 Nature 网页正文 | 支撑“时频小波变换被用于航空轨迹预测” |
| WTFTP+ 2026 | 阅读 Nature 网页正文 | 支撑“WTFTP 后续扩展到 multi-horizon flight trajectory prediction” |
| FlightBERT++ 2024 | 下载本地 PDF 并转文本阅读 | 支撑其“non-autoregressive multi-horizon”“horizon-aware context generator”“differential-prompted decoder”等描述；同时据此修正 BibTeX 标题、作者与 DOI |
| MQTransformer 2020 | 下载本地 PDF 并转文本阅读 | 支撑“multi-horizon forecasting 中不同 horizon 不宜共享单一压缩上下文”“decoder-encoder attention” |
| Gneiting & Raftery 2007 | 下载本地 PDF 并转文本阅读 | 支撑“严格适定评分规则”“sharpness subject to calibration” |
| Guo et al. 2017 | 官方 PMLR 页面 | 支撑“现代神经网络可能存在误校准” |
| Li et al. 2025 | 阅读 Nature 网页摘要与方法段落 | 支撑“噪声鲁棒 Transformer 与 hybrid positional encoding 的轨迹预测研究” |
| Shan et al. 2025 | 阅读 Nature 网页正文与图注/方法段落 | 支撑“稀疏卫星观测下的深度学习轨迹预测框架” |
| H3 docs | 官方文档页面 | 支撑 H3 的层级索引与邻域操作接口 |

## 仅作高层归类、未在正文中使用细节结论的文献

| 文献 | 当前使用边界 |
|---|---|
| TrAISformer 2021 | 仅用于说明“离散空间/离散 token 的 Transformer 轨迹预测是一条相关路线”；正文未使用其更细的方法细节或具体数值结论 |
| Sahr et al. 2003 | 仅用于说明 DGGS / 多分辨率离散全球网格的概念背景；正文未展开其理论细节 |
| Brier 1950 | 仅用于交代 Brier 分数的经典来源；正文未引申其历史背景之外的额外结论 |

## 本次前半部分修改的原则

1. 不把“全球”写成实验事实。当前数据来自 ADRR 派生的一月商业航班轨迹，主要覆盖欧洲及欧洲上空。
2. 不把点式轨迹预测文献硬说成区域概率预测文献。相关工作中明确区分“连续点预测”和“区域概率预测”。
3. 对无法拿到全文的文献，只在标题或摘要明确支持的范围内使用，不扩写模型细节。
4. 对能拿到全文的关键文献，优先用它们来支撑前半部分的主线：开放数据、multi-horizon 建模、概率评分与校准。

## 备注

- 原中文版前半部分参考文献数量偏少。对这类“航空轨迹预测 + 大规模开放数据 + 概率预测”论文，13 篇通常不够支撑完整立论；本次已把前半部分相关引用扩展并重组到更合理的规模。
- 当前更重要的不是单纯继续堆文献数量，而是保证每条引用与正文论断一一对应。
