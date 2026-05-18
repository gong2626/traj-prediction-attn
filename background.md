**question 1:调研一下当前文献，如果想在方法上进行创新有哪些可行的方向，重点关注你说的五个方法创新点中的面向缺失轨迹的 uncertainty-aware forecasting**

**结论**
我看了一轮当前文献后，判断比较明确：

当前如果你想在“方法上”做出像样的创新，最有价值的切入点确实不是再换一个 `GRU/Transformer` 主干，而是做 **面向缺失轨迹的 uncertainty-aware regional forecasting**。  
这个方向有两个优势：

- 学术上站得住，因为它处在三个活跃方向的交叉点：`不规则/缺失时间序列建模`、`轨迹预测的不确定性量化与校准`、`ADS-B/飞机轨迹预测`
- 工程上也贴合你的真实需求，因为你的数据问题本来就不是“没有模型”，而是“轨迹不连续、观测稀疏、输出需要带置信度的区域而不是单点”

我基于文献的整体判断是：**“飞机轨迹预测”这条线已经有不少 encoder/decoder 架构创新，但“把缺失本身当成信号，并把不确定性做成可校准的区域输出”这一块仍然明显不饱和。**  
所以这是最值得押的方向。

**重点方向**
如果聚焦你这个项目，我建议的方法创新点定义成：

`从稀疏/缺失 ADS-B 观测中，进行多时距、区域级、带校准不确定性的飞机动向预测`

这个题目比“又提一个新网络”更像论文。

具体看，文献里已经形成了几条成熟但还没被你这个任务完整拼起来的线：

- **缺失模式本身是有信息的**。GRU-D 明确提出 missingness 可能是 informative missingness，不该只做插值后丢掉；它把 `masking` 和 `time interval` 直接送进模型来建模缺失模式。  
  来源：Che et al., *Recurrent Neural Networks for Multivariate Time Series with Missing Values*, Scientific Reports 2018  
  https://www.nature.com/articles/s41598-018-24271-9

- **不规则时间序列不一定要先强行重采样**。Latent ODE、GRU-ODE-Bayes、Neural CDE、NCDSSM 这一串工作，核心都是直接在连续时间里建模稀疏观测。  
  这对 ADS-B 很关键，因为你现在的数据缺口不是小噪声，而是时间间隔本身就不稳定。  
  来源：  
  Rubanova et al., *Latent Ordinary Differential Equations for Irregularly-Sampled Time Series*, NeurIPS 2019  
  https://proceedings.neurips.cc/paper/2019/hash/42a6845a557bef704ad8ac9cb4461d43-Abstract.html  
  De Brouwer et al., *GRU-ODE-Bayes*, NeurIPS 2019  
  https://papers.nips.cc/paper/8957-gru-ode-bayes-continuous-modeling-of-sporadically-observed-time-series  
  Kidger et al., *Neural Controlled Differential Equations for Irregular Time Series*, NeurIPS 2020  
  https://papers.nips.cc/paper/2020/hash/4a5876b450b45371f6cfe5047ac8cd45-Abstract.html  
  Ansari et al., *Neural Continuous-Discrete State Space Models for Irregularly-Sampled Time Series*, ICML 2023  
  https://proceedings.mlr.press/v202/ansari23a.html

- **“先插值再预测”不是唯一范式，越来越多工作在做 joint impute-and-learn**。BRITS、CSDI、GRU-M 都是这条线。  
  对你来说，这意味着可以把“缺失段恢复”和“未来区域预测”放在同一个学习框架里，而不是把插值当成纯预处理。  
  来源：  
  Cao et al., *BRITS*, NeurIPS 2018  
  https://proceedings.neurips.cc/paper_files/paper/2018/hash/734e6bfcd358e25ac1db0a4241b95651-Abstract.html  
  Tashiro et al., *CSDI*, NeurIPS 2021  
  https://proceedings.neurips.cc/paper/2021/hash/cfe8504bda37b575c70ee1a8276f3486-Abstract.html  
  Pachal et al., *GRU-M*, ACML 2025  
  https://proceedings.mlr.press/v260/pachal25a.html

- **轨迹预测里“输出概率”不等于“输出可信不确定性”**。这点在 trajectory 领域已经被反复指出。  
  CCTR 直接指出现有 trajectory prediction 的方差/分布输出往往并不可靠，需要专门 calibration。  
  来源：Cao et al., *CCTR: Calibrating Trajectory Prediction for Uncertainty-Aware Motion Planning in Autonomous Driving*, AAAI 2024  
  https://research.tudelft.nl/en/publications/cctr-calibrating-trajectory-prediction-for-uncertainty-aware-moti/

- **真正可发表的不确定性输出，越来越强调 coverage guarantee，而不是只看 softmax/confidence**。  
  时间序列和轨迹上，Conformal TSF、SPCI、MultiDimSPCI、CAFHT 已经把“有保证的预测区间/区域/轨迹带”这件事做起来了。  
  这和你要的“一个或多个带置信度的区域”天然契合。  
  来源：  
  Stankeviciute et al., *Conformal Time-series Forecasting*, NeurIPS 2021  
  https://proceedings.neurips.cc/paper_files/paper/2021/hash/312f1ba2a72318edaaa995a67835fad5-Abstract.html  
  Xu & Xie, *Sequential Predictive Conformal Inference for Time Series*, ICML 2023  
  https://proceedings.mlr.press/v202/xu23r.html  
  Xu et al., *Conformal prediction for multi-dimensional time series by ellipsoidal sets*, ICML 2024  
  https://proceedings.mlr.press/v235/xu24m.html  
  Zhou et al., *Conformalized Adaptive Forecasting of Heterogeneous Trajectories*, ICML 2024  
  https://proceedings.mlr.press/v235/zhou24l.html

基于这几条线，我觉得你最值得做的不是“缺失后补齐再按老模型跑”，而是下面这个组合：

1. 用 **缺失感知编码器** 代替纯固定步长 GRU。  
   最稳的是 `GRU-D 风格增强版` 或 `GRU-ODE-Bayes / NCDE`。
2. 保留你现在已经比较合适的 **H3 分层区域输出**。  
   这个比继续做经纬度点回归更适合长时距动向预测。
3. 让模型显式输出三类不确定性信号。  
   `aleatoric`：本来就多模态的航向分岔  
   `epistemic`：样本少、机型少、区域冷门  
   `gap-induced`：因为观测断裂导致的额外不确定性
4. 最后加一个 **conformal calibration** 层，把原始概率变成“有覆盖保证的候选区域集”。

这样你的论文点就不再是“我也做了个区域预测模型”，而是：

- 缺失不是简单噪声，而是作为可学习信号进入预测
- 不确定性会随 gap 扩大而合理增长
- 输出不是未校准 softmax，而是可解释的区域集和覆盖保证

这比单纯换主干网络强很多。

**其他可行方向**
如果不只盯一个方向，我觉得当前文献下还有 4 个可做，但优先级低于上面这条：

- **连续时间编码 + H3 分层解码**
  价值在于把“不规则观测”处理得更自然。适合你这种 ADS-B 断续数据。
  但如果没有后面的 calibration，论文亮点还是差一点。

- **联合插补与预测**
  可以参考 CSDI、GRU-M，把缺失段重建作为辅助任务。
  但单做这个更像“数据处理增强”，不如“缺失感知 + 校准区域输出”完整。

- **Conformal 区域预测**
  单独拿出来也能成一条论文线，尤其适合“未来区域集合”输出。
  但如果底层编码器还是普通重采样 GRU， novelty 会偏弱。

- **epistemic uncertainty 的 prototype / retrieval / memory bank**
  对稀有机型、稀有航路很有帮助。
  适合做 second-stage 增强，不适合当第一篇主线。

- **evidential 输出**
  Deep Evidential Regression/EDL 类方法单次前向就能给不确定性，工程上很诱人。  
  但我不建议你把它作为唯一主力，因为最近已有工作专门质疑 evidential 方法原生 UQ 的可靠性。  
  来源：  
  Amini et al., *Deep Evidential Regression*, NeurIPS 2020  
  https://proceedings.neurips.cc/paper/2020/hash/aab085461de182608ee9f607f3f7d18f-Abstract.html  
  Shen et al., *Are Uncertainty Quantification Capabilities of Evidential Deep Learning a Mirage?*, NeurIPS 2024  
  https://proceedings.neurips.cc/paper_files/paper/2024/hash/c3177be226ee12e34d6ba3b5e6fe6a5b-Abstract-Conference.html

再说回飞机领域本身。2022 到 2025 的飞机轨迹文献，主流还是在做更强的序列骨干、直接多时距预测、鲁棒预训练、噪声鲁棒 Transformer 之类，例如：

- Wu et al., 2022，聚类 + CNN-BiLSTM 时空特征网络  
  https://www.mdpi.com/2079-9292/11/21/3453
- Guo et al., *FlightBERT++*, AAAI 2024，非自回归多时距飞行轨迹预测  
  DOI: https://doi.org/10.1609/aaai.v38i1.27763
- Li & Fang, *Noise robust aircraft trajectory prediction via autoregressive transformers with hybrid positional encoding*, Scientific Reports 2025  
  https://www.nature.com/articles/s41598-025-96512-7
- 2025 年综述也明确提到 ADS-B 依然是主流数据源，但原始数据普遍存在噪声、缺失和区域覆盖不均。  
  https://www.mdpi.com/2076-3417/15/19/10739

我的综合判断是：

- **最推荐的论文主线**：`Gap-aware + uncertainty-aware + conformalized H3 regional forecasting`
- **最不推荐的论文主线**：单纯把 GRU 换成 Transformer 或 iTransformer
- **最现实的实现顺序**：  
  先做 `缺失感知编码器 + gap bucket 评估`  
  再做 `uncertainty head`  
  最后做 `conformal 区域校准`

如果你要，我下一步可以直接给你出一版更具体的 **论文级方法设计草案**，包括：

- 方法框架图怎么画
- 模型各模块怎么命名
- loss 怎么设计
- 实验表格怎么设
- 哪些 baseline 必须做
- 哪些指标最能把论文讲成立。

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

**question 2:这种长时，粗粒度的区域预测如果使用开源的民航轨迹数据集的话研究背景好写吗？**
好写，而且**比“长时精细轨迹点预测”更好写**。

你的这个方向如果基于开源民航 ADS-B 轨迹数据来写研究背景，逻辑上其实很顺，因为它天然满足 4 个学术叙事点：

- **问题重要**：全球空中交通规模大，长时预测对空管协同、流量管理、冲突预警、异常监视都有价值。
- **数据真实且可复现**：OpenSky 这类公开数据已经是飞机轨迹研究里非常常见的数据源，审稿人容易接受。OpenSky 官方也明确提供实时/历史数据接口、科学数据集和机体元数据。  
  来源：OpenSky Data https://opensky-network.org/data/  
  OpenSky Scientific Datasets https://opensky-network.org/data/scientific
- **任务设定合理**：公开 ADS-B 数据本来就有噪声、缺失、覆盖不均，用它做几个小时后的“区域级出现概率”比做“精确轨迹点”更符合数据质量和业务现实。
- **方法创新空间清楚**：公开 ADS-B 的缺失和不规则采样，正好能支撑你做 `uncertainty-aware forecasting`，这比单纯换个 backbone 更像论文问题。

**为什么这个背景比精细点预测更容易成立**
因为公开民航轨迹数据的局限，反过来正好帮你把问题定义得更合理。

OpenSky 官方 FAQ 直接说了两点：  
一是它有全球覆盖，但**欧洲和美国覆盖最好**；二是 ADS-B/Mode S 数据本身就是从开放广播信号和众包接收网络来的，所以区域覆盖并不均匀。  
来源：OpenSky FAQ https://opensky-network.org/about/faq  
OpenSky Coverage https://opensky-network.org/about/network

而近年的综述也明确把 ADS-B 的问题写得很清楚：它是最常用的数据源，因为全球可得、时空分辨率高、开放性强；但同时也有**噪声、缺失值和区域覆盖不均**。  
来源：2025 综述 https://www.mdpi.com/2076-3417/15/19/10739

这意味着你完全可以把背景写成这样：

- 开源民航 ADS-B 轨迹是当前数据驱动飞行轨迹研究的主流数据基础；
- 但这类数据在长时预测场景下面临观测中断、局部缺失、采样不规则、区域覆盖不均等问题；
- 因此，与其追求长时精细点位外推，不如研究**长时、粗粒度、带不确定性的区域级动向预测**；
- 这种输出形式更贴近实际监视、流量研判和风险预警需求。

这个叙事是自然的，不牵强。

**真正难写的地方不在背景，而在“应用场景和数据源是否一致”**
这里要分两种情况。

第一种，你写的是**全球民航场景**。  
那就非常顺。背景可以直接挂在：

- air traffic management
- trajectory intent inference
- surveillance resilience
- strategic flow awareness
- long-horizon regional occupancy forecasting

第二种，你想最后落到“南海周边外国飞机动向研判”这种场景。  
那就没那么顺了。因为开源民航数据和目标应用之间存在明显 domain gap：

- 民航航路更规范
- 机型分布、任务模式、巡航策略都更稳定
- 开源 ADS-B 对特定非民航目标的代表性有限

这种情况下，公开民航数据更适合作为：

- 方法验证数据
- 预训练或原型验证数据
- 可复现实验平台

而不太适合作为“最终应用场景已经被充分覆盖”的证据。

也就是说：

- **写全球民航长时区域预测**：背景很好写
- **写特定敏感区域的非民航目标动向预测**：背景可以写，但实验支撑会比较虚

**从论文角度，公开民航数据最适合你把题目改写成这类方向**
我建议优先写成下面这种口径，而不是“军政方向”口径：

- 基于公开 ADS-B 轨迹的长时区域级飞机动向预测
- 面向不规则和缺失观测的区域级飞行意图预测
- 面向空域态势感知的 uncertainty-aware long-horizon regional forecasting
- 面向公开民航轨迹数据的多时距区域出现概率预测

这样背景会明显更稳，也更容易和公开数据对上。

**文献上能支撑你写什么背景**
你至少可以很顺地写 3 段：

1. **ADS-B 已经成为主流数据基础**
   很多飞机轨迹预测工作都直接用 OpenSky 或类似 ADS-B 数据。  
   例如 2022 年这篇工作就明确使用了 OpenSky 的公开 ADS-B 数据，并且专门提到原始 ADS-B 轨迹有重复、缺失，需要清洗和重构。  
   来源：https://www.mdpi.com/2079-9292/11/21/3453

2. **ADS-B 缺失/中断本身就是安全和预测问题**
   2023 年 Scientific Reports 这篇文章就是围绕 ADS-B 失效条件下的轨迹预测来写的，说明“观测不完整情况下做预测”本身就是合理研究问题。  
   来源：https://www.nature.com/articles/s41598-023-46914-2

3. **开放数据促进可复现，但也带来粗粒度任务需求**
   OpenSky 官方公开提供历史数据、科学数据集和工具，但覆盖并不均匀；终端区数据集如 TartanAviation 也说明开放航空数据正在支持更细分的研究场景。  
   来源：  
   OpenSky Data https://opensky-network.org/data/  
   TartanAviation https://www.nature.com/articles/s41597-025-04775-6

**所以，结论是**
如果你把任务定义为：

`使用开源民航 ADS-B 轨迹数据，预测未来 0.5/1/2/5 小时飞机可能出现的区域集合及其置信度，并显式处理缺失观测带来的不确定性`

这个研究背景是**好写的，而且是比长时精细轨迹点预测更自然的**。

但要注意两点：

- 不要把公开民航数据硬说成能完整代表所有目标类型和所有区域场景。
- 不要把任务写成“精确航迹外推”，而要明确写成“区域级、多模态、带不确定性的动向预测”。

**一句更直接的话**
用开源民航数据做这个题目，背景不是问题；  
真正要下功夫的是把“为什么做区域预测而不是点预测”“为什么必须做 uncertainty-aware”这两件事讲透。

如果你要，我下一步可以直接帮你写一版**论文引言/研究背景初稿提纲**，按中文论文或英文论文两种风格都可以。

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

**question 3:好，那就使用开源民航数据，按照你说的论文定位和建议的改造路线，基于当前项目列一个改造计划单，如果你需要事先阅读某些文献的完整版可以跟我讲，我会上传给你**
可以。按现在这个项目的基础，我建议不要推倒重来，而是把它升级成一个**公开民航 ADS-B 的长时、区域级、缺失感知、带不确定性输出的预测框架**。这样既能保留你已经做好的工程骨架，又能逐步把它抬到论文定位。

我先给你一个面向论文和项目交付都能落地的改造计划单。这个计划默认目标是：

- 数据源：开源民航 ADS-B 数据，优先 `OpenSky`
- 任务定义：预测未来 `0.5/1/2/5 小时` 的**出现区域集合**、对应置信度、以及 `airborne/landed`
- 论文定位：**面向不规则与缺失观测的 uncertainty-aware long-horizon regional flight movement forecasting**
- 工程要求：保留当前按机型训练、模型注册表、预测接口、批量训练能力

**一、先定论文定位**
我建议把论文贡献点收敛成 3 个，不要太散。

1. 提出一个面向公开 ADS-B 缺失轨迹的长时区域预测任务。
2. 提出一个缺失感知的多时距 H3 区域预测模型。
3. 提出一个可校准的不确定性输出机制，把 softmax 概率变成更可靠的候选区域集。

对应到你当前项目，最该保留的是：

- H3 分层区域预测框架
- 多 horizon 输出
- landed/airborne 状态头
- 按机型训练与模型路由
- 预测接口和注册表

最该升级的是：

- 当前固定步长重采样 + 普通 GRU 编码
- 当前“概率输出但没有严格不确定性建模/校准”
- 当前实验协议还不够论文化

**二、改造总路线**
我建议按 6 个阶段做，每一阶段都能单独验收，不容易改崩。

---

**阶段 0：基准定义与数据协议**
目标是把“论文到底测什么”先固定住。

要做的事：

- 明确使用哪类公开数据。
  - 首选：`OpenSky historical / scientific datasets`
- 明确研究范围。
  - 建议先做“全球民航场景”
  - 不要一开始就写成南海或特殊敏感场景
- 明确预测任务。
  - 输入：过去 `60~120 分钟` 的不规则 ADS-B 观测
  - 输出：未来 `30/60/120/300 分钟` 的 H3 区域集合 + 置信度 + landed/airborne
- 明确数据切分。
  - 必做：严格**时间切分**
  - 可做：按 aircraft 切分 / 按航班切分 / 按机场冷启动切分
- 明确评估指标。
  - 继续保留：`NLL / Brier / ECE / top-k / candidate cover / neighbor ring hit`
  - 新增重点：按缺失强度分桶评估

当前项目对应改动：

- [config/region_forecast.yaml](/e:/traj2/config/region_forecast.yaml:1)
- [regional_forecast/data/preprocess.py](/e:/traj2/regional_forecast/data/preprocess.py:1)
- [regional_forecast/pipeline/evaluate.py](/e:/traj2/regional_forecast/pipeline/evaluate.py:1)

新增建议文件：

- `docs/benchmark_protocol.md`
- `config/region_forecast_opensky.yaml`

阶段验收结果：

- 一份固定的数据协议
- 一份固定的实验协议
- 一份论文表格模板草案

---

**阶段 1：把数据管线改成“论文数据集构建器”**
这是最先该落地的技术阶段。

当前项目已经能读 CSV、切 mission、重采样、构建 H3 数据集，但还不够论文化。需要升级为能稳定处理公开 ADS-B 的数据构建管线。

要做的事：

1. 增加 OpenSky 风格字段兼容。
   - 常见字段如 `icao24 / callsign / time / lat / lon / velocity / heading / vertrate / onground / baroaltitude`
2. 增加航班级或 aircraft+time 级 mission segmentation。
3. 增加严格时间切分模式。
   - 不能只按 `_aircraft_id` 随机切
4. 增加缺失模式统计。
   - 每个样本记录：
     - 总缺失率
     - 最大 gap
     - gap 次数
     - 观测覆盖率
5. 增加 gap bucket 标签。
   - 例如：
     - `clean`
     - `light_gap`
     - `medium_gap`
     - `severe_gap`
6. 保存更完整的数据集 metadata。
   - 论文复现实验要靠这个

当前项目需重点改的模块：

- [regional_forecast/data/preprocess.py](/e:/traj2/regional_forecast/data/preprocess.py:1)
- [regional_forecast/config.py](/e:/traj2/regional_forecast/config.py:1)
- [regional_forecast/pipeline/build_dataset.py](/e:/traj2/regional_forecast/pipeline/build_dataset.py:1)

建议新增：

- `regional_forecast/data/opensky_schema.py`
- `regional_forecast/data/splitters.py`
- `regional_forecast/data/gap_analysis.py`

阶段验收结果：

- 能从公开 ADS-B 原始轨迹稳定构建训练/验证/测试集
- 每个样本自带缺失统计信息
- 数据切分方式可配置且可追溯

---

**阶段 2：建立论文 baseline 组**
现在项目里 baseline 还不够。论文阶段必须先把 baseline 组立起来。

最少要有这几组：

1. `Markov`  
   你已经有了，可以保留。

2. `Flat GRU`
   不用分层 H3，直接预测目标分辨率 cell。

3. `Hierarchical GRU`
   你当前主模型，作为现有 strongest baseline。

4. `Gap-agnostic Transformer`
   可以简单上一个 encoder baseline，不追求最强，只为说明不是“谁 backbone 大谁赢”。

5. `GRU-D style baseline`
   这是后续主线的前置版本，先做成 baseline 再往主模型升级。

要做的事：

- 把模型注册成统一接口
- 统一训练、预测、评估输出格式
- 所有 baseline 走同一套数据与指标

当前项目需重点改的模块：

- [regional_forecast/models/forecast_model.py](/e:/traj2/regional_forecast/models/forecast_model.py:1)
- [regional_forecast/models/baseline.py](/e:/traj2/regional_forecast/models/baseline.py:1)
- [regional_forecast/pipeline/train.py](/e:/traj2/regional_forecast/pipeline/train.py:1)
- [regional_forecast/pipeline/evaluate.py](/e:/traj2/regional_forecast/pipeline/evaluate.py:1)

建议新增：

- `regional_forecast/models/grud_model.py`
- `regional_forecast/models/transformer_model.py`

阶段验收结果：

- 你可以跑一张正式论文 baseline 表
- 你能知道“提升到底来自模型，还是来自数据处理”

---

**阶段 3：主方法第一版，Gap-aware encoder**
这是最核心的方法改造，优先级最高。

我建议第一版不要直接上最复杂的 Neural CDE 或 Latent ODE，而是先做一个**Gap-aware Hierarchical H3 Forecaster**，用 GRU-D 思路升级你现有模型。

核心思想：

- 保留当前 H3 分层解码头
- 把编码器从“普通固定步长 GRU”升级成“缺失感知编码器”
- 显式输入：
  - `mask`
  - `delta_t`
  - `gap bucket`
  - `since_last_observed`
  - 观测衰减特征
- 让模型学会“gap 越大，越不应该过度自信”

建议方法名可以先叫：

- `GapAware-H3`
- 或 `UGR-H3`：Uncertainty-aware Gap-robust Regional H3 Forecaster

要做的事：

1. 在 sample 中加入更丰富的 missingness 特征。
2. 实现 GRU-D 风格的 decay/imputation 机制。
3. 保留现有 hierarchical coarse-to-fine 解码。
4. 做 ablation：
   - 无 gap feature
   - 只有 mask
   - mask + delta_t
   - full gap-aware encoder

当前项目重点改动：

- [regional_forecast/data/preprocess.py](/e:/traj2/regional_forecast/data/preprocess.py:1)
- [regional_forecast/models/forecast_model.py](/e:/traj2/regional_forecast/models/forecast_model.py:1)

建议新增：

- `regional_forecast/models/gap_aware_encoder.py`

阶段验收结果：

- 在高缺失样本桶上明显优于普通 GRU
- 论文里第一个核心贡献成立

---

**阶段 4：主方法第二版，Uncertainty-aware prediction**
这是第二个核心创新点。

你现在已经有“候选区域 + 置信度”，但严格来说，这还是模型分布输出，不等于可靠不确定性。

这一阶段分两层做。

第一层：模型内部 uncertainty  
建议先做最稳的两种，不要上太花的：

- `MC Dropout`
- `Deep Ensemble`

输出：

- 预测分布均值
- epistemic uncertainty
- entropy / confidence
- 不确定性随 gap 增长的曲线

第二层：输出校准  
建议做：

- 温度缩放或 Dirichlet/分桶校准
- conformal set calibration，把 top-k cell 变成“有覆盖控制的区域集”

要做的事：

1. 增加 inference-time ensemble / MC dropout
2. 增加 uncertainty metrics：
   - NLL
   - Brier
   - ECE
   - risk-coverage
   - set size vs coverage
3. 增加 conformal calibration
4. 导出“校准后的候选区域集”

当前项目重点改动：

- [regional_forecast/pipeline/evaluate.py](/e:/traj2/regional_forecast/pipeline/evaluate.py:1)
- [regional_forecast/pipeline/predict.py](/e:/traj2/regional_forecast/pipeline/predict.py:1)
- [regional_forecast/evaluation/metrics.py](/e:/traj2/regional_forecast/evaluation/metrics.py:1)

建议新增：

- `regional_forecast/uncertainty/ensemble.py`
- `regional_forecast/uncertainty/calibration.py`
- `regional_forecast/uncertainty/conformal.py`

阶段验收结果：

- 你不只是在报 softmax prob
- 你能说“缺失越严重，模型不确定性越高”
- 你能给出更有说服力的区域集覆盖结果

---

**阶段 5：连续时间升级版，可选**
这是论文冲更强 novelty 时再做，不是第一优先级。

如果阶段 3 和 4 做出来已经不错，可以考虑把编码器再进一步升级为：

- `Neural CDE`
- 或 `GRU-ODE-Bayes`
- 或 lightweight continuous-time encoder

这个阶段的价值是：

- 更自然处理不规则采样
- 比单纯重采样更有方法亮点

但风险也更高：

- 实现复杂
- 训练更难稳定
- 工程负担更大

所以我建议把它作为：

- `paper v2`
- 或论文中的增强版/附加实验

建议新增：

- `regional_forecast/models/continuous_time_encoder.py`

阶段验收结果：

- 如果成功，会显著增强方法新意
- 如果失败，也不影响前 4 阶段成文

---

**阶段 6：论文实验与写作资产生成**
这一步不是简单“跑结果”，而是专门为论文准备。

必须产出的实验：

1. 主结果表
   - 各 horizon 上的 NLL / top-k / candidate cover / ECE

2. 缺失强度分桶表
   - clean / light / medium / severe

3. 校准结果表
   - ECE
   - coverage
   - set size

4. 消融实验表
   - 无 gap-aware
   - 无 hierarchy
   - 无 uncertainty calibration
   - 无 landed head

5. 泛化实验
   - 机场冷启动
   - 区域冷启动
   - 稀有机型

6. 可视化图
   - gap 大小时的预测区域变化
   - uncertainty heatmap
   - reliability diagram
   - coverage-size curve

建议新增：

- `tools/plot_reliability_diagram.py`
- `tools/plot_gap_bucket_metrics.py`
- `tools/plot_region_set_coverage.py`
- `tools/export_paper_tables.py`

---

**三、和你当前项目的对应关系**
不是所有东西都要重写。

建议保留：

- H3 regionizer 和 connected-region 输出
- hierarchical coarse-to-fine 预测框架
- 数据集 artifact 结构
- 训练/评估/predict 主流程
- model registry
- 按机型训练与 fallback
- Python API

建议重构：

- 样本构造逻辑
- split 协议
- encoder
- uncertainty 评估与推理

建议延后处理：

- retrieval/memory bank
- graph bias
- aircraft-type adapter/MoE
- 多模态数据融合

这些不是当前第一篇论文最关键的。

**四、推荐的实际实施顺序**
如果让我带着你一步步做，我会按这个顺序下命令：

1. `第 1 步：重做数据协议和切分逻辑`
2. `第 2 步：补 baseline 组`
3. `第 3 步：实现 gap-aware encoder`
4. `第 4 步：实现 uncertainty estimation`
5. `第 5 步：实现 conformal calibration`
6. `第 6 步：补论文图表工具和 README/实验说明`

这是最稳的顺序。

**五、你如果愿意上传全文，最值得先给我的文献**
我不需要你先上传很多，先给 4 篇最有用的就够了。优先级如下：

1. `GRU-D`
   Che et al., 2018  
   用来准确复现缺失感知 RNN 机制

2. `Neural Controlled Differential Equations`
   Kidger et al., NeurIPS 2020  
   如果后面你想做连续时间版本，这篇最关键

3. `CCTR`
   AAAI 2024  
   用来参考 trajectory uncertainty calibration 的论文写法和实验组织

4. `Conformal Time-series Forecasting` 或 `CAFHT`
   用来设计“区域集合覆盖保证”的校准部分

如果你方便，再加 2 篇：

5. `Latent ODE` 或 `GRU-ODE-Bayes`
6. `一篇公开 ADS-B / OpenSky 飞机轨迹预测论文`
   最好是你后面准备引用为飞机领域 baseline 的那篇

**六、我建议你现在就做的决定**
为了避免后面方向飘掉，我建议你现在先拍板这 3 件事：

1. 论文先不做连续时间主方法，先做 `Gap-aware + Uncertainty-aware + Conformal`
2. 数据先用 `OpenSky` 公开数据体系
3. 先把第一版目标定成：
   - `普通重采样 Hierarchical GRU`
   - 升级为 `Gap-aware Hierarchical GRU`
   - 再做 `uncertainty calibration`

这样路线最稳。

如果你同意，我下一步可以直接给你出一个**更工程化的改造任务清单**，按“文件级别”展开成：

- 每一步要改哪些文件
- 新增哪些类/函数
- 每一步如何验收
- 哪一步风险最大

这样你后面就可以继续像之前那样，一步一步让我实现。