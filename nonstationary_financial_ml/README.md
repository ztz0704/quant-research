# Nonstationary Financial ML

2026-10-12 新增 `run_research_v2.cmd`：共享动态原始数据、统一下一开盘至后日开盘简单收益、
Ridge/HGB 固定比较、独立历史校准段、完整 OOS 身份与自融资长仓诊断。输出在
`outputs/research_v2/historical_integration`，合成测试另存 `engineering_validation`。
这是 V2 历史集成；V3 已实现开盘风险优化并通过工程测试，正式 PIT 实证仍受阻。
操作与缺失字段见[数据接入说明](../README.md)。旧 base 配置用于复现原结果。

基于真实 A 股缓存的横截面收益预测、非平稳性和模型增量研究工程。默认是真实数据；数据缺失直接报错，合成数据只用于隔离的工程验证。九个阶段已经完成，阶段记录见 [outputs/ (阶段实验与测试证据)](README.md)。四个项目的完整中文操作手册见 [../alpha_research_playbook/README.md](../alpha_research_playbook/README.md)。

**当前结论：尚未发现可靠的 ML 增量 alpha。** 软件能运行、模型能预测与 alpha 被确认是不同的验收标准。这里报告实测结果及限制，不保证盈利。

## Project Overview

目标是回答：在按日期严格分割、标签到期处理、交易成本与风险暴露约束下，复杂模型是否比简单基准提供增量？分布漂移、关系漂移、训练窗口和重训频率如何影响结果？

实现包含七种模型、滚动/扩展时间验证、日度 IC/HAC、日期区块 bootstrap、分位数组合、成本敏感性、分布漂移、实际重训、单特征/分组置换、重要性稳定性、滞后市场状态、扰动、参数/种子敏感性和真实输入未来扰动检查。Notebook 读取诊断产物，核心研究逻辑在可测试模块中。

## Why Nonstationarity Matters

特征分布变化不等于预测关系变化。原始与处理后 PSI/KS、原始均值/标准差/分位数分别监控分布；成熟特征 IC、系数和模型年龄诊断关系与衰减。漂移统计不自动触发重训；重训是否改善需要同一未来日期的比较。

## Why Financial ML Can Fail

重复试验、股票行相关、标签重叠、未来股票池、行业/市值暴露、成本、模型年龄和制度变化都可能让拟合结果失效。随机拆分不适用于这里的主检验；不把横截面股票数量当独立时间样本数。R² 为负或 RankIC 为正都不能单独决定可交易性。

## Research Questions

1. 非线性模型是否在同样的未来日期和信息集下优于 Ridge？
2. 收益排序在中性化与成本后是否仍存在？
3. 分布变化、特征关系变化与模型年龄是否能解释性能变化？
4. 不同窗口和实际重训规则是否稳定，而非测试期偶然赢家？
5. 增量是否经得住日期区块不确定性、参数、种子和数据扰动？

## Repository Structure

```text
configs/                 base.yaml 真实数据；engineering.yaml 隔离合成工程验证
data/raw/                本项目真实输入快照
data/processed/          历史特征、未来标签、到期日期与选择掩码
src/financial_ml/        数据、特征、时序、模型、评价、漂移、诊断与报告模块
scripts/                 独立 CLI、导出、验收、ML→风险项目连接
tests/                   时序与数据不变量、已知信号/空信号、费用等测试
outputs/ (阶段实验与测试证据)         九阶段实施记录
notebooks/               四本已执行的诊断与最终报告 Notebook
outputs/models/          每折保存的模型
outputs/predictions/     OOS 预测与外部信号
outputs/tables/           核心 CSV、实验边界、来源与验收清单
outputs/figures/          24 张科学图
outputs/engineering_validation/  合成对照，不混入真实报告
outputs/portfolio_bridge/        派生的历史连接实验
```

## Data Schema

一行一个 `(date, asset)`，键唯一且排序，至少包含 `date, asset, return, market_cap, industry`。收益为小数，市值正数，不允许无穷值；CSV 股票代码按字符串读取。当前实现要求每个资产覆盖同一个日期网格，生命周期股票池需额外开发，不能把缺失日期直接删掉。

`return` 是截至该日的已实现单期收益；`tradable` 表示该日已知交易状态，`feature_asof < date` 标记缓存预滞后特征的可用时点。标准 schema 支持额外列，但任意供应商数值不会自动被当作已验证特征。缺失活跃目标与不可交易状态有显式掩码；当前可选股票不根据未来收益是否缺失来决定。

## Real-Data Adapter

`configs/base.yaml` 使用相邻 `risk_aware_portfolio/data/processed/ashare_panel.parquet`，来源是既有 AKShare/Sina 缓存，复制为本项目独立快照，并保存来源路径和 SHA。当前共 **21,420 行、60 只固定股票、357 个日期**，包含 2022 年预热和 2023 年数据。数据最后日期是 2023-12-29，不是当前实时数据。

本项目的 SHA 和严格时序只证明文件身份及内部流程，**不认证供应商历史发布时点**。更完整的原始来源记录位于风险项目 `data/raw/ashare/data_manifest.json`。连接脚本额外核验该原始来源链。固定事后股票池、粗行业和历史股本可用性限制必须保留在结论中。

## Synthetic Data Design

`configs/engineering.yaml` 明确设 `source: synthetic`，资产用 `SYN` 前缀，数据与输出隔离。设计包括固定弱信号、空信号、系数符号突变、均值漂移、相关性变化和噪声放大；所有标签由定义的下一期机制生成。它们用于验证代码对已知机制的行为，不是 A 股实证，也不用于挑选真实策略。

```bat
uv run python scripts\prepare_data.py --config configs\engineering.yaml
uv run python scripts\build_features.py --config configs\engineering.yaml
uv run python scripts\train_models.py --config configs\engineering.yaml
uv run python scripts\evaluate_models.py --config configs\engineering.yaml
```

## Feature Engineering

默认 10 个特征：缓存预滞后的 size、momentum、volatility、liquidity，以及历史 rev_1、rev_5、mom_20、mom_60、vol_20、vol_60。新增收益特征额外滞后一日，以保守地保证时点；保留 `raw__` 列供原始分布漂移诊断。滚动函数只向过去读取。

特征组在配置中声明；增加特征需要明确其原始数据、可用时间、所用窗口和方向，并更新研究协议。没有真实财务字段时不伪造 value/quality 因子。

## Preprocessing

每日横截面 MAD/分位数缩尾、标准化，以及可选行业/风格残差中性化。它们只使用决策日可见横截面，不用全期间统计量。默认当前完整且可交易特征样本不足 20 只时整日不预测，数量判断不使用未来标签。零方差输出显式为零，缺失保留并排除预测。行业和风格中性化是线性横截面诊断，不能代替完整现金组合的风险控制。

## Label Construction

可配置 1/5/20 日等未来复合收益：`target(t)=prod(1+r(t+1...t+h))-1`，同时记录 `label_end` 和方向。当前默认 `h=1`。未来不可交易/缺失影响目标有效性，不回溯改变当日预测资格。

多日标签重叠需要 purge 和正确的误差处理；诊断组合按期限取非重叠决策。更换目标期限必须重跑整个流程，并重新审查成交/持有时点。

## Temporal Validation Design

默认 Rolling：126 个日期 Train → 42 个日期 Validation → 42 个日期 Test，两个边界各 2 个日期 embargo，测试步长 42 日，测试折不重叠。支持 Expanding。

训练/验证行用实际 `label_end` purge：边界以后才成熟的标签不能用于此前拟合。超参数根据 Validation 的平均日度 RankIC 选择，再用 Test 前已成熟的 Train+Validation 拟合。测试预测可包含最终未成熟目标，它们仍保留预测身份，但不进入标签评价。

## Models

Dummy、Linear、Ridge、Lasso、Elastic Net、Random Forest 和 Histogram Gradient Boosting 已运行。LightGBM 为可选注册模型，默认不需要安装。树模型早停不使用随机内部时序切分；主实验保存所有折参数、拟合截止日期和模型，限制数值线程以提高运行确定性。

复杂模型不是默认赢家，比较必须在相同的时间、目标、股票覆盖和风险暴露下进行。

## Evaluation Metrics

MAE/RMSE/R²、逐日 Pearson IC/RankIC、未年化 ICIR、HAC 标准误及有限模型家族的 Holm 校正；常数 Dummy 的日度 IC 未定义，不能伪造为可排序预测。

日期配对模型比较使用日期区块 bootstrap，当前 300 次、10 日区块，用于探索性区间。多模型/多参数事后筛选仍需要新的独立时期；不能把所有 bootstrap/敏感性结果当成多重检验已解决。

## Portfolio Diagnostics

十组分位数、长仓最高组、研究用多空组、预测分布、换手及 0/5/10/20 bps 成本。多空是研究诊断，不假设普通 A 股可自由融券。Dummy 长仓为等权、多空为零，避免常数分数按代码顺序选股票。

交易费用按漂移后旧权重到新权重的 **半 L1 换手**计价。当前完整成交、连续权重、线性费率，未实现真实整手/涨跌停/可卖库存账。活跃持仓目标缺失时该期收益为 NaN；完整期间 CAGR/Sharpe 为 N/A，图不会在缺失后拼接伪造的累计净值。参考基准是同固定篮子的费用为零等权，不是官方指数。

## Drift Diagnostics

PSI 使用训练期固定分箱、尾部无限边界及伪计数；KS 报告统计量，不把汇总股票行当独立 iid 样本给出确定性显著结论。分别记录原始特征与模型输入，以及滚动原始矩、特征 IC 和系数。特征 IC 只在目标到期后可用于诊断。

## Model Decay and Retraining

模型年龄曲线，以及 63/126/189 日和 Expanding 窗口 × 5/21/63 日重训的 **12 个真实拟合实验**。只使用当时成熟标签，统一未来日期，参数冻结于第一次过去验证；记录拟合次数及运行时间。不把测试期表现最佳的周期自动设为新默认。

## Feature Importance and Stability

OOS 日期内单特征与联合分组置换，保持横截面日期身份；分组内特征共用置换顺序。报告验证损失/IC 变化、线性系数、RF impurity、相邻折排名/Top-3/符号稳定性。重要性不是因果归因；强相关特征可以互相替代。rho=0.1/0.9 的已知相关工程对照单独保存。

## Regime Analysis

用过去的市场平均收益/波动/横截面离散度与趋势划分状态，阈值冻结于第一个训练段。六个状态是可重叠的边际分组，不是六个互斥且独立的市场。存档状态日期、样本数和规则，不从未来结果挑选阈值。

## Robustness Checks

特征噪声、只在训练目标添加噪声、10% 特征缺失并用训练均值处理、删除过去重要特征、显式相关性反事实；RF 三种 seed；RF/HGB 共 32 个声明参数组合 × 4 折的敏感性。此类测试描述脆弱性，不提供新的真实历史时期。

额外对真实第一折的未来特征、标签和收益大幅修改后，重新拟合全部七模型，核对历史预测与调参记录完全不变。保存的 28 个模型重新预测，与归档 raw OOS 逐值一致。

## Main Findings

当前 OOS 日期 **2023-03-30—2023-12-06**，168 日期、4 折、**70,553 条预测**。每个模型 10,079 条预测，10,078 条有有效未来标签。

| 模型/预测 | 平均 RankIC | RankIC 的 HAC p |
|---|---:|---:|
| Ridge raw | 0.021580 | 0.242545 |
| Linear raw | 0.021505 | 约 0.247 |
| Elastic Net raw | 0.022426 | 约 0.238 |
| Lasso raw | 0.020358 | 约 0.286 |
| Boosting raw | 0.017778 | 约 0.149 |
| Random Forest raw | −0.011188 | 约 0.409 |
| Ridge neutralized | 0.001302 | 约 0.907 |

表中 p 对应 `ic_summary.csv` 的 **日度 RankIC 均值**检验。全部 12 个有限检验 Holm p=1.0。Ridge R²≈−0.00488；中性化后关系显著减弱，提示可能主要重学已有暴露，但不是因果证明。完整测量、漂移/重训/重要性/状态/扰动结论见 `outputs/research_report.md`，不要只摘出最大一项。

## Connection to Risk-Aware Portfolio

已运行固定 Ridge 的历史 OOS 连接实验，原数据和供应商清单先核验，再派生独立数据，不改变原风险项目结果。使用风险项目已有 Python：

```bat
cd /d C:\Users\19388\Desktop\项目\risk_aware_portfolio
set PYTHONPATH=src
.venv\Scripts\python.exe ..\nonstationary_financial_ml\scripts\run_portfolio_bridge.py --model ridge --prediction raw_prediction --rebalance-frequency daily
```

默认日调仓对齐一日目标；以前的周调仓实验保留，不能把它解释成一周收益预测。风险项目将外部信号标准化并乘固定 `scale=0.0006`，它是分数尺度假设，不是 ML 预期收益校准。

日调仓连接的 Risk+TC 净年化≈−1.29%、Sharpe≈−0.045，相对同篮子基准日均超额 HAC p≈0.765。连接完成不等于确认 alpha。信号角色、源哈希、四组合、未来扰动与报告位于 `outputs/portfolio_bridge/ridge_raw_prediction_daily`。

## How to Run

在 CMD 中运行。环境由 `uv.lock` 固定；当前已安装，不需重复安装才能读报告。

```bat
cd /d C:\Users\19388\Desktop\项目\nonstationary_financial_ml
run_real_research.cmd
```

分步命令：

```bat
uv sync --dev
uv run pytest -q
uv run python scripts\prepare_data.py
uv run python scripts\build_features.py
uv run python scripts\train_models.py
uv run python scripts\evaluate_models.py
uv run python scripts\analyze_drift.py
uv run python scripts\run_diagnostics.py
uv run python scripts\generate_report.py
uv run python scripts\export_signal.py --model ridge --prediction raw_prediction
uv run python scripts\verify_notebooks.py
uv run python scripts\verify_artifacts.py --require-notebooks
```

配置实验在以上核心研究命令后加 `--config configs\your_experiment.yaml`，使用独立输出目录。Notebook 执行脚本当前验证默认真实配置的四本 Notebook。报告生成会重写 Notebook，之后重新执行再验收。改变配置或数据后必须重训并重建相应产物；不混用旧预测和新表。

## Outputs

七个要求的汇总表均存在：`model_performance.csv`、`feature_importance_summary.csv`、`drift_summary.csv`、`window_retraining_summary.csv`、`regime_summary.csv`、`robustness_summary.csv`、`portfolio_summary.csv`。另外保存日度 IC、折边界、调参/拟合、配对区间、置换稳定性、费用路径、种子和参数敏感性、原始矩与未来扰动检查。

24 张图覆盖 IC/RankIC、分位组、净值与缺失覆盖、费用/换手、重要性/稳定性、原始/输入漂移、年龄、窗口、重训、状态和敏感性。四本 Notebook 已执行：`01_data_diagnostics`、`02_model_comparison`、`03_drift_analysis`、`04_final_report`。

`outputs/predictions/ridge_raw_prediction_signal.csv` 是历史 OOS 的唯一 `(date, asset, alpha)` 分数导出，旁边 JSON 明确未经收益校准、非完整风险窗口覆盖、非实时指令。`oos_predictions.parquet` 保留所有模型、raw/neutralized、折和模型年龄。

## Reproducibility

存档配置、随机 seed、uv.lock、包版本、输入/特征/预测 SHA、源与脚本 SHA、图 SHA、模型、拟合日期和执行 Notebook SHA。读取评价产物会拒绝数据/配置/预测身份不一致。`verify_artifacts.py` 还从真实原始快照重建特征/标签，检查保存模型预测、成熟标签与未来扰动。

当前实际验收：**34 项测试通过**，Ruff 通过，28 模型预测复现，7 个未来扰动检查通过，24 图与 4 个执行 Notebook 核对通过。完整记录见 `outputs/tables/test_results.xml`、`notebook_execution.json` 和 `artifact_verification.json`。

## Limitations

固定事后股票池带幸存者选择风险；样本仅四测试折；粗行业/风格与流通市值并非认证的 PIT 历史；行情/标签时点不认证供应商出版历史；同年反复分析并非独立确认。诊断组合不含完整涨跌停/整手/T+1 库存、分红现金、借券、容量或真实成交；桥接继承理想化收盘交易和部分求解器不精确状态。OOS 信号额外滞后不等于实际可在对应收盘价成交。

PSI、系数、重要性、状态分析均为诊断，不自动构成因果解释。改变研究范围需重新登记多重试验族。当前没有实时数据更新、券商下单或生产运维服务。

## What I Would Do Next

冻结一到两个可解释假设及否决标准；补齐较长的日历、历史股票池、公司行动和可用时点；在新开发/验证期间比较 Ridge 与一个非线性模型的成本后配对增量。增加训练期收益校准与真实交易账，再使用同股票/同日期的逐笔数据评估执行。最终只在真正未触碰的时期评价冻结方案。

## Interview Talking Points

能解释标签到期 purge 与额外特征滞后的区别；为何日度 IC/日期区块比股票行 iid 检验合适；分布漂移和关系漂移为何不同；中性化后的增量与整体收益的区别；为何常数模型不能按代码排序挑股票；怎样用数据/模型/代码哈希及未来扰动测试证明内部时序；怎样用真实负面结果否定看似好看的研究假设。


## V3 公共历史 bundle

新数据共用相邻 Alpha 的 `data/historical_v3/range`。已有 `research_v3_pipeline.load_inputs` 支持 `data.source: bundle`、`bundle_directory: ../Robust_Alpha_Discovery/data/historical_v3/range`、显式 `load_end: 2025-12-31`，按 SHA 和预读日期过滤加载。详细采集操作与正式限制见[总 README](../README.md)。当前是 11 股 EXPLORATORY_ONLY，未知 as-of 留空，资格/有效标签为零；不要降低最小横截面、伪造时间或交易状态来强行训练。

V3 用 Ridge 与 HGB 同股票日样本比较，校准仅用已成熟训练标签；特征和标签来自同一交易所日历。风险项目通过自己的 Python 环境和 Parquet/JSON `ashare-open-events-v3` 运行。冻结协议已经迁移到 Alpha `configs/protocols/research_v2.txt`，内容和原 SHA 不变。科学历史结果保留在 `outputs/v3_validation/research_results.md`，既有 `outputs/research_v3`、模型及验证日志保留。
