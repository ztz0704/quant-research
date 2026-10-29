# A 股量化研究：四个项目总入口

**2026-10-13：** V3 开盘事件记账、风险优化和跨项目接口已通过工程验收。
统一真实历史采集器已下载 11 股的 2015—2025 样本，包含退市与停牌；
数据为 **EXPLORATORY_ONLY**，正式 PIT 实证仍被阻断。
[历史数据及命令](#统一真实历史数据2026-10-13) ·
[冻结工程协议](Robust_Alpha_Discovery/configs/protocols/research_v2.txt) ·
[历史 V3 科学报告](nonstationary_financial_ml/outputs/v3_validation/research_results.md)。

整理日期：2026-10-13。本目录包含因子发现、金融机器学习、风险组合和微观结构执行四个 Python 项目。本 README 汇总使用顺序、真实数据入口和结果位置；详细设计和参数见各项目自己的 README。

**可以直接运行真实历史数据研究和添加量价因子。当前结果尚未确认可实盘盈利的 alpha，也没有四项目贯通的券商交易系统。** 已生成的报告、图表和历史预测可以直接打开，无需申请数据账号或重新训练。现有缓存是历史样本，不能作为今日选股数据。

实盘操作与补齐路径见 [从研究到 A 股实盘](alpha_research_playbook/README.md)：从券商 App 的手动委托、前向纸面记录，到近期行情核验、整手订单规划和账户接口。

**2026-10-05 数据质量补充：**近期接口实测发现 AKShare 腾讯 `000` 前缀股票的成交量单位问题；新核验工具已加入限定版本/实现的修正，原缓存未改写。旧因子缓存也有相关异常迹象，成交量/流动性相关结论需重建数据并另存实验后复核。详情与原始证据见上述实盘说明。

## 1. 四个项目分别做什么

| 项目 | 用途 | 真实数据运行入口 | 项目 README |
|---|---|---|---|
| `Robust_Alpha_Discovery` | 计算自定义量价因子，做训练/验证筛选、多重检验、冻结评价和日期分数导出 | `configs/mine_real.yaml` + `run_factor_mining.cmd` | [因子发现](Robust_Alpha_Discovery/README.md) |
| `nonstationary_financial_ml` | 比较简单与非线性模型，检验未来预测、漂移、重训和模型增量 | 默认 `configs/base.yaml` + `run_real_research.cmd` | [金融机器学习](nonstationary_financial_ml/README.md) |
| `risk_aware_portfolio` | 将信号转为受仓位、行业、风格、风险和成本约束的组合，做收益归因 | `configs/ashare.yaml` + `scripts/validate_ashare.py` | [风险组合](risk_aware_portfolio/README.md) |
| `market_microstructure_execution` | 从真实逐笔重建订单簿，研究短周期信号、成交、库存、费用和执行损失 | 默认 `configs/base.yaml` + 准备、训练、回放脚本 | [微观结构与执行](market_microstructure_execution/README.md) |

目录结构：

```text
项目/
├── README.md                       本总入口
├── Robust_Alpha_Discovery/          因子发现
├── nonstationary_financial_ml/      金融机器学习；第四个项目在这里
├── risk_aware_portfolio/            风险组合
├── market_microstructure_execution/ 微观结构与执行
├── alpha_research_playbook/         中文操作手册、研究协议和报告导航
└── tmp/                            临时文件与依赖缓存
```

建议从**因子发现**开始：提出经济假设、固定公式和方向，再检验历史表现。ML 用于检验组合特征是否有增量；风险项目检验暴露与成本；执行项目检验能否成交。四个工具各自回答不同问题，复杂模型或优化器不会自动产生有效 alpha。

## 2. 最快开始：先打开已有结果

双击 [open_reports.cmd](alpha_research_playbook/open_reports.cmd)，可打开操作手册和已有报告。它只打开文件；Markdown 文件需要本机有相应编辑器关联。新增的批量因子挖掘结果可从下面第一行直接打开。

| 文件 | 先看什么 |
|---|---|
| [批量因子挖掘报告](Robust_Alpha_Discovery/reports/generated/factor_mining/mining_report.md) | 23 个候选的筛选结论和拒绝理由 |
| [原有真实因子实验](Robust_Alpha_Discovery/reports/generated/akshare_smoke/validation_report.md) | IC、因子收益多重检验、walk-forward 与费用敏感性 |
| [ML 图文报告](nonstationary_financial_ml/outputs/research_report.html) / [Markdown](nonstationary_financial_ml/outputs/research_report.md) | 7 个模型、未来测试、漂移与增量检验 |
| [风险组合报告](risk_aware_portfolio/outputs/ashare/research_report.md) | 四种组合的净收益、暴露、换手、费用和归因 |
| [微观结构报告](market_microstructure_execution/outputs/research_report.html) / [Markdown](market_microstructure_execution/outputs/research_report.md) | 重建来源、信号、成交账、库存与限制 |
| [ML→风险组合连接报告](nonstationary_financial_ml/outputs/portfolio_bridge/ridge_raw_prediction_daily/research_report.md) | 已实际运行的 Ridge 日调仓组合连接 |
| [本地数据与产物清单](alpha_research_playbook/outputs/readiness.md) | 四个项目的数据维度、文件和已有结果 |

下文命令均为 **Windows CMD**。工作目录若移动到其他位置，替换 `cd /d` 后的路径。各项目使用自己的 `pyproject.toml`、`uv.lock` 和 `.venv`，不要混成一个公共环境；已配置环境以 Python 3.12 为主。

仅重新盘点本地结果：

```bat
cd /d "C:\Users\19388\Desktop\项目"
nonstationary_financial_ml\.venv\Scripts\python.exe alpha_research_playbook\check_readiness.py
```

该清单检查文件和数据维度；完整的重算核验使用各项目验收脚本。阅读报告不需要执行 `uv sync`；首次部署或依赖缺失时再同步环境，同步可能需要联网下载依赖。

## 3. 直接挖因子：从真实缓存开始

### 3.1 运行现有候选

```bat
cd /d "C:\Users\19388\Desktop\项目\Robust_Alpha_Discovery"
uv sync --dev
run_factor_mining.cmd
```

批处理依次运行 `mine`、`evaluate` 和产物核验。默认使用 AKShare 获取的 40 股 2018—2020 年真实缓存；2018 年训练、2019 年验证。2020 年是已查看过的探索测试样本，不是未接触的最终测试集。

**本轮 23 个候选中 0 个通过。** 空选择有明确记录，本次评价没有加载测试行情，也没有强行生成选股分数。这说明当前样本与规则下没有候选达到要求，不说明程序不能研究因子。

按以下顺序读结果：

| 产物 | 用途 |
|---|---|
| [factor_catalog.csv](Robust_Alpha_Discovery/reports/generated/factor_mining/factor_catalog.csv) | 全部公式、方向与候选家族 |
| [factor_screening.csv](Robust_Alpha_Discovery/reports/generated/factor_mining/factor_screening.csv) | 训练/验证 IC、HAC、Holm、成本后增量、覆盖率与拒绝原因 |
| `reports/generated/factor_mining/factor_values.parquet` | 逐股逐日原始因子与横截面处理值，可用 pandas 读取 |
| [portfolio_diagnostics.csv](Robust_Alpha_Discovery/reports/generated/factor_mining/portfolio_diagnostics.csv) | 不同费率下的收益、基准与换手 |
| [selection.json](Robust_Alpha_Discovery/reports/generated/factor_mining/selection.json) | 冻结选择及配置、代码、输入身份 |
| [trials.jsonl](Robust_Alpha_Discovery/reports/generated/factor_mining/trials.jsonl) | 每次试验的追加记录，包括失败与空选择 |
| [mining_verification.json](Robust_Alpha_Discovery/reports/generated/factor_mining/mining_verification.json) | 真实因子重建、哈希与未来行情扰动检查 |

### 3.2 添加自己的量价因子

先复制配置，保留上一轮：

```bat
copy configs\mine_real.yaml configs\local_mine_v2.yaml
```

编辑副本，将 `output.directory` 改为 `reports/generated/mine_v2`。在现有 `custom_signals` 中加入定义，并把名称加入现有 `signals` 列表。例如：

```yaml
# 合并到副本对应字段，不是用这段替换整个配置文件。
signals:
  - rev_1
  - my_reversal_10d
custom_signals:
  my_reversal_10d:
    expression: "-ts_sum(ret_1, 10) / (ts_std(ret_1, 20) + 0.000001)"
    direction: 1
    description: "事先声明：波动调整后的十日反转"
```

公式展示可运行接口，本身没有有效性承诺。运行前记录经济理由、候选范围、固定方向、时间区间、费用和否决标准；不要看完测试结果再翻转方向或删除失败候选来降低校正 p 值。

```bat
uv run python scripts\mine_factors.py mine --config configs\local_mine_v2.yaml
uv run python scripts\mine_factors.py evaluate --config configs\local_mine_v2.yaml
uv run python scripts\verify_mining.py --config configs\local_mine_v2.yaml
```

表达式支持 OHLC、成交量、成交额、`ret_1`、历史滚动运算和当日横截面排名；禁止负向 delay、未来标签及任意 Python 执行。当前因子在收盘后形成，目标是下一开盘进入、再下一开盘退出。通过研究筛选后才可使用 `score` 导出指定日期分数；当前默认缓存止于 2020 年，不能生成今日指令。

下载新历史数据、导入真实 CSV、公式语法、封存评价和分数导出的完整步骤见 [直接挖因子说明](Robust_Alpha_Discovery/README.md)。当前接口是沪深量价因子，财务公告因子和北京市场尚未接入。

> 配置选择：`mine_real.yaml` 是新的直接挖因子入口；`akshare_smoke.yaml` 是原有真实实验；该项目默认 `base.yaml` 使用合成数据，不能将默认 demo 当作 A 股实证。

## 4. 运行金融机器学习项目

```bat
cd /d "C:\Users\19388\Desktop\项目\nonstationary_financial_ml"
run_real_research.cmd
```

批处理同步依赖、运行测试、准备真实数据、构建特征、训练和评价模型、分析漂移、生成诊断报告、导出历史信号并核验 Notebook 与产物。完整重训和诊断耗时较长；已有结果不需要为阅读而重跑。

默认复用相邻风险项目的真实面板，再保存自己的输入快照。源文件缺失会报错，不回退到合成行情。已运行 7 种模型、10 个历史特征和 4 个未来测试折，保存 28 个模型，产生 70,553 条历史 OOS 预测。

主要结果位置：

- `outputs/tables/fold_boundaries.csv`、`model_fits.csv`、`tuning_results.csv`：时间边界、标签到期与验证调参。
- `outputs/tables/model_performance.csv`、`ic_summary.csv`、`paired_model_comparison.csv`：模型表现和按日期配对的增量。
- `outputs/tables/drift_summary.csv`、`window_retraining_summary.csv`、`importance_stability.csv`：漂移、实际重训与特征稳定性。
- `outputs/predictions/ridge_raw_prediction_signal.csv`：历史 OOS 分数，不是今日预测或已校准的预期收益。
- [artifact_verification.json](nonstationary_financial_ml/outputs/tables/artifact_verification.json)：模型重放、未来扰动与产物身份核验。

改特征或预测期限时，重新构建标签到期日期、训练和评价；不要只改调仓频率。新增因子不会自动进入 ML，必须按股票、日期和可用时点适配。分步命令与诊断解释见 [ML README](nonstationary_financial_ml/README.md) 和 [中文操作手册](alpha_research_playbook/README.md)。

## 5. 运行风险组合与 ML→组合连接

### 5.1 复现原有真实信号组合

```bat
cd /d "C:\Users\19388\Desktop\项目\risk_aware_portfolio"
uv sync --extra real-data
uv run --extra real-data python scripts\prepare_data.py --existing --config configs\ashare.yaml
uv run --extra real-data python scripts\validate_ashare.py
```

必须使用 `--existing --config configs\ashare.yaml`。默认工程数据准备入口使用合成数据；真实数据的来源、配置与缓存需要共同通过审核。真实配置使用 60 股缓存、2022 年预热和 2023 年评价，原信号是固定五日反转。

检查净收益之外的行业/风格暴露、换手、费用、风险贡献、归因、求解器状态和约束可行性。外部信号要求唯一的 `(date, asset)` 键和 `alpha`，并满足必要日期与可交易资产覆盖，不能静默填充旧分数。细节见 [真实数据验证说明](risk_aware_portfolio/README.md)。

### 5.2 已经跑通的 Ridge 日调仓连接

先确保 ML 历史产物和风险项目环境存在，再运行：

```bat
cd /d "C:\Users\19388\Desktop\项目\risk_aware_portfolio"
set PYTHONPATH=src
.venv\Scripts\python.exe ..\nonstationary_financial_ml\scripts\run_portfolio_bridge.py --model ridge --prediction raw_prediction --rebalance-frequency daily
```

桥接脚本审核原始来源，创建独立派生面板与信号，运行四种组合和未来扰动检查；原风险数据与配置不被替换。结果位于 `nonstationary_financial_ml/outputs/portfolio_bridge/ridge_raw_prediction_daily`，决策日期为 2023-03-30 至 2023-12-06。

这里的 `alpha` 被风险项目作为横截面分数重新标准化，并乘固定 `scale=0.0006`。这不是经训练期校准的 ML 日收益预测。日调仓用于对齐一日预测目标，此前周调仓实验另行保留，没有根据测试收益选择默认。

## 6. 运行微观结构与执行研究

```bat
cd /d "C:\Users\19388\Desktop\项目\market_microstructure_execution"
uv sync
uv run python scripts\prepare_data.py
uv run python scripts\build_features.py
uv run python scripts\train_signal.py
uv run python scripts\run_backtest.py
uv run python scripts\generate_report.py
uv run python scripts\verify_notebooks.py
uv run python scripts\verify_artifacts.py --require-notebooks
```

默认配置使用公开深市真实逐笔委托/成交数据，以固定来源版本和 SHA 重建 000400.SZ 在 2022-06-01 的订单簿。主样本有 14,179 个有效快照；当前实时盘口探测是辅助来源，没有混入历史训练。

先看未来信号检验，再看成交、拒单、库存、现金与费用的事件账。当前主要结果是持仓盯市，不能解释为已完成的日内套利。Maker 排队和冲击参数尚未用真实成交反馈校准；一个股票日无法证明跨日期、跨股票的执行优势。

下一轮执行研究应固定同一父订单、初始已结算库存、截止时间与风险预算，比较成交损失、完成率和残余库存，再扩大独立股票日。来源与规则详见 [项目 README](market_microstructure_execution/README.md)。

## 7. 当前真实数据与 alpha 证据

| 项目/实验 | 当前真实样本 | 当前结论 |
|---|---|---|
| 新因子挖掘 | 40 股，2018—2020；筛选使用 2018—2019 | 23 个候选全部未通过；拒绝理由有记录 |
| 原有因子实验 | 同一 40 股缓存 | 五个候选的因子收益多重检验未通过；IC 检验与因子收益检验不能混用 |
| ML | 60 股，2022 年预热至 2023 年末；未来测试 168 个日期 | Ridge 原始 RankIC 约 0.0216，HAC p 约 0.243；尚无可靠增量证据 |
| 原风险组合 | 60 股，2023 年评价 | 部分组合有正收益，但相对 Baseline 改善的校正检验不足以确认 alpha |
| Ridge→风险组合 | 同一 60 股历史样本，2023 年日调仓 | Risk+TC 诊断净年化约 −1.29%，相对等权基准的日均超额 HAC p 约 0.765 |
| 微观结构 | 000400.SZ，2022-06-01，一个股票日 | 该股票日的 OOS 预测有信息；成交、库存与未平仓限制使实盘盈利尚未得到证明 |

这些结论来自上面链接的本地产物。测试通过、文件哈希一致、未来扰动不改变过去，验证的是工程行为；它们不能认证供应商历史发布时点，也不能替代经济和交易有效性。

**成本口径要对齐：**因子和 ML 诊断按半 L1 换手计线性费用，风险项目报告完整 L1 换手；同样的 bps 数字不能直接比较。最低佣金、印花税、非线性冲击和实际成交约束还需单独核对。

## 8. 如何让下一轮研究有实际功效

1. **写试验协议。** 使用 [research_protocol.md](alpha_research_playbook/deployment/templates/research_protocol.txt)，固定假设、候选范围、方向、信息时点、持有期、基准、费用、数据切分与否决条件，登记所有尝试。
2. **补齐可追溯数据。** 获取真实历史交易日历、上市/退市和股票池、原始价格与复权事件、逐日交易状态；财务因子需公告可用时间与修订版本。当前固定事后股票池不能代表无偏全市场。
3. **先筛选因子，再检验模型增量。** 用同一日期、股票池和目标比较简单基准与新增因子/模型；保留失败候选，多重检验覆盖事先声明的候选族。
4. **冻结后评价新期间。** 只在训练/验证段调参；已查看的期间保持探索标记。测试后修改方案形成新试验，不能继续把同一测试段称为未接触样本。
5. **在统一规则下检验净增量。** 将同一信号接入风险组合，控制暴露、持有期和费用；再用匹配股票日的订单簿检验同一父订单的执行损失。
6. **增加未来纸面运行。** 补齐数据更新、异常处理、整手、涨跌停、结算库存、账户对账与调度后，记录新的未来预测和模拟成交。当前工程尚未实现完整实盘部署。

四项目连接时必须先统一这些内容：

| 内容 | 当前状态与需要的工作 |
|---|---|
| 因子→ML | 新因子表尚未自动接入 ML；40 股 2018—2020 与 60 股 2023 样本不对齐 |
| 收益区间 | 因子入口是收盘后决策、下一开盘至后日开盘；ML/风险当前使用下一期收盘收益规则，需显式统一 |
| ML→风险 | 历史 Ridge 连接已实际跑通；分数尺度仍是假设，需要另行做过去数据上的收益校准 |
| 风险→执行 | 当前订单簿是 2022 年一个股票日，无法覆盖 2023 年日频组合；权重到父订单、现金、整手和可卖库存尚需适配 |
| 每日运行 | 现有历史缓存与模型不构成实时服务；需要新数据、来源审核、明确可用时点和未来运行记录 |

## 9. 文档导航与常见问题

- [四项目详细中文操作手册](alpha_research_playbook/README.md)：分步运行、读表、结果解释和下一轮研究计划。
- [直接挖因子说明](Robust_Alpha_Discovery/README.md)：自定义公式、真实数据下载/导入、冻结评价与分数导出。
- [ML 项目说明](nonstationary_financial_ml/README.md)：模型、时间切分、漂移、诊断与产物验收。
- [A 股风险数据验证](risk_aware_portfolio/README.md)：来源、缺口、停牌、信号和记账边界。
- [微观结构项目说明](market_microstructure_execution/README.md)：逐笔来源、订单簿重建、执行假设与账本。

**“ML 在 tmp 里吗？”** 第四个正式项目是 `nonstationary_financial_ml`；`tmp` 用于临时文件和缓存。

**“四个能直接挖因子吗？”** 因子发现项目提供直接公式研究入口；另外三个分别检验模型增量、风险组合和执行可行性。四者配合用于研究，但并未自动贯通。

**“为什么没有选股名单？”** 当前筛选没有通过候选，而且缓存不是最新行情；程序拒绝输出强行挑选或过期分数。

**“必须申请数据权限才能看结果吗？”** 已有本地结果和缓存可以直接使用。新增下载取决于网络与上游接口；长期 PIT 数据和多股票日逐笔数据需要另行确认合法来源与覆盖，不能用合成数据填补缺口。

**“看到正收益就找到 alpha 了吗？”** 还需要相对基准、暴露、费用、完整收益覆盖、多重试验和新的未来样本证据。当前报告保留这些限制，未将历史最佳结果提升为盈利承诺。


## 统一真实历史数据（2026-10-13）

三个中低频项目共用 Alpha 项目中的历史采集器。Python 3.12、uv、pandas、pyarrow；正式数据入口仍是已有 V3 `read_bundle`，不创建新研究项目。 canonical 文件只保留一份，metadata 指定文件名，旧 `panel.parquet` bundle 继续兼容。

当前来源顺序：仅从环境变量检测 Tushare Token，未配置便记录 `not_configured`；实测 BaoStock 匿名免费服务；用 AKShare 东方财富、腾讯、退市资料及日历交叉核对；本地缓存只盘点、保留，不混入新数据。没有购买、索取凭证或输出凭证明文。已验证的接口、列、日期和失败均见 `Robust_Alpha_Discovery/data/historical_v3/source_access.json`。

已完成 10 股×5 交易日测试及 2015 年动态生命周期测试。主数据是同一已验证样本的 2015—2025 扩展：11 只股票，2,674 个交易日，29,414 个股票日期行，19,785 行原始成交价格；保留招商地产、宏源证券、国恒退、暴风退四只历史退市股票。2015 年测试还包含当年新上市的暴风，以及 345 行明确停牌记录。

主 bundle：`Robust_Alpha_Discovery/data/historical_v3/range/`，包括 `canonical_market_panel.parquet`、`calendar.parquet`、生命周期、复权事件、历史发现列表、动态成员表与独立供应商参考表。范围中的未上市、已退市和停牌价格留空。11 股是采集工程样本，不能代表全市场历史股票池，也不按未来表现筛选 Alpha。

**结论：EXPLORATORY_ONLY，未达到正式研究要求。** 事件日期不等于历史可用时间。所有无法认证的 `*_asof` 留 NaT，因此正式资格及收益标签均不开放。行业、日历和指数成员供应商快照独立保存；不将当前行业或列表投射到过去。总/流通市值、官方限价、方向成交资格及完整分红现金/股数/支付/退市结算账仍缺失。历史下载成功不构成 PIT 认证。

价格为人民币，成交量为股，金额为人民币；BaoStock 换手率由百分数转小数，东方财富成交量由手乘 100。沪深各一只股票的 5 日 OHLCV/金额与东方财富一致，两份 2015—2025 日历一致；AKShare 1.19.1 腾讯 `sz000001` 成交量存在 100 倍差异，该补充来源被排除，未静默改写。

### 采集、恢复和验收

```bat
cd /d C:\Users\19388\Desktop\项目\Robust_Alpha_Discovery
uv sync --dev --python 3.12
uv run robust-alpha-history all --config configs\historical_data.yaml
uv run robust-alpha-history crosscheck --config configs\historical_data.yaml
uv run robust-alpha-history verify --bundle data\historical_v3\range
uv run python scripts\audit_pit_bundle_v3.py --bundle data\historical_v3\range --load-end 2025-12-31 --output reports\generated\historical_data\v3_audit
```

`all` 依次探测、5 日、年度、11 股全范围、批量门禁。接口错误按 1/2 秒指数退避最多 3 次，串行请求间隔至少 1 秒；每个调用有进程超时。成功股票年度分区按原始/Parquet SHA 恢复；失败、空分区及未解释缺日单独记录，不跳过、不补合成价格。原始档案是 SDK 规范化之前的解码响应 JSON，保留下载 UTC 时间、接口参数、SDK 版本和 SHA；不是底层网络报文。工作临时文件和 checkpoint 可更新，`archive/raw` 永不覆写。

`max_assets: 1000` 提供长期批量入口，阶段不过即停止。默认 `allow_exploratory_bulk: false`，本次千股范围被 PIT 门禁阻断；500—1000 股并未宣称下载完成。需要扩大探索采集时可明确修改此设置；仍保留 EXPLORATORY_ONLY 和全部缺失证据，不改变正式研究门禁。批量选择保留退市股票，按生命周期扩展，不使用单独当前上市名单。

已有 V3 标签契约保持：t 收盘后决策，下一交易所开盘进入、再下一开盘退出，调整开盘价简单收益，退出日收盘才成熟。调整因子只向后关联已生效记录；首个记录前留空，不能从缺失历史制造 1。公司行动绝不由因子比例模拟现金或股数。

### 协议、结果与说明文件

冻结工程协议位于 `Robust_Alpha_Discovery/configs/protocols/research_v2.txt`，迁移保留逐字节内容及 SHA `71aec05da228e6865c42437434159e46ae725a4f30bcef5563ae2df90619f998`。采集 2024—2025 行情并不运行 Alpha 或解除研究留出门禁。本次没有确认性 Alpha 报告。

原 V3 科学结果与图迁入 `nonstationary_financial_ml/outputs/v3_validation/research_results.md`，历史实验、模型、账本、日志和测试证据保留。新采集改变了实现身份，旧实验的过去验证结论保留，不冒充新代码已重新训练验证。项目使用说明集中在此总 README 和各子项目 README；冻结契约、数据 manifest 和实验报告仍按用途保留。说明清理清单位于 `Robust_Alpha_Discovery/reports/generated/historical_data/documentation_cleanup.json`。
