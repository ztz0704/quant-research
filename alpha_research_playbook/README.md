# 研究使用与执行辅助工具

四项目操作、历史数据契约和验收结果统一见[总 README](../README.md)。本目录保留辅助工具，避免另存一份重复操作手册。

`check_readiness.py` 只读取已有实验产物、输入维度及文件身份，不认证 PIT 或可实盘 Alpha。使用各研究项目锁定的 Python 环境运行；查看参数执行 `--help`。

`deployment/recent_data_probe.py` 是近期日线探测，不能替代历史动态股票池。`deployment/order_planner.py` 生成手工复核的订单建议，要求 targets、quotes、account、calendar、execution-at 与新 output 目录；输入模板在 `deployment/templates`，保留原始输入 SHA 和计划身份。

```bat
..\Robust_Alpha_Discovery\.venv\Scripts\python.exe deployment\order_planner.py --help
..\Robust_Alpha_Discovery\.venv\Scripts\python.exe deployment\recent_data_probe.py --help
```

执行前须由真实账户提供可用现金、已结算库存和当时的执行报价、停牌/限价状态。账户字段不能从研究 NAV 推导。计划工具没有券商下单权限；没有经过净值、约束、成本、容量及纸面前向检验的分数不能当可交易预期收益。

冻结实验协议模板存放于 `deployment/templates/research_protocol.txt`。`replay_v3_risk_stage.py` 仅允许明确的历史风险阶段重放，并核对输入/实现身份；`render_v3_report.py` 只绘制已经验证的科学结果。历史 V3 图和报告集中在相邻 ML 的 `outputs/v3_validation`。
