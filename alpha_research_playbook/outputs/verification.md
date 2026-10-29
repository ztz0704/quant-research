# 本次本地交付核验

日期：2026-10-05。范围是研究工程和历史缓存，不是可交易 alpha 认证。

- 四个项目的真实配置、面板和报告均存在；维度清单见 readiness.md / readiness.json。
- 补建 Nonstationary Financial ML 的九个阶段完成。独立 uv 环境 pytest 34 passed，Ruff 通过；两个日期类型依赖弃用提示保留，不影响本次结果。
- 从本项目真实原始快照重建特征/标签，与归档精确一致；28 个保存模型复现 OOS raw 预测最大差异均为 0。
- 七个模型的完整未来特征/标签/收益扰动检查均通过，历史预测和过去调参不变。
- 24 张图已生成并通过身份核验；三张总览覆盖全部图，已逐一目视检查标题、轴、图例和非空内容，未发现裁切。总览只供 QA，原始图在 ML outputs/figures。
- 四本 Notebook 实际执行，执行后身份验收通过。
- ML→风险组合的日调仓连接已完成；原始供应商数据先核验，派生数据与结果独立存放；前缀和未来扰动检查权重最大差异为 0。周调仓旧实验保留。
- 使用手册、研究协议/试验日志模板和报告打开器已生成。

验收的机器可读证据：nonstationary_financial_ml/outputs/tables/artifact_verification.json、notebook_execution.json、test_results.xml；连接证据在 outputs/portfolio_bridge/ridge_raw_prediction_daily。

本轮没有确认 ML 增量 alpha。固定事后股票池、供应商历史可用时间、近期数据、真实成交/容量、动态股票生命周期和多股票日订单簿等缺口仍存在。没有用合成样本替代这些缺口，也没有生成券商实际订单。
