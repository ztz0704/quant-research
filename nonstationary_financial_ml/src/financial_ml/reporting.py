"""Scientific figures and measured research narrative, with explicit empirical limits."""

from __future__ import annotations

import html
import importlib.metadata
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from financial_ml.evaluation import load_research
from financial_ml.utils import output, path, sha256, write_json


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    """Render compact tables without an optional tabulate dependency."""
    rows = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for values in frame[columns].itertuples(index=False, name=None):
        cells = [f"{value:.5f}" if isinstance(value, (float, np.floating)) and np.isfinite(value) else ("N/A" if pd.isna(value) else str(value)) for value in values]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def figures(cfg: dict) -> list[Path]:
    """Plot all models or ex-ante declared reference views, without selecting winners."""
    destination = output(cfg)
    tables = destination / "tables"
    saved = []
    plt.rcParams.update({"figure.dpi": 140, "axes.grid": True, "grid.alpha": .2, "font.size": 9})

    def read(name):
        return pd.read_csv(tables / f"{name}.csv")

    def save(name, title, ylabel, xlabel=""):
        fig = plt.gcf()
        if len(fig.axes) in (1, 2):
            axis = fig.axes[0]
            axis.set(title=title, ylabel=ylabel, xlabel=xlabel)
        else:
            fig.suptitle(title)
        fig.tight_layout()
        filename = destination / f"figures/{name}.png"
        fig.savefig(filename, bbox_inches="tight")
        plt.close(fig)
        saved.append(filename)

    metrics = read("model_performance")
    pivot = metrics.pivot(index="model", columns="prediction", values="mean_rank_ic")
    pivot.plot.bar(figsize=(9, 4))
    plt.axhline(0, color="black", linewidth=.8)
    save("model_ic_comparison", "All-model OOS Rank IC; undefined constant IC is omitted", "Mean daily Rank IC", "Model")
    ic = read("daily_ic")
    ic["date"] = pd.to_datetime(ic.date)
    raw = ic.loc[ic.prediction.eq("raw_prediction")]
    for metric, filename in [("ic", "rolling_ic"), ("rank_ic", "rolling_rank_ic")]:
        plt.figure(figsize=(10, 4))
        for model, group in raw.groupby("model", sort=True):
            plt.plot(group.date, group[metric].rolling(21, min_periods=10).mean(), label=model)
        plt.legend(ncol=4, fontsize=7)
        save(filename, f"21-session rolling {metric}; all declared models", metric, "Decision date")
    plt.figure(figsize=(9, 4))
    valid_models = [(model, group.rank_ic.dropna()) for model, group in raw.groupby("model") if group.rank_ic.notna().any()]
    plt.boxplot([values for _, values in valid_models], tick_labels=[name for name, _ in valid_models])
    save("ic_distribution", "Daily cross-sectional OOS Rank IC distribution", "Rank IC", "Model")
    deciles = read("decile_returns")
    plt.figure(figsize=(9, 4))
    for model, group in deciles.loc[deciles.prediction.eq("raw_prediction")].groupby("model"):
        mean = group.groupby("quantile").mean_forward_return.mean()
        plt.plot(mean.index, mean * 10000, marker="o", label=model)
    plt.legend(ncol=3, fontsize=7)
    save("decile_returns", "All-model mean forward return by prediction decile", "Mean forward return (bps)", "Prediction decile")
    daily = read("portfolio_daily")
    daily["date"] = pd.to_datetime(daily.date)
    primary = daily.loc[daily.prediction.eq("raw_prediction") & daily.cost_bps_per_half_l1.eq(10)]
    for kind, filename in [("long_only", "cumulative_long_only"), ("long_short", "cumulative_long_short")]:
        plt.figure(figsize=(10, 4))
        for model, group in primary.loc[primary.portfolio.eq(kind)].groupby("model"):
            nav = (1 + group.net_return).cumprod(skipna=False)
            plt.plot(group.date, nav - 1, label=model)
        plt.legend(ncol=4, fontsize=7)
        save(filename, f"Diagnostic {kind}, 10bps per half-L1; path stops on missing return", "Cumulative diagnostic return", "Decision date")
    summary = read("portfolio_summary")
    chosen = summary.loc[summary.prediction.eq("raw_prediction") & summary.portfolio.eq("long_only")]
    view = chosen.loc[chosen.cost_bps_per_half_l1.isin([0, 10])]
    view.pivot(index="model", columns="cost_bps_per_half_l1", values="annualized_return").plot.bar(figsize=(9, 4))
    save("gross_net_performance", "Diagnostic long-only return; incomplete series have N/A performance", "Geometric annualized diagnostic return", "Model")
    chosen.loc[chosen.cost_bps_per_half_l1.eq(10)].set_index("model").mean_turnover.plot.bar(figsize=(9, 4))
    save("turnover_comparison", "All-model long-only half-L1 turnover", "Mean half-L1 turnover per rebalance", "Model")
    importance = read("feature_importance_summary")
    heat = importance.loc[~importance.feature.str.startswith("group:")].pivot_table(index="feature", columns="model", values="permutation_rank_ic_loss", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(9, 5))
    color = ax.imshow(heat, aspect="auto", cmap="coolwarm")
    ax.set(xticks=range(len(heat.columns)), xticklabels=heat.columns, yticks=range(len(heat.index)), yticklabels=heat.index)
    ax.tick_params(axis="x", rotation=45)
    fig.colorbar(color, ax=ax, label="Held-out daily Rank IC loss")
    save("feature_importance", "Date-preserving OOS permutation importance; no causal interpretation", "Feature", "Model")
    stability = read("importance_stability")
    plt.figure(figsize=(9, 4))
    for model, group in stability.groupby("model"):
        plt.plot(group.current_fold, group.rank_correlation, marker="o", label=model)
    plt.legend(ncol=4, fontsize=7)
    save("importance_stability", "Importance rank correlation between consecutive OOS folds", "Rank correlation", "Current fold")
    relation = read("feature_relationships")
    relation["date"] = pd.to_datetime(relation.date)
    plt.figure(figsize=(10, 4))
    for feature, group in relation.groupby("feature"):
        plt.plot(group.date, group.rolling_rank_ic, label=feature)
    plt.legend(ncol=5, fontsize=7)
    save("feature_ic_over_time", "21-session feature Rank IC; outcomes available only after label_end", "Rolling feature Rank IC", "Feature date")
    drift = read("drift_summary")
    for representation in ("raw", "model_input"):
        heat = drift.loc[drift.representation.eq(representation)].pivot(index="feature", columns="fold", values="psi")
        fig, ax = plt.subplots(figsize=(8, 5))
        color = ax.imshow(heat, aspect="auto", cmap="viridis")
        ax.set(xticks=range(len(heat.columns)), xticklabels=heat.columns, yticks=range(len(heat.index)), yticklabels=heat.index)
        fig.colorbar(color, ax=ax, label="PSI with training-only bins")
        save(f"feature_drift_psi_{representation}", f"Training-to-test PSI ({representation})", "Feature", "OOS fold")
    moments = read("feature_moments")
    moments["date"] = pd.to_datetime(moments.date)
    fig, axes = plt.subplots(2, 2, figsize=(11, 6))
    available = set(moments.feature)
    views = [name for name in ("size", "momentum", "volatility", "liquidity") if name in available]
    views += [name for name in sorted(available) if name not in views]
    for feature, axis in zip(views[:4], axes.flat, strict=True):
        group = moments.loc[moments.feature.eq(feature)]
        axis.plot(group.date, group["mean"], label="mean")
        axis.fill_between(group.date, group.q10, group.q90, alpha=.2, label="q10-q90")
        axis.set(title=feature, xlabel="Decision date", ylabel="Raw feature units")
        axis.legend(fontsize=7)
    save("feature_distribution_drift", "Raw feature distributions; cross-sectional normalization can hide level drift", "")
    age = read("model_decay")
    plt.figure(figsize=(9, 4))
    for model, group in age.groupby("model"):
        mean = group.groupby("model_age_sessions").rank_ic.mean().rolling(5, min_periods=1).mean()
        plt.plot(mean.index, mean, label=model)
    plt.legend(ncol=4, fontsize=7)
    save("model_decay_curve", "Mean Rank IC by model age; four folds, limited independent evidence", "Rank IC", "Market sessions since last training observation")
    age["date"] = pd.to_datetime(age.date)
    joined = primary.loc[primary.portfolio.eq("long_only")].merge(age[["date", "model", "model_age_sessions"]], on=["date", "model"], validate="one_to_one")
    plt.figure(figsize=(9, 4))
    for model, group in joined.groupby("model"):
        mean = group.groupby("model_age_sessions").net_return.mean().rolling(5, min_periods=1).mean()
        plt.plot(mean.index, mean * 10000, label=model)
    plt.legend(ncol=4, fontsize=7)
    save("return_vs_model_age", "Diagnostic net return by model age", "Mean horizon return (bps)", "Model age (market sessions)")
    cadence = read("window_retraining_summary")
    plt.figure(figsize=(8, 4))
    group = cadence.loc[cadence.frequency.eq(21)]
    plt.bar(group.window.astype(str), group.mean_rank_ic)
    save("performance_vs_training_window", "Ridge common-date OOS study; parameters fixed from past validation", "Mean Rank IC", "Training sessions / expanding")
    plt.figure(figsize=(8, 4))
    for window, group in cadence.groupby("window"):
        plt.plot(group.frequency, group.mean_rank_ic, marker="o", label=str(window))
    plt.legend(title="Window", fontsize=7)
    save("performance_vs_retraining_frequency", "Ridge actual-refit cadence study; no test winner selected", "Mean Rank IC", "Market sessions between refits")
    regime = read("regime_summary")
    heat = regime.pivot(index="regime", columns="model", values="mean_rank_ic")
    fig, ax = plt.subplots(figsize=(9, 5))
    color = ax.imshow(heat, cmap="coolwarm", aspect="auto")
    ax.set(xticks=range(len(heat.columns)), xticklabels=heat.columns, yticks=range(len(heat.index)), yticklabels=heat.index)
    ax.tick_params(axis="x", rotation=45)
    fig.colorbar(color, ax=ax, label="Mean Rank IC")
    save("regime_performance", "Causal first-train thresholds; marginal regimes overlap", "Regime", "Model")
    sensitivity = read("hyperparameter_sensitivity")
    rf = sensitivity.loc[sensitivity.model.eq("random_forest")]
    heat = rf.pivot_table(index="max_depth", columns="min_samples_leaf", values="mean_rank_ic", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(7, 4))
    color = ax.imshow(heat, cmap="coolwarm", aspect="auto")
    ax.set(xticks=range(len(heat.columns)), xticklabels=heat.columns, yticks=range(len(heat.index)), yticklabels=heat.index)
    fig.colorbar(color, ax=ax, label="Mean exploratory OOS Rank IC")
    save("hyperparameter_sensitivity", "RF surface averaged over folds/other parameters; fixed seed, diagnostic only", "max_depth", "min_samples_leaf")
    seeds = read("seed_sensitivity")
    seeds.pivot(index="fold", columns="seed", values="prediction_correlation_to_declared_seed").plot(marker="o", figsize=(8, 4))
    save("prediction_stability", "RF prediction correlation across seeds on identical OOS observations", "Pearson correlation", "OOS fold")
    distribution = read("prediction_distribution")
    distribution["date"] = pd.to_datetime(distribution.date)
    plt.figure(figsize=(10, 4))
    for model, group in distribution.loc[distribution.prediction.eq("raw_prediction")].groupby("model"):
        plt.plot(group.date, group["std"] * 10000, label=model)
    plt.legend(ncol=4, fontsize=7)
    save("prediction_distribution", "All-model cross-sectional prediction scale", "Prediction cross-sectional std (bps)", "Decision date")
    plt.figure(figsize=(9, 4))
    for model, group in chosen.groupby("model"):
        plt.plot(group.cost_bps_per_half_l1, group.annualized_return, marker="o", label=model)
    plt.legend(ncol=4, fontsize=7)
    save("transaction_cost_sensitivity", "Diagnostic long-only cost sensitivity; fixed prediction paths", "Geometric annualized return", "Cost bps per half-L1 turnover")
    return saved


def report(cfg: dict, generated_figures: list[Path]) -> None:
    """State what was measured, what failed and what the dataset cannot establish."""
    destination = output(cfg)
    panel, predictions, features = load_research(cfg)
    data = json.loads((destination / "tables/data_manifest.json").read_text(encoding="utf-8"))
    metric = pd.read_csv(destination / "tables/model_performance.csv")
    portfolio = pd.read_csv(destination / "tables/portfolio_summary.csv")
    passed = metric.loc[metric.holm_p.le(.05) & metric.mean_rank_ic.gt(0)]
    first, last = predictions.date.min().date(), predictions.date.max().date()
    folds = predictions.fold.nunique()
    nature = "真实 A 股探索性研究" if data["empirical"] else "合成工程验证：不属于 A 股结果"
    text = [f"# Nonstationary Financial ML：{nature}", "",
            f"数据：{data.get('provider', 'controlled_engineering_simulation')}，{data['rows']:,} 行、{data['assets']} 个标的。样本外决策日期 {first} 至 {last}，{predictions.date.nunique()} 个交易日、{folds} 个时间窗、{len(cfg['models']['enabled'])} 类模型、{len(features)} 个特征。", "",
            "结论：当前是探索性历史研究。真实数据、无未来信息的程序和样本外评价已跑通；固定幸存股票池、粗行业划分、股本/复权历史的发布时点未认证，以及成交/容量模型的限制，仍然阻止将结果认定为可交易 alpha。", "",
            f"在本次已声明模型及原始/中性化预测比较中，有 {len(passed)} 个正平均 Rank IC 通过 HAC+Holm 5% 校正。这只针对当前比较集合，不能校正未记录的研究尝试，也不替代从未查看的新样本。", "",
            "## 预测与风险暴露", "",
            markdown_table(metric, ["model", "prediction", "mean_rank_ic", "r2", "hac_p", "holm_p"]), "",
            "原始预测与中性化预测都被完整报告。中性化以当日行业、规模、动量、波动暴露投影，不使用未来收益；这些暴露的来源仍受数据限制。原始 IC 消失于中性化后，说明预测可能主要重学已知暴露，不能把该差异解释成因果证明。", "",
            "## 经济评价", "",
            "每个决策使用当时可见的分数选股，收益是理论收盘(t)→收盘(t+h)。量化组合默认是每日、充分成交、分数等权；long-short 为研究诊断，不假设普通 A 股账户能自由融券。税费、涨跌停、整手、可卖库存、分红现金、订单规模和实际执行尚未成为该模块的成交账本。", "",
            "成本单位为每半 L1 换手 bps；风险项目采用另一种单边 L1 定义，两者不能直接比较数值。上一期持仓先按已知收益漂移再计算换手。多日标签的组合只在每 h 个日期决策，避免把重叠 h 日收益每日复利。", "",
            "主动持仓收益缺失时记为 NaN；完整序列的年化收益、Sharpe 不计算，图上的累计路径从该点停止。被排除的标签和未产生预测的当前不可交易股票均保留计数，不能用删股/重配掩盖失败。基准是费用前同研究篮子的等权收益，尚非官方指数或严格风险模型截距 alpha。", "",
            markdown_table(portfolio.loc[portfolio.prediction.eq("raw_prediction") & portfolio.portfolio.eq("long_only") & portfolio.cost_bps_per_half_l1.eq(10)],
                           ["model", "return_coverage", "annualized_return", "sharpe", "mean_turnover", "annualized_active_mean"]), "",
            "## 时间验证与复现", "",
            "日期横截面不会被随机拆到不同分区。每窗先 train→过去 validation 选参数，再用两者中成熟的标签拟合并预测未来 test；训练和验证 label_end 必须早于对应后续边界。默认保留两个市场日期间隔，测试日期不重叠。仅使用最后完整测试窗，剩余不足窗的尾部不自动拼接。", "",
            "所有数值标准化/截尾均在当前日期内处理。特征使用滞后收益和已声明 as-of 的缓存暴露；程序时间安全不等于供应商历史发布时点完整。没有伪造缺失 value、quality 或把流通市值称作总市值。Dummy 为过去训练标签均值，恒定横截面预测的 IC 不存在，故显示 N/A。", "",
            "## 漂移、重训练与重要性", "",
            "同时分析原始特征和模型输入，防止日期内标准化掩盖原始分布变化。PSI 分箱仅从训练样本生成，并保留尾部；KS 统计量用于分布诊断，不以股票行数作为独立有效样本来解释其 p 值。概念漂移的 feature-return IC 必须等 label_end 后才可知。", "",
            "重训练/窗口对比在共同样本外日期实际拟合，固定首次过去验证选定的 Ridge 参数。全部配置结果都报告，不从测试结果自动选择最优频率。模型年龄只有四窗的重复观测，不能据此声称稳定的半衰期。", "",
            "置换重要性在真实测试日期内打乱资产，相关特征组共同置换；单特征可能被相关替代特征分担信息。系数、impurity、单特征和分组置换均不等于因果贡献。受控相关预测变量、结构突变和噪声加倍实验位于 engineering_validation，明确是合成验证。", "",
            "## 稳健性与模型比较", "",
            "保留所有特征噪声、训练标签噪声、缺失特征、过去模型选定的 dropout、相关性改变、Random Forest seed 和完整树参数网格。参数网格的测试成绩只作探索性脆弱性诊断，不能再把其中最佳点称为未触碰样本外成果。", "",
            "同日模型 IC 差异使用日期块 bootstrap。未来特征、标签与收益被修改后，第一真实窗的历史预测和验证选参必须逐值不变；实际结果见 no_lookahead_checks.csv。漂移与成绩共同变化只能说明关联，不能证明漂移导致模型失效。", "",
            "## 怎样进入更有实际效用的研究", "",
            "先固定股票生命周期/历史状态/真实可交易价格与特征可用时间，统一多个项目的收益时间定义。再在开发数据固定假设、模型、成本、基准和退出条件，保留未查看的后续区间；现有 2023 样本已经看过，不能重新称为最终 holdout。", "",
            "本项目提供 OOS 信号导出和与相邻 risk_aware_portfolio 的独立桥接实验。桥接仍按风险项目的固定 score→预期收益系数构造仓位，必须作为假设进行敏感性研究；它证明接口连接，不证明 alpha 或真实成交。日频预测不能直接替换一秒订单簿预测；后者需要同标的/日期的订单簿、真实队列/成交反馈和适用交易规则。", "",
            "## 全部图表", ""]
    for filename in generated_figures:
        text += [f"![{filename.stem}](figures/{filename.name})", ""]
    (destination / "research_report.md").write_text("\n".join(text), encoding="utf-8")
    body = "<h1>Nonstationary Financial ML — 真实 A 股研究</h1>" + "<p>探索性历史样本；当前没有经过未触碰样本、PIT 股票池和真实执行确认的 alpha。</p>"
    body += metric.to_html(index=False, float_format=lambda value: f"{value:.5f}", na_rep="N/A")
    body += "<p>完整口径、限制和复现说明见同目录 research_report.md。</p>"
    for filename in generated_figures:
        body += f"<h2>{html.escape(filename.stem)}</h2><img src='figures/{filename.name}' style='max-width:100%'>"
    (destination / "research_report.html").write_text("<!doctype html><html><head><meta charset='utf-8'><style>body{font-family:Arial,sans-serif;max-width:1200px;margin:32px auto}table{border-collapse:collapse}td,th{padding:5px;border:1px solid #ddd}</style></head><body>" + body + "</body></html>", encoding="utf-8")
    packages = ["numpy", "pandas", "scipy", "scikit-learn", "statsmodels", "matplotlib", "pyarrow", "joblib"]
    write_json({"config": cfg, "package_versions": {name: importlib.metadata.version(name) for name in packages},
                "uv_lock_sha256": sha256(path("uv.lock")), "input_sha256": sha256(path(cfg["data"]["input_path"])),
                "predictions_sha256": sha256(destination / "predictions/oos_predictions.parquet"),
                "figure_sha256": {filename.name: sha256(filename) for filename in generated_figures},
                "code_sha256": {filename.relative_to(path(".")).as_posix(): sha256(filename)
                                for folder in (path("src"), path("scripts")) for filename in sorted(folder.rglob("*.py"))},
                "kind": "empirical_exploratory" if data["empirical"] else "synthetic_engineering"}, destination / "tables/reproducibility_manifest.json")
