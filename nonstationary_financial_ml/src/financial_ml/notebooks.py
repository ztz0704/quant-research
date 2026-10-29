"""Read-only diagnostic notebooks; model and research logic stays in src."""

import nbformat

from financial_ml.utils import ROOT, output


def write_notebooks(cfg: dict) -> None:
    """Create lightweight evidence viewers with no hidden fitting or feature logic."""
    folder = ROOT / "notebooks"
    if cfg["data"]["source"] == "synthetic":
        folder = folder / "engineering_validation"
    folder.mkdir(parents=True, exist_ok=True)
    directory = output(cfg).relative_to(ROOT).as_posix()
    common = f"from pathlib import Path\nimport pandas as pd\nfrom IPython.display import Image, display\nresult = Path.cwd() / {directory!r}\n"
    definitions = [
        ("01_data_diagnostics", "数据与时间：固定真实股票篮子、缓存来源、标签/特征日期、缺失值需要明确。供应商发布时点未完整认证。",
         ["fold_boundaries", "model_fits"], ["feature_distribution_drift"]),
        ("02_model_comparison", "全部模型与原始/中性化预测；平均 Rank IC、HAC、多重校正、成本和基准。预测分数不等于 alpha。",
         ["model_performance", "portfolio_summary"], ["model_ic_comparison", "gross_net_performance"]),
        ("03_drift_analysis", "原始分布与标准化后的分布都检查；漂移和性能变化的关联不是因果识别。概念漂移需等待未来标签成熟。",
         ["drift_summary", "window_retraining_summary", "importance_stability"], ["feature_drift_psi_raw", "model_decay_curve"]),
        ("04_final_report", "当前结果是固定历史样本上的探索性研究；没有未触碰样本与真实执行确认的可交易 alpha。",
         ["paired_model_comparison", "regime_summary", "no_lookahead_checks"], ["cumulative_long_only", "transaction_cost_sensitivity"]),
    ]
    for name, description, tables, images in definitions:
        notebook = nbformat.v4.new_notebook()
        notebook.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
        notebook.cells = [nbformat.v4.new_markdown_cell(f"# {name}\n\n{description}"), nbformat.v4.new_code_cell(common)]
        for table in tables:
            notebook.cells.append(nbformat.v4.new_code_cell(f"display(pd.read_csv(result / 'tables/{table}.csv').head(30))"))
        for image in images:
            notebook.cells.append(nbformat.v4.new_code_cell(f"display(Image(filename=str(result / 'figures/{image}.png')))"))
        nbformat.write(notebook, folder / f"{name}.ipynb")
