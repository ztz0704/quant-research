"""Book distributions, out-of-sample statistics and stability plots."""

import pandas as pd
from sklearn.metrics import auc, roc_curve

from market_microstructure.reporting.plots import COLORS, local_time, new_plot, save_plot
from market_microstructure.utils.paths import project_path


def book_figures(features: pd.DataFrame) -> list[str]:
    """Plot the entire accepted real sample, including distribution tails."""
    paths = []
    histograms = (("imbalance_1", "order_imbalance_distribution", "L1 order imbalance", "Imbalance (unitless)"),
                  ("spread_bps", "spread_distribution", "Quoted bid-ask spread", "Spread (bps)"),
                  ("total_depth", "depth_distribution", "Five-level displayed depth", "Depth (shares)"),
                  ("microprice_deviation_bps", "microprice_deviation_distribution", "Microprice minus mid", "Deviation (bps)"))
    for column, name, title, xlabel in histograms:
        fig, ax = new_plot(f"{title} | entire accepted sample", xlabel, "Snapshot count")
        ax.hist(features[column].dropna(), bins=60, color="#126b85", alpha=0.85)
        paths.append(save_plot(fig, name))
    fig, ax = new_plot("Spread through the continuous sessions", "Exchange time (Asia/Shanghai)", "Spread (bps)")
    ax.plot(local_time(features.timestamp), features.spread_bps, lw=0.7, color="#126b85")
    paths.append(save_plot(fig, "spread_by_time"))
    for column, name, xlabel in (("realized_vol_bps", "spread_vs_volatility", "Trailing realized volatility (bps)"),
                                 ("total_depth", "spread_vs_liquidity", "Five-level depth (shares)")):
        fig, ax = new_plot("Spread and contemporaneous market state", xlabel, "Spread (bps)")
        ax.scatter(features[column], features.spread_bps, s=7, alpha=0.12, color="#126b85", rasterized=True)
        paths.append(save_plot(fig, name))
    return paths


def signal_figures(predictions: pd.DataFrame, selected_model: str, rolling_window: int) -> list[str]:
    """Saved OOS predictions only; no final-fold model used to re-predict history."""
    paths = []
    fig, ax = new_plot("Signal quantiles and future mid returns | OOS", "Signal quantile (descriptive bins)", "Mean future return (bps)")
    for model in predictions.model.unique():
        table = pd.read_csv(project_path(f"outputs/tables/quantiles_{model}.csv"))
        ax.plot(table["quantile"], table.mean_return_bps, marker="o", color=COLORS[model], label=model)
    ax.axhline(0, color="black", lw=0.7)
    ax.legend()
    paths.append(save_plot(fig, "signal_quantile_vs_future_return"))
    decay = pd.read_csv(project_path("outputs/tables/signal_decay.csv"))
    fig, ax = new_plot(f"Signal decay | {selected_model}, OOS", "Horizon (retained 1-second observations)", "Pearson IC (unitless)")
    for asset, part in decay.groupby("asset"):
        ax.plot(part.horizon_events, part.ic, marker="o", label=asset)
    ax.set_xscale("log")
    ax.axhline(0, color="black", lw=0.7)
    ax.legend()
    paths.append(save_plot(fig, "signal_decay_curve"))
    selected = predictions.loc[predictions.model == selected_model].dropna(subset=["direction"])
    fig, ax = new_plot(f"One-vs-rest ROC | {selected_model}, OOS", "False positive rate (fraction)", "True positive rate (fraction)")
    for label in (-1, 0, 1):
        column = f"prob_{label}"
        target = selected.direction.eq(label)
        if column in selected and target.nunique() == 2:
            fpr, tpr, _ = roc_curve(target, selected[column])
            ax.plot(fpr, tpr, label=f"class {label:+d}, AUC={auc(fpr, tpr):.3f}")
    ax.plot([0, 1], [0, 1], "--", color="#8796a5", label="random ranking")
    ax.legend()
    paths.append(save_plot(fig, "roc_curve"))
    linear = predictions.loc[predictions.model == "linear"].dropna(subset=["future_return_bps"])
    fig, ax = new_plot("Linear return predictions and realized labels | OOS", "Predicted return (bps)", "Future mid return (bps)")
    density = ax.hexbin(linear.prediction_bps, linear.future_return_bps, gridsize=40, mincnt=1, cmap="Blues")
    fig.colorbar(density, ax=ax, label="Observation count")
    paths.append(save_plot(fig, "prediction_vs_realized_return"))
    importance = pd.read_csv(project_path("outputs/tables/feature_importance.csv"))
    coefficients = importance.loc[importance.model == "linear"]
    largest = coefficients.groupby("feature").coefficient_bps.apply(lambda x: x.abs().mean()).nlargest(5).index
    fig, ax = new_plot("OLS coefficient drift | training-only HAC fits", "Walk-forward fold (index)", "Coefficient (bps / training std)")
    for feature in largest:
        part = coefficients.loc[coefficients.feature == feature].groupby("fold").coefficient_bps.mean()
        ax.plot(part.index, part, label=feature)
    ax.legend(fontsize=8)
    paths.append(save_plot(fig, "coefficient_stability"))
    forest = importance.loc[importance.model == "forest"]
    mean = forest.groupby("feature").permutation_mean.mean().sort_values()
    variation = forest.groupby("feature").permutation_mean.std().reindex(mean.index).fillna(0)
    fig, ax = new_plot("Validation permutation importance | fold variation", "Balanced accuracy decrease (fraction; +/- fold std)", "Feature")
    ax.barh(mean.index, mean, xerr=variation, color="#639453", alpha=0.85)
    ax.tick_params(axis="y", labelsize=8)
    paths.append(save_plot(fig, "feature_importance"))
    rolling = pd.read_csv(project_path("outputs/tables/rolling_ic.csv"))
    fig, ax = new_plot(f"Trailing OOS signal IC | {selected_model}", "Exchange time (Asia/Shanghai)", f"IC (unitless; {rolling_window} observations)")
    ax.plot(local_time(rolling.timestamp), rolling.rolling_ic, color="#a64462", lw=0.8)
    ax.axhline(0, color="black", lw=0.7)
    paths.append(save_plot(fig, "rolling_ic"))
    return paths
