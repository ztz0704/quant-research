"""Shared static scientific plot styling, labels and artifact persistence."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from market_microstructure.utils.paths import project_path

plt.switch_backend("Agg")

COLORS = {"mid": "#8796a5", "taker": "#126b85", "maker": "#dd8b2e",
          "linear": "#126b85", "logistic": "#a64462", "forest": "#639453"}


def new_plot(title: str, xlabel: str, ylabel: str) -> tuple:
    """Consistent readable axes; every figure declares units."""
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "figure.dpi": 120, "savefig.dpi": 150})
    fig, ax = plt.subplots(figsize=(9, 4.8), layout="constrained")
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.grid(alpha=0.18)
    ax.set_axisbelow(True)
    return fig, ax


def save_plot(fig, name: str) -> str:
    """Write a self-contained PNG and release Matplotlib resources."""
    path = project_path(f"outputs/figures/{name}.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return str(Path("figures") / path.name).replace("\\", "/")


def local_time(values: pd.Series) -> pd.Series:
    """Render exchange-local wall time; storage remains timezone-aware UTC."""
    return pd.to_datetime(values, utc=True).dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)


def empty_annotation(ax, message: str) -> None:
    """Absence is shown explicitly, rather than a fabricated zero statistic."""
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes,
            bbox={"facecolor": "#f3f5f7", "edgecolor": "none", "pad": 15})
