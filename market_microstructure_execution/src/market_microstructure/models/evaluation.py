"""Prediction metrics, quantile monotonicity and dependent-sample uncertainty."""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def correlation(x: np.ndarray | pd.Series, y: np.ndarray | pd.Series, rank: bool = False) -> float:
    """Return NaN for absent/constant data rather than an invented correlation."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(spearmanr(x, y).statistic if rank else np.corrcoef(x, y)[0, 1])


def regression_metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    """Evaluate returns in basis points; MSE therefore has bps squared units."""
    valid = np.isfinite(y) & np.isfinite(prediction)
    y, prediction = np.asarray(y)[valid], np.asarray(prediction)[valid]
    if len(y) < 2:
        return {}
    return {"r2": r2_score(y, prediction), "mse_bps2": mean_squared_error(y, prediction),
            "mae_bps": mean_absolute_error(y, prediction),
            "pearson": correlation(y, prediction), "spearman": correlation(y, prediction, True)}


def classification_metrics(
    y: np.ndarray, prediction: np.ndarray, probabilities: np.ndarray, classes: np.ndarray,
) -> dict[str, float]:
    """Macro precision/recall, balanced accuracy and properly mapped binary/multiclass AUC."""
    valid = np.isfinite(y)
    y, prediction, probabilities = y[valid], prediction[valid], probabilities[valid]
    if not len(y):
        return {}
    auc = np.nan
    if len(np.unique(y)) == len(classes) and len(classes) > 1:
        auc = roc_auc_score(y, probabilities[:, 1]) if len(classes) == 2 else roc_auc_score(
            y, probabilities, labels=classes, multi_class="ovr", average="macro",
        )
    return {"accuracy": accuracy_score(y, prediction),
            "balanced_accuracy": balanced_accuracy_score(y, prediction),
            "precision": precision_score(y, prediction, average="macro", zero_division=0),
            "recall": recall_score(y, prediction, average="macro", zero_division=0), "roc_auc": auc}


def quantile_analysis(frame: pd.DataFrame, quantiles: int = 10) -> pd.DataFrame:
    """Descriptive OOS bins only; never feed these full-sample edges back to a strategy."""
    valid = frame[["signal", "future_return_bps"]].dropna().copy()
    if valid.empty or valid.signal.nunique() < 2:
        return pd.DataFrame(columns=["quantile", "mean_return_bps", "mean_signal", "count"])
    valid["quantile"] = pd.qcut(valid.signal, quantiles, labels=False, duplicates="drop") + 1
    return valid.groupby("quantile", observed=True).agg(
        mean_return_bps=("future_return_bps", "mean"), mean_signal=("signal", "mean"),
        count=("signal", "size"),
    ).reset_index()


def block_bootstrap_ic(frame: pd.DataFrame, block_events: int, samples: int, seed: int) -> dict:
    """Resample contiguous timestamp blocks, retaining same-time cross-asset dependence."""
    if block_events < 1 or samples < 2:
        raise ValueError("Bootstrap requires positive block length and >=2 samples")
    grouped = [g[["signal", "future_return_bps"]].to_numpy() for _, g in frame.groupby("timestamp")]
    if len(grouped) < block_events:
        return {"ic_low": np.nan, "ic_high": np.nan}
    starts = np.arange(0, len(grouped), block_events)
    blocks = [np.concatenate(grouped[s:s + block_events]) for s in starts]
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(samples):
        sample = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])
        estimates.append(correlation(sample[:, 0], sample[:, 1]))
    low, high = np.nanquantile(estimates, [0.025, 0.975])
    return {"ic_low": float(low), "ic_high": float(high)}
