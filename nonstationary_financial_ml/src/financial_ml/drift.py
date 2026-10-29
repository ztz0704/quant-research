"""Training-reference distribution drift and outcome-available concept diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from financial_ml.evaluation import load_research
from financial_ml.metrics import correlation, daily_ic
from financial_ml.temporal import partitions, walk_forward
from financial_ml.utils import output


def distribution_shift(train, test, bins: int = 10) -> dict:
    """PSI uses train-only bins, infinite tails and fixed smoothing for empty bins."""
    train = np.asarray(pd.Series(train).dropna(), dtype=float)
    test = np.asarray(pd.Series(test).dropna(), dtype=float)
    if not len(train) or not len(test):
        raise ValueError("Drift requires nonempty reference and observed samples")
    inner = np.unique(np.quantile(train, np.linspace(0, 1, bins + 1)[1:-1]))
    edges = np.r_[-np.inf, inner, np.inf]
    p = np.histogram(train, bins=edges)[0].astype(float) + 0.5
    q = np.histogram(test, bins=edges)[0].astype(float) + 0.5
    p, q = p / p.sum(), q / q.sum()
    std = train.std(ddof=0)
    return {"psi": float(np.sum((p - q) * np.log(p / q))), "ks_statistic": float(ks_2samp(train, test).statistic),
            "mean_shift": float(test.mean() - train.mean()), "std_shift": float(test.std(ddof=0) - std),
            "q10_shift": float(np.quantile(test, .1) - np.quantile(train, .1)),
            "median_shift": float(np.median(test) - np.median(train)),
            "q90_shift": float(np.quantile(test, .9) - np.quantile(train, .9)), "train_rows": len(train), "test_rows": len(test)}


def analyze(cfg: dict) -> dict[str, pd.DataFrame]:
    """These are retrospective diagnostics, not future-informed retraining triggers."""
    panel, predictions, names = load_research(cfg)
    tables = output(cfg) / "tables"
    shifts, relationships, moments, age = [], [], [], []
    for fold in walk_forward(panel, cfg["validation"]):
        train, _, test = partitions(panel, fold)
        for name in names:
            for representation, column in [("raw", f"raw__{name}"), ("model_input", name)]:
                shifts.append({"fold": fold.number, "feature": name, "representation": representation,
                               **distribution_shift(train[column], test[column], cfg["drift"]["psi_bins"]),
                               "test_start": fold.test_start, "test_end": fold.test_end})
    for date, section in panel.loc[panel.eligible].groupby("date", sort=True):
        for name in names:
            valid = section.loc[section.target_valid]
            relationships.append({"date": date, "feature": name, "feature_rank_ic": correlation(valid[name], valid.target),
                                  "available_after": valid.label_end.max()})
            values = section[f"raw__{name}"]
            moments.append({"date": date, "feature": name, "mean": values.mean(), "std": values.std(),
                            "q10": values.quantile(.1), "median": values.median(), "q90": values.quantile(.9)})
    concept = pd.DataFrame(relationships)
    concept["rolling_rank_ic"] = concept.groupby("feature", sort=False).feature_rank_ic.transform(lambda s: s.rolling(cfg["drift"]["rolling_window"], min_periods=10).mean())
    for (model, fold), section in predictions.groupby(["model", "fold"], sort=True):
        ic = daily_ic(section, "raw_prediction")
        ages = section.groupby("date").model_age_sessions.first()
        for _, row in ic.iterrows():
            age.append({"date": row.date, "model": model, "fold": fold,
                        "model_age_sessions": ages.loc[row.date], "rank_ic": row.rank_ic})
    frames = {"drift_summary": pd.DataFrame(shifts), "feature_relationships": concept,
              "feature_moments": pd.DataFrame(moments), "model_decay": pd.DataFrame(age)}
    for name, frame in frames.items():
        frame.to_csv(tables / f"{name}.csv", index=False)
    return frames
