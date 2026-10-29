"""Actually refit models under alternative past windows and retraining cadences."""

from __future__ import annotations

import json
import time

import pandas as pd
from threadpoolctl import threadpool_limits

from financial_ml.evaluation import load_research
from financial_ml.metrics import prediction_metrics
from financial_ml.models import make_model
from financial_ml.portfolio import performance, portfolio_daily
from financial_ml.utils import output


@threadpool_limits.wrap(limits=1)
def refit_predictions(panel: pd.DataFrame, features: list[str], dates: pd.DatetimeIndex, name: str,
                      parameters: dict, cfg: dict, window: int | None, frequency: int) -> tuple[pd.DataFrame, list[dict]]:
    """All labels must have matured before today's fit; no random validation split."""
    calendar = pd.DatetimeIndex(sorted(panel.date.unique()))
    predictions, fits = [], []
    model = None
    fit_until = None
    for position, date in enumerate(dates):
        decision_index = calendar.get_indexer([date])[0]
        if model is None or position % frequency == 0:
            end_index = decision_index - cfg["validation"]["embargo_days"]
            start_index = max(0, end_index - window) if window is not None else 0
            train = panel.loc[panel.date.ge(calendar[start_index]) & panel.date.lt(calendar[end_index])
                              & panel.eligible & panel.target_valid & panel.label_end.lt(date)]
            if len(train) < cfg["validation"]["min_train_rows"]:
                raise ValueError(f"Insufficient cadence training rows at {date}: {len(train)}")
            model = make_model(name, parameters, cfg["seed"])
            started = time.perf_counter()
            model.fit(train[features].to_numpy(), train.target.to_numpy())
            fit_until = train.date.max()
            fits.append({"date": date, "train_start": train.date.min(), "train_end": fit_until,
                         "max_label_end": train.label_end.max(), "train_dates": train.date.nunique(), "rows": len(train),
                         "training_seconds": time.perf_counter() - started, "frequency": frequency, "window": window or "expanding"})
        test = panel.loc[panel.date.eq(date) & panel.eligible].copy()
        test["raw_prediction"] = model.predict(test[features].to_numpy())
        test["model"] = name
        test["trained_until"] = fit_until
        test["model_age_sessions"] = decision_index - calendar.get_indexer([fit_until])[0]
        predictions.append(test)
    return pd.concat(predictions, ignore_index=True), fits


def study(cfg: dict) -> None:
    """Report the full matrix; no best cadence or window is chosen using test scores."""
    panel, original, features = load_research(cfg)
    destination = output(cfg)
    name = cfg["robustness"]["selected_model"]
    fit_table = pd.read_csv(destination / "tables/model_fits.csv")
    params = json.loads(fit_table.loc[fit_table.model.eq(name)].sort_values("fold").iloc[0].parameters)
    max_window = max(cfg["robustness"]["window_days"])
    calendar = pd.DatetimeIndex(sorted(panel.date.unique()))
    earliest = calendar[max_window + cfg["validation"]["embargo_days"]]
    common = pd.DatetimeIndex(sorted(original.loc[original.date.ge(earliest)].date.unique()))
    if len(common) < 20:
        raise ValueError("Not enough common OOS dates for window/cadence comparison")
    records, fit_records, daily_records = [], [], []
    for window in [*cfg["robustness"]["window_days"], None]:
        for frequency in cfg["retraining"]["experiments"]:
            predicted, fits = refit_predictions(panel, features, common, name, params, cfg, window, frequency)
            metrics = prediction_metrics(predicted, "raw_prediction")
            portfolio = portfolio_daily(predicted, "raw_prediction", "long_only", cfg)
            selected = portfolio.loc[portfolio.cost_bps_per_half_l1.eq(10)]
            records.append({"model": name, "window": window or "expanding", "frequency": frequency,
                            "refits": len(fits), "total_training_seconds": sum(fit["training_seconds"] for fit in fits),
                            **metrics, **performance(selected, cfg["labels"]["forward_horizon"])})
            fit_records.extend(fits)
            portfolio["window"], portfolio["frequency"] = window or "expanding", frequency
            daily_records.append(portfolio)
    pd.DataFrame(records).to_csv(destination / "tables/window_retraining_summary.csv", index=False)
    pd.DataFrame(fit_records).to_csv(destination / "tables/cadence_fits.csv", index=False)
    pd.concat(daily_records, ignore_index=True).to_csv(destination / "tables/cadence_portfolio_daily.csv", index=False)
