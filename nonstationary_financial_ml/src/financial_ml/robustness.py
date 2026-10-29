"""Declared perturbation, seed, parameter and paired time-block diagnostics."""

from __future__ import annotations

import itertools
import json

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from financial_ml.evaluation import load_research
from financial_ml.metrics import correlation, daily_ic, prediction_metrics
from financial_ml.models import make_model
from financial_ml.temporal import partitions, walk_forward
from financial_ml.training import select_model
from financial_ml.utils import output


def block_mean_ci(values, replications: int = 300, block_days: int = 10, seed: int = 8) -> tuple[float, float]:
    """Moving-block percentile CI; input is one paired observation per date."""
    sample = np.asarray(pd.Series(values).dropna(), float)
    if len(sample) < max(10, block_days):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(replications):
        starts = rng.integers(0, len(sample) - block_days + 1, int(np.ceil(len(sample) / block_days)))
        draw = np.concatenate([sample[start:start + block_days] for start in starts])[:len(sample)]
        means.append(draw.mean())
    return tuple(np.quantile(means, [.025, .975]))


@threadpool_limits.wrap(limits=1)
def perturbations(cfg: dict) -> None:
    """Perturb historical fits or observable test features; actual targets stay fixed."""
    panel, _, features = load_research(cfg)
    destination = output(cfg)
    name = cfg["robustness"]["selected_model"]
    records = []
    for fold in walk_forward(panel, cfg["validation"]):
        train, validation, test = partitions(panel, fold)
        fit = pd.concat([train, validation]).sort_values(["date", "asset"])
        saved = joblib.load(destination / f"models/fold_{fold.number}_{name}.joblib")
        parameters = saved["parameters"]
        importance = np.abs(saved["estimator"].coef_) if hasattr(saved["estimator"], "coef_") else saved["estimator"].feature_importances_
        dropped = features[int(np.argmax(importance))]
        for scenario in ("baseline", "feature_noise", "training_label_noise", "missing_features", "past_top_feature_dropout", "correlation_increase"):
            rng = np.random.default_rng(cfg["seed"] + fold.number)
            xfit, xtest, y = fit[features].copy(), test[features].copy(), fit.target.to_numpy().copy()
            chosen = features.copy()
            if scenario == "feature_noise":
                xtest += rng.normal(0, cfg["robustness"]["feature_noise_std"], xtest.shape)
            elif scenario == "training_label_noise":
                y += rng.normal(0, np.std(y) * cfg["robustness"]["label_noise_std"], len(y))
            elif scenario == "missing_features":
                missing = rng.random(xtest.shape) < .1
                xtest = xtest.mask(missing).fillna(xfit.mean())
            elif scenario == "past_top_feature_dropout":
                chosen.remove(dropped)
            elif scenario == "correlation_increase":
                # A declared counterfactual predictor shift, not a new historical tape.
                for matrix in (xfit, xtest):
                    matrix[features[1]] = .9 * matrix[features[0]] + np.sqrt(1 - .9 ** 2) * matrix[features[1]]
            model = make_model(name, parameters, cfg["seed"])
            model.fit(xfit[chosen].to_numpy(), y)
            result = test[["date", "asset", "target"]].copy()
            result["prediction"] = model.predict(xtest[chosen].to_numpy())
            records.append({"fold": fold.number, "model": name, "scenario": scenario, "dropout_feature": dropped if scenario == "past_top_feature_dropout" else "",
                            **prediction_metrics(result, "prediction", max(5, cfg["labels"]["forward_horizon"] - 1))})
    pd.DataFrame(records).to_csv(destination / "tables/robustness_summary.csv", index=False)


@threadpool_limits.wrap(limits=1)
def parameter_and_seed_study(cfg: dict) -> None:
    """Evaluate the entire declared grid on OOS data solely as exploratory diagnostics."""
    panel, _, features = load_research(cfg)
    destination = output(cfg)
    parameters, seeds = [], []
    for fold in walk_forward(panel, cfg["validation"]):
        train, val, test = partitions(panel, fold)
        fit = pd.concat([train, val])
        for name, grid in cfg["robustness"]["sensitivity"].items():
            for combination in itertools.product(*grid.values()):
                candidate = dict(zip(grid, combination, strict=True))
                model = make_model(name, candidate, cfg["seed"]).fit(fit[features].to_numpy(), fit.target.to_numpy())
                predicted = test[["date", "target"]].copy()
                predicted["prediction"] = model.predict(test[features].to_numpy())
                parameters.append({"fold": fold.number, "model": name, "parameters": json.dumps(candidate, sort_keys=True),
                                   "mean_rank_ic": daily_ic(predicted).rank_ic.mean(), **candidate})
        saved = joblib.load(destination / f"models/fold_{fold.number}_random_forest.joblib")
        for seed in cfg["robustness"]["seeds"]:
            model = make_model("random_forest", saved["parameters"], seed).fit(fit[features].to_numpy(), fit.target.to_numpy())
            predicted = test[["date", "target"]].copy()
            predicted["prediction"] = model.predict(test[features].to_numpy())
            baseline = saved["estimator"].predict(test[features].to_numpy())
            seeds.append({"fold": fold.number, "model": "random_forest", "seed": seed,
                          "mean_rank_ic": daily_ic(predicted).rank_ic.mean(),
                          "prediction_correlation_to_declared_seed": correlation(baseline, predicted.prediction, False)})
    pd.DataFrame(parameters).to_csv(destination / "tables/hyperparameter_sensitivity.csv", index=False)
    pd.DataFrame(seeds).to_csv(destination / "tables/seed_sensitivity.csv", index=False)


def paired_comparison(cfg: dict) -> None:
    """Compare model and Ridge IC on the same dates, with block uncertainty."""
    destination = output(cfg)
    frame = pd.read_csv(destination / "tables/daily_ic.csv")
    matrix = frame.loc[frame.prediction.eq("raw_prediction")].pivot(index="date", columns="model", values="rank_ic")
    rows = []
    for model in matrix:
        difference = (matrix[model] - matrix["ridge"]).dropna()
        low, high = block_mean_ci(difference, cfg["robustness"]["bootstrap_replications"], cfg["robustness"]["block_days"], cfg["seed"])
        rows.append({"model": model, "reference": "ridge", "dates": len(difference), "mean_rank_ic_difference": difference.mean(),
                     "block_ci_low": low, "block_ci_high": high})
    pd.DataFrame(rows).to_csv(destination / "tables/paired_model_comparison.csv", index=False)


@threadpool_limits.wrap(limits=1)
def future_perturbation(cfg: dict) -> None:
    """Refit the first real fold after changing future features, targets and regimes."""
    panel, _, features = load_research(cfg)
    first = walk_forward(panel, cfg["validation"])[0]
    train, val, test = partitions(panel, first)
    cutoff = sorted(test.date.unique())[len(test.date.unique()) // 2]
    changed = panel.copy()
    future = changed.date.gt(cutoff)
    changed.loc[future, features] *= 100
    changed.loc[future, "target"] = .8
    changed.loc[future, "target_valid"] = False
    changed.loc[future, "return"] = .5
    changed_train, changed_val, changed_test = partitions(changed, first)
    checks = []
    for name in cfg["models"]["enabled"]:
        before, before_params, _, _ = select_model(name, train, val, features, cfg)
        after, after_params, _, _ = select_model(name, changed_train, changed_val, features, cfg)
        historic = test.loc[test.date.le(cutoff)]
        revised = changed_test.loc[changed_test.date.le(cutoff)]
        a, b = before.predict(historic[features].to_numpy()), after.predict(revised[features].to_numpy())
        difference = float(np.max(np.abs(a - b)))
        passed = np.array_equal(a, b) and before_params == after_params
        checks.append({"model": name, "cutoff": cutoff, "rows_checked": len(a), "max_prediction_difference": difference,
                       "parameters_identical": before_params == after_params, "passed": passed})
        if not passed:
            raise AssertionError(f"Future perturbation changed history: {name}")
    pd.DataFrame(checks).to_csv(output(cfg) / "tables/no_lookahead_checks.csv", index=False)


def synthetic_shift_study(cfg: dict) -> None:
    """Known coefficient reversal and residual-volatility shock are engineering tests."""
    destination = output(cfg) / "engineering_validation"
    destination.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg["seed"])
    x = rng.normal(size=(12000, 3))
    noise = rng.normal(0, .012, len(x))
    y = .0015 * x[:, 0] + noise
    model = make_model("ridge", {"alpha": 1}, cfg["seed"]).fit(x[:6000], y[:6000])
    pred = model.predict(x[6000:])
    rows = []
    for scenario, target in [("stationary", y[6000:]), ("concept_sign_reversal", -.0015 * x[6000:, 0] + noise[6000:]),
                             ("residual_noise_2x", .0015 * x[6000:, 0] + 2 * noise[6000:])]:
        rows.append({"scenario": scenario, "pearson": correlation(pred, target, False), "rows": len(pred), "empirical": False})
    pd.DataFrame(rows).to_csv(destination / "structural_break_noise.csv", index=False)
