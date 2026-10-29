"""Held-out, date-preserving grouped permutation and importance stability."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from financial_ml.evaluation import load_research
from financial_ml.metrics import correlation, daily_ic
from financial_ml.models import make_model
from financial_ml.temporal import partitions, walk_forward
from financial_ml.utils import output


def permute_within_dates(frame: pd.DataFrame, columns: list[str], seed: int) -> pd.DataFrame:
    """Permute assets within each observed date; one permutation per feature group."""
    rng = np.random.default_rng(seed)
    result = frame.copy()
    for _, index in frame.groupby("date", sort=True).groups.items():
        permutation = rng.permutation(len(index))
        result.loc[index, columns] = frame.loc[index, columns].to_numpy()[permutation]
    return result


@threadpool_limits.wrap(limits=1)
def study(cfg: dict) -> None:
    """Diagnostics use held-out observations; they never refit or select features."""
    panel, _, names = load_research(cfg)
    destination = output(cfg)
    records, coefficients = [], []
    groups = {name: [name] for name in names}
    groups.update({f"group:{key}": [name for name in values if name in names]
                   for key, values in cfg["drift"]["groups"].items()})
    groups = {key: values for key, values in groups.items() if values}
    for fold in walk_forward(panel, cfg["validation"]):
        _, _, test = partitions(panel, fold)
        test = test.loc[test.target_valid]
        for model_name in cfg["models"]["enabled"]:
            model = joblib.load(destination / f"models/fold_{fold.number}_{model_name}.joblib")["estimator"]
            predicted = test[["date", "target"]].copy()
            predicted["prediction"] = model.predict(test[names].to_numpy())
            baseline = daily_ic(predicted).rank_ic.mean()
            if hasattr(model, "coef_"):
                for feature, value in zip(names, model.coef_, strict=True):
                    coefficients.append({"fold": fold.number, "model": model_name, "feature": feature, "kind": "linear_coefficient", "value": value})
            if hasattr(model, "feature_importances_"):
                for feature, value in zip(names, model.feature_importances_, strict=True):
                    coefficients.append({"fold": fold.number, "model": model_name, "feature": feature, "kind": "impurity_noncausal", "value": value})
            for group_name, columns in groups.items():
                losses = []
                for repeat in range(cfg["drift"]["importance_repeats"]):
                    changed = permute_within_dates(test, columns, cfg["seed"] + repeat + 100 * fold.number)
                    predicted["prediction"] = model.predict(changed[names].to_numpy())
                    losses.append(baseline - daily_ic(predicted).rank_ic.mean())
                records.append({"fold": fold.number, "model": model_name, "feature": group_name,
                                "permutation_rank_ic_loss": np.mean(losses), "repeat_std": np.std(losses),
                                "baseline_rank_ic": baseline, "test_rows": len(test)})
    importance = pd.DataFrame(records)
    stability = []
    individual = importance.loc[~importance.feature.str.startswith("group:")]
    for model_name, group in individual.groupby("model", sort=True):
        matrix = group.pivot(index="fold", columns="feature", values="permutation_rank_ic_loss")
        for previous, current in zip(matrix.index[:-1], matrix.index[1:]):
            a, b = matrix.loc[previous], matrix.loc[current]
            finite = a.notna() & b.notna()
            overlap = len(set(a.dropna().nlargest(3).index) & set(b.dropna().nlargest(3).index)) / 3 if finite.sum() >= 3 else np.nan
            stability.append({"model": model_name, "previous_fold": previous, "current_fold": current,
                              "rank_correlation": correlation(a.loc[finite], b.loc[finite]), "top3_overlap": overlap,
                              "sign_agreement": float((np.sign(a.loc[finite]) == np.sign(b.loc[finite])).mean()) if finite.any() else np.nan})
    importance.to_csv(destination / "tables/feature_importance_summary.csv", index=False)
    pd.DataFrame(coefficients).to_csv(destination / "tables/model_coefficients.csv", index=False)
    pd.DataFrame(stability).to_csv(destination / "tables/importance_stability.csv", index=False)
    correlated_experiment(cfg)


@threadpool_limits.wrap(limits=1)
def correlated_experiment(cfg: dict) -> None:
    """Controlled correlated predictors belong to engineering validation only."""
    destination = output(cfg) / "engineering_validation"
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    for rho in (.1, .9):
        for seed in cfg["robustness"]["seeds"]:
            rng = np.random.default_rng(seed)
            x = rng.normal(size=6000)
            z = rho * x + np.sqrt(1 - rho * rho) * rng.normal(size=len(x))
            y = .00075 * (x + z) + rng.normal(0, .012, len(x))
            frame = pd.DataFrame({"date": np.repeat(np.arange(100), 60), "x": x, "z": z, "target": y})
            train, test = frame.iloc[:3600], frame.iloc[3600:]
            for name in ("linear", "ridge", "random_forest"):
                parameters = {"n_estimators": 30, "max_depth": 4, "min_samples_leaf": 40} if name == "random_forest" else {}
                model = make_model(name, parameters, seed).fit(train[["x", "z"]].to_numpy(), train.target)
                baseline = correlation(model.predict(test[["x", "z"]].to_numpy()), test.target, False)
                for group, columns in [("x", ["x"]), ("z", ["z"]), ("joint", ["x", "z"])]:
                    changed = permute_within_dates(test, columns, seed)
                    rows.append({"rho": rho, "observed_correlation": correlation(x, z, False), "seed": seed, "model": name,
                                 "permuted_group": group, "baseline_pearson": baseline,
                                 "pearson_loss": baseline - correlation(model.predict(changed[["x", "z"]].to_numpy()), test.target, False),
                                 "coefficient_x": model.coef_[0] if hasattr(model, "coef_") else np.nan,
                                 "coefficient_z": model.coef_[1] if hasattr(model, "coef_") else np.nan,
                                 "impurity_x": model.feature_importances_[0] if hasattr(model, "feature_importances_") else np.nan})
    pd.DataFrame(rows).to_csv(destination / "correlated_predictors.csv", index=False)
    (destination / "README.md").write_text("# 工程受控实验\n\n本目录是合成相关预测变量实验，不属于 A 股经验结果。单个变量的置换重要性可能被替代信息压低；联合置换用于比较，不表示因果识别。\n", encoding="utf-8")
