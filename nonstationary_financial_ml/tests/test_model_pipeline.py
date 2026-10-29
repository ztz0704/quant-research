from io import BytesIO

import joblib
import numpy as np
import pytest

from financial_ml.models import make_model


@pytest.mark.parametrize("name", ["dummy", "linear", "ridge", "lasso", "elastic_net", "random_forest", "boosting"])
def test_fit_predict_reload(name):
    rng = np.random.default_rng(8)
    x = rng.normal(size=(300, 3))
    y = 0.001 * x[:, 0] + rng.normal(0, 0.012, 300)
    params = {"n_estimators": 10, "max_depth": 3} if name == "random_forest" else ({"max_iter": 10} if name == "boosting" else {})
    model = make_model(name, params).fit(x, y)
    values = model.predict(x[-20:])
    assert np.isfinite(values).all()
    model.fit(x, y)
    assert np.array_equal(values, model.predict(x[-20:]))
    target = BytesIO()
    joblib.dump(model, target)
    target.seek(0)
    assert np.array_equal(values, joblib.load(target).predict(x[-20:]))


def test_stationary_and_no_signal_generalization():
    rng = np.random.default_rng(81)
    x = rng.normal(size=(12000, 3))
    y = 0.0015 * x[:, 0] + rng.normal(0, 0.012, len(x))
    model = make_model("ridge", {"alpha": 1}).fit(x[:6000], y[:6000])
    assert np.corrcoef(model.predict(x[6000:]), y[6000:])[0, 1] > 0.05
    null = rng.normal(0, 0.012, len(x))
    model.fit(x[:6000], null[:6000])
    assert abs(np.corrcoef(model.predict(x[6000:]), null[6000:])[0, 1]) < 0.04


def test_full_synthetic_temporal_pipeline_signal_and_null():
    import pandas as pd

    from financial_ml.data import synthetic
    from financial_ml.features import build_features
    from financial_ml.labels import add_labels
    from financial_ml.metrics import daily_ic
    from financial_ml.temporal import partitions, walk_forward
    from financial_ml.training import select_model

    cfg = {"seed": 8, "features": {"additional_return_lag": 1, "winsorize": "mad", "standardize": True, "neutralize": False},
           "models": {"grids": {"ridge": [{"alpha": 1.}, {"alpha": 100.}]}},
           "validation": dict(mode="rolling", train_days=250, validation_days=63, test_days=63, step_days=63, embargo_days=2)}
    for signal in (True, False):
        panel, features = build_features(synthetic(days=450, assets=40, seed=8, stationary=True, signal=signal), cfg)
        panel = add_labels(panel, 1)
        fold = walk_forward(panel, cfg["validation"])[0]
        train, val, test = partitions(panel, fold)
        model, _, _, _ = select_model("ridge", train, val, features, cfg)
        predicted = pd.DataFrame({"date": test.date, "target": test.target, "prediction": model.predict(test[features].to_numpy())})
        ic = daily_ic(predicted).rank_ic.mean()
        assert ic > .04 if signal else abs(ic) < .08
