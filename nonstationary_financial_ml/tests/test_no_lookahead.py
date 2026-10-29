from financial_ml.data import synthetic
from financial_ml.features import build_features


def test_future_returns_and_features_do_not_change_history():
    panel = synthetic(days=100, assets=10)
    cfg = {"features": {"additional_return_lag": 1, "winsorize": "mad", "standardize": True, "neutralize": False}}
    before, names = build_features(panel, cfg)
    cutoff = panel.date.unique()[80]
    changed = panel.copy()
    changed.loc[changed.date > cutoff, "return"] = 0.5
    changed.loc[changed.date > cutoff, "x"] = 10000
    after, _ = build_features(changed, cfg)
    assert before.loc[before.date <= cutoff, names].equals(after.loc[after.date <= cutoff, names])


def test_small_current_cross_section_abstains_without_using_future_halts():
    panel = synthetic(days=100, assets=10)
    cfg = {"features": {"additional_return_lag": 1, "winsorize": "mad",
                        "standardize": True, "neutralize": False, "min_cross_section": 8}}
    reference, names = build_features(panel, cfg)
    date = panel.date.unique()[80]
    changed = panel.copy()
    changed.loc[(changed.date == date) & changed.asset.isin(sorted(changed.asset.unique())[:3]), "tradable"] = False
    changed.loc[changed.date > date, "tradable"] = False
    actual, _ = build_features(changed, cfg)
    assert reference.loc[reference.date == date, "eligible"].all()
    assert not actual.loc[actual.date == date, "eligible"].any()
    assert reference.loc[reference.date < date, names + ["eligible"]].equals(
        actual.loc[actual.date < date, names + ["eligible"]])
