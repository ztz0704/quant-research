"""Small model registry; all external boosting libraries remain optional."""

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge


def make_model(name: str, parameters: dict | None = None, seed: int = 8):
    """Construct an unfitted estimator; early stopping never creates random folds."""
    values = dict(parameters or {})
    if name == "dummy":
        return DummyRegressor(strategy="mean", **values)
    if name == "linear":
        return LinearRegression(**values)
    if name == "ridge":
        return Ridge(**values)
    if name == "lasso":
        return Lasso(max_iter=10000, **values)
    if name == "elastic_net":
        return ElasticNet(max_iter=10000, **values)
    if name == "random_forest":
        return RandomForestRegressor(random_state=seed, n_jobs=1, **values)
    if name == "boosting":
        return HistGradientBoostingRegressor(random_state=seed, early_stopping=False, **values)
    if name == "lightgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(random_state=seed, n_jobs=1, verbosity=-1, **values)
    raise ValueError(f"Unknown model: {name}")
