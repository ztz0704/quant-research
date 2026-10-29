"""Small sklearn pipelines; every fitted transform belongs to a training fold."""

from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_model(name: str, settings: dict[str, Any], seed: int) -> Pipeline:
    """Construct deterministic linear, L2 logistic, or nonlinear forest baseline."""
    if name == "linear":
        estimator = LinearRegression()
    elif name == "logistic":
        if settings["regularization"] <= 0:
            raise ValueError("Logistic C must be positive")
        estimator = LogisticRegression(C=settings["regularization"], max_iter=2000,
                                       random_state=seed)
    elif name == "forest":
        estimator = RandomForestClassifier(
            n_estimators=settings["forest_trees"], max_depth=settings["forest_max_depth"],
            min_samples_leaf=settings["forest_min_leaf"], random_state=seed, n_jobs=1,
        )
    else:
        raise ValueError(f"Unknown model {name}; use linear, logistic or forest")
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()), ("model", estimator),
    ])
