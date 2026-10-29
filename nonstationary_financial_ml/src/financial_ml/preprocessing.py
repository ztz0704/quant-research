"""Date-local transformations without full-sample estimates."""

from __future__ import annotations

import numpy as np
import pandas as pd


def winsorize(values: pd.Series, method: str = "mad") -> pd.Series:
    """Clip finite values within the current cross section only."""
    if method == "mad":
        center = values.median()
        width = 5 * 1.4826 * (values - center).abs().median()
        return values.clip(center - width, center + width) if width > 0 else values.copy()
    if method == "percentile":
        return values.clip(values.quantile(0.01), values.quantile(0.99))
    if method in ("none", None):
        return values.copy()
    raise ValueError(f"Unknown winsorization: {method}")


def standardize(values: pd.Series) -> pd.Series:
    """Constant cross sections have zero scores; missingness stays explicit."""
    std = values.std(ddof=0)
    return (values - values.mean()) / std if std > 1e-12 else values * 0.0


def residualize(values: np.ndarray, section: pd.DataFrame, styles: tuple[str, ...] = ("size",)) -> np.ndarray:
    """Remove same-date industry and declared style exposure, with an intercept."""
    industry = pd.get_dummies(section["industry"], dtype=float).to_numpy()
    columns = [np.ones((len(section), 1)), industry]
    for style in styles:
        if style not in section:
            raise ValueError(f"Missing neutralization exposure: {style}")
        columns.append(standardize(section[style]).to_numpy().reshape(-1, 1))
    x = np.column_stack(columns)
    return values - x @ np.linalg.lstsq(x, values, rcond=None)[0]


def transform(frame: pd.DataFrame, names: list[str], settings: dict) -> pd.DataFrame:
    """Use currently feature-ready, currently tradable rows for date-local statistics."""
    result = frame.copy()
    for _, index in result.loc[result["eligible"]].groupby("date", sort=True).groups.items():
        section = result.loc[index]
        for name in names:
            values = winsorize(section[name], settings["winsorize"])
            if settings["neutralize"]:
                values = pd.Series(residualize(values.to_numpy(), section), index=index)
            if settings["standardize"]:
                values = standardize(values)
            result.loc[index, name] = values
    return result
