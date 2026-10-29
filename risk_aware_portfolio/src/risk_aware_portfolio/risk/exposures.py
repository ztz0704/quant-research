"""Fixed, ex-ante factor vocabulary with an extensible exposure-builder interface."""

from typing import Protocol

import numpy as np
import pandas as pd

from risk_aware_portfolio.preprocessing.standardization import zscore
from risk_aware_portfolio.utils.config import Config


class ExposureBuilder(Protocol):
    """A production builder may compute styles from historical point-in-time inputs."""

    def __call__(self, cross_section: pd.DataFrame, config: Config) -> pd.DataFrame: ...


def industry_exposures(cross_section: pd.DataFrame, config: Config) -> pd.DataFrame:
    """All industry indicators for constraints, including the reference industry."""
    industries = config["risk_model"]["industries"]
    if not cross_section["industry"].isin(industries).all():
        raise ValueError("Unknown industry; revise the ex-ante vocabulary explicitly")
    return pd.DataFrame(
        {
            f"industry:{name}": cross_section["industry"].eq(name).to_numpy(dtype=float)
            for name in industries
        },
        index=cross_section["asset"],
    )


def build_exposures(cross_section: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Intercept, J-1 industry dummies, and same-date standardized style exposures.

    The reference is the first configured industry, never inferred from future rows.
    No style forward filling is performed.
    """
    styles = config["risk_model"]["styles"]
    missing = set(styles) - set(cross_section.columns)
    if missing:
        raise ValueError(f"Missing risk exposures: {sorted(missing)}")
    industry = industry_exposures(cross_section, config)
    result = pd.DataFrame({"market": np.ones(len(cross_section))}, index=cross_section["asset"])
    result = pd.concat([result, industry.iloc[:, 1:]], axis=1)
    for name in styles:
        result[name] = zscore(cross_section[name].to_numpy())
    return result
