"""Same-date alpha residualization with explicit industry/style modes."""

import numpy as np
import pandas as pd

from risk_aware_portfolio.preprocessing.standardization import zscore
from risk_aware_portfolio.preprocessing.winsorization import mad_winsorize
from risk_aware_portfolio.risk.exposures import build_exposures
from risk_aware_portfolio.utils.config import Config


def neutralize(values: np.ndarray, exposures: pd.DataFrame, mode: str = "joint") -> np.ndarray:
    """Remove chosen risk exposures with an intercept and OLS projection."""
    industries = [c for c in exposures if c.startswith("industry:")]
    styles = [c for c in exposures if c not in industries + ["market"]]
    selections = {
        "industry": industries,
        "size": ["size"],
        "style": styles,
        "joint": industries + styles,
    }
    if mode not in selections:
        raise ValueError(f"Unknown neutralization mode: {mode}")
    columns = ["market"] + selections[mode]
    x = exposures[columns].to_numpy()
    y = np.asarray(values, dtype=float)
    if not np.isfinite(y).all():
        raise ValueError("Nonfinite alpha")
    coefficients, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    return y - x @ coefficients


def preprocess_alpha(cross_section: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Keep original input, raw standardized alpha and neutralized alpha.

    scale is a fixed ex-ante daily expected-return-per-z assumption, not an IC fit
    on the evaluation sample. Input alpha is treated as a score, not a daily return.
    """
    settings = config["alpha"]
    original = cross_section["alpha"].to_numpy()
    if settings.get("input_mode", "score") == "expected_return":
        if settings["winsorize"]:
            raise ValueError("Expected return calibration cannot be clipped by score preprocessing")
        if not np.isfinite(original).all():
            raise ValueError("Nonfinite expected return")
        residual = neutralize(original, build_exposures(cross_section, config),
                              settings["neutralization"])
        return pd.DataFrame({"raw_input": original, "raw_alpha": original,
                             "neutralized_alpha": residual,
                             "selected_alpha": residual if settings["neutralize"] else original},
                            index=cross_section["asset"])
    if settings.get("input_mode", "score") != "score":
        raise ValueError("Unknown alpha input_mode")
    cleaned = (
        mad_winsorize(original, settings["mad_multiplier"]) if settings["winsorize"] else original
    )
    raw = zscore(cleaned) * settings["scale"]
    residual = neutralize(raw, build_exposures(cross_section, config), settings["neutralization"])
    neutral = zscore(residual) * settings["scale"]
    return pd.DataFrame(
        {
            "raw_input": original,
            "raw_alpha": raw,
            "neutralized_alpha": neutral,
            "selected_alpha": neutral if settings["neutralize"] else raw,
        },
        index=cross_section["asset"],
    )
