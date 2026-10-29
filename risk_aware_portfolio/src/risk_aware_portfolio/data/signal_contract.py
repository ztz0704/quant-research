"""Return interval and unit gates for the legacy close-to-close risk engine."""

import pandas as pd

CLOSE_CONVENTION = "close_to_next_close_simple"


def require_engine_compatible(panel: pd.DataFrame, config: dict) -> None:
    """Reject open forecasts BEFORE estimating close-return covariance or trading.

    This engine uses close-return drifted holdings and close-return attribution.
    Passing a next-open forecast requires a separate event/accounting risk engine;
    changing a date or return column cannot make the interval compatible.
    """
    declared = config["backtest"].get("return_convention", CLOSE_CONVENTION)
    conventions = (
        set(panel.return_convention.dropna()) if "return_convention" in panel else {declared}
    )
    if conventions != {CLOSE_CONVENTION} or declared != CLOSE_CONVENTION:
        raise ValueError(
            "Incompatible return convention: close risk engine cannot trade open intervals"
        )
    if "return_unit" in panel and not panel.return_unit.eq("decimal_simple").all():
        raise ValueError("Risk engine requires decimal simple returns")
    mode = config["alpha"].get("input_mode", "score")
    if mode not in {"score", "expected_return"}:
        raise ValueError("Unknown alpha input_mode")
    if mode == "expected_return":
        fields = {"model_fit_cutoff", "calibration_fit_cutoff", "return_unit", "return_convention"}
        if fields - set(panel):
            raise ValueError("Calibrated expected returns require timing/unit metadata")
        for field in ("model_fit_cutoff", "calibration_fit_cutoff"):
            values = pd.to_datetime(panel[field], errors="raise")
            if values.isna().any() or values.gt(panel.date + pd.Timedelta(hours=18)).any():
                raise ValueError("Expected return uses future fit information")
