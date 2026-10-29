"""Real training-quote risk inputs; uncalibrated impact remains an explicit sensitivity."""

from dataclasses import replace
from itertools import product

import numpy as np
import pandas as pd

from market_microstructure.optimal_execution.almgren_chriss import (
    ExecutionParameters,
    optimal_inventory,
    schedule_costs,
)
from market_microstructure.optimal_execution.schedules import comparison_schedules, round_lots


def training_inputs(features: pd.DataFrame, cutoff: pd.Timestamp, settings: dict) -> tuple[dict, np.ndarray]:
    """Estimate only on quotes before the first OOS fold; exclude overnight/lunch jumps.

    A local volume forecast averages past flow at matching positions in repeated
    duration-length clock blocks. It is a simple historical forecast, not actual VWAP.
    """
    history = features.loc[features.timestamp <= cutoff].copy()
    if history.asset.nunique() != 1 or len(history) < 20:
        raise ValueError("Optimal-execution calibration needs one asset and sufficient training history")
    dt = history.timestamp.diff().dt.total_seconds()
    moves = history.mid.diff()
    session = history.get("session_id", history.timestamp.dt.date.astype(str))
    cadence = dt.loc[dt > 0].median()
    normal = dt.between(cadence * 0.5, cadence * 1.5) & session.eq(session.shift())
    sigma = np.sqrt(np.mean(moves.loc[normal] ** 2 / dt.loc[normal]))
    spread = float(history.spread.median() / 2)
    launch = features.loc[features.timestamp > cutoff, "timestamp"].min()
    intervals = settings["intervals"]
    tau = settings["duration_seconds"] / intervals
    phase = ((history.timestamp.astype("int64") / 1e9 - launch.value / 1e9)
             % settings["duration_seconds"] // tau).astype(int).clip(0, intervals - 1)
    profile = history.groupby(phase).trade_size.mean().reindex(range(intervals))
    profile = profile.fillna(history.trade_size.mean()).clip(lower=1).to_numpy()
    audit = {"asset": history.asset.iloc[0], "training_last_timestamp": history.timestamp.max().isoformat(),
             "hypothetical_execution_launch_timestamp": launch.isoformat(), "training_rows": len(history),
             "sigma_cny_per_sqrt_second": float(sigma), "half_spread_cny": spread,
             "risk_source": "real training-only mid increments and median spread",
             "volume_source": "past clock-phase mean flow; ex-ante proxy, not realized future volume",
             "impact_source": "configured eta/gamma; not calibrated from own-order executions"}
    return audit, profile


def optimal_experiments(features: pd.DataFrame, cutoff: pd.Timestamp,
                        config: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Complete parameter grid, including analytically optimal and rounded lot schedules."""
    settings = config["optimal_execution"]
    audit, forecast = training_inputs(features, cutoff, settings)
    initial = ExecutionParameters(settings["order_size"], settings["intervals"], settings["duration_seconds"],
                                  audit["sigma_cny_per_sqrt_second"], settings["eta"], settings["gamma"],
                                  audit["half_spread_cny"])
    rows, trajectories = [], []
    grid = product(settings["risk_aversions"], settings["sigma_multipliers"],
                   settings["eta_multipliers"], settings["size_multipliers"])
    for experiment, (risk, sigma, eta, size) in enumerate(grid):
        parameters = replace(initial, sigma=initial.sigma * sigma,
                             eta=initial.eta * eta, order_size=initial.order_size * size)
        schedules = comparison_schedules(parameters.order_size, parameters.intervals, forecast)
        schedules["Almgren_Chriss"] = optimal_inventory(parameters, risk)
        for name, inventory in schedules.items():
            rounded = round_lots(inventory, config["market"]["lot_size"])
            rounded_cost = schedule_costs(rounded, parameters, risk)
            rows.append({"experiment": experiment, "schedule": name, "risk_aversion_per_cny": risk,
                         "sigma_multiplier": sigma, "eta_multiplier": eta, "size_multiplier": size,
                         "sigma_cny_per_sqrt_second": parameters.sigma, "eta_cny_second_per_share": parameters.eta,
                         "gamma_cny_per_share": parameters.gamma, "order_size_shares": parameters.order_size,
                         "parameter_status": "real risk inputs; uncalibrated impact sensitivity",
                         **schedule_costs(inventory, parameters, risk),
                         "rounded_expected_cost_cny": rounded_cost["expected_cost_cny"],
                         "rounded_variance_cny2": rounded_cost["cost_variance_cny2"]})
            for step, remaining in enumerate(inventory):
                trajectories.append({"experiment": experiment, "schedule": name, "step": step,
                                     "elapsed_seconds": step * parameters.tau, "inventory_shares": remaining,
                                     "trade_shares": inventory[step - 1] - remaining if step else 0,
                                     "rounded_inventory_shares": rounded[step],
                                     "rounded_trade_shares": rounded[step - 1] - rounded[step] if step else 0})
    audit["forecast_volume_profile"] = forecast.tolist()
    audit["eta"] = initial.eta
    audit["gamma"] = initial.gamma
    return pd.DataFrame(rows), pd.DataFrame(trajectories), audit
