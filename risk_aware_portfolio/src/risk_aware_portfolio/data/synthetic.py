"""Reproducible time-varying factor DGP with separate truth artifacts."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from risk_aware_portfolio.utils.config import Config


@dataclass(frozen=True)
class SyntheticData:
    """Panel plus truth kept outside the estimation/optimization interface."""

    panel: pd.DataFrame
    factor_returns: pd.DataFrame
    factor_covariance: pd.DataFrame
    truth: pd.DataFrame


def generate_synthetic(config: Config) -> SyntheticData:
    """Generate r[t+1] = X[t] f[t+1] + mu[t] + epsilon[t+1].

    All innovations are drawn sequentially. The alpha signal contains a genuine
    residual component and incidental style exposure. Truth is for validation only.
    """
    settings, risk = config["synthetic"], config["risk_model"]
    rng = np.random.default_rng(settings["seed"])
    n, t = settings["n_assets"], settings["n_dates"]
    styles, industries = risk["styles"], risk["industries"]
    dates = pd.bdate_range(settings["start_date"], periods=t)
    assets = np.array([f"asset_{i:04d}" for i in range(n)])
    sectors = np.arange(n) % len(industries)
    names = ["market"] + [f"industry:{s}" for s in industries[1:]] + styles
    k = len(names)
    loadings = rng.normal(size=(k, k))
    corr = loadings @ loadings.T
    corr /= np.sqrt(np.outer(np.diag(corr), np.diag(corr)))
    sigma = settings["factor_volatility"] ** 2 * (0.7 * np.eye(k) + 0.3 * corr)
    state = rng.normal(size=(n, len(styles)))
    signal_state = rng.normal(size=n)
    previous_x, previous_mu = None, np.zeros(n)
    panels, truths, factors = [], [], []
    for date in dates:
        state = 0.96 * state + np.sqrt(1 - 0.96**2) * rng.normal(size=state.shape)
        standardized = (state - state.mean(axis=0)) / state.std(axis=0)
        x = np.column_stack(
            [np.ones(n)]
            + [(sectors == j).astype(float) for j in range(1, len(industries))]
            + [standardized]
        )
        factor = rng.multivariate_normal(np.zeros(k), sigma)
        epsilon = rng.normal(0, settings["specific_volatility"], n)
        factor_component = np.zeros(n) if previous_x is None else previous_x @ factor
        returns = factor_component + previous_mu + epsilon
        signal_state = 0.8 * signal_state + 0.6 * rng.normal(size=n)
        signal = signal_state + 0.5 * standardized[:, 1]
        mu = settings["alpha_strength"] * (signal - signal.mean()) / signal.std()
        frame = pd.DataFrame(
            {
                "date": date,
                "asset": assets,
                "return": returns,
                "alpha": signal,
                "market_cap": np.exp(22 + standardized[:, 0]),
                "industry": np.array(industries)[sectors],
            }
        )
        for j, name in enumerate(styles):
            frame[name] = standardized[:, j]
        panels.append(frame)
        truths.append(
            pd.DataFrame(
                {
                    "date": date,
                    "asset": assets,
                    "true_factor_return": factor_component,
                    "true_alpha_return": previous_mu,
                    "true_specific_return": epsilon,
                }
            )
        )
        factors.append(factor)
        previous_x, previous_mu = x, mu
    panel = pd.concat(panels, ignore_index=True).sort_values(["date", "asset"])
    return SyntheticData(
        panel.reset_index(drop=True),
        pd.DataFrame(factors, index=dates, columns=names),
        pd.DataFrame(sigma, index=names, columns=names),
        pd.concat(truths, ignore_index=True),
    )
