"""Asset covariance construction with auditable PSD validation and repair."""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger("risk_aware_portfolio")


@dataclass(frozen=True)
class CovarianceAudit:
    before: np.ndarray
    after: np.ndarray
    repaired: bool
    symmetry_error: float

    def metrics(self) -> dict[str, float]:
        """Summarize the numerical state; eigenvalue vectors remain separately saved."""
        smallest, largest = float(self.after.min()), float(self.after.max())
        return {
            "smallest_eigenvalue": smallest,
            "largest_eigenvalue": largest,
            "condition_number": largest / smallest if smallest > 0 else np.inf,
            "symmetry_error": self.symmetry_error,
            "repaired": float(self.repaired),
            "min_eigenvalue_before": float(self.before.min()),
        }


def ensure_psd(matrix: np.ndarray, floor: float = 1e-10) -> tuple[np.ndarray, CovarianceAudit]:
    """Symmetrize roundoff and clip eigenvalues, always warning on actual repair."""
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1] or not np.isfinite(values).all():
        raise ValueError("Covariance must be a finite square matrix")
    if floor < 0:
        raise ValueError("Eigenvalue floor must be nonnegative")
    symmetry_error = float(np.max(np.abs(values - values.T)))
    if symmetry_error > 1e-8:
        raise ValueError(f"Materially asymmetric covariance: {symmetry_error}")
    symmetric = (values + values.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    clipped = np.maximum(eigenvalues, floor)
    repaired = bool(np.any(eigenvalues < floor))
    if repaired:
        logger.warning(
            "PSD repair min_before=%.6g min_after=%.6g clipped=%d",
            eigenvalues.min(),
            clipped.min(),
            np.sum(eigenvalues < floor),
        )
        symmetric = (eigenvectors * clipped) @ eigenvectors.T
        symmetric = (symmetric + symmetric.T) / 2
    return symmetric, CovarianceAudit(eigenvalues, clipped, repaired, symmetry_error)


def build_asset_covariance(
    exposures: pd.DataFrame,
    factor_covariance: pd.DataFrame,
    specific_variance: pd.Series,
    floor: float = 1e-10,
) -> tuple[pd.DataFrame, CovarianceAudit]:
    """Construct X F X' + D with strict factor and asset alignment."""
    if not exposures.columns.equals(factor_covariance.index):
        raise ValueError("Factor alignment mismatch")
    if not factor_covariance.columns.equals(factor_covariance.index):
        raise ValueError("Factor covariance labels mismatch")
    if not exposures.index.equals(specific_variance.index):
        raise ValueError("Specific risk alignment mismatch")
    if (specific_variance < 0).any() or not np.isfinite(specific_variance).all():
        raise ValueError("Invalid specific variance")
    x = exposures.to_numpy()
    matrix = x @ factor_covariance.to_numpy() @ x.T + np.diag(specific_variance)
    repaired, audit = ensure_psd(matrix, floor)
    return pd.DataFrame(repaired, index=exposures.index, columns=exposures.index), audit
