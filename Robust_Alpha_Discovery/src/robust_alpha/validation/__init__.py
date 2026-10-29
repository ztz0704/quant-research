"""Walk-forward, multiple-testing, bootstrap, and regime validation."""

from robust_alpha.validation.bootstrap import joint_moving_block_max_t
from robust_alpha.validation.multiple_testing import adjust_pvalues
from robust_alpha.validation.regimes import build_market_regimes, regime_ic_summary
from robust_alpha.validation.splits import (
    RESEARCH_PHASES,
    filter_evaluation_dates,
    filter_walk_forward_years,
    resolve_research_window,
)
from robust_alpha.validation.walk_forward import (
    attach_walk_forward_score,
    walk_forward_ic_selection,
)

__all__ = [
    "adjust_pvalues",
    "attach_walk_forward_score",
    "build_market_regimes",
    "joint_moving_block_max_t",
    "RESEARCH_PHASES",
    "filter_evaluation_dates",
    "filter_walk_forward_years",
    "regime_ic_summary",
    "resolve_research_window",
    "walk_forward_ic_selection",
]
