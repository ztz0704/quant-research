"""Data loading, validation, universe, alignment, and synthetic fixtures."""

from robust_alpha.data.align import add_open_to_open_target
from robust_alpha.data.loader import load_panel, save_panel
from robust_alpha.data.quality import panel_quality_summary
from robust_alpha.data.schema import normalize_and_validate_panel
from robust_alpha.data.synthetic import generate_synthetic_panel
from robust_alpha.data.universe import add_eligibility

__all__ = [
    "add_eligibility",
    "add_open_to_open_target",
    "generate_synthetic_panel",
    "load_panel",
    "normalize_and_validate_panel",
    "panel_quality_summary",
    "save_panel",
]
