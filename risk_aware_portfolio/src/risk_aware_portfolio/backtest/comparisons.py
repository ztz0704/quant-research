"""Shared predeclared strategy matrix; actual fees apply to every strategy."""

STRATEGIES = (
    ("Baseline", "baseline", True, False, 0.0),
    ("Risk-Aware Portfolio", "risk_aware", False, False, 0.0),
    ("Risk-Aware + TC", "risk_aware_tc", False, None, None),
    ("Neutralized Alpha Portfolio", "neutralized", False, True, None),
)
