"""Restricted, causal factor expressions. No eval, labels, attributes or forward shifts."""

from __future__ import annotations

import ast
import operator
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

FIELDS = {"open", "high", "low", "close", "volume", "amount", "ret_1"}
FUNCTIONS = {
    "delay": 2,
    "ts_mean": 2,
    "ts_std": 2,
    "ts_sum": 2,
    "ts_min": 2,
    "ts_max": 2,
    "ts_rank": 2,
    "ts_corr": 3,
    "cs_rank": 1,
    "abs": 1,
    "log": 1,
}
OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


def parse_expression(expression: str) -> ast.Expression:
    """Reject unknown fields, arbitrary Python, negative delays and invalid windows."""
    if not isinstance(expression, str) or not expression.strip() or len(expression) > 2000:
        raise ValueError("Factor expression must be a nonempty string of at most 2000 chars")
    tree = ast.parse(expression, mode="eval")
    if len(list(ast.walk(tree))) > 150:
        raise ValueError("Factor expression is too complex")

    def check(node: ast.AST) -> None:
        if isinstance(node, ast.Expression):
            check(node.body)
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, int | float):
                raise ValueError("Only numeric constants are supported")
            if not np.isfinite(node.value):
                raise ValueError("Constants must be finite")
        elif isinstance(node, ast.Name) and node.id in FIELDS:
            return
        elif isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
            check(node.left)
            check(node.right)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd | ast.USub):
            check(node.operand)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id
            if name not in FUNCTIONS or len(node.args) != FUNCTIONS[name] or node.keywords:
                raise ValueError(f"Invalid factor function or arguments: {name}")
            if name == "delay" or name.startswith("ts_"):
                window = node.args[-1]
                minimum = 0 if name == "delay" else 2 if name == "ts_corr" else 1
                if (
                    not isinstance(window, ast.Constant)
                    or type(window.value) is not int
                    or not minimum <= window.value <= 2600
                ):
                    raise ValueError(f"{name} needs a literal integer in [{minimum}, 2600]")
            for argument in node.args:
                check(argument)
        else:
            raise ValueError(f"Unsupported factor expression element: {type(node).__name__}")

    check(tree)
    return tree


def validate_definitions(definitions: Mapping[str, Any], reserved: set[str]) -> None:
    if not isinstance(definitions, Mapping):
        raise ValueError("custom_signals must be a mapping")
    for name, spec in definitions.items():
        if (
            not isinstance(name, str)
            or not name.isidentifier()
            or name in reserved
            or name in FIELDS
            or name.startswith(("z__", "target", "entry_", "exit_"))
        ):
            raise ValueError(f"Invalid or reserved custom signal name: {name}")
        if not isinstance(spec, Mapping) or type(spec.get("direction", 1)) is not int:
            raise ValueError(f"Custom signal {name} needs a mapping and integer direction")
        if spec.get("direction", 1) not in {-1, 1}:
            raise ValueError("Factor direction must be declared as +1 or -1 before evaluation")
        parse_expression(spec.get("expression"))


def evaluate_expression(panel: pd.DataFrame, expression: str) -> pd.Series:
    """All time-series functions operate backwards inside each asset's market grid."""
    tree = parse_expression(expression)
    fields = {name: panel[name].astype(float) for name in FIELDS - {"ret_1"}}
    fields["ret_1"] = panel.groupby("asset", sort=False)["close"].pct_change(fill_method=None)

    def series(value: Any) -> pd.Series:
        return value if isinstance(value, pd.Series) else pd.Series(value, index=panel.index)

    def evaluate(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return fields[node.id]
        if isinstance(node, ast.UnaryOp):
            value = evaluate(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp):
            return OPERATORS[type(node.op)](evaluate(node.left), evaluate(node.right))
        name = node.func.id
        arguments = [evaluate(argument) for argument in node.args]
        value = series(arguments[0])
        if name == "abs":
            return value.abs()
        if name == "log":
            return np.log(value.where(value > 0))
        if name == "cs_rank":
            visible = value.where(panel["eligible"] if "eligible" in panel else True)
            return visible.groupby(panel["date"], sort=False).rank(pct=True, method="average")
        grouped = value.groupby(panel["asset"], sort=False)
        window = arguments[-1]
        if name == "delay":
            return grouped.shift(window)
        if name == "ts_corr":
            other = series(arguments[1])
            result = pd.Series(np.nan, index=panel.index)
            for index in panel.groupby("asset", sort=False).groups.values():
                result.loc[index] = value.loc[index].rolling(window).corr(other.loc[index])
            return result
        method = name.removeprefix("ts_")
        if method == "rank":
            return grouped.transform(lambda part: part.rolling(window).rank(pct=True))
        return grouped.transform(lambda part: getattr(part.rolling(window), method)())

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return series(evaluate(tree)).replace([np.inf, -np.inf], np.nan).astype(float)
