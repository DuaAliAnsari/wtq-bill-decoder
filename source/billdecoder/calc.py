"""Safe evaluator for the arithmetic the model asks for.

The model writes small Python expressions; this module checks them against a strict whitelist and
evaluates them, so every number in a final answer comes from code rather than from the model's head.
"""
from __future__ import annotations

import ast
import math
import statistics
from typing import Any

MAX_EXPR_LENGTH = 400
MAX_INT_CONSTANT = 10_000_000


def mean(values) -> float | None:
    values = [v for v in values if v is not None]
    return statistics.fmean(values) if values else None


def median(values) -> float | None:
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def safe_div(a, b):
    if a is None or b in (None, 0):
        return None
    return a / b


def pct(part, whole):
    ratio = safe_div(part, whole)
    return None if ratio is None else ratio * 100


FUNCTIONS = {
    "sum": sum, "min": min, "max": max, "len": len, "abs": abs, "round": round, "sorted": sorted,
    "mean": mean, "median": median, "pct": pct, "safe_div": safe_div,
}

_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp, ast.Constant, ast.Name,
    ast.Call, ast.List, ast.Tuple, ast.Dict, ast.Subscript, ast.Slice, ast.ListComp, ast.GeneratorExp,
    ast.comprehension, ast.Load, ast.Store,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.USub, ast.UAdd, ast.Not,
    ast.And, ast.Or, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Is, ast.IsNot, ast.In, ast.NotIn,
)


class UnsafeExpression(ValueError):
    pass


def _validate(tree: ast.AST, allowed_names: set[str]) -> None:
    bound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.comprehension):
            for target in ast.walk(node.target):
                if isinstance(target, ast.Name):
                    bound.add(target.id)
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise UnsafeExpression(f"{type(node).__name__} is not allowed")
        if isinstance(node, ast.Name):
            if node.id.startswith("_") or node.id not in allowed_names | bound:
                raise UnsafeExpression(f"unknown name '{node.id}'")
        if isinstance(node, ast.Call):
            if not (isinstance(node.func, ast.Name) and node.func.id in FUNCTIONS) or node.keywords:
                raise UnsafeExpression("only plain calls to the allowed helper functions are permitted")
        if isinstance(node, ast.Constant):
            if isinstance(node.value, int) and abs(node.value) > MAX_INT_CONSTANT:
                raise UnsafeExpression("constant too large")
            if not isinstance(node.value, (int, float, str, bool, type(None))):
                raise UnsafeExpression("unsupported constant")


def evaluate(expr: str, variables: dict[str, Any]) -> Any:
    """Evaluate one whitelisted expression. Raises UnsafeExpression / ValueError / TypeError on failure."""
    if not isinstance(expr, str) or not expr.strip():
        raise UnsafeExpression("empty expression")
    if len(expr) > MAX_EXPR_LENGTH:
        raise UnsafeExpression("expression too long")
    tree = ast.parse(expr.strip(), mode="eval")
    _validate(tree, set(variables) | set(FUNCTIONS))
    code = compile(tree, "<expr>", "eval")
    namespace = {"__builtins__": {}, **FUNCTIONS, **variables}  # one dict: comprehensions only see globals
    return eval(code, namespace)  # noqa: S307 - AST is whitelisted


def jsonable(value: Any) -> Any:
    """Make an evaluation result JSON-friendly and tidy (round floats, cap list length)."""
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return round(value, 4)
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in list(value)[:60]]
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in list(value.items())[:60]}
    return str(value)


def run_calculations(calculations: Any, variables: dict[str, Any]) -> dict[str, Any]:
    """Run a list of {"name", "expr"}; failures are reported per item instead of aborting the batch."""
    results: dict[str, Any] = {}
    if not isinstance(calculations, list):
        return results
    scope = dict(variables)
    for item in calculations[:20]:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name.isidentifier() or name.startswith("_") or name in FUNCTIONS or name in variables:
            continue
        try:
            value = evaluate(item.get("expr"), scope)
            results[name] = jsonable(value)
            scope[name] = value  # later calculations may reuse earlier ones
        except Exception as exc:  # noqa: BLE001 - any failure just means "not computed"
            results[name] = f"could not be computed ({type(exc).__name__})"
    return results
