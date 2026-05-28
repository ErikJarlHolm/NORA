"""
Calculator – numeric computation utilities available to NORA.

These functions are exposed to the LLM as "tools" so the model can
request precise calculations rather than relying on its own arithmetic.
"""

from __future__ import annotations

import math
import statistics
from typing import Any

import numpy as np
import pandas as pd


# ── Basic statistics ──────────────────────────────────────────────────────────

def describe_series(values: list[float]) -> dict[str, float]:
    """Return comprehensive descriptive statistics for a list of numbers."""
    arr = np.array([v for v in values if v is not None and not math.isnan(v)], dtype=float)
    if arr.size == 0:
        return {}
    q1, median, q3 = np.percentile(arr, [25, 50, 75])
    return {
        "n": float(arr.size),
        "sum": float(arr.sum()),
        "mean": float(arr.mean()),
        "median": float(median),
        "mode": float(statistics.mode(arr.tolist())) if arr.size > 1 else float(arr[0]),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "var": float(arr.var(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "range": float(arr.max() - arr.min()),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
        "cv_pct": float(arr.std(ddof=1) / arr.mean() * 100) if arr.mean() != 0 else 0.0,
    }


# ── Growth & change ───────────────────────────────────────────────────────────

def percentage_change(old: float, new: float) -> float:
    """Return percentage change from *old* to *new*."""
    if old == 0:
        return float("inf") if new > 0 else float("-inf")
    return (new - old) / abs(old) * 100


def cagr(start: float, end: float, years: float) -> float:
    """Compound Annual Growth Rate."""
    if start <= 0 or years <= 0:
        raise ValueError("start og years må være positive")
    return (end / start) ** (1 / years) - 1


def yoy_growth(series: list[float]) -> list[float | None]:
    """Year-over-year growth rates for a time series."""
    result: list[float | None] = [None]
    for i in range(1, len(series)):
        result.append(percentage_change(series[i - 1], series[i]))
    return result


# ── Financial calculations ────────────────────────────────────────────────────

def npv(rate: float, cash_flows: list[float]) -> float:
    """Net Present Value."""
    return float(sum(cf / (1 + rate) ** t for t, cf in enumerate(cash_flows)))


def irr(cash_flows: list[float], guess: float = 0.1) -> float:
    """Internal Rate of Return (Newton-Raphson)."""
    from scipy.optimize import brentq

    def npv_fn(r: float) -> float:
        return sum(cf / (1 + r) ** t for t, cf in enumerate(cash_flows))

    try:
        return float(brentq(npv_fn, -0.999, 10.0))
    except ValueError:
        return float("nan")


def payback_period(initial: float, annual_cash_flow: float) -> float:
    """Simple payback period in years."""
    if annual_cash_flow <= 0:
        return float("inf")
    return initial / annual_cash_flow


# ── Currency conversion ───────────────────────────────────────────────────────

def convert_currency(
    amount: float,
    from_currency: str,
    to_currency: str,
    rates: dict[str, float],
) -> float:
    """
    Convert *amount* from *from_currency* to *to_currency*.

    *rates* should be {currency: units_per_base} where base = NOK
    (as returned by web_fetcher.get_norges_bank_rates).
    """
    from_currency = from_currency.upper()
    to_currency = to_currency.upper()

    if from_currency == to_currency:
        return amount

    # rates are expressed as "X NOK per 1 foreign currency"
    # so: amount (foreign) → NOK → target foreign
    if from_currency == "NOK":
        if to_currency not in rates:
            raise KeyError(f"Valuta ikke tilgjengelig: {to_currency}")
        return amount / rates[to_currency]
    elif to_currency == "NOK":
        if from_currency not in rates:
            raise KeyError(f"Valuta ikke tilgjengelig: {from_currency}")
        return amount * rates[from_currency]
    else:
        if from_currency not in rates or to_currency not in rates:
            missing = [c for c in [from_currency, to_currency] if c not in rates]
            raise KeyError(f"Valuta ikke tilgjengelig: {missing}")
        amount_nok = amount * rates[from_currency]
        return amount_nok / rates[to_currency]


# ── DataFrame utilities ───────────────────────────────────────────────────────

def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return extended describe() for all numeric columns in a DataFrame."""
    num = df.select_dtypes(include="number")
    if num.empty:
        return pd.DataFrame()
    desc = num.describe(percentiles=[0.25, 0.5, 0.75]).T
    desc["sum"] = num.sum()
    desc["cv_%"] = num.std() / num.mean() * 100
    desc["missing"] = df[num.columns].isna().sum()
    desc["missing_%"] = desc["missing"] / len(df) * 100
    return desc.round(4)


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation matrix for numeric columns."""
    return df.select_dtypes(include="number").corr().round(4)


def detect_outliers_iqr(series: pd.Series, multiplier: float = 1.5) -> pd.Series:
    """Return boolean mask of outliers using the IQR method."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return (series < q1 - multiplier * iqr) | (series > q3 + multiplier * iqr)


def aggregate_by(df: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
    """Group by *group_col* and compute sum/mean/count for *value_col*."""
    return (
        df.groupby(group_col)[value_col]
        .agg(["sum", "mean", "count", "min", "max"])
        .round(4)
        .reset_index()
    )


# ── Regression (simple linear) ────────────────────────────────────────────────

def linear_regression(x: list[float], y: list[float]) -> dict[str, float]:
    """Fit y = a + b*x and return coefficients + R²."""
    from scipy.stats import linregress

    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_value ** 2),
        "p_value": float(p_value),
        "std_err": float(std_err),
    }


# ── Safe eval for arithmetic expressions ─────────────────────────────────────

_SAFE_NAMES = {
    k: v for k, v in math.__dict__.items() if not k.startswith("_")
}
_SAFE_NAMES.update({"abs": abs, "round": round, "min": min, "max": max, "sum": sum})


def safe_eval(expression: str) -> Any:
    """
    Evaluate a mathematical expression string safely.
    Only standard math functions are available – no builtins, no imports.
    """
    try:
        return eval(expression, {"__builtins__": {}}, _SAFE_NAMES)  # noqa: S307
    except Exception as exc:
        raise ValueError(f"Ugyldig uttrykk: {expression!r} – {exc}") from exc
