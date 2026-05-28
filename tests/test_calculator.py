"""Tests for NORA calculator module."""

import math
import pytest
from nora.calculator import (
    describe_series,
    percentage_change,
    cagr,
    npv,
    linear_regression,
    safe_eval,
    convert_currency,
    yoy_growth,
)


def test_describe_series_basic():
    stats = describe_series([10, 20, 30, 40, 50])
    assert stats["n"] == 5
    assert stats["mean"] == 30.0
    assert stats["sum"] == 150.0
    assert stats["min"] == 10.0
    assert stats["max"] == 50.0


def test_describe_series_empty():
    assert describe_series([]) == {}


def test_percentage_change():
    assert percentage_change(100, 150) == pytest.approx(50.0)
    assert percentage_change(200, 100) == pytest.approx(-50.0)
    assert percentage_change(0, 10) == float("inf")


def test_cagr():
    # 100 → 200 over 5 years ≈ 14.87 %
    result = cagr(100, 200, 5)
    assert result == pytest.approx(0.1487, abs=1e-3)


def test_npv():
    # Initial investment -1000, returns 400/year for 3 years, 10% rate
    flows = [-1000, 400, 400, 400]
    result = npv(0.10, flows)
    assert result == pytest.approx(-0.526, abs=1.0)


def test_linear_regression():
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]  # y = 2x
    res = linear_regression(x, y)
    assert res["slope"] == pytest.approx(2.0)
    assert res["intercept"] == pytest.approx(0.0)
    assert res["r_squared"] == pytest.approx(1.0)


def test_safe_eval():
    assert safe_eval("2 + 2") == 4
    assert safe_eval("sqrt(9)") == pytest.approx(3.0)
    assert safe_eval("1500 * 1.25 / 3") == pytest.approx(625.0)


def test_safe_eval_rejects_dangerous():
    with pytest.raises((ValueError, NameError)):
        safe_eval("__import__('os').system('echo bad')")


def test_convert_currency():
    rates = {"USD": 10.5, "EUR": 11.5}
    # 100 USD → NOK
    assert convert_currency(100, "USD", "NOK", rates) == pytest.approx(1050.0)
    # 1050 NOK → USD
    assert convert_currency(1050, "NOK", "USD", rates) == pytest.approx(100.0)
    # 100 USD → EUR
    result = convert_currency(100, "USD", "EUR", rates)
    assert result == pytest.approx(100 * 10.5 / 11.5, rel=1e-4)


def test_yoy_growth():
    series = [100, 110, 121]
    result = yoy_growth(series)
    assert result[0] is None
    assert result[1] == pytest.approx(10.0)
    assert result[2] == pytest.approx(10.0)
