"""
Web fetcher – retrieves supplementary data from trusted, authoritative sources.

Trusted sources used by NORA:
  - Norges Bank  : exchange rates (NOK)
  - ECB           : EUR reference rates
  - SSB           : Norwegian statistics (Statistics Norway)
  - World Bank    : global macroeconomic indicators
  - Eurostat      : EU statistics
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any, Optional

import httpx

log = logging.getLogger(__name__)

# HTTP client with sensible defaults
_CLIENT_KWARGS = {"timeout": 20.0, "headers": {"User-Agent": "NORA-Agent/0.1 (ErikJarlHolm)"}}


# ── Norges Bank exchange rates ────────────────────────────────────────────────

def get_norges_bank_rates(
    base: str = "NOK",
    currencies: Optional[list[str]] = None,
    on_date: Optional[date] = None,
) -> dict[str, float]:
    """
    Fetch exchange rates from Norges Bank's official data API.
    Returns {currency_code: rate_in_NOK} (or inverse if base != NOK).

    Docs: https://data.norges-bank.no/api/
    """
    currencies = currencies or ["USD", "EUR", "GBP", "SEK", "DKK", "CHF", "JPY"]
    freq = "B"  # business day

    results: dict[str, float] = {}
    date_str = on_date.strftime("%Y-%m-%d") if on_date else datetime.today().strftime("%Y-%m-%d")

    for cur in currencies:
        url = (
            f"https://data.norges-bank.no/api/data/EXR/{freq}.{cur}.NOK.SP"
            f"?startPeriod={date_str}&endPeriod={date_str}&format=sdmx-json&locale=no"
        )
        try:
            with httpx.Client(**_CLIENT_KWARGS) as client:
                r = client.get(url)
                r.raise_for_status()
                data = r.json()
                obs = data["data"]["dataSets"][0]["series"]["0:0:0:0"]["observations"]
                # Take the most recent observation
                rate = float(list(obs.values())[-1][0])
                results[cur] = rate
                log.info("Norges Bank: 1 %s = %.4f NOK", cur, rate)
        except Exception as exc:
            log.warning("Norges Bank kurs for %s feilet: %s", cur, exc)

    return results


# ── ECB reference rates ───────────────────────────────────────────────────────

def get_ecb_rates() -> dict[str, float]:
    """
    Fetch latest EUR reference rates from the European Central Bank.
    Returns {currency_code: units_per_EUR}.

    Source: https://www.ecb.europa.eu/stats/eurofxref/eurofxref-daily.xml
    """
    url = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-daily.xml"
    try:
        with httpx.Client(**_CLIENT_KWARGS) as client:
            r = client.get(url)
            r.raise_for_status()

        from xml.etree import ElementTree as ET
        root = ET.fromstring(r.text)
        ns = {"gesmes": "http://www.gesmes.org/xml/2002-08-01", "ecb": "http://www.ecb.int/vocabulary/2002-08-01/eurofxref"}
        rates: dict[str, float] = {"EUR": 1.0}
        for cube in root.findall(".//ecb:Cube[@currency]", ns):
            rates[cube.attrib["currency"]] = float(cube.attrib["rate"])
        log.info("ECB: %d valutakurser hentet", len(rates))
        return rates
    except Exception as exc:
        log.warning("ECB kurs feilet: %s", exc)
        return {}


# ── SSB (Statistics Norway) ───────────────────────────────────────────────────

def get_ssb_statistic(table_id: str, query: dict) -> dict[str, Any]:
    """
    Fetch data from SSB's JSON-stat API.

    Args:
        table_id: SSB table ID, e.g. "12880" (consumer price index).
        query:    JSON-stat query dict (see SSB API docs).

    Returns raw JSON response dict.

    Docs: https://data.ssb.no/api/
    """
    url = f"https://data.ssb.no/api/v0/no/table/{table_id}"
    try:
        with httpx.Client(**_CLIENT_KWARGS) as client:
            r = client.post(url, json=query)
            r.raise_for_status()
            data = r.json()
            log.info("SSB tabell %s: hentet OK", table_id)
            return data
    except Exception as exc:
        log.warning("SSB tabell %s feilet: %s", table_id, exc)
        return {}


def get_ssb_kpi(year: Optional[int] = None, month: Optional[int] = None) -> dict[str, Any]:
    """Convenience wrapper: fetch Norwegian CPI (KPI) from SSB table 03013."""
    today = datetime.today()
    y = year or today.year
    m = month or today.month
    period = f"{y}M{m:02d}"

    query = {
        "query": [
            {"code": "Tid", "selection": {"filter": "item", "values": [period]}},
            {"code": "ContentsCode", "selection": {"filter": "item", "values": ["KpiAlle"]}},
        ],
        "response": {"format": "json-stat2"},
    }
    return get_ssb_statistic("03013", query)


# ── World Bank ────────────────────────────────────────────────────────────────

def get_world_bank_indicator(
    indicator: str,
    country: str = "NO",
    mrv: int = 5,
) -> list[dict[str, Any]]:
    """
    Fetch a World Bank indicator for a country.

    Args:
        indicator: e.g. "NY.GDP.MKTP.CD" (GDP in USD).
        country:   ISO 3166-1 alpha-2 code (default: Norway).
        mrv:       Most recent values to retrieve.

    Docs: https://datahelpdesk.worldbank.org/knowledgebase/articles/898581
    """
    url = (
        f"https://api.worldbank.org/v2/country/{country}/indicator/{indicator}"
        f"?format=json&mrv={mrv}"
    )
    try:
        with httpx.Client(**_CLIENT_KWARGS) as client:
            r = client.get(url)
            r.raise_for_status()
            payload = r.json()
            records = payload[1] if isinstance(payload, list) and len(payload) > 1 else []
            log.info("World Bank %s/%s: %d records", country, indicator, len(records))
            return records
    except Exception as exc:
        log.warning("World Bank %s/%s feilet: %s", country, indicator, exc)
        return []


# ── Eurostat ──────────────────────────────────────────────────────────────────

def get_eurostat_dataset(dataset_id: str, params: Optional[dict] = None) -> dict[str, Any]:
    """
    Fetch a dataset from Eurostat's JSON API.

    Args:
        dataset_id: e.g. "prc_hicp_manr" (HICP inflation).
        params:     Optional query parameters.

    Docs: https://ec.europa.eu/eurostat/web/json-and-unicode-web-services
    """
    url = f"https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/{dataset_id}"
    try:
        with httpx.Client(**_CLIENT_KWARGS) as client:
            r = client.get(url, params=params or {})
            r.raise_for_status()
            data = r.json()
            log.info("Eurostat %s: hentet OK", dataset_id)
            return data
    except Exception as exc:
        log.warning("Eurostat %s feilet: %s", dataset_id, exc)
        return {}
