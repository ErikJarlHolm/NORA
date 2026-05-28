"""
NORA – core agent logic.

Orchestrates file reading, calculations, and web data fetching through
an LLM (Azure OpenAI or OpenAI) using the Assistants / tool-calling API.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import openai

from .calculator import (
    aggregate_by,
    cagr,
    convert_currency,
    correlation_matrix,
    describe_series,
    detect_outliers_iqr,
    linear_regression,
    npv,
    numeric_summary,
    percentage_change,
    safe_eval,
    yoy_growth,
)
from .config import settings
from .file_reader import FileContent, read_file, read_folder
from .web_fetcher import (
    get_ecb_rates,
    get_norges_bank_rates,
    get_ssb_kpi,
    get_world_bank_indicator,
)

log = logging.getLogger(__name__)

# ── Tool definitions (OpenAI function calling schema) ─────────────────────────

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "describe_series",
            "description": "Beregn beskrivende statistikk (gjennomsnitt, median, std, etc.) for en tallliste.",
            "parameters": {
                "type": "object",
                "properties": {
                    "values": {"type": "array", "items": {"type": "number"}, "description": "Tallverdiene"}
                },
                "required": ["values"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "percentage_change",
            "description": "Beregn prosentvis endring fra én verdi til en annen.",
            "parameters": {
                "type": "object",
                "properties": {
                    "old": {"type": "number"},
                    "new": {"type": "number"},
                },
                "required": ["old", "new"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cagr",
            "description": "Beregn sammensatt årlig vekstrate (CAGR).",
            "parameters": {
                "type": "object",
                "properties": {
                    "start": {"type": "number"},
                    "end": {"type": "number"},
                    "years": {"type": "number"},
                },
                "required": ["start", "end", "years"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "npv",
            "description": "Beregn netto nåverdi (NPV) gitt diskonteringsrente og kontantstrømmer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "rate": {"type": "number", "description": "Diskonteringsrente, f.eks. 0.08 for 8 %"},
                    "cash_flows": {"type": "array", "items": {"type": "number"}},
                },
                "required": ["rate", "cash_flows"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "linear_regression",
            "description": "Utfør enkel lineær regresjon y = a + b*x og returner koeffisienter og R².",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "array", "items": {"type": "number"}},
                    "y": {"type": "array", "items": {"type": "number"}},
                },
                "required": ["x", "y"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "safe_eval",
            "description": "Evaluer et matematisk uttrykk som en streng, f.eks. '(1500 * 1.25) / 3'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"}
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_norges_bank_rates",
            "description": "Hent valutakurser mot NOK fra Norges Bank (offisiell kilde).",
            "parameters": {
                "type": "object",
                "properties": {
                    "currencies": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Liste med valutakoder, f.eks. ['USD','EUR']",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_ecb_rates",
            "description": "Hent ECBs offisielle EUR-referansekurser.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_ssb_kpi",
            "description": "Hent norsk konsumprisindeks (KPI) fra SSB (Statistisk sentralbyrå).",
            "parameters": {
                "type": "object",
                "properties": {
                    "year": {"type": "integer"},
                    "month": {"type": "integer"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_world_bank_indicator",
            "description": "Hent makroøkonomiske indikatorer fra Verdensbanken.",
            "parameters": {
                "type": "object",
                "properties": {
                    "indicator": {"type": "string", "description": "f.eks. NY.GDP.MKTP.CD"},
                    "country": {"type": "string", "description": "ISO 2-bokstavs landkode, f.eks. 'NO'"},
                    "mrv": {"type": "integer", "description": "Antall siste verdier"},
                },
                "required": ["indicator"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "convert_currency",
            "description": "Konverter et beløp mellom valutaer ved hjelp av Norges Banks kurser.",
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {"type": "number"},
                    "from_currency": {"type": "string"},
                    "to_currency": {"type": "string"},
                },
                "required": ["amount", "from_currency", "to_currency"],
            },
        },
    },
]


# ── Tool dispatcher ───────────────────────────────────────────────────────────

_nb_rates_cache: dict[str, float] = {}


def _dispatch(name: str, args: dict) -> str:
    """Execute a tool call and return the result as a JSON string."""
    global _nb_rates_cache

    if name == "describe_series":
        return json.dumps(describe_series(**args))

    elif name == "percentage_change":
        val = percentage_change(**args)
        return json.dumps({"percentage_change": val})

    elif name == "cagr":
        val = cagr(**args)
        return json.dumps({"cagr_pct": round(val * 100, 4)})

    elif name == "npv":
        return json.dumps({"npv": npv(**args)})

    elif name == "linear_regression":
        return json.dumps(linear_regression(**args))

    elif name == "safe_eval":
        result = safe_eval(**args)
        return json.dumps({"result": result})

    elif name == "get_norges_bank_rates":
        rates = get_norges_bank_rates(**args)
        _nb_rates_cache = rates
        return json.dumps(rates)

    elif name == "get_ecb_rates":
        return json.dumps(get_ecb_rates())

    elif name == "get_ssb_kpi":
        return json.dumps(get_ssb_kpi(**args))

    elif name == "get_world_bank_indicator":
        records = get_world_bank_indicator(**args)
        simplified = [{"year": r.get("date"), "value": r.get("value")} for r in records]
        return json.dumps(simplified)

    elif name == "convert_currency":
        if not _nb_rates_cache:
            _nb_rates_cache = get_norges_bank_rates()
        result = convert_currency(**args, rates=_nb_rates_cache)
        return json.dumps({"converted": result})

    else:
        return json.dumps({"error": f"Ukjent verktøy: {name}"})


# ── OpenAI client factory ─────────────────────────────────────────────────────

def _build_client() -> tuple[openai.OpenAI | openai.AzureOpenAI, str]:
    """Return (client, model_name) depending on settings."""
    if settings.use_azure:
        client = openai.AzureOpenAI(
            api_key=settings.azure_openai_api_key,
            azure_endpoint=settings.azure_openai_endpoint,
            api_version=settings.azure_openai_api_version,
        )
        return client, settings.azure_openai_deployment
    else:
        client = openai.OpenAI(api_key=settings.openai_api_key)
        return client, settings.openai_model


# ── NORA system prompt ────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
Du er NORA – Numerical Operations & Results Assistant.
Du er en ekspert på å lese, forstå og analysere dokumenter som inneholder tallmateriale.

Oppgavene dine inkluderer:
- Lese og forstå tallene OG tekstkonteksten rundt dem
- Gjengi tall nøyaktig fra kildedokumentene
- Utføre presise, komplekse beregninger (statistikk, finans, vekst, konvertering, regresjon)
- Hente oppdaterte tall fra internett NÅR det er nødvendig (valutakurser, KPI, statistikk)
- Alltid bruke kun pålitelige, offisielle kilder (Norges Bank, SSB, ECB, Verdensbanken, Eurostat)

Svar alltid på norsk med mindre brukeren ber om noe annet.
Vis beregningstrinn tydelig. Oppgi kilde når du bruker data fra internett.
Vær presis: si eksplisitt hvilke tall du har funnet i hvilken fil.
""".strip()


# ── Main agent class ──────────────────────────────────────────────────────────

class Nora:
    """
    NORA agent – load files, then chat.

    Usage:
        agent = Nora()
        agent.load_folder()          # or agent.load_file("myfile.xlsx")
        response = agent.ask("Hva er total omsetning?")
        print(response)
    """

    def __init__(self) -> None:
        self.client, self.model = _build_client()
        self.file_contents: list[FileContent] = []
        self.history: list[dict] = []

    # ── File loading ──────────────────────────────────────────────────────────

    def load_file(self, path: str | Path) -> None:
        """Load a single file into the agent's context."""
        fc = read_file(Path(path))
        self.file_contents.append(fc)
        log.info("Lastet: %s", fc.filename)

    def load_folder(self, folder: Optional[str | Path] = None) -> None:
        """Load all supported files from *folder* (default: settings.data_folder)."""
        folder = Path(folder) if folder else settings.data_folder
        self.file_contents = read_folder(folder)
        log.info("Lastet %d fil(er) fra %s", len(self.file_contents), folder)

    # ── Conversation ──────────────────────────────────────────────────────────

    def ask(self, question: str) -> str:
        """Send *question* to NORA and return her response."""
        if not self.history:
            # Build initial context from loaded files
            context = self._build_context()
            self.history = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": context},
                {"role": "assistant", "content": "Jeg har lest filene og er klar til å hjelpe. Hva vil du vite?"},
            ]

        self.history.append({"role": "user", "content": question})
        response = self._run_with_tools()
        self.history.append({"role": "assistant", "content": response})
        return response

    def reset(self) -> None:
        """Clear conversation history (keeps loaded files)."""
        self.history = []

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_context(self) -> str:
        if not self.file_contents:
            return "Ingen filer er lastet inn ennå."
        parts = ["Her er innholdet i de innlastede filene:\n"]
        for fc in self.file_contents:
            parts.append(f"=== {fc.filename} ===")
            parts.append(fc.text[:8000])  # truncate very large files
            if len(fc.text) > 8000:
                parts.append(f"[... {len(fc.text) - 8000} tegn utelatt ...]")
        return "\n\n".join(parts)

    def _run_with_tools(self) -> str:
        """Run the model with tool-calling loop until a final response is produced."""
        messages = list(self.history)

        for _ in range(10):  # max 10 tool rounds
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )
            msg = completion.choices[0].message

            if msg.tool_calls:
                messages.append(msg)
                for tc in msg.tool_calls:
                    args = json.loads(tc.function.arguments)
                    log.debug("Tool call: %s(%s)", tc.function.name, args)
                    result = _dispatch(tc.function.name, args)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })
            else:
                return msg.content or ""

        return "Beklager, kunne ikke fullføre beregningen etter maks antall steg."
