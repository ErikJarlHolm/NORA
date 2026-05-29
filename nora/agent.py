"""
NORA – core agent logic.

Arkitektur-oversikt:
    Bygget som en Azure AI Foundry-agent med lokale verktøy for
    fillesing, beregninger og henting av data fra pålitelige nettkilder.

    Flyten er:
    1. Konfigurasjon lastes fra .env via python-dotenv / pydantic-settings
    2. Filer leses og tekst+tall trekkes ut (file_reader)
    3. Agentdefinisjon bygges med system prompt + verktøydefinisjonar
    4. Agenten registreres i Azure AI Foundry (create-kommandoen)
    5. Chat-løkken håndterer verktøykall og viser svar til bruker

Bruk:
    python -m nora.agent create        # Opprett / oppdater agenten i Foundry
    python -m nora.agent chat          # Start interaktiv samtale
    python -m nora.agent create chat   # Opprett og start samtale

    Eller via CLI: nora chat / nora create

Forutsetninger:
    - Kopier .env.example til .env og fyll inn PROJECT_ENDPOINT
    - Logg inn med: azd auth login --scope https://ai.azure.com/.default
    - Installer avhengigheter: pip install -e .
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import openai
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient

try:
    # Kjøres som pakke: python -m nora.agent
    from .calculator import (
        cagr,
        convert_currency,
        describe_series,
        linear_regression,
        npv,
        percentage_change,
        safe_eval,
        yoy_growth,
    )
    from .chart_generator import (
        create_area_chart,
        create_bar_chart,
        create_horizontal_bar_chart,
        create_line_chart,
        create_pie_chart,
        create_scatter_plot,
        create_stacked_bar_chart,
    )
    from .config import settings
    from .file_reader import FileContent, read_file, read_folder
    from .web_fetcher import (
        get_ecb_rates,
        get_norges_bank_rates,
        get_ssb_kpi,
        get_world_bank_indicator,
    )
except ImportError:
    # Kjøres som script: python agent.py (fra nora/-mappen)
    from calculator import (  # type: ignore
        cagr,
        convert_currency,
        describe_series,
        linear_regression,
        npv,
        percentage_change,
        safe_eval,
        yoy_growth,
    )
    from chart_generator import (  # type: ignore
        create_area_chart,
        create_bar_chart,
        create_horizontal_bar_chart,
        create_line_chart,
        create_pie_chart,
        create_scatter_plot,
        create_stacked_bar_chart,
    )
    from config import settings  # type: ignore
    from file_reader import FileContent, read_file, read_folder  # type: ignore
    from web_fetcher import (  # type: ignore
        get_ecb_rates,
        get_norges_bank_rates,
        get_ssb_kpi,
        get_world_bank_indicator,
    )

log = logging.getLogger(__name__)

_MAX_RETRIES = 5
_INITIAL_WAIT = 5  # seconds


def _call_with_retry(fn, *args, **kwargs):
    """Call *fn* with exponential backoff on 429 RateLimitError."""
    for attempt in range(_MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except openai.RateLimitError:
            if attempt == _MAX_RETRIES - 1:
                raise
            wait = _INITIAL_WAIT * (2 ** attempt)
            log.warning("429 rate limit – venter %ds før nytt forsøk (%d/%d)",
                        wait, attempt + 1, _MAX_RETRIES)
            time.sleep(wait)


# ── Tool definitions (Foundry / OpenAI function calling schema) ───────────────

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
    # ── Chart tools ───────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "create_bar_chart",
            "description": "Lag et stolpediagram (bar chart) og lagre som PNG-bilde lokalt på brukerens maskin. BRUK DETTE VERKTØYET når brukeren ber om søylediagram, stolpediagram eller bar chart.",
            "parameters": {
                "type": "object",
                "properties": {
                    "labels": {"type": "array", "items": {"type": "string"}, "description": "Kategorier/etiketter for x-aksen"},
                    "values": {"type": "array", "items": {"type": "number"}, "description": "Verdier for hver kategori"},
                    "title": {"type": "string", "description": "Tittel på diagrammet"},
                    "xlabel": {"type": "string", "description": "Etikett for x-aksen"},
                    "ylabel": {"type": "string", "description": "Etikett for y-aksen"},
                    "filename": {"type": "string", "description": "Filnavn (uten eller med .png)"},
                    "color": {"type": "string", "description": "Farge (hex-kode, f.eks. #FF5722)"},
                },
                "required": ["labels", "values"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_line_chart",
            "description": "Lag et linjediagram og lagre som PNG-bilde lokalt. Støtter flere linjer. BRUK DETTE når brukeren ber om linjediagram, trendgraf eller utvikling over tid.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "array", "items": {}, "description": "X-verdier (tall eller tekst/årstall)"},
                    "y": {"type": "array", "items": {"type": "number"}, "description": "Y-verdier for én linje (ikke nødvendig ved bruk av y_series)"},
                    "title": {"type": "string", "description": "Tittel på diagrammet"},
                    "xlabel": {"type": "string", "description": "Etikett for x-aksen"},
                    "ylabel": {"type": "string", "description": "Etikett for y-aksen"},
                    "filename": {"type": "string", "description": "Filnavn"},
                    "series_labels": {"type": "array", "items": {"type": "string"}, "description": "Navn på hver serie (for flere linjer)"},
                    "y_series": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}, "description": "Liste av y-verdier per serie (for flere linjer). Bruk null for manglende verdier."},
                },
                "required": ["x"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_pie_chart",
            "description": "Lag et kakediagram (pie chart) og lagre som PNG-bilde lokalt. BRUK DETTE når brukeren ber om kakediagram eller fordeling.",
            "parameters": {
                "type": "object",
                "properties": {
                    "labels": {"type": "array", "items": {"type": "string"}, "description": "Kategorier"},
                    "values": {"type": "array", "items": {"type": "number"}, "description": "Verdier for hver kategori"},
                    "title": {"type": "string", "description": "Tittel"},
                    "filename": {"type": "string", "description": "Filnavn"},
                    "explode_largest": {"type": "boolean", "description": "Om den største biten skal fremheves"},
                },
                "required": ["labels", "values"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_scatter_plot",
            "description": "Lag et spredningsdiagram (scatter plot) og lagre som PNG-bilde lokalt.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "array", "items": {"type": "number"}, "description": "X-verdier"},
                    "y": {"type": "array", "items": {"type": "number"}, "description": "Y-verdier"},
                    "title": {"type": "string", "description": "Tittel"},
                    "xlabel": {"type": "string", "description": "Etikett for x-aksen"},
                    "ylabel": {"type": "string", "description": "Etikett for y-aksen"},
                    "filename": {"type": "string", "description": "Filnavn"},
                    "add_trendline": {"type": "boolean", "description": "Om en trendlinje skal legges til"},
                },
                "required": ["x", "y"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_horizontal_bar_chart",
            "description": "Lag et horisontalt stolpediagram og lagre som PNG-bilde lokalt. Fargelegger automatisk per år dersom labels er datoer (f.eks. '2019-01').",
            "parameters": {
                "type": "object",
                "properties": {
                    "labels": {"type": "array", "items": {"type": "string"}, "description": "Kategorier"},
                    "values": {"type": "array", "items": {"type": "number"}, "description": "Verdier"},
                    "title": {"type": "string", "description": "Tittel"},
                    "xlabel": {"type": "string", "description": "Etikett for x-aksen"},
                    "ylabel": {"type": "string", "description": "Etikett for y-aksen"},
                    "filename": {"type": "string", "description": "Filnavn"},
                    "color": {"type": "string", "description": "Enkeltfarge (hex-kode). Utelat for auto-farging per år."},
                    "color_by_group": {"type": "boolean", "description": "Sett til true for å fargelegge per år/gruppe basert på labels"},
                },
                "required": ["labels", "values"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_stacked_bar_chart",
            "description": "Lag et stablet stolpediagram og lagre som PNG-bilde lokalt.",
            "parameters": {
                "type": "object",
                "properties": {
                    "labels": {"type": "array", "items": {"type": "string"}, "description": "Kategorier på x-aksen"},
                    "series_data": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}, "description": "Verdier per serie"},
                    "series_labels": {"type": "array", "items": {"type": "string"}, "description": "Navn på hver serie"},
                    "title": {"type": "string", "description": "Tittel"},
                    "xlabel": {"type": "string", "description": "Etikett for x-aksen"},
                    "ylabel": {"type": "string", "description": "Etikett for y-aksen"},
                    "filename": {"type": "string", "description": "Filnavn"},
                },
                "required": ["labels", "series_data", "series_labels"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_area_chart",
            "description": "Lag et arealdiagram og lagre som PNG-bilde lokalt.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "array", "items": {}, "description": "X-verdier (tall eller tekst)"},
                    "y": {"type": "array", "items": {"type": "number"}, "description": "Y-verdier"},
                    "title": {"type": "string", "description": "Tittel"},
                    "xlabel": {"type": "string", "description": "Etikett for x-aksen"},
                    "ylabel": {"type": "string", "description": "Etikett for y-aksen"},
                    "filename": {"type": "string", "description": "Filnavn"},
                },
                "required": ["x", "y"],
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

    # ── Chart tools ───────────────────────────────────────────────────────────
    elif name == "create_bar_chart":
        filepath = create_bar_chart(**args)
        return json.dumps({"status": "ok", "filepath": filepath})

    elif name == "create_line_chart":
        filepath = create_line_chart(**args)
        return json.dumps({"status": "ok", "filepath": filepath})

    elif name == "create_pie_chart":
        filepath = create_pie_chart(**args)
        return json.dumps({"status": "ok", "filepath": filepath})

    elif name == "create_scatter_plot":
        filepath = create_scatter_plot(**args)
        return json.dumps({"status": "ok", "filepath": filepath})

    elif name == "create_horizontal_bar_chart":
        filepath = create_horizontal_bar_chart(**args)
        return json.dumps({"status": "ok", "filepath": filepath})

    elif name == "create_stacked_bar_chart":
        filepath = create_stacked_bar_chart(**args)
        return json.dumps({"status": "ok", "filepath": filepath})

    elif name == "create_area_chart":
        filepath = create_area_chart(**args)
        return json.dumps({"status": "ok", "filepath": filepath})

    else:
        return json.dumps({"error": f"Ukjent verktøy: {name}"})


# ── Foundry client ────────────────────────────────────────────────────────────

def get_client() -> AIProjectClient:
    """
    Opprett Foundry-klient med DefaultAzureCredential.

    DefaultAzureCredential prøver autentiseringsmetoder i rekkefølge:
        1. Miljøvariabler (AZURE_CLIENT_ID osv.)
        2. Managed Identity (i Azure-miljø)
        3. Azure CLI (az login)
        4. Azure Developer CLI (azd auth login --scope https://ai.azure.com/.default)

    Kaster ValueError hvis PROJECT_ENDPOINT ikke er konfigurert.
    """
    if not settings.project_endpoint:
        raise ValueError(
            "PROJECT_ENDPOINT er ikke satt. "
            "Kopier .env.example til .env og fyll inn endepunktet."
        )
    credential = DefaultAzureCredential()
    return AIProjectClient(endpoint=settings.project_endpoint, credential=credential)


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
- Lage grafer, kakediagram, linjediagram og andre visualiseringer som PNG-bilder

VIKTIG OM GRAFVERKTØY:
Du har verktøy som create_bar_chart, create_line_chart, create_pie_chart, create_scatter_plot,
create_horizontal_bar_chart, create_stacked_bar_chart og create_area_chart.
Disse verktøyene kjører LOKALT på brukerens maskin og lagrer PNG-filer direkte til disk.
Når brukeren ber om et diagram eller graf, BRUK ALLTID disse verktøyene – ikke gi brukeren
et Python-skript. Verktøyene fungerer og skriver til riktig mappe automatisk.

Tekniske detaljer om verktøyene:
- create_bar_chart og create_horizontal_bar_chart fargelegger automatisk per år dersom
  labels inneholder datoer (f.eks. '2019-01', '2020-03'). Du trenger IKKE spørre om farger.
- create_line_chart støtter flere serier via y_series + series_labels (da trengs ikke y).
  None-verdier i y_series håndteres automatisk (hopp over manglende data).
- Alle verktøy legger til grid-linjer automatisk for lettere avlesning.

VIKTIG OM BESLUTNINGER:
- Ikke still brukeren unødvendige oppfølgingsspørsmål. Ta EGNE beslutninger basert på hva
  brukeren ba om.
- Hvis brukerens forespørsel er tydelig, LAG DIAGRAMMET UMIDDELBART uten å spørre om
  alternativer eller avklaringer.
- Du kan stille MAKSIMALT ETT oppfølgingsspørsmål dersom noe virkelig er uklart.
  Etter det MÅ du ta en beslutning og lage diagrammet.
- Brukeren vil ha resultater, ikke diskusjon om diagramtyper.

Svar alltid på norsk med mindre brukeren ber om noe annet.
Vis beregningstrinn tydelig. Oppgi kilde når du bruker data fra internett.
Vær presis: si eksplisitt hvilke tall du har funnet i hvilken fil.

Når du lager grafer/diagrammer:
- Velg diagramtype som passer best til dataene (kakediagram for andeler, linjediagram for trender, osv.)
- Gi diagrammet en beskrivende tittel
- Bruk meningsfulle filnavn som beskriver innholdet
- Bekreft til brukeren hvor filen ble lagret
""".strip()


def _to_foundry_tools(tools: list[dict]) -> list[dict]:
    """Konverter fra OpenAI-format til Foundry FunctionTool-format.

    OpenAI:  {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}
    Foundry: {"type": "function", "name": ..., "description": ..., "parameters": ..., "strict": false}
    """
    foundry_tools = []
    for tool in tools:
        fn = tool["function"]
        foundry_tools.append({
            "type": "function",
            "name": fn["name"],
            "description": fn.get("description", ""),
            "parameters": fn.get("parameters", {}),
            "strict": False,
        })
    return foundry_tools


def build_agent_definition(file_context: str) -> dict:
    """Bygg agentdefinisjon med verktøy for registrering i Foundry."""
    instructions = SYSTEM_PROMPT
    if file_context:
        instructions += f"\n\n{file_context}"
    return {
        "kind": "prompt",
        "instructions": instructions,
        "model": settings.model_deployment_name,
        "tools": _to_foundry_tools(TOOLS),
    }


# ── Main agent class ──────────────────────────────────────────────────────────

class Nora:
    """
    NORA agent – last filer, opprett agent i Foundry, chat.

    Bruk:
        agent = Nora()
        agent.load_folder()                   # eller agent.load_file("fil.xlsx")
        agent.create_or_update_agent()        # registrer/oppdater i Foundry
        response = agent.ask("Hva er total omsetning?")
        print(response)
    """

    def __init__(self) -> None:
        self.file_contents: list[FileContent] = []
        self.conversation_history: list[dict] = []
        self._client: Optional[AIProjectClient] = None
        self._openai_client = None
        self._last_response_id: Optional[str] = None

    # ── File loading ──────────────────────────────────────────────────────────

    def load_file(self, path: str | Path) -> None:
        """Last inn én fil i agentens kontekst."""
        fc = read_file(Path(path))
        self.file_contents.append(fc)
        log.info("Lastet: %s", fc.filename)

    def load_folder(self, folder: Optional[str | Path] = None) -> None:
        """Last inn alle støttede filer fra *folder* (standard: settings.data_folder)."""
        folder = Path(folder) if folder else settings.data_folder
        self.file_contents = read_folder(folder)
        log.info("Lastet %d fil(er) fra %s", len(self.file_contents), folder)

    # ── Foundry agent management ──────────────────────────────────────────────

    def create_or_update_agent(self) -> None:
        """Opprett en ny versjon av NORA-agenten i Foundry."""
        client = self._get_client()
        file_context = self._build_context()
        definition = build_agent_definition(file_context)

        log.info("Oppretter / oppdaterer agent '%s' i Foundry ...", settings.agent_name)
        result = client.agents.create_version(settings.agent_name, {"definition": definition})
        log.info(
            "Agent '%s' klar  |  versjon: %s",
            settings.agent_name,
            result.get("version", "ukjent"),
        )
        print(f"\n✅  Agent '{settings.agent_name}' er klar i Foundry.\n")

    # ── Conversation ──────────────────────────────────────────────────────────

    def ask(self, question: str) -> str:
        """Send *question* til NORA og returner svaret."""
        openai_client = self._get_openai_client()

        # Inject file context as developer message on first interaction
        if not self.conversation_history and self.file_contents:
            file_context = self._build_context()
            self.conversation_history.append({
                "type": "message",
                "role": "developer",
                "content": file_context,
            })

        self.conversation_history.append({"type": "message", "role": "user", "content": question})

        # Use previous_response_id for follow-up questions to avoid
        # resending the full file context (saves tokens / avoids 429).
        if self._last_response_id:
            response = _call_with_retry(
                openai_client.responses.create,
                model=settings.model_deployment_name,
                input=[{"type": "message", "role": "user", "content": question}],
                extra_body={
                    "agent_reference": {
                        "type": "agent_reference",
                        "name": settings.agent_name,
                    },
                    "previous_response_id": self._last_response_id,
                },
            )
        else:
            response = _call_with_retry(
                openai_client.responses.create,
                model=settings.model_deployment_name,
                input=self.conversation_history,
                extra_body={
                    "agent_reference": {
                        "type": "agent_reference",
                        "name": settings.agent_name,
                    }
                },
            )

        # ── Verktøy-loop ──────────────────────────────────────────────────────
        while True:
            tool_calls = [
                item for item in (response.output or [])
                if getattr(item, "type", None) == "function_call"
            ]
            if not tool_calls:
                break

            tool_outputs = []
            for tc in tool_calls:
                tool_args = json.loads(tc.arguments or "{}")
                log.info("Verktøykall: %s(%s)", tc.name, tool_args)
                tool_result = _dispatch(tc.name, tool_args)
                tool_outputs.append({
                    "type": "function_call_output",
                    "call_id": tc.call_id,
                    "output": tool_result,
                })

            response = _call_with_retry(
                openai_client.responses.create,
                model=settings.model_deployment_name,
                input=tool_outputs,
                extra_body={
                    "agent_reference": {
                        "type": "agent_reference",
                        "name": settings.agent_name,
                    },
                    "previous_response_id": response.id,
                },
            )

        self._last_response_id = response.id
        answer = response.output_text or ""
        self.conversation_history.append({"type": "message", "role": "assistant", "content": answer})
        return answer

    def reset(self) -> None:
        """Tøm samtalehistorikk (beholder innlastede filer)."""
        self.conversation_history = []
        self._last_response_id = None

    # ── Private helpers ───────────────────────────────────────────────────────

    def _get_client(self) -> AIProjectClient:
        if self._client is None:
            self._client = get_client()
        return self._client

    def _get_openai_client(self):
        if self._openai_client is None:
            self._openai_client = self._get_client().get_openai_client()
        return self._openai_client

    def _build_context(self) -> str:
        if not self.file_contents:
            return ""
        parts = ["Her er det fullstendige innholdet i de innlastede filene:\n"]
        for fc in self.file_contents:
            parts.append(f"=== {fc.filename} ===")
            parts.append(fc.text)
        return "\n\n".join(parts)


# ── Inngangspunkt (python -m nora.agent create chat) ──────────────────────────

def main() -> None:
    """Parse kommandolinjeargumenter og kjør valgt(e) kommando(er)."""
    args = set(sys.argv[1:])

    if not args or args.isdisjoint({"create", "chat"}):
        print(__doc__)
        sys.exit(0)

    agent = Nora()

    with_files = "chat" in args
    if with_files:
        log.info("Laster filer fra %s", settings.data_folder)
        agent.load_folder()

    if "create" in args:
        agent.create_or_update_agent()

    if "chat" in args:
        _interactive_chat(agent)


def _interactive_chat(agent: Nora) -> None:
    """Interaktiv chat-løkke i terminalen."""
    print("\n🔢  NORA – Numerical Operations & Results Assistant")
    if agent.file_contents:
        names = ", ".join(fc.filename for fc in agent.file_contents)
        print(f"    Lastede filer: {names}")
    print("    Skriv spørsmålet ditt, eller 'avslutt' for å avslutte.\n")

    while True:
        try:
            user_input = input("Du: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAvslutter.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"avslutt", "exit", "quit"}:
            print("Samtalen er avsluttet.")
            break

        try:
            answer = agent.ask(user_input)
        except Exception as exc:
            log.error("Feil: %s", exc)
            print(f"\n⚠️  Feil: {exc}\n")
            continue

        print(f"\nNORA: {answer}\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()
