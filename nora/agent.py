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
from pathlib import Path
from typing import Optional

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
    from config import settings  # type: ignore
    from file_reader import FileContent, read_file, read_folder  # type: ignore
    from web_fetcher import (  # type: ignore
        get_ecb_rates,
        get_norges_bank_rates,
        get_ssb_kpi,
        get_world_bank_indicator,
    )

log = logging.getLogger(__name__)

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

Svar alltid på norsk med mindre brukeren ber om noe annet.
Vis beregningstrinn tydelig. Oppgi kilde når du bruker data fra internett.
Vær presis: si eksplisitt hvilke tall du har funnet i hvilken fil.
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

        # Inject file context as system message on first interaction
        if not self.conversation_history and self.file_contents:
            file_context = self._build_context()
            self.conversation_history.append({"role": "system", "content": file_context})

        self.conversation_history.append({"role": "user", "content": question})

        response = openai_client.responses.create(
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

            response = openai_client.responses.create(
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

        answer = response.output_text or ""
        self.conversation_history.append({"role": "assistant", "content": answer})
        return answer

    def reset(self) -> None:
        """Tøm samtalehistorikk (beholder innlastede filer)."""
        self.conversation_history = []

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
        parts = ["Her er innholdet i de innlastede filene:\n"]
        for fc in self.file_contents:
            parts.append(f"=== {fc.filename} ===")
            parts.append(fc.text[:8000])
            if len(fc.text) > 8000:
                parts.append(f"[... {len(fc.text) - 8000} tegn utelatt ...]")
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
