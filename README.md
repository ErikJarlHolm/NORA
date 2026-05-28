# NORA – Numerical Operations & Results Assistant

> **NORA** er en AI-agent som leser dokumenter (Excel, CSV, PDF, Word), forstår tallmaterialet i kontekst, utfører komplekse beregninger og kan hente oppdaterte data fra pålitelige internettkilder.

---

## Funksjoner

| Kategori | Detaljer |
|---|---|
| **Fillesing** | Excel (`.xlsx`, `.xls`), CSV, PDF, Word (`.docx`), tekst |
| **Statistikk** | Beskrivende statistikk, korrelasjonsmatrise, outlier-deteksjon |
| **Finans** | NPV, IRR, CAGR, tilbakebetalingstid |
| **Regresjon** | Enkel lineær regresjon (y = a + bx) med R² |
| **Valuta** | Konvertering via live-kurser fra **Norges Bank** og **ECB** |
| **Statistikk-API** | **SSB** (KPI, m.m.), **Verdensbanken**, **Eurostat** |
| **Beregningsmotor** | Sikker eval av matematiske uttrykk |

---

## Arkitektur

```
nora/
├── agent.py        ← Kjerneobjektet Nora – koordinerer alt
├── cli.py          ← Kommandolinjegrensesnitt (typer + rich)
├── config.py       ← Innstillinger via .env
├── file_reader.py  ← Leser filer til FileContent-objekter
├── calculator.py   ← Beregningsfunksjoner (eksponert som LLM-verktøy)
└── web_fetcher.py  ← Pålitelige nettkilder: Norges Bank, SSB, ECB, WB
```

### Dataflyt

```
Fil(er) ──► file_reader ──► FileContent
                                │
                                ▼
                         agent.Nora (LLM + tool-calling)
                           │           │
                     calculator    web_fetcher
                           │           │
                    Beregningsresultater  Live-data (kurs, statistikk)
                                │
                                ▼
                          Svar til bruker
```

---

## Kom i gang

### 1. Installer avhengigheter

```bash
pip install -e ".[dev]"
```

### 2. Konfigurer

Kopier `.env.example` til `.env` og fyll inn API-nøkler:

```bash
copy .env.example .env
# Rediger .env med din Azure OpenAI- eller OpenAI-nøkkel
```

### 3. Legg filer i datamappen

Standard datamappe er:
```
C:\Users\erikholm\OneDrive - Atea\Documents\Kunder\Atea AI Norge\Agent tallknusing
```
Legg Excel-, CSV-, PDF- eller Word-filer her.

### 4. Start NORA

```bash
nora chat
```

Alternativer:
```bash
nora chat --file rapport.xlsx          # last én fil
nora chat --folder C:\mine\filer       # bruk annen mappe
nora chat --verbose                    # vis debug-logger
nora info                              # vis konfigurasjon
```

---

## Eksempelspørsmål

- *"Hva er total omsetning per kvartal?"*
- *"Beregn CAGR fra 2020 til 2024 for linje 'Revenue'."*
- *"Hvor mye er 150 000 USD i NOK i dag?"*
- *"Finn outliere i kolonnen 'kostnad'."*
- *"Hva er korrelasjonen mellom salg og margin?"*
- *"Beregn NPV med 8 % rente og disse kontantstrømmene: -500 000, 150 000, 200 000, 250 000."*
- *"Hent siste KPI fra SSB og juster tallene for inflasjon."*

---

## Pålitelige nettkilder

NORA bruker kun offisielle, tillitsvekkende datakilder:

| Kilde | Data |
|---|---|
| [Norges Bank](https://data.norges-bank.no/api/) | Valutakurser NOK |
| [ECB](https://www.ecb.europa.eu/stats/eurofxref/) | EUR-referansekurser |
| [SSB](https://data.ssb.no/api/) | Norsk statistikk (KPI, m.m.) |
| [Verdensbanken](https://api.worldbank.org/) | Globale makroindikatorer |
| [Eurostat](https://ec.europa.eu/eurostat/) | EU-statistikk |

---

## Kjør tester

```bash
pytest tests/ -v
```

---

## Lisens

Intern Atea AI Norge – ikke for distribusjon.
