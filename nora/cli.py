"""
NORA CLI – command-line interface.

Usage examples:
  nora chat                          # interaktiv chat, laster standardmappen
  nora chat --file rapport.xlsx      # last én spesifikk fil
  nora chat --folder C:\mine\filer   # last en annen mappe
  nora info                          # vis konfigurasjon
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from .agent import Nora
from .config import settings

app = typer.Typer(
    name="nora",
    help="NORA – Numerical Operations & Results Assistant",
    no_args_is_help=True,
)
console = Console()


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )


@app.command()
def chat(
    file: Optional[Path] = typer.Option(None, "--file", "-f", help="Last inn én enkelt fil"),
    folder: Optional[Path] = typer.Option(None, "--folder", help="Last inn alle filer i mappen"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Vis debug-logger"),
) -> None:
    """Start en interaktiv chat med NORA."""
    _setup_logging("DEBUG" if verbose else settings.log_level)

    console.print(
        Panel.fit(
            "[bold cyan]NORA – Numerical Operations & Results Assistant[/bold cyan]\n"
            "Skriv spørsmålet ditt og trykk Enter. Skriv [bold]exit[/bold] for å avslutte.",
            title="👩‍💻 NORA",
        )
    )

    agent = Nora()

    with console.status("Laster filer…"):
        if file:
            agent.load_file(file)
        else:
            target = folder or settings.data_folder
            agent.load_folder(target)

    if not agent.file_contents:
        console.print("[yellow]Ingen filer funnet. Legg filer i datamappen og prøv igjen.[/yellow]")
        console.print(f"Datamappe: [bold]{settings.data_folder}[/bold]")
    else:
        names = ", ".join(fc.filename for fc in agent.file_contents)
        console.print(f"[green]✓ Lastet {len(agent.file_contents)} fil(er):[/green] {names}\n")

    while True:
        try:
            question = Prompt.ask("[bold cyan]Du[/bold cyan]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Avslutter.[/dim]")
            break

        if question.strip().lower() in {"exit", "quit", "avslutt", "bye"}:
            console.print("[dim]Ha det bra![/dim]")
            break

        if not question.strip():
            continue

        with console.status("NORA tenker…"):
            try:
                answer = agent.ask(question)
            except Exception as exc:
                console.print(f"[red]Feil: {exc}[/red]")
                continue

        console.print(Panel(Markdown(answer), title="[bold green]NORA[/bold green]", border_style="green"))


@app.command()
def info() -> None:
    """Vis gjeldende konfigurasjon."""
    _setup_logging(settings.log_level)
    console.print("[bold]NORA konfigurasjon[/bold]")
    console.print(f"  Backend    : {'Azure OpenAI' if settings.use_azure else 'OpenAI'}")
    console.print(f"  Modell     : {settings.azure_openai_deployment if settings.use_azure else settings.openai_model}")
    console.print(f"  Datamappe  : {settings.data_folder}")
    console.print(f"  Finnes     : {'✓' if settings.data_folder.exists() else '✗ (mangler)'}")


if __name__ == "__main__":
    app()
