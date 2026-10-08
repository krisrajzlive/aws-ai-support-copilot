from __future__ import annotations

import json
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from copilot import __version__
from copilot.aws import make_session
from copilot.config import Settings
from copilot.doctor import DoctorReport, Status, run_doctor

app = typer.Typer(help="AWS AI Support Copilot", no_args_is_help=True, add_completion=False)
console = Console()

_STYLE = {Status.AVAILABLE: "green", Status.DENIED: "red", Status.ERROR: "yellow"}


def _render(report: DoctorReport, settings: Settings) -> None:
    table = Table(title=f"Service availability ({settings.aws_region})")
    for col in ("Service", "Capability", "Status", "Detail"):
        table.add_column(col)
    for r in report.results:
        table.add_row(r.service, r.capability, f"[{_STYLE[r.status]}]{r.status}[/]", r.detail)
    console.print(table)
    if report.selected_model:
        console.print(f"Selected Bedrock model: [bold]{report.selected_model}[/bold]")
    else:
        console.print("[yellow]No Bedrock model responded; pipeline will use stub backends.[/]")


@app.command()
def doctor(
    as_json: Annotated[bool, typer.Option("--json", help="Emit machine-readable JSON.")] = False,
) -> None:
    """Probe each AWS AI service with the configured profile and report what is usable."""
    settings = Settings()
    report = run_doctor(make_session(settings), settings)
    if as_json:
        console.print_json(json.dumps(report.to_dict()))
    else:
        _render(report, settings)


@app.command()
def config() -> None:
    """Show the effective configuration (no secrets are stored here)."""
    settings = Settings()
    console.print_json(settings.model_dump_json())


@app.command()
def version() -> None:
    """Print the package version."""
    console.print(__version__)
