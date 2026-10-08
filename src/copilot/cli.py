from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from copilot import __version__
from copilot.aws import make_session
from copilot.config import Settings
from copilot.doctor import DoctorReport, Status, run_doctor
from copilot.personas import load_personas
from copilot.pipeline import analyze_case, speak_reply

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
def analyze(
    text: Annotated[str, typer.Option(help="Customer message, in any language.")] = "",
    document: Annotated[Path | None, typer.Option(help="Scanned page or receipt image.")] = None,
    image: Annotated[Path | None, typer.Option(help="Product photo.")] = None,
    audio: Annotated[
        Path | None, typer.Option(help="Voicemail or call recording (wav/mp3).")
    ] = None,
    reply_language: Annotated[
        str | None, typer.Option(help="Defaults to the input language.")
    ] = None,
    persona: Annotated[
        str | None, typer.Option(help="Force a persona instead of routing by category.")
    ] = None,
    speak: Annotated[
        Path | None, typer.Option(help="Write a spoken MP3 reply here (needs tts_backend=polly).")
    ] = None,
) -> None:
    """Turn a support request into a structured case and a drafted reply."""
    settings = Settings()
    session = make_session(settings)
    case = analyze_case(
        session,
        settings,
        text=text,
        document=document.read_bytes() if document else None,
        image=image.read_bytes() if image else None,
        audio=audio.read_bytes() if audio else None,
        audio_format=audio.suffix.lstrip(".").lower() if audio else "wav",
        reply_language=reply_language,
        persona=persona,
    )
    console.print_json(case.model_dump_json())
    if speak:
        speak.write_bytes(speak_reply(session, settings, case))
        console.print(f"Spoken reply written to {speak}")


@app.command()
def personas(
    check: Annotated[
        bool, typer.Option("--check", help="Call each persona's models to see which respond.")
    ] = False,
) -> None:
    """List personas, their routing categories and models."""
    settings = Settings()
    persona_set = load_personas(settings.personas_file or None)
    table = Table(title="Support personas")
    for col in ("Persona", "Handles", "Models (in order)", "Temp"):
        table.add_column(col)
    for p in persona_set.personas.values():
        handles = ", ".join(p.categories) or "escalation rules"
        table.add_row(p.name, handles, ", ".join(p.models), str(p.temperature))
    console.print(table)
    r = persona_set.routing
    console.print(
        f"Escalate to [bold]{r.escalation}[/bold] on priority {list(r.escalate_priorities)} or "
        f"negative sentiment with priority {list(r.escalate_negative_priorities)}; "
        f"default persona: [bold]{r.default}[/bold]."
    )
    if check:
        models = dict.fromkeys(m for p in persona_set.personas.values() for m in p.models)
        probe_settings = settings.model_copy(update={"bedrock_models": ",".join(models)})
        for result in run_doctor(make_session(settings), probe_settings).results:
            if result.service == "bedrock":
                console.print(f"  [{_STYLE[result.status]}]{result.status}[/]  {result.capability}")


@app.command()
def config() -> None:
    """Show the effective configuration (no secrets are stored here)."""
    settings = Settings()
    console.print_json(settings.model_dump_json())


@app.command()
def version() -> None:
    """Print the package version."""
    console.print(__version__)
