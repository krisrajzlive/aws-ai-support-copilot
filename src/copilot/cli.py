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
from copilot.graph import CaseGraph
from copilot.knowledge import (
    KnowledgeBase,
    bedrock_embeddings,
    load_or_build,
    sync_knowledge_base,
)
from copilot.personas import load_personas
from copilot.pipeline import analyze_case, speak_reply
from copilot.services.lex import LexIntake

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
def intake(
    say: Annotated[
        list[str] | None,
        typer.Option("--say", help="Scripted customer message; repeat for each turn."),
    ] = None,
) -> None:
    """Chat with the Lex intake bot, then run the collected case through the pipeline."""
    settings = Settings()
    if not settings.lex_bot_id:
        raise typer.BadParameter("Set COPILOT_LEX_BOT_ID (run scripts/create_lex_bot.py first).")
    session = make_session(settings)
    lex = LexIntake(
        session.client("lexv2-runtime"),
        settings.lex_bot_id,
        settings.lex_bot_alias_id,
        settings.lex_locale,
    )
    scripted = iter(say or [])
    first = next(scripted) if say else typer.prompt("You")
    if say:
        console.print(f"[green]You:[/green] {first}")
    turn = lex.send(first)
    while True:
        for message in turn.messages:
            console.print(f"[cyan]Bot:[/cyan] {message}")
        if turn.fulfilled or turn.action == "Close":
            break
        reply = next(scripted, None) if say else typer.prompt("You")
        if reply is None:
            console.print("[yellow]Scripted messages ran out before the bot finished.[/]")
            return
        if say:
            console.print(f"[green]You:[/green] {reply}")
        turn = lex.send(reply)
    if not turn.fulfilled or turn.intent != "ReportProblem":
        return
    console.print()
    console.print("[bold]Opening a case...[/bold]")
    case = analyze_case(session, settings, text=lex.case_text(turn))
    console.print_json(case.model_dump_json())


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


kb_app = typer.Typer(help="Policy knowledge base (LlamaIndex).", no_args_is_help=True)
app.add_typer(kb_app, name="kb")


@kb_app.command("build")
def kb_build() -> None:
    """Embed kb/policies with Bedrock, save the index locally and sync it to S3 when configured."""
    from copilot.services import storage

    settings = Settings()
    session = make_session(settings)
    kb = KnowledgeBase.build(settings.kb_docs_dir, bedrock_embeddings(settings))
    kb.save(settings.kb_index_dir)
    console.print(f"Index built from {settings.kb_docs_dir} and saved to {settings.kb_index_dir}")
    if settings.s3_bucket:
        count = storage.upload_dir(
            session.client("s3"), settings.s3_bucket, "kb-index/", settings.kb_index_dir
        )
        console.print(f"Synced {count} files to s3://{settings.s3_bucket}/kb-index/")


@kb_app.command("sync")
def kb_sync(
    from_s3: Annotated[
        bool, typer.Option("--from-s3", help="Treat the S3 bucket as the source of truth.")
    ] = False,
) -> None:
    """Update the index incrementally: add new documents, re-embed edited ones, drop removed."""
    settings = Settings()
    s3 = make_session(settings).client("s3")
    report = sync_knowledge_base(settings, s3=s3, from_s3=from_s3)
    console.print(
        f"inserted {report.inserted}, updated {report.updated}, deleted {report.deleted}, "
        f"unchanged {report.unchanged}"
    )
    if not report.changed:
        console.print("Index was already up to date.")


@kb_app.command("search")
def kb_search(query: Annotated[str, typer.Argument(help="Question to look up.")]) -> None:
    """Show the policy excerpts the pipeline would give the model for this query."""
    settings = Settings()
    session = make_session(settings)
    kb = load_or_build(settings, s3=session.client("s3"))
    for snippet in kb.search(query):
        console.print(f"[bold]{snippet.source}[/bold]  score {snippet.score}")
        console.print(snippet.text)
        console.print()


@app.command()
def graph() -> None:
    """Print the support-case workflow as a Mermaid diagram."""
    settings = Settings()
    console.print(CaseGraph(make_session(settings), settings).mermaid(), markup=False)


@app.command()
def config() -> None:
    """Show the effective configuration (no secrets are stored here)."""
    settings = Settings()
    console.print_json(settings.model_dump_json())


@app.command()
def version() -> None:
    """Print the package version."""
    console.print(__version__)
