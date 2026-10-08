"""Run a batch of varied support messages through the pipeline and show how each was routed.

    uv run python scripts/evaluate_routing.py [fixtures/support_requests.json]

Prints one row per request: persona, why it was chosen, which models answered, and the reply.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from copilot.aws import make_session
from copilot.config import Settings
from copilot.pipeline import analyze_case

DEFAULT_FILE = Path(__file__).resolve().parent.parent / "fixtures" / "support_requests.json"


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_FILE
    requests = json.loads(path.read_text(encoding="utf-8"))
    settings = Settings()
    session = make_session(settings)
    console = Console()

    table = Table(title=f"Routing results ({len(requests)} requests)", show_lines=True)
    for col in ("Request", "Category / priority", "Sentiment", "Persona (reason)", "Reply model"):
        table.add_column(col)
    replies: list[tuple[str, str]] = []
    for item in requests:
        case = analyze_case(session, settings, text=item["text"])
        table.add_row(
            item["id"],
            f"{case.category} / {case.priority}",
            case.sentiment or "-",
            f"{case.persona} ({case.persona_reason})",
            case.model_id,
        )
        replies.append((item["id"], case.reply))
    console.print(table)
    for request_id, reply in replies:
        console.print(f"\n[bold]{request_id}[/bold]\n{reply}")


if __name__ == "__main__":
    main()
