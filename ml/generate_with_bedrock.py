"""Generate varied synthetic support tickets with a Bedrock model (no real customer data).

    uv run python ml/generate_with_bedrock.py [--per-call 12] [--calls 2]

For each category and each writing style the model produces distinct messages. Styles in
TEST_STYLES are held out of training, so test accuracy measures unseen customer voices, not
unseen sentences. Output: ml/data/tickets_llm.csv (text,label,split,source).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from copilot.aws import make_session
from copilot.config import Settings
from copilot.services.llm import ChatLLM

CATEGORIES = {
    "billing": "charges, double charges, refunds, invoices, subscriptions, payment problems",
    "shipping": "late or lost parcels, tracking, delivery address changes, delivery questions",
    "product_defect": "products that are broken, damaged, faulty, missing parts or not working",
    "account_access": "login problems, passwords, lockouts, verification codes, account security",
    "other": "general questions, feedback, store information, partnerships, gift options",
}
TRAIN_STYLES = ["formal and polite", "angry, with strong emotion", "terse, with typos"]
TEST_STYLES = ["long and rambling with background story", "non-native English speaker"]

SYSTEM = (
    "You write realistic but entirely fictional customer-support messages for a classifier "
    "training set. Never include real names, emails or phone numbers. Return only a JSON array "
    "of strings."
)


def parse_list(text: str) -> list[str]:
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        raise ValueError("no JSON array in reply")
    return [str(m).strip() for m in json.loads(match.group(0)) if str(m).strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--per-call", type=int, default=12)
    parser.add_argument("--calls", type=int, default=2)
    parser.add_argument("--out", type=Path, default=Path("ml/data/tickets_llm.csv"))
    args = parser.parse_args()

    settings = Settings()
    session = make_session(settings)
    llm = ChatLLM(session.client("bedrock-runtime"), ["amazon.nova-pro-v1:0"], settings.aws_region)

    rows: list[tuple[str, str, str, str]] = []
    for style, split in [(s, "train") for s in TRAIN_STYLES] + [(s, "test") for s in TEST_STYLES]:
        for label, description in CATEGORIES.items():
            seen: set[str] = set()
            for call in range(args.calls):
                prompt = (
                    f"Write {args.per_call} different customer messages (1-3 sentences each) about "
                    f"this topic: {description}. Writing style: {style}. Vary the situations and "
                    f"vocabulary; this is batch {call + 1}, so avoid repeating earlier batches."
                )
                try:
                    texts = parse_list(llm.complete(SYSTEM, prompt, 1500, 0.9))
                except (ValueError, json.JSONDecodeError, RuntimeError) as exc:
                    print(f"  skipped {label}/{style} batch {call + 1}: {exc}")
                    continue
                for t in texts:
                    if t.lower() not in seen:
                        seen.add(t.lower())
                        rows.append((t, label, split, "llm"))
            print(f"{split:5} {label:15} {style:42} {len(seen)} messages")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["text", "label", "split", "source"])
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
