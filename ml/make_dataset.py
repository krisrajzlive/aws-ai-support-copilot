"""Generate a synthetic, labelled support-ticket dataset (no real customer data).

    uv run python ml/make_dataset.py [--rows 800] [--out ml/data/tickets.csv]

Messages are assembled from templates, product words and noise. The last templates of each
category are reserved for the test split, so test accuracy measures unseen phrasing.
Categories match the pipeline's triage categories.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

PRODUCTS = ["blender", "kettle", "headphones", "laptop charger", "desk lamp", "backpack", "monitor"]

TEMPLATES: dict[str, list[str]] = {
    "billing": [
        "I was charged twice for my {product} and need a refund",
        "there is an unknown charge on my card from your store",
        "my invoice shows the wrong amount for order {n}",
        "please cancel my subscription and refund last month's payment",
        "the discount code was not applied and I paid the full price",
        "I need a receipt for the payment I made for the {product}",
        "you overcharged me on order {n}, please correct the bill",
    ],
    "shipping": [
        "my parcel for order {n} has not arrived yet",
        "the tracking for my {product} has not updated in a week",
        "delivery is late and nobody told me why",
        "the courier left my package at the wrong address",
        "can you change the delivery address for order {n}",
        "my order {n} says delivered but I never received it",
        "how long will shipping to my city take",
    ],
    "product_defect": [
        "the {product} I received is broken",
        "my {product} stopped working after two days",
        "the {product} arrived damaged with a cracked case",
        "the {product} is defective and makes a strange noise",
        "there is a part missing from my {product}",
        "my {product} overheats and shuts down by itself",
        "the {product} does not turn on at all",
    ],
    "account_access": [
        "I cannot log in to my account",
        "I did not request this password reset email",
        "my account is locked after too many attempts",
        "I forgot my password and the reset link is expired",
        "someone else may have accessed my account",
        "the verification code never arrives on my phone",
        "I need to change the email address on my account",
    ],
    "other": [
        "do you ship to other countries",
        "what are your opening hours",
        "can I speak to someone about a bulk order",
        "do you offer a gift wrapping service",
        "I would like to leave feedback about your website",
        "is the {product} available in other colours",
        "how do I become a reseller",
    ],
}

OPENERS = ["", "", "Hi, ", "Hello, ", "Hey team, ", "Good morning. ", "Urgent: ", "Please help. "]
CLOSERS = [
    "",
    "",
    " Thanks.",
    " Please advise.",
    " This is frustrating.",
    " Regards, Sam",
    " ASAP!",
]


HELD_OUT_TEMPLATES = 2  # the last N templates of every category only appear in the test split


def make_rows(count: int, seed: int) -> list[tuple[str, str, str]]:
    """Rows of (text, label, split). Test rows use templates never seen in training."""
    rng = random.Random(seed)
    labels = list(TEMPLATES)
    rows = []
    for i in range(count):
        label = labels[i % len(labels)]
        templates = TEMPLATES[label]
        index = rng.randrange(len(templates))
        split = "test" if index >= len(templates) - HELD_OUT_TEMPLATES else "train"
        body = templates[index].format(product=rng.choice(PRODUCTS), n=rng.randint(10000, 99999))
        text = f"{rng.choice(OPENERS)}{body}.{rng.choice(CLOSERS)}".strip()
        rows.append((text, label, split))
    rng.shuffle(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", type=int, default=800)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=Path("ml/data/tickets.csv"))
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["text", "label", "split"])
        writer.writerows(make_rows(args.rows, args.seed))
    print(f"Wrote {args.rows} rows to {args.out}")


if __name__ == "__main__":
    main()
