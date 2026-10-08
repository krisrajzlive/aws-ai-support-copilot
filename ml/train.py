"""Train the support-ticket classifier and export it as JSON.

Runs unchanged in three places:

    uv run --group ml python ml/train.py                   # locally
    python ml/train.py                                      # in a SageMaker notebook instance
    # as a SageMaker script-mode training job (SM_CHANNEL_TRAIN / SM_MODEL_DIR are honoured)

Outputs `model.json` (weights for copilot.services.classifier) and `metrics.json`.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from copilot.services.classifier import TicketClassifier


def load_rows(directory: Path) -> list[dict[str, str]]:
    """Every tickets*.csv in the directory; rows without a source are template-generated."""
    rows: list[dict[str, str]] = []
    for path in sorted(directory.glob("tickets*.csv")):
        with path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                row.setdefault("split", "train")
                rows.append({**row, "source": row.get("source") or "template"})
    if not rows:
        raise SystemExit(f"no tickets*.csv found in {directory}")
    return rows


def handwritten_accuracy(model: TicketClassifier, path: Path) -> dict[str, float] | None:
    """Score on the hand-written English fixtures, which the model never saw in any form."""
    if not path.exists():
        return None
    items = [
        i
        for i in json.loads(path.read_text(encoding="utf-8"))
        if "expected_category" in i and i.get("lang", "en") == "en"
    ]
    if not items:
        return None
    hits = sum(model.predict(i["text"]).label == i["expected_category"] for i in items)
    return {"handwritten_accuracy": hits / len(items), "handwritten_rows": len(items)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train", default=os.environ.get("SM_CHANNEL_TRAIN", "ml/data"))
    parser.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "ml/model"))
    parser.add_argument("--c", type=float, default=10.0, help="inverse regularisation strength")
    parser.add_argument("--fixtures", default="fixtures/support_requests.json")
    args = parser.parse_args()

    rows = load_rows(Path(args.train))
    train_rows = [r for r in rows if r["split"] == "train"]
    test_rows = [r for r in rows if r["split"] == "test"]
    x_train, y_train = [r["text"] for r in train_rows], [r["label"] for r in train_rows]
    x_test, y_test = [r["text"] for r in test_rows], [r["label"] for r in test_rows]

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    model = LogisticRegression(C=args.c, max_iter=2000)
    model.fit(vectorizer.fit_transform(x_train), y_train)
    sk_pred = model.predict(vectorizer.transform(x_test))

    out = Path(args.model_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "model.json").write_text(
        json.dumps(
            {
                "classes": model.classes_.tolist(),
                "vocabulary": {t: int(i) for t, i in vectorizer.vocabulary_.items()},
                "idf": vectorizer.idf_.tolist(),
                "coef": model.coef_.tolist(),
                "intercept": model.intercept_.tolist(),
                "ngram_max": 2,
            }
        ),
        encoding="utf-8",
    )

    # The exported pure-Python model must agree with scikit-learn on every held-out message.
    exported = TicketClassifier.load(out / "model.json")
    mismatches = sum(exported.predict(t).label != p for t, p in zip(x_test, sk_pred, strict=True))
    report = classification_report(y_test, sk_pred, output_dict=True, zero_division=0)
    metrics = {
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "train_rows": len(x_train),
        "test_rows": len(x_test),
        "export_mismatches": mismatches,
    }
    for source in sorted({r["source"] for r in test_rows}):
        idx = [i for i, r in enumerate(test_rows) if r["source"] == source]
        hits = sum(sk_pred[i] == y_test[i] for i in idx)
        metrics[f"test_accuracy_{source}"] = hits / len(idx)
        metrics[f"test_rows_{source}"] = len(idx)
    # How trustworthy is the confidence? Accuracy and coverage if low-confidence cases fall back.
    scored = [exported.predict(x) for x in x_test]
    for threshold in (0.4, 0.5, 0.6, 0.7):
        kept = [(p, y) for p, y in zip(scored, y_test, strict=True) if p.confidence >= threshold]
        metrics[f"coverage_at_{threshold}"] = len(kept) / len(y_test)
        metrics[f"accuracy_at_{threshold}"] = (
            sum(p.label == y for p, y in kept) / len(kept) if kept else 0.0
        )
    hand = handwritten_accuracy(exported, Path(args.fixtures))
    if hand:
        metrics.update(hand)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(classification_report(y_test, sk_pred, zero_division=0))
    print(json.dumps(metrics, indent=2))
    if mismatches:
        raise SystemExit("exported model disagrees with scikit-learn")


if __name__ == "__main__":
    main()
