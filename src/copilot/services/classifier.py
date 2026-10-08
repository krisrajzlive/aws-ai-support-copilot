"""Ticket classifier inference with no ML dependencies.

`ml/train.py` fits TF-IDF + logistic regression (scikit-learn, locally or in a SageMaker notebook)
and exports the weights to JSON. This module reproduces the vectoriser and the linear model in
plain Python so the pipeline and Lambda package stay small.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

_TOKEN = re.compile(r"(?u)\b\w\w+\b")  # scikit-learn's default token pattern


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float


class TicketClassifier:
    def __init__(
        self,
        classes: list[str],
        vocabulary: dict[str, int],
        idf: list[float],
        coef: list[list[float]],
        intercept: list[float],
        ngram_max: int = 2,
    ):
        self.classes = classes
        self._vocab = vocabulary
        self._idf = idf
        self._coef = coef
        self._intercept = intercept
        self._ngram_max = ngram_max

    @classmethod
    def load(cls, path: str | Path) -> TicketClassifier:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            data["classes"],
            data["vocabulary"],
            data["idf"],
            data["coef"],
            data["intercept"],
            data.get("ngram_max", 2),
        )

    def _features(self, text: str) -> dict[int, float]:
        tokens = _TOKEN.findall(text.lower())
        terms = list(tokens)
        for n in range(2, self._ngram_max + 1):
            terms += [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
        counts: dict[int, float] = {}
        for term in terms:
            idx = self._vocab.get(term)
            if idx is not None:
                counts[idx] = counts.get(idx, 0.0) + 1.0
        weighted = {i: c * self._idf[i] for i, c in counts.items()}
        norm = math.sqrt(sum(v * v for v in weighted.values()))
        return {i: v / norm for i, v in weighted.items()} if norm else {}

    def predict(self, text: str) -> Prediction:
        feats = self._features(text)
        logits = [
            sum(w[i] * v for i, v in feats.items()) + b
            for w, b in zip(self._coef, self._intercept, strict=True)
        ]
        top = max(logits)
        exps = [math.exp(x - top) for x in logits]
        total = sum(exps)
        best = max(range(len(exps)), key=exps.__getitem__)
        return Prediction(self.classes[best], exps[best] / total)
