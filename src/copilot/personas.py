"""Support personas and the routing rules that choose between them."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Persona:
    name: str
    description: str
    categories: tuple[str, ...]
    models: tuple[str, ...]
    prompt: str
    temperature: float = 0.3
    max_tokens: int = 500


@dataclass(frozen=True)
class Routing:
    default: str = "general"
    escalation: str = "escalation"
    escalate_priorities: tuple[str, ...] = ("urgent",)
    escalate_negative_priorities: tuple[str, ...] = ("high",)


@dataclass(frozen=True)
class PersonaSet:
    personas: dict[str, Persona]
    routing: Routing = field(default_factory=Routing)

    def select(
        self,
        category: str,
        priority: str,
        sentiment: str | None,
        override: str | None = None,
    ) -> tuple[Persona, str]:
        """Pick a persona and say why, so routing decisions are explainable."""
        if override:
            if override not in self.personas:
                raise ValueError(
                    f"unknown persona '{override}'; choose from {sorted(self.personas)}"
                )
            return self.personas[override], "override"
        r = self.routing
        if priority in r.escalate_priorities:
            return self.personas[r.escalation], f"escalation: priority {priority}"
        if sentiment == "NEGATIVE" and priority in r.escalate_negative_priorities:
            return self.personas[
                r.escalation
            ], f"escalation: negative sentiment, {priority} priority"
        for persona in self.personas.values():
            if category in persona.categories:
                return persona, f"category: {category}"
        return self.personas[r.default], "default"


def _parse(data: dict[str, Any]) -> PersonaSet:
    routing_data = data.get("routing", {})
    routing = Routing(
        default=routing_data.get("default", "general"),
        escalation=routing_data.get("escalation", "escalation"),
        escalate_priorities=tuple(routing_data.get("escalate_priorities", ["urgent"])),
        escalate_negative_priorities=tuple(
            routing_data.get("escalate_negative_priorities", ["high"])
        ),
    )
    personas: dict[str, Persona] = {}
    for name, raw in data.get("personas", {}).items():
        models = tuple(raw.get("models", ()))
        if not models or not raw.get("prompt"):
            raise ValueError(f"persona '{name}' needs a prompt and at least one model")
        personas[name] = Persona(
            name=name,
            description=raw.get("description", ""),
            categories=tuple(raw.get("categories", ())),
            models=models,
            prompt=raw["prompt"].strip(),
            temperature=float(raw.get("temperature", 0.3)),
            max_tokens=int(raw.get("max_tokens", 500)),
        )
    for needed in (routing.default, routing.escalation):
        if needed not in personas:
            raise ValueError(f"routing refers to missing persona '{needed}'")
    return PersonaSet(personas, routing)


def load_personas(path: str | Path | None = None) -> PersonaSet:
    """Load personas from a TOML file, or the built-in defaults when no path is given."""
    if path:
        text = Path(path).read_text(encoding="utf-8")
    else:
        text = resources.files("copilot").joinpath("default_personas.toml").read_text("utf-8")
    return _parse(tomllib.loads(text))
