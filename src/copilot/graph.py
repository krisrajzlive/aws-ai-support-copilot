"""The support-case workflow as a LangGraph state machine.

    transcribe -> attachments -> language -> guard_input
    guard_input -> insights -> triage -> classify -> route -> retrieve -> draft
    guard_input -> blocked                       (guardrail intervened on the input)
    draft -> review -> localize                  (escalations, when human review is required)
    draft -> localize
    review -> blocked                            (reviewer rejected the draft)
    localize -> archive,  blocked -> archive

State is kept as plain dicts and bytes so it can be checkpointed. The checkpointer also lets the
`review` node pause for a human decision and resume later with the same thread id.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from typing import Any, TypedDict

from botocore.exceptions import BotoCoreError, ClientError
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from copilot.aws import CLIENT_CONFIG
from copilot.config import Settings
from copilot.knowledge import KnowledgeBase, load_or_build, split_queries
from copilot.models import SupportCase
from copilot.personas import PersonaSet, load_personas
from copilot.services import comprehend, storage, vision
from copilot.services.classifier import TicketClassifier
from copilot.services.guardrails import GuardrailService
from copilot.services.llm import ChatLLM
from copilot.services.transcribe import AwsTranscriber, Transcriber
from copilot.services.translate import Translator

CATEGORIES = ["billing", "shipping", "product_defect", "account_access", "other"]
PRIORITIES = ["low", "medium", "high", "urgent"]

TRIAGE_PROMPT = (
    "You are a customer-support triage assistant. Using only the case facts provided, return a "
    "single JSON object with keys: summary (max 2 sentences), category (one of "
    f"{CATEGORIES}) and priority (one of {PRIORITIES}). Priority guide: low = a general question; "
    "medium = a normal problem the customer wants fixed (late parcel, wrong charge, faulty item); "
    "high = repeated failure, major financial loss or a vulnerable customer; urgent = safety, "
    "legal threat or possible account compromise. Output JSON only."
)

BASE_REPLY_RULES = (
    "Write only the reply text addressed to the customer, in English, as plain text with no "
    "markdown and no preamble. Never state delivery times, prices, availability or company "
    "policies unless they appear in {sources}, and never describe website pages, buttons, "
    "forms or internal processes; when the customer needs one of these, say a team member will "
    "confirm it. Never invent order numbers, amounts or dates. Do not use bracketed template "
    "placeholders such as [Your Name]; sign off as 'Customer Support Team'."
)


def reply_rules(placeholders: dict[str, str], has_policy: bool) -> str:
    """Only advertise tokens and policy sources that exist, so the model cannot copy examples."""
    sources = "the case facts or the policy excerpts" if has_policy else "the case facts"
    rules = BASE_REPLY_RULES.format(sources=sources)
    names = [t for t in placeholders if t.startswith("[[NAME_")]
    others = [t for t in placeholders if t not in names]
    greeting = (
        f"Greet the customer with the token {names[0]} exactly as written."
        if names
        else "The customer's name is unknown: greet them with a plain 'Hello,'."
    )
    rules = f"{rules} {greeting} Never address the customer by a country, product or order number."
    if others:
        rules += (
            f" These tokens stand for other customer details and may be mentioned only when "
            f"relevant, never as a greeting: {', '.join(others)}."
        )
    return f"{rules} Use no other bracketed tokens."


def case_prompt(case: SupportCase, include_entities: bool = True) -> str:
    parts = [f"Customer message (PII redacted): {case.english_text}"]
    if case.sentiment:
        parts.append(f"Detected sentiment: {case.sentiment}")
    if include_entities and case.entities:
        parts.append(f"Entities: {', '.join(case.entities)}")
    if case.document_lines:
        parts.append("Attached document text:\n" + "\n".join(case.document_lines[:40]))
    if case.image_labels:
        parts.append(f"Attached photo shows: {', '.join(case.image_labels)}")
    return "\n".join(parts)


class CaseState(TypedDict, total=False):
    case: dict[str, Any]
    text: str
    document: bytes | None
    image: bytes | None
    audio: bytes | None
    audio_format: str
    reply_language: str | None
    persona_override: str | None
    placeholders: dict[str, str]
    snippets: list[dict[str, Any]]


@dataclass(frozen=True)
class CaseRun:
    """Outcome of starting or resuming a case: finished, or paused awaiting human review."""

    thread_id: str
    case: SupportCase | None = None
    review: dict[str, Any] | None = None

    @property
    def pending(self) -> bool:
        return self.review is not None


def _case(state: CaseState) -> SupportCase:
    return SupportCase(**state["case"])


def _save(case: SupportCase, **extra: Any) -> dict[str, Any]:
    return {"case": case.model_dump(), **extra}


class CaseGraph:
    """Builds the workflow around one AWS session; reuse it so paused reviews can be resumed."""

    def __init__(
        self,
        session: Any,
        settings: Settings,
        personas: PersonaSet | None = None,
        knowledge: KnowledgeBase | None = None,
    ):
        self.settings = settings
        self.personas = personas or load_personas(settings.personas_file or None)
        self._session = session
        self._clients: dict[str, Any] = {}
        self._client_lock = threading.Lock()
        region = settings.aws_region
        runtime = self._client("bedrock-runtime")
        self._runtime = runtime
        self.translate_llm = ChatLLM(runtime, settings.bedrock_model_list, region)
        self.triage_llm = ChatLLM(runtime, settings.triage_model_list, region)
        self.translator = Translator(
            self._client("translate"), self.translate_llm, settings.translate_backend
        )
        self.guardrail = (
            GuardrailService(
                self._client("bedrock"), settings.guardrail_id, settings.guardrail_version
            )
            if settings.guardrail_id
            else None
        )
        self._comprehend = self._client("comprehend")
        self._knowledge = knowledge
        self._classifier: TicketClassifier | None = None
        self._checkpointer = InMemorySaver()
        self.app = self._build()

    def _client(self, name: str) -> Any:
        """One cached client per service: reuse keeps TLS connections warm (a fresh S3 client
        cost ~0.7 s per call). The lock matters because boto3 sessions are not thread-safe."""
        with self._client_lock:
            if name not in self._clients:
                self._clients[name] = self._session.client(name, config=CLIENT_CONFIG)
            return self._clients[name]

    # ---- nodes -------------------------------------------------------------------------------

    def transcribe(self, state: CaseState) -> dict[str, Any]:
        audio = state.get("audio")
        if not audio:
            return {}
        s = self.settings
        transcriber = (
            AwsTranscriber(self._client("transcribe"), self._client("s3"), s.transcribe_bucket)
            if s.transcribe_backend == "aws"
            else Transcriber(self._runtime)
        )
        text = f"{state.get('text', '')} {transcriber.transcribe(audio, state['audio_format'])}"
        case = _case(state)
        case.backends["transcribe"] = transcriber.backend
        return _save(case, text=text.strip())

    def attachments(self, state: CaseState) -> dict[str, Any]:
        case = _case(state)
        if state.get("document"):
            case.document_lines = vision.extract_document_lines(
                self._client("textract"), state["document"]
            )
        if state.get("image"):
            case.image_labels = vision.detect_image_labels(
                self._client("rekognition"), state["image"]
            )
        return _save(case)

    def language(self, state: CaseState) -> dict[str, Any]:
        case = _case(state)
        text = state.get("text", "")
        case.original_text = text
        case.source_language = (
            comprehend.detect_language(self._comprehend, text) if text.strip() else "en"
        )
        case.english_text = self.translator.translate(text, case.source_language, "en")
        return _save(case)

    def guard_input(self, state: CaseState) -> dict[str, Any]:
        if not self.guardrail:
            return {}
        case = _case(state)
        checked = self.guardrail.apply(case.english_text, "INPUT")
        case.english_text = checked.text
        if checked.blocked:
            case.blocked, case.reply, case.reply_en = True, checked.text, checked.text
        return _save(case)

    def insights(self, state: CaseState) -> dict[str, Any]:
        case = _case(state)
        found = comprehend.analyze_english(self._comprehend, case.english_text)
        case.sentiment, case.entities, case.pii_types = (
            found.sentiment,
            found.entities,
            found.pii_types,
        )
        case.english_text = found.redacted_text
        return _save(case, placeholders=found.placeholders)

    def triage(self, state: CaseState) -> dict[str, Any]:
        case = _case(state)
        result = self.triage_llm.complete_json(TRIAGE_PROMPT, case_prompt(case))
        case.summary = str(result.get("summary", ""))
        case.category = str(result.get("category", "other"))
        case.category_source = "llm"
        case.priority = str(result.get("priority", "medium"))
        case.models["triage"] = self.triage_llm.model_id
        return _save(case)

    def classify(self, state: CaseState) -> dict[str, Any]:
        """A trained classifier overrides the LLM's category only when it is confident."""
        path = self.settings.classifier_path
        if not path:
            return {}
        if self._classifier is None:
            self._classifier = TicketClassifier.load(path)
        case = _case(state)
        prediction = self._classifier.predict(case.english_text)
        case.category_confidence = round(prediction.confidence, 3)
        if prediction.confidence >= self.settings.classifier_min_confidence:
            case.category, case.category_source = prediction.label, "classifier"
        return _save(case)

    def route(self, state: CaseState) -> dict[str, Any]:
        case = _case(state)
        chosen, reason = self.personas.select(
            case.category, case.priority, case.sentiment, state.get("persona_override")
        )
        case.persona, case.persona_reason = chosen.name, reason
        return _save(case)

    def _kb(self) -> KnowledgeBase | None:
        if self._knowledge is None and self.settings.kb_enabled:
            self._knowledge = load_or_build(self.settings, s3=self._client("s3"))
        return self._knowledge

    def retrieve(self, state: CaseState) -> dict[str, Any]:
        """Ground the reply in company policy text instead of the model's imagination.

        A message with several issues is split into sub-queries that are searched in parallel
        inside the knowledge base and merged.
        """
        case = _case(state)
        try:
            kb = self._kb()
            snippets = kb.search_many(split_queries(case.english_text)) if kb else []
        except Exception as exc:  # the knowledge base must never take a case down
            case.backends["knowledge"] = f"unavailable ({type(exc).__name__})"
            return _save(case, snippets=[])
        if kb:
            case.backends["knowledge"] = "llamaindex"
        case.policy_sources = sorted({x.source for x in snippets})
        case.policy_excerpts = [x.text for x in snippets]
        return _save(case, snippets=[{"source": x.source, "text": x.text} for x in snippets])

    def draft(self, state: CaseState) -> dict[str, Any]:
        case = _case(state)
        persona = self.personas.personas[case.persona]
        snippets = state.get("snippets", [])
        user = f"{case_prompt(case, include_entities=False)}\n"
        if snippets:
            excerpts = "\n\n".join(f"[{s['source']}]\n{s['text']}" for s in snippets)
            user += f"Policy excerpts (the only policy facts you may state):\n{excerpts}\n"
        user += f"Triage summary: {case.summary}\nPriority: {case.priority}"
        reply_llm = ChatLLM(self._runtime, persona.models, self.settings.aws_region)
        reply = reply_llm.complete(
            f"{persona.prompt}\n\n{reply_rules(state.get('placeholders', {}), bool(snippets))}",
            user,
            max_tokens=persona.max_tokens,
            temperature=persona.temperature,
        )
        if self.guardrail:
            reply = self.guardrail.apply(reply, "OUTPUT").text
        case.reply_en = case.reply_redacted = reply
        case.model_id = reply_llm.model_id
        case.models["reply"] = reply_llm.model_id
        return _save(case)

    def review(self, state: CaseState) -> dict[str, Any]:
        """Pause an escalated case until a person approves, edits or rejects the draft."""
        case = _case(state)
        decision = interrupt(
            {
                "case_id": case.case_id,
                "persona": case.persona,
                "reason": case.persona_reason,
                "summary": case.summary,
                "draft": case.reply_redacted,
            }
        )
        if not decision.get("approved", False):
            case.blocked = True
            case.reply = case.reply_en = case.reply_redacted = ""
            case.backends["review"] = "rejected by reviewer"
        else:
            edited = decision.get("reply")
            if edited:
                case.reply_en = case.reply_redacted = edited
            case.human_reviewed = True
        return _save(case)

    def localize(self, state: CaseState) -> dict[str, Any]:
        case = _case(state)
        placeholders = state.get("placeholders", {})
        target = state.get("reply_language") or case.source_language
        translated = self.translator.translate(case.reply_en, "en", target)
        case.reply = comprehend.fill_placeholders(translated, placeholders)
        case.reply_en = comprehend.fill_placeholders(case.reply_en, placeholders)
        case.reply_language = target
        case.backends = {
            **case.backends,
            "translate": self.translator.backend,
            "llm": "bedrock",
            "guardrail": "bedrock-guardrails" if self.guardrail else "comprehend-pii-redaction",
        }
        return _save(case)

    def finish_blocked(self, state: CaseState) -> dict[str, Any]:
        case = _case(state)
        case.backends = {
            **case.backends,
            "translate": self.translator.backend,
            "guardrail": "bedrock-guardrails" if self.guardrail else "none",
        }
        return _save(case)

    def archive(self, state: CaseState) -> dict[str, Any]:
        """Store the redacted record in S3; an archive failure never fails the case."""
        s = self.settings
        if not (s.s3_bucket and s.archive_cases):
            return {}
        case = _case(state)
        try:
            case.archive_key = storage.archive_case(self._client("s3"), s.s3_bucket, case)
            case.backends["archive"] = "s3"
        except (ClientError, BotoCoreError) as exc:
            case.backends["archive"] = f"failed ({type(exc).__name__})"
        return _save(case)

    # ---- edges -------------------------------------------------------------------------------

    @staticmethod
    def after_guard(state: CaseState) -> str:
        return "blocked" if state["case"].get("blocked") else "insights"

    def after_draft(self, state: CaseState) -> str:
        needs_review = (
            self.settings.require_escalation_review
            and state["case"].get("persona") == self.personas.routing.escalation
        )
        return "review" if needs_review else "localize"

    @staticmethod
    def after_review(state: CaseState) -> str:
        return "blocked" if state["case"].get("blocked") else "localize"

    def _build(self):
        g = StateGraph(CaseState)
        for name in (
            "transcribe",
            "attachments",
            "language",
            "guard_input",
            "insights",
            "triage",
            "classify",
            "route",
            "retrieve",
            "draft",
            "review",
            "localize",
            "archive",
        ):
            g.add_node(name, getattr(self, name))
        g.add_node("blocked", self.finish_blocked)

        g.add_edge(START, "transcribe")
        g.add_edge("transcribe", "attachments")
        g.add_edge("attachments", "language")
        g.add_edge("language", "guard_input")
        g.add_conditional_edges(
            "guard_input", self.after_guard, {"blocked": "blocked", "insights": "insights"}
        )
        for a, b in (
            ("insights", "triage"),
            ("triage", "classify"),
            ("classify", "route"),
            ("route", "retrieve"),
            ("retrieve", "draft"),
        ):
            g.add_edge(a, b)
        g.add_conditional_edges(
            "draft", self.after_draft, {"review": "review", "localize": "localize"}
        )
        g.add_conditional_edges(
            "review", self.after_review, {"blocked": "blocked", "localize": "localize"}
        )
        g.add_edge("localize", "archive")
        g.add_edge("blocked", "archive")
        g.add_edge("archive", END)
        return g.compile(checkpointer=self._checkpointer)

    # ---- running -----------------------------------------------------------------------------

    def _outcome(self, thread_id: str, result: dict[str, Any]) -> CaseRun:
        if "__interrupt__" in result:
            return CaseRun(thread_id, review=result["__interrupt__"][0].value)
        return CaseRun(thread_id, case=SupportCase(**result["case"]))

    def start(
        self,
        *,
        text: str = "",
        document: bytes | None = None,
        image: bytes | None = None,
        audio: bytes | None = None,
        audio_format: str = "wav",
        reply_language: str | None = None,
        persona: str | None = None,
    ) -> CaseRun:
        thread_id = uuid.uuid4().hex
        self.translator.backend = "none"
        case = SupportCase(case_id=thread_id[:12], original_text=text)
        result = self.app.invoke(
            {
                "case": case.model_dump(),
                "text": text,
                "document": document,
                "image": image,
                "audio": audio,
                "audio_format": audio_format,
                "reply_language": reply_language,
                "persona_override": persona,
            },
            {"configurable": {"thread_id": thread_id}},
        )
        return self._outcome(thread_id, result)

    def resume(self, thread_id: str, *, approved: bool, reply: str | None = None) -> CaseRun:
        result = self.app.invoke(
            Command(resume={"approved": approved, "reply": reply}),
            {"configurable": {"thread_id": thread_id}},
        )
        return self._outcome(thread_id, result)

    def mermaid(self) -> str:
        return self.app.get_graph().draw_mermaid()
