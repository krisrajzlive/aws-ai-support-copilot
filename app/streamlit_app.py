"""Web UI for the support copilot.

    uv run --group ui streamlit run app/streamlit_app.py

Three tabs: analyse a request (text, receipt, photo, audio), chat with the Lex intake bot, and
inspect the personas and routing rules. Uses the same configuration as the CLI (.env, COPILOT_*).
"""

from __future__ import annotations

import streamlit as st
from botocore.exceptions import BotoCoreError, ClientError

from copilot.aws import make_session
from copilot.config import Settings
from copilot.doctor import Status, run_doctor
from copilot.graph import CaseGraph, CaseRun
from copilot.models import SupportCase
from copilot.personas import load_personas
from copilot.services.lex import LexIntake

st.set_page_config(page_title="AWS AI Support Copilot", page_icon="🎧", layout="wide")


@st.cache_resource
def get_settings() -> Settings:
    return Settings()


@st.cache_resource
def get_session():
    return make_session(get_settings())


@st.cache_resource
def get_graph() -> CaseGraph:
    """One workflow per server process so a paused review can be resumed on the next click."""
    return CaseGraph(get_session(), get_settings())


settings = get_settings()
personas = load_personas(settings.personas_file or None)


def show_case(case: SupportCase) -> None:
    """Render one analysed case: routing decision first, then the reply, then the evidence."""
    tiles = [
        ("Category", case.category.replace("_", " ")),
        ("Priority", case.priority),
        ("Sentiment", (case.sentiment or "n/a").title()),
        ("Persona", case.persona),
    ]
    for column, (label, value) in zip(st.columns(4), tiles, strict=True):
        column.caption(label)
        column.markdown(f"#### {value}")

    source = case.category_source or "llm"
    confidence = f" ({case.category_confidence:.0%})" if case.category_confidence else ""
    st.caption(
        f"Routed by **{case.persona_reason}**. Category decided by the **{source}**{confidence}. "
        f"Triage model `{case.models.get('triage', '-')}`, "
        f"reply model `{case.models.get('reply', '-')}`."
    )
    if case.blocked:
        st.error("Blocked by the guardrail: " + case.reply)
        return

    st.subheader("Drafted reply")
    st.info(case.reply)
    if case.reply_language and case.reply_language != "en":
        with st.expander("English version"):
            st.write(case.reply_en)

    st.write("**Summary:** " + case.summary)
    left, right = st.columns(2)
    with left:
        st.write(f"**Detected language:** {case.source_language}")
        if case.pii_types:
            st.write("**PII redacted before any model call:** " + ", ".join(case.pii_types))
        if case.entities:
            st.write("**Entities:** " + ", ".join(case.entities))
    with right:
        if case.document_lines:
            st.write("**Document text (Textract)**")
            st.code("\n".join(case.document_lines), language=None)
        if case.image_labels:
            st.write("**Photo labels (Rekognition):** " + ", ".join(case.image_labels))
    if case.policy_excerpts:
        with st.expander(f"Policy used: {', '.join(case.policy_sources)}"):
            for excerpt in case.policy_excerpts:
                st.markdown(excerpt)
                st.divider()
    footer = []
    if case.human_reviewed:
        footer.append("approved by a human reviewer")
    if case.archive_key:
        footer.append(f"redacted record archived to S3 as `{case.archive_key}`")
    if footer:
        st.caption(" · ".join(footer))
    with st.expander("What the model saw (redacted) and backends used"):
        st.write(case.english_text)
        st.json(case.backends)
    with st.expander("Full JSON"):
        st.json(case.model_dump())


def run_pipeline(text: str, **kwargs) -> CaseRun | None:
    try:
        with st.spinner("Analysing..."):
            return get_graph().start(text=text, **kwargs)
    except (ClientError, BotoCoreError, RuntimeError) as exc:
        st.error(f"AWS call failed: {exc}")
        st.caption("Credentials may have expired. Re-run `aws configure --profile ...` and reload.")
        return None


def render_run(key: str) -> None:
    """Show a finished case, or the approve/edit/reject panel for an escalated one."""
    run: CaseRun | None = st.session_state.get(key)
    if run is None:
        return
    if not run.pending:
        show_case(run.case)
        return
    review = run.review
    st.warning(f"Escalated case ({review['reason']}). Review the drafted reply before it is sent.")
    st.write("**Summary:** " + review["summary"])
    edited = st.text_area("Reply to send", value=review["draft"], key=f"{key}-draft", height=160)
    st.caption("Customer details appear as tokens such as [[NAME_1]] and are restored on approval.")
    approve, reject = st.columns(2)
    if approve.button("Approve and send", type="primary", key=f"{key}-approve"):
        st.session_state[key] = get_graph().resume(run.thread_id, approved=True, reply=edited)
        st.rerun()
    if reject.button("Reject draft", key=f"{key}-reject"):
        st.session_state[key] = get_graph().resume(run.thread_id, approved=False)
        st.rerun()


# ---- sidebar ---------------------------------------------------------------------------------
with st.sidebar:
    st.title("Support Copilot")
    st.caption(f"Profile `{settings.aws_profile or 'default chain'}` in `{settings.aws_region}`")
    persona_choice = st.selectbox(
        "Persona", ["Route automatically", *personas.personas], help="Force a persona for testing."
    )
    reply_language = st.text_input(
        "Reply language code", "", placeholder="same as customer, e.g. fr"
    )
    settings.require_escalation_review = st.checkbox(
        "Human review for escalations",
        value=settings.require_escalation_review,
        help="Pause urgent or angry cases so a person approves or edits the reply first.",
    )
    st.divider()
    if st.button("Check AWS services"):
        report = run_doctor(get_session(), settings)
        for r in report.results:
            icon = {Status.AVAILABLE: "✅", Status.DENIED: "⛔", Status.ERROR: "⚠️"}[r.status]
            st.write(f"{icon} {r.service} · {r.capability}")

tab_analyse, tab_lex, tab_personas = st.tabs(["Analyse a request", "Lex intake chat", "Personas"])

# ---- tab 1: analyse --------------------------------------------------------------------------
with tab_analyse:
    text = st.text_area(
        "Customer message (any language)",
        height=120,
        placeholder="Hola, mi pedido 48213 llegó roto. Soy Maria Lopez, maria@example.com.",
    )
    up1, up2, up3 = st.columns(3)
    document = up1.file_uploader("Receipt or document", type=["png", "jpg", "jpeg"])
    image = up2.file_uploader("Product photo", type=["png", "jpg", "jpeg"])
    audio = up3.file_uploader("Voice message", type=["wav", "mp3"])

    if st.button("Analyse", type="primary", disabled=not (text.strip() or audio)):
        st.session_state.analyse = run_pipeline(
            text,
            document=document.getvalue() if document else None,
            image=image.getvalue() if image else None,
            audio=audio.getvalue() if audio else None,
            audio_format=audio.name.rsplit(".", 1)[-1].lower() if audio else "wav",
            reply_language=reply_language.strip() or None,
            persona=None if persona_choice == "Route automatically" else persona_choice,
        )
    render_run("analyse")

# ---- tab 2: Lex chat -------------------------------------------------------------------------
with tab_lex:
    if not settings.lex_bot_id:
        st.warning(
            "Set COPILOT_LEX_BOT_ID in .env (run `scripts/create_lex_bot.py`) to use the bot."
        )
    else:
        if "lex" not in st.session_state:
            st.session_state.lex = LexIntake(
                get_session().client("lexv2-runtime"),
                settings.lex_bot_id,
                settings.lex_bot_alias_id,
                settings.lex_locale,
            )
            st.session_state.chat = []
            st.session_state.lex_case = None
        lex: LexIntake = st.session_state.lex

        if st.button("Start over"):
            for key in ("lex", "chat", "lex_case"):
                st.session_state.pop(key, None)
            st.rerun()

        for role, message in st.session_state.chat:
            st.chat_message(role).write(message)

        done = st.session_state.lex_case is not None
        if prompt := st.chat_input("Tell the bot what happened", disabled=done):
            st.session_state.chat.append(("user", prompt))
            turn = lex.send(prompt)
            for message in turn.messages:
                st.session_state.chat.append(("assistant", message))
            if turn.fulfilled and turn.intent == "ReportProblem":
                st.session_state.lex_case = run_pipeline(lex.case_text(turn))
            st.rerun()

        if st.session_state.get("lex_case"):
            st.divider()
            st.subheader("Case opened from the conversation")
            render_run("lex_case")

# ---- tab 3: personas -------------------------------------------------------------------------
with tab_personas:
    st.write("Each request is triaged by a small model, then handled by one of these personas.")
    st.dataframe(
        [
            {
                "Persona": p.name,
                "Handles": ", ".join(p.categories) or "escalation rules",
                "Models (in order)": " → ".join(p.models),
                "Temperature": p.temperature,
                "Description": p.description,
            }
            for p in personas.personas.values()
        ],
        hide_index=True,
        use_container_width=True,
    )
    r = personas.routing
    st.caption(
        f"Escalates to **{r.escalation}** on priority {list(r.escalate_priorities)} or negative "
        f"sentiment with priority {list(r.escalate_negative_priorities)}. Default: **{r.default}**."
    )
    with st.expander("System prompts"):
        for p in personas.personas.values():
            st.markdown(f"**{p.name}**")
            st.code(p.prompt, language=None)
