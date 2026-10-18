"""A local Streamlit version of the Still study assistant preview."""

from __future__ import annotations

import uuid
from hashlib import sha256

import streamlit as st

from agents import rag, supervisor
from server import generate_answer, load_config
from study_assistant.services.study_plans import generate_plan_answer


st.set_page_config(
    page_title="Still — your study space",
    page_icon="✳",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700&display=swap');
      :root { --ink:#303831; --muted:#899087; --green:#648066; --paper:#f8f9f7; }
      html, body, [class*="css"] { font-family:'DM Sans',Arial,sans-serif; }
      .stApp { background:var(--paper); color:var(--ink); }
      [data-testid="stSidebar"] { background:#fff; border-right:1px solid #e9ece8; }
      [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:#596458; }
      h1, h2, h3 { font-family:'Manrope','DM Sans',sans-serif; color:var(--ink); }
      .brand { font:600 22px 'Manrope',sans-serif; letter-spacing:-1px; margin:0 0 1.5rem; }
      .brand-mark { display:inline-grid; place-items:center; width:29px; height:29px; margin-right:8px;
        border-radius:9px 9px 9px 3px; background:#5b795f; color:#fff; font:italic 18px Georgia,serif; }
      .welcome { text-align:center; padding:2.6rem 0 1rem; }
      .welcome-mark { display:grid; place-items:center; width:42px; height:42px; margin:0 auto 18px;
        border:1px solid #e4e9e1; background:#fff; border-radius:14px 14px 14px 4px; color:#6e8a69; font-size:20px; }
      .overline { color:#8b9489; font-size:11px; letter-spacing:2px; font-weight:700; }
      .welcome h1 { font-size:clamp(2.1rem,5vw,3rem); line-height:1.15; letter-spacing:-2px; margin:.4rem 0; }
      .welcome h1 em { color:var(--green); font:italic 400 1em Georgia,serif; }
      .welcome-copy { color:var(--muted); line-height:1.8; margin:.8rem 0 1.8rem; }
      .stButton > button { border-color:#e3e8e1; border-radius:8px; color:#4a5149; background:#fff; }
      .stButton > button:hover { border-color:#b7c7b2; color:#49664d; }
      [data-testid="stChatMessage"] { border:1px solid #ebeee8; border-radius:10px; background:#fff; }
      [data-testid="stChatInput"] { border-color:#dfe6dd; }
      .privacy-note { color:#929a90; font-size:12px; text-align:center; padding-top:1rem; }
      @media(max-width:700px) { .welcome { padding-top:1rem; } }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_state() -> None:
    st.session_state.setdefault("sessions", [])
    st.session_state.setdefault("active_session_id", None)
    st.session_state.setdefault("pending_prompt", None)
    st.session_state.setdefault("rag_index", None)
    st.session_state.setdefault("rag_file_id", None)
    st.session_state.setdefault("rag_error_file_id", None)
    st.session_state.setdefault("rag_error", None)
    st.session_state.setdefault("rag_consent", False)
    st.session_state.setdefault("rag_consent_file_id", None)


def active_session() -> dict | None:
    session_id = st.session_state.active_session_id
    return next(
        (item for item in st.session_state.sessions if item["id"] == session_id),
        None,
    )


def make_reply(
    messages: list[dict],
    agent_mode: str,
    quiz_format: str,
    planning_details: dict,
    rag_index: dict | None = None,
    rag_consent: bool = False,
) -> dict:
    """Route a request and optionally ground it in retrieved course PDF passages."""
    question = next(
        (item["text"] for item in reversed(messages) if item.get("role") == "user"),
        "",
    )
    passages = []
    document = None
    try:
        agent_id = supervisor.route_request(question, agent_mode, messages)
        use_document = rag_index is not None and (
            agent_id != "planner" or supervisor.study_planner_agent.requests_document(question)
        )
        if use_document:
            if not rag_consent:
                return {
                    "agent": agent_id,
                    "text": "Check the course-material consent box before I send selected PDF passages and your question to the answer provider (Groq for study plans; Tavily for other agents).",
                }
            passages = rag.retrieve_passages(rag_index, question)
            if not passages and agent_id != "planner":
                return {
                    "agent": agent_id,
                    "text": f"I couldn't find a relevant passage in {rag_index['file_name']}. Try a question using terms from the PDF.",
                }
            document = {
                "name": rag_index["file_name"],
                "text": rag.format_passages(passages) if passages else "",
            }

        def answer_with_tavily(prompt: str) -> str:
            result = generate_answer(
                "tavily",
                [{"role": "user", "text": prompt}],
                load_config(),
                document=document,
                document_source_only=document is not None,
            )
            return result["answer"]

        def answer_with_groq(prompt: str) -> str:
            return generate_plan_answer(
                prompt,
                question,
                load_config(),
                has_document=document is not None,
            )

        response = supervisor.dispatch(
            question,
            answer_with_tavily,
            mode=agent_mode,
            history=messages,
            quiz_format=quiz_format,
            planning_details=planning_details,
            planner_answer=answer_with_groq,
            planning_document=document if supervisor.study_planner_agent.requests_document(question) else None,
        )
        if passages:
            labels = rag.source_labels(passages)
            response["text"] += "\n\n**Sources:** " + "; ".join(labels)
        return response
    except ValueError as error:
        return {"text": str(error)}
    except Exception as error:
        print(f"Streamlit answer error: {type(error).__name__}")
        return {"text": "I couldn't get an answer. Check the Streamlit terminal for details."}


def queue_prompt(prompt: str) -> None:
    st.session_state.pending_prompt = prompt


def submit_prompt(
    prompt: str,
    agent_mode: str,
    quiz_format: str,
    planning_details: dict,
) -> None:
    prompt = prompt.strip()
    if not prompt:
        return
    session = active_session()
    if session is None:
        session = {"id": str(uuid.uuid4()), "title": prompt[:42], "messages": []}
        st.session_state.sessions.insert(0, session)
        st.session_state.active_session_id = session["id"]
    session["messages"].append({"role": "user", "text": prompt})
    with st.spinner("Choosing the right study agent and preparing your answer..."):
        reply = make_reply(session["messages"], agent_mode, quiz_format, planning_details, rag_index, rag_consent)
    session["messages"].append({"role": "assistant", **reply})


init_state()

with st.sidebar:
    st.markdown('<div class="brand"><span class="brand-mark">s</span>still</div>', unsafe_allow_html=True)
    if st.button("＋  New study session", use_container_width=True):
        st.session_state.active_session_id = None
        st.rerun()
    st.caption("YOUR SPACE")
    st.markdown("🟢  Study assistant  \n📚  Explore a topic")
    st.caption("COURSE MATERIAL")
    uploaded_pdf = st.file_uploader(
        "Drag and drop a course PDF here",
        type=["pdf"],
        help="Limit 200 MB per file. PDF text is indexed in this session. Scanned PDFs need OCR before upload.",
        key="course_pdf_upload",
    )
    rag_index = None
    rag_consent = False
    if uploaded_pdf is None:
        st.session_state.rag_index = None
        st.session_state.rag_file_id = None
        st.session_state.rag_error = None
        st.session_state.rag_error_file_id = None
        st.session_state.rag_consent = False
        st.session_state.rag_consent_file_id = None
    else:
        pdf_bytes = uploaded_pdf.getvalue()
        file_id = sha256(uploaded_pdf.name.encode("utf-8") + pdf_bytes).hexdigest()
        if st.session_state.rag_file_id != file_id and st.session_state.rag_error_file_id != file_id:
            st.session_state.rag_index = None
            st.session_state.rag_file_id = file_id
            st.session_state.rag_error = None
            st.session_state.rag_consent = False
            st.session_state.rag_consent_file_id = file_id
            try:
                with st.spinner("Extracting PDF text and creating local embeddings..."):
                    index = rag.build_index(pdf_bytes, uploaded_pdf.name)
                index["file_id"] = file_id
                st.session_state.rag_index = index
            except (ValueError, RuntimeError) as error:
                st.session_state.rag_error = str(error)
                st.session_state.rag_error_file_id = file_id
            except Exception as error:
                print(f"PDF indexing error: {type(error).__name__}")
                st.session_state.rag_error = "I couldn't index this PDF. Check that it is readable and try again."
                st.session_state.rag_error_file_id = file_id
        if st.session_state.rag_error_file_id == file_id:
            st.error(st.session_state.rag_error)
        elif st.session_state.rag_index is not None:
            rag_index = st.session_state.rag_index
            st.caption(f"Indexed {len(rag_index['chunks'])} passages from {rag_index['file_name']}.")
            if st.session_state.rag_consent_file_id != file_id:
                st.session_state.rag_consent = False
                st.session_state.rag_consent_file_id = file_id
            rag_consent = st.checkbox(
                "Allow selected passages and my question to be sent to the answer provider (Groq for study plans; Tavily for other agents)",
                key="rag_consent",
                help="The PDF is indexed locally and kept in this Streamlit session. Only retrieved passages are sent to the relevant answer provider after you enable this.",
            )
            st.caption("The first upload downloads the local multilingual embedding model.")

    with st.expander("STUDY AGENT", expanded=True):
        agent_mode = st.selectbox(
            "Choose an agent",
            list(supervisor.MODE_TO_AGENT),
            index=0,
            help="Automatic routing picks a tutor, quiz, or planner from your request.",
        )
        quiz_format = "Mixed"
        planning_details = {}
        if agent_mode == "Quiz Agent":
            quiz_format = st.selectbox(
                "Quiz format",
                sorted(supervisor.quiz_agent.QUESTION_TYPES),
            )
        elif agent_mode == "Study Planner Agent":
            st.caption("Optional planning details")
            planning_details = {
                "subjects": st.text_input("Subjects", max_chars=120),
                "goal": st.text_input("Study goal", max_chars=120),
                "minutes_per_day": st.number_input(
                    "Available minutes per day", min_value=5, max_value=720, value=60, step=5
                ),
                "deadline": st.text_input("Deadline (optional)", max_chars=120),
            }
    with st.expander("RECENT SESSIONS", expanded=True):
        if not st.session_state.sessions:
            st.caption("Your recent sessions will appear here.")
        for saved in st.session_state.sessions:
            if st.button(saved["title"], key=f"session-{saved['id']}", use_container_width=True):
                st.session_state.active_session_id = saved["id"]
                st.rerun()
    st.divider()
    st.caption("◇  A calm place to learn.\n\nYour questions stay in this browser session.")

session = active_session()
if session is None or not session["messages"]:
    st.markdown(
        '<div class="welcome"><div class="welcome-mark">✳</div>'
        '<div class="overline">YOUR STUDY SPACE</div>'
        '<h1>What would you like<br>to <em>understand?</em></h1>'
        '<div class="welcome-copy">Bring a question, a topic, or a half-formed thought.<br>We’ll work through it one step at a time.</div></div>',
        unsafe_allow_html=True,
    )
    col1, col2, col3 = st.columns(3)
    starters = [
        ("Understand an idea", "Explain the seasons in simple steps", col1),
        ("Practice a skill", "Give me a practice question about fractions", col2),
        ("Make a small plan", "Help me make a short study plan", col3),
    ]
    for label, prompt, column in starters:
        with column:
            if st.button(label, key=f"starter-{label}", use_container_width=True):
                queue_prompt(prompt)
                st.rerun()
else:
    for message in session["messages"]:
        with st.chat_message(message["role"], avatar="✳" if message["role"] == "assistant" else None):
            st.markdown(message["text"])
            if message.get("agent"):
                st.caption(f"{message['agent'].title()} Agent")
            if message.get("hint"):
                with st.expander(message["hint_label"]):
                    st.write(message["hint"])

st.markdown('<div class="privacy-note">Still is a study companion. Check important facts with your class materials.</div>', unsafe_allow_html=True)
typed_prompt = st.chat_input("Ask anything you’re learning…", max_chars=2000)
prompt_to_send = st.session_state.pending_prompt or typed_prompt
st.session_state.pending_prompt = None
if prompt_to_send:
    submit_prompt(prompt_to_send, agent_mode, quiz_format, planning_details)
    st.rerun()
