"""
Streamlit frontend for the RAG Research Assistant.

Key fixes over the previous version:
  * Explicit timeouts on every request (indexing needs a long one).
  * Backend status shown in the sidebar so a cold start is visible, not silent.
  * Upload sends model_choice, and all tuning params go in the form body.
  * Server error details surface instead of a generic failure message.
  * Chat history is written once and rendered from a single place.
"""

import os
import uuid

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

CHAT_TIMEOUT = 120      # LLM calls
UPLOAD_TIMEOUT = 300    # indexing: cold start can add 60s on free tiers
HEALTH_TIMEOUT = 180

MODELS = [
    "openai/gpt-oss-20b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]

MODE_CHAT = "💬 Chat"
MODE_RAG = "📄 RAG"
MODE_AGENT = "🤖 Agent"

st.set_page_config(page_title="RAG Assistant", page_icon="🔬", layout="wide")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

st.session_state.setdefault("session_id", str(uuid.uuid4()))
st.session_state.setdefault("messages", [])
st.session_state.setdefault("documents", [])
st.session_state.setdefault("selected_doc", None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def error_detail(exc: Exception) -> str:
    """Pull the server's `detail` field out of an HTTP error when present."""
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            return response.json().get("detail", response.text[:300])
        except ValueError:
            return response.text[:300] or str(exc)
    if isinstance(exc, requests.Timeout):
        return "The request timed out. The backend may be waking up — try again."
    if isinstance(exc, requests.ConnectionError):
        return f"Could not reach the backend at {API_URL}."
    return str(exc)


@st.cache_data(ttl=20, show_spinner=False)
def backend_is_up() -> bool:
    try:
        resp = requests.get(f"{API_URL}/health", timeout=HEALTH_TIMEOUT)
        return resp.status_code == 200
    except requests.RequestException:
        return False


def post_json(path: str, payload: dict, timeout: int = CHAT_TIMEOUT) -> dict:
    resp = requests.post(f"{API_URL}{path}", json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def render_sources(sources: list[str]) -> None:
    if not sources:
        return
    with st.expander(f"🔍 Sources ({len(sources)})"):
        for i, src in enumerate(sources, 1):
            st.caption(f"**{i}.** {src}")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚙️ Settings")

    if backend_is_up():
        st.success("Backend online", icon="✅")
    else:
        st.warning("Backend unreachable or starting up", icon="⏳")

    model_choice = st.selectbox("Model", MODELS)
    mode = st.radio("Mode", [MODE_CHAT, MODE_RAG, MODE_AGENT])

    if mode == MODE_RAG:
        st.divider()
        uploaded_file = st.file_uploader(
            "Upload a document",
            type=["pdf", "txt", "md"],
            help="Max 10 MB. Files are discarded when the server restarts.",
        )

        with st.expander("Retrieval parameters"):
            chunk_size = st.slider("Chunk size", 100, 2000, 1000, step=100)
            chunk_overlap = st.slider("Chunk overlap", 0, 500, 200, step=50)
            top_k = st.slider("Top K", 1, 10, 4)

        if uploaded_file is not None and st.button("Process document", type="primary"):
            with st.spinner("Indexing… first run can take a minute on a cold server."):
                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type or "application/octet-stream",
                    )
                }
                form = {
                    "session_id": st.session_state.session_id,
                    "model_choice": model_choice,
                    "chunk_size": str(chunk_size),
                    "chunk_overlap": str(chunk_overlap),
                    "top_k": str(top_k),
                }
                try:
                    resp = requests.post(
                        f"{API_URL}/upload",
                        files=files,
                        data=form,
                        timeout=UPLOAD_TIMEOUT,
                    )
                    resp.raise_for_status()
                    result = resp.json()

                    st.success(
                        f"Indexed {result['chunks']} chunks from "
                        f"{result['document']} in {result.get('seconds', '?')}s"
                    )
                    if result["document"] not in st.session_state.documents:
                        st.session_state.documents.append(result["document"])
                    st.session_state.selected_doc = result["document"]
                except requests.RequestException as exc:
                    st.error(f"Upload failed: {error_detail(exc)}")

        if st.session_state.documents:
            st.markdown("**Indexed documents**")
            st.session_state.selected_doc = st.selectbox(
                "Active document",
                st.session_state.documents,
                index=st.session_state.documents.index(st.session_state.selected_doc)
                if st.session_state.selected_doc in st.session_state.documents
                else 0,
                key="doc_selector",
            )

    st.divider()
    if st.button("🗑️ Clear chat"):
        try:
            requests.post(
                f"{API_URL}/clear/{st.session_state.session_id}",
                timeout=HEALTH_TIMEOUT,
            )
        except requests.RequestException:
            pass
        st.session_state.messages = []
        st.session_state.documents = []
        st.session_state.selected_doc = None
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

    st.caption(f"Session `{st.session_state.session_id[:8]}`")


# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------

st.title("🧠 RAG Assistant")
st.caption("Plain chat, document Q&A over your own PDFs, and a tool-using agent.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        render_sources(msg.get("sources", []))


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

user_input = st.chat_input("Ask something…")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        payload = {
            "session_id": st.session_state.session_id,
            "message": user_input,
            "model": model_choice,
        }
        response, sources = "", []

        try:
            if mode == MODE_CHAT:
                with st.spinner("Thinking…"):
                    response = post_json("/chat", payload)["response"]

            elif mode == MODE_RAG:
                if not st.session_state.documents:
                    response = "Please upload and process a document first."
                else:
                    payload["document_name"] = st.session_state.selected_doc
                    with st.spinner("Searching the document…"):
                        data = post_json("/rag-query", payload)
                    response = data["response"]
                    sources = data.get("sources", [])

            else:  # MODE_AGENT
                with st.spinner("Running the agent…"):
                    response = post_json("/agent", payload)["response"]

            st.markdown(response)
            render_sources(sources)

        except requests.RequestException as exc:
            response = f"Something went wrong: {error_detail(exc)}"
            sources = []
            st.error(response)

        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "sources": sources,
        })