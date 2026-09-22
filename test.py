import os
import uuid

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="RAG Assistant", page_icon="🔬", layout="wide")

# -----------------------
# SESSION STATE
# -----------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "documents" not in st.session_state:
    st.session_state.documents = []

# -----------------------
# SIDEBAR
# -----------------------
with st.sidebar:
    st.title("⚙️ Settings")

    model_choice = st.selectbox(
        "Model",
        ["openai/gpt-oss-20b", "llama3-70b-8192", "mixtral-8x7b-32768"]
    )

    mode = st.radio("Mode", ["💬 Chat", "📄 RAG", "🤖 Agent"])

    if mode == "📄 RAG":
        uploaded_file = st.file_uploader("Upload file", type=["pdf", "txt"])

        with st.expander("Retrieval Parameters"):
            chunk_size = st.slider("Chunk Size", 100, 2000, 1000)
            chunk_overlap = st.slider("Chunk Overlap", 0, 500, 200)
            top_k = st.slider("Top K", 1, 10, 4)

        if uploaded_file:
            if st.button("Process Document"):
                with st.spinner("Indexing..."):
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                    params = {
                        "session_id": st.session_state.session_id,
                        "chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap,
                        "top_k": top_k,
                    }
                    try:
                        resp = requests.post(f"{API_URL}/upload", files=files, data=params)
                        resp.raise_for_status()
                        result = resp.json()
                        st.success(f"Indexed {result['chunks']} chunks from {result['document']}")
                        if result["document"] not in st.session_state.documents:
                            st.session_state.documents.append(result["document"])
                    except Exception as e:
                        st.error(f"Upload failed: {e}")

        if st.session_state.documents:
            st.markdown("**Indexed Documents**")
            selected_doc = st.selectbox(
                "Active Document",
                st.session_state.documents,
                key="doc_selector"
            )
            st.session_state.selected_doc = selected_doc

    if st.button("🗑️ Clear Chat"):
        try:
            requests.post(f"{API_URL}/clear/{st.session_state.session_id}")
        except Exception:
            pass
        st.session_state.messages = []
        st.session_state.documents = []
        st.rerun()

# -----------------------
# CHAT DISPLAY
# -----------------------
st.title("🧠 RAG Assistant")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("🔍 Sources"):
                for src in msg["sources"]:
                    st.caption(src)

# -----------------------
# INPUT
# -----------------------
user_input = st.chat_input("Ask something...")

if user_input:
    # Show user immediately
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            payload = {
                "session_id": st.session_state.session_id,
                "message": user_input,
                "model": model_choice,
            }

            try:
                if mode == "💬 Chat":
                    resp = requests.post(f"{API_URL}/chat", json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                    response = data["response"]
                    sources = []

                elif mode == "📄 RAG":
                    if not st.session_state.documents:
                        response = "Please upload a document first."
                        sources = []
                    else:
                        payload["document_name"] = st.session_state.get("selected_doc")
                        resp = requests.post(f"{API_URL}/rag-query", json=payload)
                        resp.raise_for_status()
                        data = resp.json()
                        response = data["response"]
                        sources = data.get("sources", [])

                else:  # 🤖 Agent
                    resp = requests.post(f"{API_URL}/agent", json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                    response = data["response"]
                    sources = []

                st.markdown(response)
                if sources:
                    with st.expander("🔍 Sources"):
                        for src in sources:
                            st.caption(src)

                # Persist for future reruns
                st.session_state.messages.append({
                    "role": "user",
                    "content": user_input,
                })
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "sources": sources,
                })

            except Exception as e:
                error_msg = "Something went wrong processing your request."
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "user", "content": user_input
                })
                st.session_state.messages.append({
                    "role": "assistant", "content": error_msg
                })