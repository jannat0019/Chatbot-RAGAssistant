import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from pathlib import Path

from src.rag_pipeline import build_rag_chain
from src.agent import build_agent
from src.memory import ConversationMemory
from src.chat import plain_chat
from src.llm import CreateLLM

load_dotenv()

# -----------------------
# PAGE CONFIG
# -----------------------
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🔬",
    layout="wide"
)

# -----------------------
# SESSION STATE
# -----------------------
if "memory" not in st.session_state:
    st.session_state.memory = ConversationMemory()

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "agent" not in st.session_state:
    st.session_state.agent = None

if "mode" not in st.session_state:
    st.session_state.mode = "💬 Chat"

# -----------------------
# SIDEBAR
# -----------------------
with st.sidebar:
    st.title("⚙️ Settings")

    model_choice = st.selectbox(
        "Model",
        ["llama-3.1-8b-instant", "llama3-70b-8192", "mixtral-8x7b-32768"]
    )

    mode = st.radio(
        "Mode",
        ["💬 Chat", "📄 RAG"]
    )

    st.session_state.mode = mode

    

    llm=CreateLLM(model_choice)

    if mode == "📄 RAG":
        uploaded_file = st.file_uploader("Upload file", type=["pdf", "txt"])

        if uploaded_file:
            if st.button("Process Document"):
                suffix=Path(uploaded_file.name).suffix

                tmp_file=tempfile.NamedTemporaryFile(delete=False,suffix=suffix)

                tmp_file.write(uploaded_file.getbuffer())
                tmp_file.close()
                tmp_path=tmp_file.name

                chain, chunks = build_rag_chain(tmp_path,llm)

                st.session_state.rag_chain = chain
                st.success(f"Indexed {chunks} chunks")

    
    if st.button("🗑️ Clear Chat"):
        st.session_state.memory.clear()
        st.rerun()

# -----------------------
# CHAT DISPLAY
# -----------------------
st.title("🧠 RAG Assistant")


    # ---------------- RAG --------
for msg in st.session_state.memory.get_messages():
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -----------------------
# INPUT
# -----------------------
user_input = st.chat_input("Ask something...")

if user_input:
    st.session_state.memory.add_message("user", user_input)
    print(user_input)
    with st.spinner("Thinking..."):
        try:
            mode = st.session_state.mode

            # -------- CHAT --------
            if mode == "💬 Chat":
                response = plain_chat(
                    user_input,
                    st.session_state.memory.get_langchain_history(),
                    llm
                    
                )
            elif mode=="📄 RAG":
                if not st.session_state.rag_chain:
                    response="Please upload the document first"
                else:
                    result=st.session_state.rag_chain.invoke({
                        "question":user_input,
                        "chat_history": st.session_state.memory.get_langchain_history()
                    })
                   
                    response=result["answer"]



            
          

            st.session_state.memory.add_message("assistant", response)

        except Exception as e:
            st.session_state.memory.add_message("assistant", str(e))

    st.rerun()