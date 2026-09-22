"""
FastAPI backend for the RAG Research Assistant.

Key fixes over the previous version:
  * Embedding model is warmed up once at startup, not per-upload.
  * Blocking indexing work runs in a threadpool so it never stalls the event loop.
  * Sessions have a TTL and a hard cap so vector stores don't leak.
  * Upload size is capped and internal errors are no longer leaked to clients.
"""

import logging
import os
import sys
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Allow imports from project root
sys.path.append(str(Path(__file__).parent.parent))

from src.agent import build_agent
from src.chat import plain_chat
from src.llm import CreateLLM
from src.memory import ConversationMemory
from src.rag_pipeline import build_rag_chain, warm_up

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", 10 * 1024 * 1024))  # 10 MB
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", 3600))        # 1 hour
MAX_SESSIONS = int(os.getenv("MAX_SESSIONS", 50))
MAX_DOCS_PER_SESSION = int(os.getenv("MAX_DOCS_PER_SESSION", 3))
ALLOWED_SUFFIXES = {".pdf", ".txt", ".md"}
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "openai/gpt-oss-20b")

ALLOWED_ORIGINS = [
    o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()
]

# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the embedding model before the first request arrives."""
    logger.info("Warming up embedding model...")
    started = time.perf_counter()
    try:
        await run_in_threadpool(warm_up)
        logger.info("Embeddings ready in %.1fs", time.perf_counter() - started)
    except Exception:
        # Don't kill the container - /health will still report degraded state.
        logger.exception("Embedding warm-up failed; first upload will be slow")
    yield
    sessions.clear()


app = FastAPI(title="RAG Assistant API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=128)
    message: str = Field(..., min_length=1, max_length=8000)
    model: str = DEFAULT_MODEL


class RAGQueryRequest(ChatRequest):
    document_name: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    sources: list[str] = []


# ---------------------------------------------------------------------------
# Session store
# ---------------------------------------------------------------------------

sessions: dict[str, dict] = {}


def _evict_stale_sessions() -> None:
    """Drop expired sessions, then trim to MAX_SESSIONS oldest-first."""
    now = time.time()
    for key in [
        k for k, v in sessions.items()
        if now - v.get("last_seen", 0) > SESSION_TTL_SECONDS
    ]:
        sessions.pop(key, None)
        logger.info("Evicted expired session %s", key)

    while len(sessions) > MAX_SESSIONS:
        oldest = min(sessions, key=lambda k: sessions[k].get("last_seen", 0))
        sessions.pop(oldest, None)
        logger.info("Evicted session %s (capacity)", oldest)


def get_or_create_session(session_id: str) -> dict:
    _evict_stale_sessions()
    session = sessions.get(session_id)
    if session is None:
        session = {
            "memory": ConversationMemory(),
            "rag_chains": {},   # doc_name -> chain
            "agent": None,
            "agent_model": None,
        }
        sessions[session_id] = session
    session["last_seen"] = time.time()
    return session


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Plain chat with conversation memory, no retrieval."""
    session = get_or_create_session(req.session_id)
    try:
        llm = CreateLLM(req.model)
        response = plain_chat(
            req.message,
            session["memory"].get_langchain_history(),
            llm,
        )
    except Exception:
        logger.exception("Chat failed")
        raise HTTPException(status_code=502, detail="The model call failed.")

    session["memory"].add_message("user", req.message)
    session["memory"].add_message("assistant", response)
    return ChatResponse(response=response)


@app.post("/upload")
async def upload_document(
    session_id: str = Form(...),
    model_choice: str = Form(DEFAULT_MODEL),
    file: UploadFile = File(...),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200),
    top_k: int = Form(4),
):
    """Index an uploaded document into a per-session vector store."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_SUFFIXES))}",
        )

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Limit is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
        )

    # Sanity-check the tuning knobs before doing expensive work.
    chunk_size = max(100, min(chunk_size, 2000))
    chunk_overlap = max(0, min(chunk_overlap, chunk_size - 1))
    top_k = max(1, min(top_k, 10))

    session = get_or_create_session(session_id)
    doc_name = Path(file.filename).name

    if (
        len(session["rag_chains"]) >= MAX_DOCS_PER_SESSION
        and doc_name not in session["rag_chains"]
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Session limit of {MAX_DOCS_PER_SESSION} documents reached. "
                "Clear the chat to start over."
            ),
        )

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        llm = CreateLLM(model_choice)

        started = time.perf_counter()
        # CRITICAL: indexing is CPU-bound and synchronous. Running it here on the
        # event loop would block every other request until it finished.
        chain, chunk_count = await run_in_threadpool(
            build_rag_chain,
            tmp_path,
            llm,
            chunk_size,
            chunk_overlap,
            top_k,
        )
        elapsed = time.perf_counter() - started
        logger.info("Indexed %s (%d chunks) in %.1fs", doc_name, chunk_count, elapsed)

        session["rag_chains"][doc_name] = chain
        return {
            "status": "indexed",
            "chunks": chunk_count,
            "document": doc_name,
            "seconds": round(elapsed, 2),
        }

    except ValueError as exc:
        # Raised by the pipeline for documents that are too large to index.
        raise HTTPException(status_code=400, detail=str(exc))
    except HTTPException:
        raise
    except Exception:
        logger.exception("Document processing failed for %s", doc_name)
        raise HTTPException(
            status_code=500,
            detail="Indexing failed. Please try a different document.",
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/rag-query", response_model=ChatResponse)
def rag_query(req: RAGQueryRequest):
    """Answer a question against an indexed document."""
    session = sessions.get(req.session_id)
    if not session or not session["rag_chains"]:
        raise HTTPException(
            status_code=400,
            detail="No document indexed for this session. Upload one first.",
        )
    session["last_seen"] = time.time()

    chain = None
    if req.document_name:
        chain = session["rag_chains"].get(req.document_name)
    if chain is None:
        chain = next(iter(session["rag_chains"].values()))

    try:
        result = chain.invoke({
            "question": req.message,
            "chat_history": session["memory"].get_langchain_history(),
        })
    except Exception:
        logger.exception("RAG query failed")
        raise HTTPException(status_code=502, detail="Query failed. Please try again.")

    answer = result.get("answer", "")
    sources = [
        doc.page_content[:150].replace("\n", " ").strip()
        for doc in result.get("source_documents", [])
    ]

    session["memory"].add_message("user", req.message)
    session["memory"].add_message("assistant", answer)
    return ChatResponse(response=answer, sources=sources)


@app.post("/agent", response_model=ChatResponse)
def agent_chat(req: ChatRequest):
    """ReAct agent with tool access."""
    session = get_or_create_session(req.session_id)

    # Rebuild the agent if the user switched models mid-session.
    if session["agent"] is None or session["agent_model"] != req.model:
        try:
            llm = CreateLLM(req.model)
            session["agent"] = build_agent(req.model, llm)
            session["agent_model"] = req.model
        except Exception:
            logger.exception("Agent construction failed")
            raise HTTPException(status_code=500, detail="Could not start the agent.")

    try:
        result = session["agent"].invoke({
            "input": req.message,
            "chat_history": session["memory"].get_langchain_history(),
        })
        response = result.get("output") or str(result)
    except Exception:
        logger.exception("Agent execution failed")
        response = "The agent encountered an error processing your request."

    session["memory"].add_message("user", req.message)
    session["memory"].add_message("assistant", response)
    return ChatResponse(response=response)


@app.post("/clear/{session_id}")
def clear_session(session_id: str):
    """Drop all state for a session, freeing its vector stores."""
    sessions.pop(session_id, None)
    return {"status": "cleared"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "active_sessions": len(sessions),
    }