# # src/httpx_compat.py

# import httpx


# # Patch synchronous client
# _original_client_init = httpx.Client.__init__


# def _patched_client_init(self, *args, **kwargs):
#     if "proxies" in kwargs:
#         kwargs["proxy"] = kwargs.pop("proxies")

#     _original_client_init(self, *args, **kwargs)


# httpx.Client.__init__ = _patched_client_init


# # Patch asynchronous client
# _original_async_client_init = httpx.AsyncClient.__init__


# def _patched_async_client_init(self, *args, **kwargs):
#     if "proxies" in kwargs:
#         kwargs["proxy"] = kwargs.pop("proxies")

#     _original_async_client_init(self, *args, **kwargs)


# httpx.AsyncClient.__init__ = _patched_async_client_init


import hashlib
import logging
from functools import lru_cache
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

EMBED_MODEL = "BAAI/bge-small-en-v1.5"   # 384-dim, ONNX, ~130MB
MAX_CHUNKS = 1500                         # guard against huge uploads

SYSTEM_TEMPLATE = """You are an expert research assistant.

Use the provided document context to answer the user's question accurately and concisely.
If the answer is not present in the context, say the information is not available in the
document. Do not make up information.

Relevant document excerpts:
{context}

Previous conversation:
{chat_history}
"""


@lru_cache(maxsize=1)
def get_embeddings():
    """Single shared embedding model for the whole process."""
    logger.info("Loading embedding model %s", EMBED_MODEL)
    return FastEmbedEmbeddings(model_name=EMBED_MODEL, threads=2)


def warm_up():
    """Call once at startup so the first user request isn't the cold start."""
    get_embeddings().embed_query("warm up")


def load_and_split_document(file_path: str, chunk_size: int = 800,
                            chunk_overlap: int = 150):
    path = Path(file_path)
    loader = (
        PyPDFLoader(file_path)
        if path.suffix.lower() == ".pdf"
        else TextLoader(file_path, encoding="utf-8")
    )
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    if len(chunks) > MAX_CHUNKS:
        raise ValueError(
            f"Document produces {len(chunks)} chunks (limit {MAX_CHUNKS}). "
            "Try a smaller file or a larger chunk size."
        )
    return chunks


def build_rag_chain(file_path: str, llm, chunk_size=1000,
                    chunk_overlap=200, top_k=4):
    chunks = load_and_split_document(file_path, chunk_size, chunk_overlap)

    vectorstore = InMemoryVectorStore.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": top_k},
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(SYSTEM_TEMPLATE),
        HumanMessagePromptTemplate.from_template("{question}"),
    ])

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        verbose=False,
    )
    return chain, len(chunks)