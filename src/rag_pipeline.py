import httpx
_original_client_init = httpx.Client.__init__

def _patched_client_init(self, *args, **kwargs):
    if "proxies" in kwargs:
        kwargs["proxy"] = kwargs.pop("proxies")
    _original_client_init(self, *args, **kwargs)

httpx.Client.__init__ = _patched_client_init


# Patch httpx.AsyncClient (NEW - this was missing)
_original_async_client_init = httpx.AsyncClient.__init__

def _patched_async_client_init(self, *args, **kwargs):
    if "proxies" in kwargs:
        kwargs["proxy"] = kwargs.pop("proxies")
    _original_async_client_init(self, *args, **kwargs)

httpx.AsyncClient.__init__ = _patched_async_client_init



from pathlib import Path

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)





SYSTEM_TEMPLATE = """
You are an expert research assistant.

Use the provided document context to answer the user's question accurately and concisely.

If the answer is not present in the context, clearly say that the information is not available in the document. Do not make up information.

Relevant document excerpts:
{context}

Previous Conversation:
{chat_history}
"""

HUMAN_TEMPLATE = "{question}"


def load_and_split_document(
    file_path: str,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
):
    """Load a document and split into chunks."""

    path = Path(file_path)

    if path.suffix.lower() == ".pdf":
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding="utf-8")

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    return chunks


def build_rag_chain(
    file_path: str,
    llm
):
    # Load and split document
    chunks = load_and_split_document(file_path)

    # Embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
         cache_folder=r"D:\Models\hugging-face"
    )

    # Vector store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="research_docs",
    )

    # Retriever
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
        },
    )

    # LLM
    

    # Prompt
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                SYSTEM_TEMPLATE
            ),
            HumanMessagePromptTemplate.from_template(
                HUMAN_TEMPLATE
            ),
        ]
    )

    # Conversational Retrieval Chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        combine_docs_chain_kwargs={
            "prompt": qa_prompt
        },
        verbose=False,
    )

    return chain, len(chunks)