# RAG Research Assistant

An AI-powered research assistant built with:
- Streamlit
- LangChain
- Groq
- FAISS
- Retrieval Augmented Generation (RAG)

Features:
- Multi-turn chat
- Document Q&A
- Conversational memory
- Vector search

Architecture:

Document
 ↓
Chunking
 ↓
Embeddings
 ↓
Chroma DB
 ↓
Retriever
 ↓
LLM
