import logging
from typing import List, TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from pgvector import Vector

from .config import settings
from .db import get_connection
from .web_search import search_web

logger = logging.getLogger(__name__)

class RAGState(TypedDict):
    question: str
    doc_context: str
    web_context: str
    context: str
    doc_sources: List[str]
    web_sources: List[str]
    answer: str


def _retrieve_chunks(question: str) -> List[str]:
    embeddings = GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        google_api_key=settings.google_api_key,
    )
    query_vector = Vector(embeddings.embed_query(question))

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT source, chunk_index, content
                FROM documents
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (query_vector, settings.top_k),
            )
            rows = cur.fetchall()

    return [
        f"Source: {source} (chunk {chunk_index})\n{content}"
        for source, chunk_index, content in rows
    ]


def _retrieve_web(question: str) -> List[str]:
    try:
        return search_web(question)
    except Exception:
        return []


def retrieve(state: RAGState) -> RAGState:
    doc_chunks = _retrieve_chunks(state["question"])
    web_chunks = _retrieve_web(state["question"])

    logger.info(
        "retrieved context: docs=%s web=%s",
        len(doc_chunks),
        len(web_chunks),
    )

    doc_context = "\n\n".join(doc_chunks)
    web_context = "\n\n".join(web_chunks)

    combined_parts = []
    if doc_context:
        combined_parts.append(f"[Documents]\n{doc_context}")
    if web_context:
        combined_parts.append(f"[Web]\n{web_context}")

    return {
        "question": state["question"],
        "doc_context": doc_context,
        "web_context": web_context,
        "context": "\n\n".join(combined_parts),
        "doc_sources": doc_chunks,
        "web_sources": web_chunks,
        "answer": "",
    }


def generate(state: RAGState) -> RAGState:
    model = ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.google_api_key,
        temperature=0.2,
    )
    system_prompt = (
        "You answer questions about macOS Tahoe (macOS 26) using the provided "
        "context only. Use both document context plus web context to answer "
        "the user's query. If the answer is not in the context, say you do not "
        "have enough information."
    )
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=f"Question: {state['question']}\n\nContext:\n{state['context']}"
        ),
    ]
    response = model.invoke(messages)
    return {
        "question": state["question"],
        "doc_context": state["doc_context"],
        "web_context": state["web_context"],
        "context": state["context"],
        "doc_sources": state["doc_sources"],
        "web_sources": state["web_sources"],
        "answer": response.content.strip(),
    }


def answer_question(question: str) -> RAGState:
    state = {"question": question}
    state = retrieve(state)  # type: ignore[assignment]
    return generate(state)  # type: ignore[return-value]
