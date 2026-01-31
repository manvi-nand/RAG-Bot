import logging
import secrets
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .embeddings import ingest_folder

from .db import init_db
from .rag import answer_question

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)


app = FastAPI(title="macOS Tahoe RAG Bot")
STATIC_DIR = Path(__file__).parent / "static"
MAX_TURNS = 6
SESSIONS: dict[str, list[dict]] = {}


class ChatRequest(BaseModel):
    question: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    doc_sources: list[str]
    web_sources: list[str]
    session_id: str

class IngestRequest(BaseModel):
    filenames: list[str] | None = None


class IngestResponse(BaseModel):
    processed: list[str]
    skipped: list[str]


@app.on_event("startup")
def startup():
    init_db()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest):
    session_id = payload.session_id or secrets.token_hex(8)
    history = SESSIONS.get(session_id, [])
    result = answer_question(payload.question, history=history)
    history.append({"role": "user", "content": payload.question})
    history.append({"role": "assistant", "content": result["answer"]})
    SESSIONS[session_id] = history[-MAX_TURNS * 2 :]
    return {
        "answer": result["answer"],
        "doc_sources": result.get("doc_sources", []),
        "web_sources": result.get("web_sources", []),
        "session_id": session_id,
    }


@app.post("/ingest", response_model=IngestResponse)
def ingest(payload: IngestRequest):
    processed, skipped = ingest_folder(Path("data"), filenames=payload.filenames)
    return {"processed": processed, "skipped": skipped}
