from pathlib import Path
from typing import Iterable, List, Tuple

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from .config import settings
from .db import get_connection


def load_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def chunk_text(text: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return splitter.split_text(text)


def embed_texts(texts: List[str]) -> List[List[float]]:
    embeddings = GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        google_api_key=settings.google_api_key,
        transport="rest"
    )
    return embeddings.embed_documents(texts)


def store_embeddings(source: str, chunks: List[str], vectors: List[List[float]]):
    if not chunks:
        return
    with get_connection() as conn:
        with conn.cursor() as cur:
            rows = [
                (source, idx, chunk, vector)
                for idx, (chunk, vector) in enumerate(zip(chunks, vectors))
            ]
            cur.executemany(
                """
                INSERT INTO documents (source, chunk_index, content, embedding)
                VALUES (%s, %s, %s, %s)
                """,
                rows,
            )
        conn.commit()


def is_source_ingested(source: str) -> bool:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM documents WHERE source = %s LIMIT 1",
                (source,),
            )
            return cur.fetchone() is not None


def delete_documents_by_source(source: str) -> None:
    """Remove all chunks for a given source (e.g. to re-ingest)."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM documents WHERE source = %s", (source,))
        conn.commit()


def ingest_pdf(path: Path, skip_if_exists: bool = True) -> bool:
    if skip_if_exists and is_source_ingested(path.name):
        return False
    text = load_pdf_text(path)
    chunks = chunk_text(text)
    vectors = embed_texts(chunks)
    store_embeddings(path.name, chunks, vectors)
    return True


def ingest_folder(
    folder: Path,
    filenames: List[str] | None = None,
    force: bool = False,
) -> Tuple[List[str], List[str]]:
    processed: List[str] = []
    skipped: List[str] = []
    pdf_paths = sorted(folder.glob("*.pdf"))
    if filenames:
        filename_set = {name.strip() for name in filenames if name.strip()}
        pdf_paths = [path for path in pdf_paths if path.name in filename_set]
    for pdf_path in pdf_paths:
        if force:
            delete_documents_by_source(pdf_path.name)
        did_ingest = ingest_pdf(pdf_path, skip_if_exists=not force)
        if did_ingest:
            processed.append(pdf_path.name)
        else:
            skipped.append(pdf_path.name)
    return processed, skipped
