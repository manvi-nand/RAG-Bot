import os

import psycopg2
from pgvector.psycopg2 import register_vector

from .config import settings


def get_connection(register: bool = True):
    database_url = os.getenv("DATABASE_URL") or os.getenv("DATABASE_PUBLIC_URL")
    if database_url:
        conn = psycopg2.connect(database_url)
    else:
        conn = psycopg2.connect(
            dbname=settings.db_name,
            user=settings.db_user,
            password=settings.db_password,
            host=settings.db_host,
            port=settings.db_port,
        )
    if register:
        register_vector(conn)
    return conn


EMBEDDING_DIM = 3072  # must match gemini-embedding-001 output


def init_db():
    with get_connection(register=False) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            # One-time migration: if documents exists with wrong vector dimensions, drop it
            cur.execute(
                """
                SELECT a.atttypmod FROM pg_attribute a
                JOIN pg_class c ON a.attrelid = c.oid
                JOIN pg_namespace n ON c.relnamespace = n.oid
                WHERE n.nspname = 'public' AND c.relname = 'documents'
                  AND a.attname = 'embedding' AND a.attnum > 0 AND NOT a.attisdropped
                """
            )
            row = cur.fetchone()
            if row is not None and row[0] != EMBEDDING_DIM:
                cur.execute("DROP TABLE documents;")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    source TEXT NOT NULL,
                    chunk_index INT NOT NULL,
                    content TEXT NOT NULL,
                    embedding VECTOR(%s) NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
                );
                """,
                (EMBEDDING_DIM,),
            )
        conn.commit()
    with get_connection(register=True):
        pass
