import psycopg2
from pgvector.psycopg2 import register_vector

from .config import settings


def get_connection(register: bool = True):
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


def init_db():
    with get_connection(register=False) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    source TEXT NOT NULL,
                    chunk_index INT NOT NULL,
                    content TEXT NOT NULL,
                    embedding VECTOR(768) NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
                );
                """
            )
        conn.commit()
    with get_connection(register=True):
        pass
