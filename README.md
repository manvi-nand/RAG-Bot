# macOS Tahoe RAG Bot

FastAPI + Postgres (pgvector) + Gemini embeddings + optional web search.

## Local run
1) Install deps
```
pip install -r requirements.txt
```

2) Set env vars
Copy `.env.example` to `.env` and fill:
- `DB_*`
- `GOOGLE_API_KEY`
- `TAVILY_API_KEY` (optional)

3) Ingest PDFs (from `data/`)
```
python ingest.py
```

4) Start API
```
uvicorn app.main:app --reload
```

Open: `http://127.0.0.1:8000/`

## Railway deployment
1) Push this repo to GitHub.
2) In Railway:
   - New Project → Deploy from GitHub
   - Add PostgreSQL database
   - Set environment variables (same as `.env`)
3) Railway will detect `Procfile` and run:
```
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

## Ingestion on Railway
Run `python ingest.py` locally or as a one-off Railway job.
