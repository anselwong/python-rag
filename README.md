# python-rag

RAG knowledge base backend, built with FastAPI and PostgreSQL + pgvector.

## Local development

Python 3.11 is recommended. Python 3.9 or later is supported.

## PostgreSQL

Production persistence uses PostgreSQL. The `chunks.embedding` column is a
pgvector column reserved for Day 6 Embedding and similarity search. Start the
local database with Docker:

```bash
docker compose up -d postgres
```

The default connection is `postgresql+psycopg://rag:rag@localhost:5432/rag`.
Override it with `DATABASE_URL` when using another PostgreSQL instance. The
`vector` extension is enabled automatically on application startup.

The tests use a temporary SQLite database only to avoid requiring a running
PostgreSQL server; application code and production configuration target
PostgreSQL.

Day 4 adds document upload and parsing dependencies: `python-multipart` handles
`multipart/form-data`, PyMuPDF extracts PDF pages, and `python-docx` extracts DOCX
paragraphs. Parsed page text is saved as metadata for the later citation pipeline.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

The API is available at `http://127.0.0.1:8000`; interactive documentation is at
`http://127.0.0.1:8000/docs`.

Run tests with:

```bash
pytest
```
