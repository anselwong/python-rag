# python-rag

RAG knowledge base backend, built with FastAPI.

## Local development

Python 3.11 is recommended. Python 3.9 or later is supported for Day 1.

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
