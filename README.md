## Contract Intelligence Assignment

FastAPI service to ingest contracts (PDF), extract key fields, run Q&A grounded in uploaded documents, and audit for risky clauses. Supports OpenAI and Google Gemini models; the user can pick the provider per request.

### Features

- Ingest 1..n PDFs; store text in page-aligned chunks with character offsets
- Extract common contract fields (parties, effective date, governing law, etc.)
- Q&A with citations to the source span; optional streaming via SSE
- Audit for risky clauses with rule-based checks and optional LLM augmentation
- Provider selection: `openai` or `gemini` per request
- Health and simple metrics endpoints

### Project structure

```
Assignment(Aviara)/
  contract-intel/
    app/
      main.py           # FastAPI app and endpoints
    Dockerfile
    requirements.txt
    smoke_test.py       # Optional end-to-end smoke test
  dummy_contract_1.pdf  # Sample inputs
  dummy_contract_2.pdf
  dummy_contract_3.pdf
  dummy_contract_4.pdf
```

### Prerequisites

- Python 3.11+
- Optionally Docker Desktop (Windows/macOS)

### Environment variables (optional)

- `OPENAI_API_KEY`: enable OpenAI for `/ask`, `/ask/stream`, `/audit`
- `GOOGLE_API_KEY` (or `GEMINI_API_KEY`): enable Gemini for the same endpoints

If neither is set, the app still works in retrieval-only mode; streaming will simulate tokens.

### Run locally (Windows PowerShell)

```powershell
Set-Location "D:\Assignment(Aviara)\contract-intel"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# optional: set LLM keys
$env:OPENAI_API_KEY="sk-..."
$env:GOOGLE_API_KEY="..."
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for interactive API docs.

### Run with Docker

```powershell
Set-Location accordingly
docker build -t contract-intel .
docker run --rm -p 8000:8000 `
  -e OPENAI_API_KEY=$env:OPENAI_API_KEY `
  -e GOOGLE_API_KEY=$env:GOOGLE_API_KEY `
  contract-intel
```

Then visit `http://localhost:8000/docs`.

### Endpoints overview

- `GET /healthz`: liveness
- `GET /metrics`: simple counters and document count
- `POST /ingest` (multipart/form-data):
  - field: `files` (repeatable) → one or more PDFs
  - returns: `{ "ingested": [{ document_id, filename, chunks_stored }] }`
- `POST /extract?document_id=...`:
  - returns: `{ document_id, extracted_fields: {...} }`
- `POST /ask` (JSON):
  - body: `{ question, document_id?, provider? }`
  - `provider`: `'openai' | 'gemini'` (optional; auto-picks if omitted)
  - returns: `{ answer, citations: [{ document_id, page, char_range }] }`
- `POST /ask/stream` (JSON):
  - body: `{ question, document_id?, provider? }`
  - returns: `text/event-stream` tokens + final citations
- `POST /audit` (JSON):
  - body: `{ document_id, webhook_url?, provider? }`
  - returns: `{ document_id, risks: [{ clause, issue, severity, page?, char_range? }] }`
  - if `webhook_url` provided, results are POSTed there as well

### Quick cURL examples (Windows PowerShell)

```powershell
# Ingest two PDFs from the parent folder
curl.exe -F "files=@..\dummy_contract_1.pdf" -F "files=@..\dummy_contract_2.pdf" http://127.0.0.1:8000/ingest

# Extract
curl.exe -X POST "http://127.0.0.1:8000/extract?document_id=YOUR_ID"

# Ask (force OpenAI)
curl.exe -H "Content-Type: application/json" `
  -d "{\"question\":\"What is the governing law?\",\"document_id\":\"YOUR_ID\",\"provider\":\"openai\"}" `
  http://127.0.0.1:8000/ask

# Stream (force Gemini)
curl.exe -N -H "Content-Type: application/json" `
  -d "{\"question\":\"Summarize payment terms\",\"document_id\":\"YOUR_ID\",\"provider\":\"gemini\"}" `
  http://127.0.0.1:8000/ask/stream

# Audit with webhook callback
curl.exe -H "Content-Type: application/json" `
  -d "{\"document_id\":\"YOUR_ID\",\"provider\":\"openai\",\"webhook_url\":\"https://webhook.site/YOUR-UUID\"}" `
  http://127.0.0.1:8000/audit
```

### Smoke test

There is a ready-to-run script at `contract-intel/smoke_test.py`:

```powershell
Set-Location "D:\Assignment(Aviara)\contract-intel"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt requests
uvicorn app.main:app --reload
# In a new terminal (same venv):
python smoke_test.py
```

If all calls return HTTP 200 and sensible JSON, the service is working end-to-end.

### Provider selection

- Per request, set `provider` to `'openai'` or `'gemini'` in `/ask`, `/ask/stream`, `/audit`
- If omitted, the app auto-picks based on available keys (OpenAI, then Gemini)

### Notes and limitations

- Documents are stored in-memory; restarting the server clears them
- Regex-based extraction is best-effort and may miss variants
- SSE streaming with Gemini is simulated by word-chunking the response text

### Troubleshooting

- Docker Desktop paused: resume from the whale menu
- Windows service control errors: run PowerShell as Administrator, or quit/relaunch Docker Desktop if user-mode
- 422 on ingest: ensure field name is `files`, and content-type is `multipart/form-data`
- Port in use: choose another port with `--port 8001`
