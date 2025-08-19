from fastapi import FastAPI, File, UploadFile, Body
from PyPDF2 import PdfReader
import uuid
import re
from fastapi import Query
import os
from typing import List, Dict, Any, Optional
import httpx
from openai import OpenAI
import google.generativeai as genai
from fastapi import Body
from dotenv import load_dotenv
from fastapi.responses import StreamingResponse
import json


app = FastAPI()
DOCUMENTS: Dict[str, Dict[str, Any]] = {}

# simple counters for /metrics
METRICS: Dict[str, int] = {
    "ingest_calls": 0,
    "extract_calls": 0,
    "ask_calls": 0,
    "ask_stream_calls": 0,
    "audit_calls": 0,
    "search_calls": 0,
    "documents": 0,
}

def _bump(name: str) -> None:
    METRICS[name] = METRICS.get(name, 0) + 1


@app.get("/healthz")
def health_check():
    return {"status": "ok", "source": "docker"}


@app.get("/metrics")
def get_metrics():
    METRICS["documents"] = len(DOCUMENTS)
    return METRICS



# Function to ingest 1..n documents
@app.post("/ingest")
async def ingest(files: List[UploadFile] = File(...)):
    """
    Ingest one or more PDF files and return their assigned `document_id`s.
    """
    _bump("ingest_calls")

    ingested: List[Dict[str, Any]] = []

    for file in files:
        reader = PdfReader(file.file)
        doc_id = str(uuid.uuid4())

        chunks: List[Dict[str, Any]] = []
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            start = 0
            # Split into smaller chunks (e.g. 300 characters each)
            while start < len(text):
                end = min(start + 300, len(text))
                chunk_text = text[start:end]
                chunks.append({
                    "page": page_num,
                    "text": chunk_text,
                    "start": start,
                    "end": end
                })
                start = end

        DOCUMENTS[doc_id] = {
            "filename": file.filename,
            "chunks": chunks
        }

        ingested.append({
            "document_id": doc_id,
            "filename": file.filename,
            "chunks_stored": len(chunks)
        })

    METRICS["documents"] = len(DOCUMENTS)
    return {"ingested": ingested}







# Function to extract structured information from the document
@app.post("/extract")
async def extract_fields(document_id: str):
    """
    Extracts key fields using heuristic regexes. Best-effort implementation
    for: parties[], effective_date, term, governing_law, payment_terms,
    termination, auto_renewal, confidentiality, indemnity,
    liability_cap { amount, currency }, signatories[] { name, title }
    """
    _bump("extract_calls")

    if document_id not in DOCUMENTS:
        return {"error": "document_id not found"}

    text = " ".join([c["text"] for c in DOCUMENTS[document_id]["chunks"]])

    fields = {
        "parties": re.findall(r"between\s+(.*?)\s+and\s+(.*?)\.", text, re.IGNORECASE),
        "effective_date": re.search(r"effective\s+date[:]?\s*(\w+\s+\d{1,2},\s*\d{4})", text, re.IGNORECASE),
        "term": re.search(r"term[:]?\s*(\d+\s*(?:months?|years?))", text, re.IGNORECASE),
        "governing_law": re.search(r"governed\s+by\s+the\s+laws\s+of\s+(.*?)(?:\.|,)", text, re.IGNORECASE),
        "payment_terms": re.search(r"payment\s+terms?[:]?\s*(.*?)(?:\.|\n)", text, re.IGNORECASE),
        "termination": re.search(r"termination[:]?\s*(.*?)(?:\.|\n)", text, re.IGNORECASE),
        "auto_renewal": re.search(r"auto[-\s]?renew\w*\s*(.*?)(?:\.|\n)", text, re.IGNORECASE),
        "confidentiality": re.search(r"\bconfidentialit\w*[:]?\s*(.*?)(?:\.|\n)", text, re.IGNORECASE),
        "indemnity": re.search(r"\bindemnif\w*[:]?\s*(.*?)(?:\.|\n)", text, re.IGNORECASE),
        "liability_cap": re.search(r"liabilit\w*\s+(?:cap|limit)[^\d$€£]*([\$€£]?\s?\d[\d,\.]*)(?:\s*(USD|EUR|GBP))?", text, re.IGNORECASE),
        "signatories": re.findall(r"\n\s*([A-Z][a-zA-Z' -]+)\s*,?\s*(?:Title|Role)\s*[:]?\s*([A-Za-z ][A-Za-z ]+)", text, re.IGNORECASE),
    }

    result: Dict[str, Any] = {}
    for key, value in fields.items():
        if not value:
            result[key] = None
        elif isinstance(value, list):
            result[key] = value
        elif key == "liability_cap":
            amount = value.group(1) if value.groups() else value.group(0)
            currency = value.group(2) if value.groups() and len(value.groups()) >= 2 else None
            result[key] = {"amount": amount, "currency": currency}
        else:
            result[key] = value.group(1) if value.groups() else value.group(0)

    return {"document_id": document_id, "extracted_fields": result}



# Function to search the document of the user for a given query 
@app.get("/search")
def search_document(document_id: str = Query(...), query: str = Query(...)):
    """
    Search for a query string inside document chunks
    """
    _bump("search_calls")
    if document_id not in DOCUMENTS:
        return {"error": "Document not found"}

    matches = []
    for chunk in DOCUMENTS[document_id]["chunks"]:
        if query.lower() in chunk["text"].lower():
            matches.append({
                "page": chunk["page"],
                "char_range": [chunk["start"], chunk["end"]],
                "snippet": chunk["text"].strip()
            })

    return {
        "document_id": document_id,
        "query": query,
        "matches": matches if matches else ["No matches found."]
    }


"""Q&A endpoints: configure OpenAI/Gemini clients once"""
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client: Optional[OpenAI] = OpenAI() if OPENAI_API_KEY else None

# Gemini config (supports GOOGLE_API_KEY or GEMINI_API_KEY)
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    try:
        GEMINI_MODEL = genai.GenerativeModel("gemini-1.5-flash")
    except Exception:
        GEMINI_MODEL = None
else:
    GEMINI_MODEL = None


def _select_provider(preferred: Optional[str]) -> Optional[str]:
    """Return 'openai', 'gemini', or None based on availability and preference."""
    normalized = (preferred or "").strip().lower()
    if normalized == "openai" and client is not None:
        return "openai"
    if normalized == "gemini" and GEMINI_MODEL is not None:
        return "gemini"
    # Auto-pick: OpenAI first, then Gemini
    if client is not None:
        return "openai"
    if GEMINI_MODEL is not None:
        return "gemini"
    return None


@app.post("/ask")
def ask_question(
    question: str = Body(...),
    document_id: Optional[str] = Body(None),
    provider: Optional[str] = Body(None, description="'openai' or 'gemini'")
):
    """
    Q&A endpoint:
    - Retrieve best chunk
    - Optionally use LLM to phrase a cleaner answer
    """
    _bump("ask_calls")
    if not DOCUMENTS:
        return {"error": "No documents available"}

    docs_to_search = (
        {document_id: DOCUMENTS.get(document_id)} if document_id else DOCUMENTS
    )
    if not docs_to_search or None in docs_to_search.values():
        return {"error": "Invalid document_id"}

    question_words = set(question.lower().split())
    best_chunk = None
    best_score = 0
    best_doc = None

    # Step 1: retrieval
    for doc_id, doc in docs_to_search.items():
        for chunk in doc["chunks"]:
            score = sum(1 for word in question_words if word in chunk["text"].lower())
            if score > best_score:
                best_score = score
                best_chunk = chunk
                best_doc = doc_id

    if not best_chunk:
        return {"answer": "Sorry, I could not find the answer.", "citations": []}

    raw_answer = best_chunk["text"].strip()

    # Step 2: LLM fallback (if API key available)
    final_answer = raw_answer
    selected = _select_provider(provider)
    try:
        if selected == "openai":
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Answer only using the provided context."},
                    {"role": "user", "content": f"Question: {question}\nContext: {raw_answer}\nAnswer concisely:"}
                ],
                max_tokens=150,
                temperature=0.2
            )
            final_answer = response.choices[0].message.content.strip()
        elif selected == "gemini":
            prompt = f"Answer the question using ONLY this context.\nQuestion: {question}\nContext: {raw_answer}\nConcise answer:"
            gem_resp = GEMINI_MODEL.generate_content(prompt)
            final_answer = (gem_resp.text or raw_answer).strip()
    except Exception:
        final_answer = raw_answer

    return {
        "answer": final_answer,
        "citations": [
            {
                "document_id": best_doc,
                "page": best_chunk["page"],
                "char_range": [best_chunk["start"], best_chunk["end"]],
            }
        ],
    }





# Function to ask a question to the document of the user and stream the response
@app.post("/ask/stream")
async def ask_stream(
    question: str = Body(...),
    document_id: Optional[str] = Body(None),
    provider: Optional[str] = Body(None)
):
    """
    Streaming Q&A:
    - Same retrieval as /ask
    - Streams LLM response token-by-token
    """
    _bump("ask_stream_calls")
    if not DOCUMENTS:
        return {"error": "No documents available"}

    docs_to_search = (
        {document_id: DOCUMENTS.get(document_id)} if document_id else DOCUMENTS
    )
    if not docs_to_search or None in docs_to_search.values():
        return {"error": "Invalid document_id"}

    question_words = set(question.lower().split())
    best_chunk = None
    best_score = 0
    best_doc = None
    for doc_id, doc in docs_to_search.items():
        for chunk in doc["chunks"]:
            score = sum(1 for word in question_words if word in chunk["text"].lower())
            if score > best_score:
                best_score = score
                best_chunk = chunk
                best_doc = doc_id

    if not best_chunk:
        return {"answer": "No relevant answer found.", "citations": []}

    # Citations (fixed, same as /ask)
    citations = [
        {
            "document_id": best_doc,
            "page": best_chunk["page"],
            "char_range": [best_chunk["start"], best_chunk["end"]],
        }
    ]

    async def event_generator():
        selected = _select_provider(provider)
        if selected == "openai":
            try:
                with client.chat.completions.stream(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "Answer only using the provided context."},
                        {"role": "user", "content": f"Question: {question}\nContext: {best_chunk['text']}\nAnswer concisely:"}
                    ],
                    max_tokens=200,
                    temperature=0.2,
                ) as stream:
                    for event in stream:
                        if event.type == "token":
                            yield f"data: {json.dumps({'token': event.token})}\n\n"
            except Exception:
                for word in best_chunk["text"].split():
                    yield f"data: {json.dumps({'token': word + ' '})}\n\n"
        elif selected == "gemini":
            try:
                # Gemini returns full text; chunk tokens ourselves for SSE feel
                prompt = f"Answer the question using ONLY this context.\nQuestion: {question}\nContext: {best_chunk['text']}\nConcise answer:"
                gem_resp = GEMINI_MODEL.generate_content(prompt)
                text = (gem_resp.text or "").strip()
                if text:
                    for word in text.split():
                        yield f"data: {json.dumps({'token': word + ' '})}\n\n"
                else:
                    for word in best_chunk["text"].split():
                        yield f"data: {json.dumps({'token': word + ' '})}\n\n"
            except Exception:
                for word in best_chunk["text"].split():
                    yield f"data: {json.dumps({'token': word + ' '})}\n\n"
        else:
            for word in best_chunk["text"].split():
                yield f"data: {json.dumps({'token': word + ' '})}\n\n"

        yield f"data: {json.dumps({'citations': citations})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/audit")
def audit_document(
    document_id: str = Body(...),
    webhook_url: Optional[str] = Body(None),
    provider: Optional[str] = Body(None)
):
    """
    Detect risky clauses (e.g., auto-renewal with <30d notice, unlimited liability,
    broad indemnity). Returns findings with severity and citations.

    If `webhook_url` is provided, POST results to that URL after analysis.
    """
    _bump("audit_calls")

    if document_id not in DOCUMENTS:
        return {"error": "Invalid document_id"}

    doc = DOCUMENTS[document_id]
    risks: List[Dict[str, Any]] = []

    combined_text = "\n\n".join([c["text"] for c in doc["chunks"]])
    selected = _select_provider(provider)
    if selected == "openai":
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a contract auditor. "
                            "Analyze the following contract text and identify any risky clauses. "
                            "Return a JSON array of objects with fields: clause, issue, severity."
                        )
                    },
                    {"role": "user", "content": combined_text},
                ],
                temperature=0.2,
                max_tokens=500,
            )
            llm_output = response.choices[0].message.content.strip()
            try:
                risks_from_llm = json.loads(llm_output)
                if isinstance(risks_from_llm, list):
                    risks.extend(risks_from_llm)
            except Exception:
                risks.append({"clause": "N/A", "issue": llm_output, "severity": "medium"})
        except Exception:
            pass
    elif selected == "gemini":
        try:
            prompt = (
                "You are a contract auditor. Analyze the following contract text and "
                "identify any risky clauses. Return a JSON array of objects with "
                "fields: clause, issue, severity (low|medium|high).\n\n" + combined_text
            )
            gem_resp = GEMINI_MODEL.generate_content(prompt)
            llm_output = (gem_resp.text or "").strip()
            if llm_output:
                try:
                    risks_from_llm = json.loads(llm_output)
                    if isinstance(risks_from_llm, list):
                        risks.extend(risks_from_llm)
                except Exception:
                    risks.append({"clause": "N/A", "issue": llm_output, "severity": "medium"})
        except Exception:
            pass

    # Rule-based checks
    for chunk in doc["chunks"]:
        text = chunk["text"]
        lowered = text.lower()

        auto_match = re.search(r"auto[-\s]?renew\w*.*?(\d{1,3})\s*(day|calendar day)", lowered)
        if auto_match:
            try:
                days = int(auto_match.group(1))
                if days < 30:
                    risks.append({
                        "clause": text.strip(),
                        "issue": f"Auto-renewal notice period is short ({days} days).",
                        "severity": "medium",
                        "page": chunk["page"],
                        "char_range": [chunk["start"], chunk["end"]],
                    })
            except Exception:
                pass

        if "unlimited liability" in lowered or "liability is unlimited" in lowered:
            risks.append({
                "clause": text.strip(),
                "issue": "Unlimited liability.",
                "severity": "high",
                "page": chunk["page"],
                "char_range": [chunk["start"], chunk["end"]],
            })

        if "indemn" in lowered and ("any and all" in lowered or "third parties" in lowered):
            risks.append({
                "clause": text.strip(),
                "issue": "Broad indemnity scope.",
                "severity": "medium",
                "page": chunk["page"],
                "char_range": [chunk["start"], chunk["end"]],
            })

    result = {"document_id": document_id, "risks": risks}

    if webhook_url:
        try:
            httpx.post(webhook_url, json=result, timeout=5.0)
        except Exception:
            pass

    return result
