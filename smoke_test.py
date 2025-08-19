import requests, pathlib

BASE = "http://127.0.0.1:8000"
PDF_PATH = pathlib.Path(r"D:\Assignment(Aviara)\dummy_contract_1.pdf")  # change if needed

def main():
    # Ingest
    with open(PDF_PATH, "rb") as f:
        r = requests.post(f"{BASE}/ingest", files=[("files", ("contract.pdf", f, "application/pdf"))])
    r.raise_for_status()
    data = r.json()
    doc_id = data["ingested"][0]["document_id"]
    print("document_id:", doc_id)

    # Extract
    r = requests.post(f"{BASE}/extract", params={"document_id": doc_id})
    r.raise_for_status()
    print("extract:", r.json())

    # Ask (auto-picks provider if keys set; or retrieval-only)
    r = requests.post(f"{BASE}/ask", json={"question": "What is the governing law?", "document_id": doc_id})
    r.raise_for_status()
    print("ask:", r.json())

    # Audit
    r = requests.post(f"{BASE}/audit", json={"document_id": doc_id})
    r.raise_for_status()
    print("audit:", r.json())

if __name__ == "__main__":
    main()