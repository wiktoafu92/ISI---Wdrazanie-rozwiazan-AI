import time
import io
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models import DocumentStatus, Settings
from app.ocr import run_donut_ocr
from app.rag import prep_vectorstore


client = TestClient(app)


# ---------------------------------------------------------
# FIXTURE 1: czyszczenie Chroma przed każdym testem
# ---------------------------------------------------------
@pytest.fixture(autouse=True)
def clean_chroma(tmp_path):
    settings = Settings()  # type: ignore

    persist_dir = Path(settings.persist_directory)
    if persist_dir.exists():
        shutil.rmtree(persist_dir)

    new_vs = prep_vectorstore()

    app.dependency_overrides = {}
    app.vector_store = new_vs

    yield

    if persist_dir.exists():
        shutil.rmtree(persist_dir)


# ---------------------------------------------------------
# FIXTURE 2: cache OCR (OCR wykonywany tylko raz)
# ---------------------------------------------------------
@pytest.fixture(scope="session")
def cached_ocr():
    settings = Settings()  # type: ignore
    image_path = Path(settings.file_path)

    assert image_path.exists(), f"Brak pliku testowego: {image_path}"

    parsed = run_donut_ocr(str(image_path))

    return parsed


# ---------------------------------------------------------
# FIXTURE 3: pełny pipeline → gotowy document_id
# ---------------------------------------------------------
@pytest.fixture
def indexed_document(cached_ocr):
    settings = Settings()  # type: ignore
    image_path = Path(settings.file_path)

    with open(image_path, "rb") as f:
        file_bytes = f.read()

    resp = client.post(
        "/documents/upload",
        files={"file": ("fv.jpg", io.BytesIO(file_bytes), "image/jpeg")}
    )
    assert resp.status_code == 202

    doc_id = resp.json()["document_id"]

    from app.main import DOCUMENTS, PARSED
    PARSED[doc_id] = cached_ocr

    DOCUMENTS[doc_id] = DOCUMENTS[doc_id].model_copy(
        update={
            "status": DocumentStatus.completed,
            "raw_text": "cached OCR",
            "structured": cached_ocr,
        }
    )

    resp2 = client.post(f"/documents/{doc_id}/index")
    assert resp2.status_code == 200
    assert resp2.json()["indexed"] is True

    return doc_id


# ---------------------------------------------------------
# 1. HEALTH CHECK
# ---------------------------------------------------------
def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ---------------------------------------------------------
# 2. OCR PIPELINE (bez cache)
# ---------------------------------------------------------
def test_full_ocr_pipeline():
    settings = Settings()  # type: ignore
    image_path = Path(settings.file_path)

    with open(image_path, "rb") as f:
        file_bytes = f.read()

    resp = client.post(
        "/documents/upload",
        files={"file": ("fv2.jpg", io.BytesIO(file_bytes), "image/jpeg")}
    )
    assert resp.status_code == 202

    doc_id = resp.json()["document_id"]

    status = None
    status_resp = None
    for _ in range(60):
        time.sleep(1)
        status_resp = client.get(f"/documents/{doc_id}")
        status = status_resp.json()["status"]
        if status == DocumentStatus.completed:
            break
        if status == DocumentStatus.failed:
            raise RuntimeError("OCR failed")

    assert status == DocumentStatus.completed

    data = status_resp.json()
    assert data["raw_text"] is not None
    assert data["structured"] is not None


# ---------------------------------------------------------
# 3. RAG SEARCH (korzysta z fixture indexed_document)
# ---------------------------------------------------------
def test_rag_search_real(indexed_document):
    query = "Jaki jest numer faktury?"

    resp = client.post("/rag/search", json={"query": query})
    assert resp.status_code == 200

    results = resp.json()["results"]
    assert len(results) > 0


# ---------------------------------------------------------
# 4. RAG ANSWER (korzysta z fixture indexed_document)
# ---------------------------------------------------------
def test_rag_answer_real(indexed_document):
    query = "Jaka jest kwota brutto?"

    resp = client.post("/rag/answer", json={"query": query})
    assert resp.status_code == 200

    data = resp.json()
    assert "answer" in data
    assert len(data["sources"]) > 0
