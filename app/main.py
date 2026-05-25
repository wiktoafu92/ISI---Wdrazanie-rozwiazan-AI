import uuid
import os
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, status

from app.models import (
    Document,
    DocumentStatus,
    UploadResponse,
    DocumentResponse,
    IndexResponse,
    RagSearchRequest,
    RagSearchResponse,
    RagAnswerRequest,
    RagAnswerResponse,
    Settings,
    DonutParsed,
)
from app.ocr import run_donut_ocr
from app.rag import prep_vectorstore, index_document, search_similar, answer_query


app = FastAPI(title="Invoices OCR + RAG API")

settings = Settings()  # type: ignore
os.environ["HF_TOKEN"] = settings.hf_token.get_secret_value()

# In-memory storage
DOCUMENTS: Dict[str, Document] = {}
PARSED: Dict[str, DonutParsed] = {}

UPLOAD_DIR = settings.ROOT.parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

vector_store = prep_vectorstore()


# ---------------- HEALTH ----------------
@app.get("/health")
def health():
    return {"status": "ok"}


# ---------------- BACKGROUND OCR ----------------
def _process_document_background(document_id: str, file_path: Path):
    doc = DOCUMENTS[document_id]

    try:
        DOCUMENTS[document_id] = doc.model_copy(update={"status": DocumentStatus.processing})

        parsed = run_donut_ocr(str(file_path))
        PARSED[document_id] = parsed

        # zapisujemy structured = DonutParsed
        structured = parsed

        raw_text = (
            f"HEADER: {parsed.header.model_dump(exclude_none=True)}\n"
            f"ITEMS: {[i.model_dump(exclude_none=True) for i in parsed.items]}\n"
            f"SUMMARY: {parsed.summary.model_dump(exclude_none=True)}"
        )

        DOCUMENTS[document_id] = doc.model_copy(
            update={
                "status": DocumentStatus.completed,
                "raw_text": raw_text,
                "structured": structured,
            }
        )

    except Exception as e:
        DOCUMENTS[document_id] = doc.model_copy(
            update={
                "status": DocumentStatus.failed,
                "error": str(e),
            }
        )


# ---------------- UPLOAD ----------------
@app.post("/documents/upload",
          response_model=UploadResponse,
          status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    if file.content_type not in ("image/jpeg", "image/jpg", "image/png"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Dozwolone formaty: .jpg, .jpeg, .png",
        )

    document_id = str(uuid.uuid4())
    filename = f"{document_id}_{file.filename}"
    file_path = UPLOAD_DIR / filename

    with open(file_path, "wb") as f:
        f.write(await file.read())

    DOCUMENTS[document_id] = Document(
        id=document_id,
        filename=str(file_path),
        status=DocumentStatus.queued,
    )

    background_tasks.add_task(_process_document_background, document_id, file_path)

    return UploadResponse(document_id=document_id, status=DocumentStatus.queued)


# ---------------- GET DOCUMENT ----------------
@app.get("/documents/{document_id}", response_model=DocumentResponse)
def get_document(document_id: str):
    doc = DOCUMENTS.get(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Brak dokumentu")

    return DocumentResponse(
        id=doc.id,
        status=doc.status,
        raw_text=doc.raw_text,
        structured=doc.structured,
        error=doc.error,
    )


# ---------------- INDEX DOCUMENT ----------------
@app.post("/documents/{document_id}/index", response_model=IndexResponse)
def index_document_endpoint(document_id: str):
    doc = DOCUMENTS.get(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Brak dokumentu")

    if doc.status != DocumentStatus.completed:
        raise HTTPException(
            status_code=409,
            detail=f"Dokument ma status {doc.status}, nie można indeksować.",
        )

    parsed = PARSED.get(document_id)
    if not parsed:
        raise HTTPException(status_code=500, detail="Brak danych OCR")

    index_document(document_id, vector_store, parsed)

    return IndexResponse(document_id=document_id, indexed=True)


# ---------------- RAG SEARCH ----------------
@app.post("/rag/search", response_model=RagSearchResponse)
def rag_search(req: RagSearchRequest):
    results = search_similar(req.query, vector_store, top_k=req.top_k)
    return RagSearchResponse(results=results)


# ---------------- RAG ANSWER ----------------
@app.post("/rag/answer", response_model=RagAnswerResponse)
def rag_answer(req: RagAnswerRequest):
    answer, chunks = answer_query(req.query, vector_store, top_k=req.top_k)
    return RagAnswerResponse(answer=answer, sources=chunks)
