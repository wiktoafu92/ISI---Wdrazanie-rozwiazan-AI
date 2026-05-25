from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, SecretStr
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    ROOT: Path = Path(__file__).resolve().parent

    model_ocr: str = "katanaml-org/invoices-donut-model-v1"
    local_path_ocr: str = str(ROOT / "donut")

    model_embeddings: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    hf_token: SecretStr

    groq_api_key: SecretStr
    model_llm: str = "openai/gpt-oss-120b"
    base_url_llm: str = "https://api.groq.com/openai/v1"
    temperature: float = 0.0

    file_path: str = str(ROOT.parent / "fv2.jpg")

    persist_directory: str = "chroma_db"

    class Config:
        env_file = str(Path(__file__).resolve().parent.parent / ".env")


class DonutHeader(BaseModel):
    invoice_no: Optional[str] = None
    invoice_date: Optional[str] = None
    seller: Optional[str] = None
    client: Optional[str] = None
    seller_tax_id: Optional[str] = None
    client_tax_id: Optional[str] = None
    iban: Optional[str] = None


class DonutItem(BaseModel):
    item_desc: Optional[str] = None
    item_qty: Optional[float] = None
    item_net_price: Optional[float] = None
    item_net_worth: Optional[float] = None
    item_vat: Optional[float] = None
    item_gross_worth: Optional[float] = None


class DonutSummary(BaseModel):
    total_items: int = 0
    total_net_worth: Optional[float] = None
    total_vat: Optional[float] = None
    total_gross_worth: Optional[float] = None


class DonutParsed(BaseModel):
    header: DonutHeader
    items: List[DonutItem]
    summary: DonutSummary


class DocumentStatus(str, Enum):
    queued = "queued"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class Document(BaseModel):
    id: str
    filename: str
    status: DocumentStatus
    error: Optional[str] = None
    raw_text: Optional[str] = None
    structured: Optional[DonutParsed] = None


class UploadResponse(BaseModel):
    document_id: str
    status: DocumentStatus


class DocumentResponse(BaseModel):
    id: str
    status: DocumentStatus
    raw_text: Optional[str] = None
    structured: Optional[DonutParsed] = None
    error: Optional[str] = None


class IndexResponse(BaseModel):
    document_id: str
    indexed: bool


class RagSearchRequest(BaseModel):
    query: str
    top_k: int = 5


class RagSearchResultChunk(BaseModel):
    document_id: str
    score: float
    text: str
    metadata: Dict[str, Any]


class RagSearchResponse(BaseModel):
    results: List[RagSearchResultChunk]


class RagAnswerRequest(BaseModel):
    query: str
    top_k: int = 5


class RagAnswerResponse(BaseModel):
    answer: str
    sources: List[RagSearchResultChunk]
