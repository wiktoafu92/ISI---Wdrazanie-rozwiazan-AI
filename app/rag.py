from typing import List, Tuple
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document

from app.models import RagSearchResultChunk, Settings, DonutParsed


@lru_cache(maxsize=1)
def prep_vectorstore() -> Chroma:
    settings = Settings()  # type: ignore

    embeddings = HuggingFaceEmbeddings(
        model_name=settings.model_embeddings
    )
    vectorstore = Chroma(
        collection_name="invoices",
        embedding_function=embeddings,
        persist_directory=str(settings.ROOT / "chroma_db"),
    )
    return vectorstore


def index_document(document_id: str, vector_store: Chroma, parsed: DonutParsed):
    docs: List[Document] = []

    # HEADER
    header_dict = parsed.header.model_dump(exclude_none=True)
    for key, value in header_dict.items():
        docs.append(
            Document(
                page_content=f"header.{key}: {value}",
                metadata={
                    "document_id": document_id,
                    "section": "header",
                    "json_key": key,
                    "value_type": type(value).__name__,
                },
            )
        )

    # ITEMS
    for idx, item in enumerate(parsed.items):
        item_id = f"item_{idx + 1}"
        item_dict = item.model_dump(exclude_none=True)
        for key, value in item_dict.items():
            docs.append(
                Document(
                    page_content=f"{item_id}.{key}: {value}",
                    metadata={
                        "document_id": document_id,
                        "section": "item",
                        "item_id": item_id,
                        "json_key": key,
                        "value_type": type(value).__name__,
                    },
                )
            )

    # SUMMARY
    summary_dict = parsed.summary.model_dump(exclude_none=True)
    for key, value in summary_dict.items():
        docs.append(
            Document(
                page_content=f"summary.{key}: {value}",
                metadata={
                    "document_id": document_id,
                    "section": "summary",
                    "json_key": key,
                    "value_type": type(value).__name__,
                },
            )
        )

    vector_store.add_documents(docs)


def search_similar(query: str, vector_store: Chroma, top_k: int = 10) -> List[RagSearchResultChunk]:
    results = vector_store.similarity_search_with_score(query, k=top_k)
    out: List[RagSearchResultChunk] = []
    for doc, score in results:
        out.append(
            RagSearchResultChunk(
                document_id=doc.metadata.get("document_id", ""),
                score=float(score),
                text=doc.page_content,
                metadata=doc.metadata,
            )
        )
    return out


def answer_query(
    query: str, vector_store: Chroma, top_k: int = 10
) -> Tuple[str, List[RagSearchResultChunk]]:
    settings = Settings()  # type: ignore

    chunks = search_similar(query, vector_store, top_k=top_k)

    context = "\n\n".join(
        [f"[doc {c.document_id}, score={c.score:.3f}]\n{c.text}" for c in chunks]
    )

    llm = ChatOpenAI(
        model=settings.model_llm,
        api_key=settings.groq_api_key.get_secret_value(),  # type: ignore
        base_url=settings.base_url_llm,
        temperature=settings.temperature,
    )

    system_prompt = (
        "Jesteś asystentem analizującym faktury i paragony. "
        "Odpowiadasz krótko, po polsku, używając wyłącznie informacji z kontekstu. "
        "Jeśli czegoś nie ma w kontekście – powiedz, że brak danych."
    )

    user_prompt = f"PYTANIE: {query}\n\nKONTEKST:\n{context}"

    resp = llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    return resp.content, chunks
