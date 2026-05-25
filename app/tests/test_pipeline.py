import json
import os
from pathlib import Path

from app.models import Settings
from app.ocr import run_donut_ocr
from app.rag import prep_vectorstore, index_document, answer_query


def test_pipeline():
    print("\n=== ŁADOWANIE USTAWIEŃ ===")
    settings = Settings()  # type: ignore
    os.environ["HF_TOKEN"] = settings.hf_token.get_secret_value()

    image_path = Path(settings.file_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Brak pliku obrazu: {image_path}")

    print(f"Używany plik OCR: {image_path}")

    # ---------------------------------------------------------
    # 1. OCR Donut
    # ---------------------------------------------------------
    print("\n=== OCR DONUT ===")
    parsed = run_donut_ocr(str(image_path))

    print("\n--- Wynik OCR (JSON) ---")
    print(json.dumps(parsed.model_dump(), indent=2, ensure_ascii=False))

    # ---------------------------------------------------------
    # 2. Indeksowanie dokumentu
    # ---------------------------------------------------------
    print("\n=== TWORZENIE VECTOR STORE ===")
    vector_store = prep_vectorstore()

    print("\n=== INDEKSOWANIE ===")
    doc_id = "test-doc-1"
    index_document(doc_id, vector_store, parsed)
    print("Indeksowanie zakończone.")

    # ---------------------------------------------------------
    # 3. Zapytanie RAG
    # ---------------------------------------------------------
    print("\n=== RAG: TEST PYTANIA ===")
    query = "Ile przedmiotów jest na fakturze?"
    answer, chunks = answer_query(query, vector_store, top_k=10)

    print("\n--- Odpowiedź modelu ---")
    print(answer)

    print("\n--- Użyte fragmenty ---")
    for c in chunks:
        print(f"[{c.document_id}] score={c.score:.3f} → {c.text}")


if __name__ == "__main__":
    test_pipeline()
