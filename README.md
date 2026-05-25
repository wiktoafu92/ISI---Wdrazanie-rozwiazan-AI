# 🚀 FastAPI + OCR Donut + RAG API

Backend oparty o FastAPI umożliwiający:

- OCR dokumentów przy użyciu modelu Donut,
- generowanie embeddingów,
- wyszukiwanie semantyczne (RAG),
- obsługę zapytań LLM (Groq / OpenAI),

Aplikacja jest w pełni konteneryzowana i gotowa do uruchomienia w Dockerze.

---

# 📚 Spis treści

- [Opis projektu](#-opis-projektu)
- [Technologie](#-technologie)
- [Struktura projektu](#-struktura-projektu)
- [Dockerfile](#-dockerfile--co-to-jest)
- [.dockerignore](#-dockerignore--co-to-jest)
- [Docker Context](#-docker-context--co-to-jest)
- [Warstwy obrazu Docker](#-jak-działają-warstwy-obrazu)
- [Optymalizacja buildów](#-jak-zoptymalizować-czas-budowy-obrazu)
- [Plik .env](#-uwaga-o-env)
- [Model OCR Donut](#-uwaga-o-modelu-ocr-donut)
- [Budowanie obrazu](#-instrukcja-budowania-obrazu)
- [Uruchamianie kontenera](#️-instrukcja-uruchamiania-kontenera)
- [Dostęp do API](#-dostęp-do-api)

---

# 📖 Opis projektu

Projekt udostępnia API do przetwarzania dokumentów z wykorzystaniem nowoczesnych modeli AI.

Funkcjonalności:

- OCR dokumentów,
- ekstrakcja tekstu,
- embeddingi semantyczne,
- wyszukiwanie kontekstowe (RAG),
- integracja z modelami LLM,
- REST API w FastAPI,
- pełna konteneryzacja Docker.

---

# 🧰 Technologie

- Python 3.13
- FastAPI
- Uvicorn
- HuggingFace Transformers
- Donut OCR
- ChromaDB
- OpenAI API
- Groq API
- Docker
- Docker Compose

---

# 📁 Struktura projektu

```bash
project/
│
├── app/
├── uploads/
├── chroma_db/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .dockerignore
└── .env
```

---

# 🧱 Dockerfile — co to jest?

`Dockerfile` to plik tekstowy zawierający instrukcje budowania obrazu Dockera.

Można go traktować jako „przepis” na stworzenie kompletnego środowiska uruchomieniowego aplikacji.

W tym projekcie Dockerfile:

- instaluje zależności Pythona,
- kopiuje kod aplikacji,
- tworzy katalogi `uploads` i `chroma_db`,
- uruchamia FastAPI przez Uvicorn,
- pobiera model OCR Donut przy pierwszym uruchomieniu.

Model OCR trafia do:

```bash
/root/.cache/huggingface/
```

wewnątrz kontenera.

---

# 🗂️ .dockerignore — co to jest?

Plik `.dockerignore` określa, które pliki NIE powinny zostać wysłane do Docker context.

Korzyści:

- szybsze buildy,
- mniejszy obraz,
- brak niepotrzebnych plików w obrazie,
- większe bezpieczeństwo.

Przykładowo ignorujemy:

```bash
__pycache__/
uploads/
chroma_db/
*.pyc
```

## ⚠️ Ważne

Plik `.env` NIE powinien być ignorowany, ponieważ `docker-compose` korzysta z niego do wstrzykiwania zmiennych środowiskowych.

---

# 📦 Docker Context — co to jest?

Docker Context to zestaw plików wysyłanych do Dockera podczas budowania obrazu. To wszystko, co znajduje się w katalogu projektu minus to, co jest w `.dockerignore`.


Dlaczego to ważne?

- większy context = wolniejszy build
- zmiana `.dockerignore` zmienia hash contextu
- zmiana hash = utrata cache

---

# 🧱 Jak działają warstwy obrazu?

Docker buduje obraz warstwa po warstwie.

Każda instrukcja w `Dockerfile` tworzy osobną warstwę:

```dockerfile
FROM python:3.13
RUN apt-get install ...
COPY requirements.txt .
RUN uv pip install -r requirements.txt
COPY app app
```

## Przykład działania cache

- zmiana `requirements.txt`
  → przebudowa warstw instalacji pakietów

- zmiana kodu aplikacji
  → przebudowa tylko ostatniej warstwy

To znacząco przyspiesza pracę.

---

# ⚡ Jak zoptymalizować czas budowy obrazu?

## ✔️ 1. COPY requirements.txt przed COPY app

Dzięki temu instalacja zależności jest cache’owana.

```dockerfile
COPY requirements.txt .
RUN uv pip install -r requirements.txt

COPY app app
```

---

## ✔️ 2. Używaj .dockerignore

Mniejszy context = szybszy build.

---

## ✔️ 3. Nie zmieniaj Dockerfile bez potrzeby

Zmiana instrukcji unieważnia cache od tego miejsca w dół.

---

## ✔️ 4. Nie trzymaj dużych plików w projekcie

Duże pliki spowalniają build.

---

# 🔀 Dlaczego kolejność instrukcji ma znaczenie?

Docker cache działa liniowo.

Przykład:

```dockerfile
COPY requirements.txt .
RUN uv pip install -r requirements.txt
COPY app app
```

Zmiana kodu aplikacji:

✅ przebudowa tylko ostatniej warstwy

Zmiana requirements:

✅ przebudowa instalacji pakietów

Zmiana wcześniejszych warstw:

❌ przebudowa całego obrazu

---

# 🔐 Uwaga o .env

Plik `.env` powinien znajdować się obok `docker-compose.yml`.

Przykład:

```env
HF_TOKEN=hf_xxx
GROQ_API_KEY=gsk_xxx
```

Docker Compose automatycznie przekazuje zmienne środowiskowe do kontenera.

## ⚠️ Ważne

Nie kopiujemy `.env` do obrazu Dockera.

---

# 🧠 Uwaga o modelu OCR Donut

Model Donut pobierany jest automatycznie przy pierwszym uruchomieniu aplikacji.

Lokalizacja cache:

```bash
/root/.cache/huggingface/
```

Korzyści:

- kolejne uruchomienia są szybsze,
- model pozostaje w kontenerze,
- brak potrzeby ręcznego pobierania.

---

# 🛠️ Instrukcja budowania obrazu

## Docker Compose

```bash
docker compose build --no-cache
```

## Docker Build

```bash
docker build -t egzamin-api:latest .
```

---

# ▶️ Instrukcja uruchamiania kontenera

## Docker Compose

```bash
docker compose up -d
```

## Docker Run

```bash
docker run -d \
  -p 8000:8000 \
  --env-file .env \
  --name api \
  egzamin-api:latest
```

---

# 🌐 Dostęp do API

## Swagger UI

```text
http://localhost:8000/docs
```

## OpenAPI JSON

```text
http://localhost:8000/openapi.json
```

---

# 📄 Licencja

Projekt przeznaczony do użytku edukacyjnego i developerskiego.

