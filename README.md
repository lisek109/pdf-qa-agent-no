PDF Q&A (RAG) – English

A lightweight solution for answering questions directly from PDF documents using Retrieval-Augmented Generation (RAG):
PyMuPDF → cleaning → chunking → embeddings → top-k context → answer (GPT-4o-mini).

Focus: simple and understandable architecture, traceability (citations + page numbers), and a clear path toward cloud deployment (Azure).

Requirements

Python 3.12.x (recommended)

pip, venv

💡 Note (Windows):
PyMuPDF==1.24.9 does not provide a wheel for Python 3.13. On 3.13, pip tries to compile from source and often fails with:

Exception: Unable to find Visual Studio


Solution: use Python 3.12 (recommended), or install Visual Studio C++ Build Tools.

Tech Stack
Backend / AI

Python 3.12

OpenAI API (LLM + embeddings)

RAG pipeline (chunking, retrieval, prompt orchestration)

Vector store: ChromaDB (local cache alternative in MVP phase)

Document Processing

PyMuPDF (fitz) – PDF text extraction

Text cleaning + chunking (RecursiveCharacterTextSplitter)

Frontend

Streamlit – UI for file upload, questions, answers, and citations

DevOps / Cloud (planned / roadmap)

Docker – containerization

Azure – deployment/hosting (e.g., App Service / Container Apps)

Terraform – Infrastructure as Code

Azure Blob Storage – PDF storage

Cosmos DB – metadata, document catalog, user/session data (planned)

Version Control

Git + GitHub (commit history, issues, CI-ready structure)

Getting Started
Run with Docker (recommended)
1) Build the image
docker build -t pdf-rag-en .

2) Run the container
docker run --rm -p 8501:8501 --env-file .env pdf-rag-en


Then open:
👉 http://localhost:8501

Example Dockerfile
FROM python:3.12-slim

WORKDIR /app

# System packages (optional, but useful for some PDF dependencies)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "main.py", "--server.address=0.0.0.0", "--server.port=8501"]

Run Locally
1) Copy environment variables
cp .env.example .env
# Add your OPENAI_API_KEY to .env

2) Create and activate a virtual environment
Windows (PowerShell)
py -3.12 -m venv .venv
# If you get "running scripts is disabled", see Troubleshooting below
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

macOS / Linux
python3.12 -m venv .venv
source .venv/bin/activate

3) Install dependencies
python -m pip install -U pip setuptools wheel
pip install -r requirements.txt

4) Start the UI
streamlit run main.py

Notes

The application provides answer traceability by showing citations and page numbers.

Designed as an MVP with a clear upgrade path to full cloud deployment on Azure.

Architecture intentionally kept simple for clarity and maintainability.






# PDF-spørsmål & svar (RAG) – norsk

En lettvektsløsning som besvarer spørsmål direkte fra PDF-dokumenter ved hjelp av RAG (Retrieval-Augmented Generation):
**PyMuPDF → rensing → chunking → embeddings → top-k kontekst → svar (GPT-4o-mini)**.

Fokus: enkel og forståelig arkitektur, sporbarhet (sitater + side), og tydelig vei mot skyløsning (Azure).

## Krav
- **Python 3.12.x** (anbefalt)
- `pip`, `venv`

> 💡 Merk (Windows): `PyMuPDF==1.24.9` mangler wheel for **Python 3.13**. På 3.13 forsøker `pip` å kompilere fra kilde og feiler ofte med  
> `Exception: Unable to find Visual Studio`. Løsning: bruk **Python 3.12** (enklest), eller installer **Visual Studio C++ Build Tools**.


## Teknologier og verktøy (Tech stack)

**Backend / AI**
- Python 3.12
- OpenAI API (LLM + embeddings)
- RAG-pipeline (chunking + retrieval + promptstyring)
- Vector store: **Chroma DB** (alternativt lokal cache i MVP-fase)

**Dokumentbehandling**
- PyMuPDF (fitz) – tekstuttrekk fra PDF
- Tekstrensing + chunking (RecursiveCharacterTextSplitter)

**Frontend**
- Streamlit (UI for opplasting, spørsmål, svar og sitater)

**DevOps / Cloud (planlagt / del av videre arbeid)**
- Docker (containerisering)
- Azure (deploy/hosting – f.eks. App Service / Container Apps)
- Terraform (IaC – opprette ressurser)
- Azure Blob Storage (lagring av PDF-er)
- Cosmos DB (metadata, dokumentkatalog, bruker-/sesjonsdata) *(planlagt / roadmap)*

**Versjonskontroll**
- Git + GitHub (commit-historikk, issues, CI-ready struktur)



## Kom i gang

## Kjør med Docker (anbefalt)

### 1) Bygg image
```bash
docker build -t pdf-rag-no .

Kjør container
docker run --rm -p 8501:8501 --env-file .env pdf-rag-no
Åpne deretter: http://localhost:8501

### Eksempel Dockerfile
```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Systempakker (valgfritt, men nyttig for enkelte PDF-avhengigheter)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "main.py", "--server.address=0.0.0.0", "--server.port=8501"]



## Kjør lokalt
1. Kopier miljøvariabler:
   ```bash
   cp .env.example .env
   # legg inn OPENAI_API_KEY i .env

2. Opprett og aktiver virtuelt miljø :

Windows PowerShell

py -3.12 -m venv .venv
# Hvis du får "running scripts is disabled", se Troubleshooting under.
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
macOS / Linux:

python3.12 -m venv .venv
source .venv/bin/activate

macOS / Linux

python3.12 -m venv .venv
source .venv/bin/activate

3. Installer avhengigheter:

python -m pip install -U pip setuptools wheel
pip install -r requirements.txt

4. Start UI:

streamlit run main.py


