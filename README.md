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


