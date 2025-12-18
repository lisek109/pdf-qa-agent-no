# PDF Q&A (RAG)

A lightweight solution for answering questions directly from PDF documents using
**Retrieval-Augmented Generation (RAG)**.

Pipeline:  
PyMuPDF → cleaning → chunking → embeddings → top-k context → answer (GPT-4o-mini).

## Focus

- Simple and understandable architecture
- Traceability (citations + page numbers)
- Clear path toward cloud deployment (Azure)

---

## Requirements

- Python 3.12.x (recommended)
- pip, venv

> 💡 **Windows note**  
> PyMuPDF==1.24.9 does not provide a wheel for Python 3.13.  
> Use Python 3.12 or install Visual Studio C++ Build Tools.

---

## Tech Stack

### Backend / AI
- Python 3.12
- OpenAI API (LLM + embeddings)
- RAG pipeline (chunking, retrieval, prompt orchestration)
- ChromaDB (vector store)

### Document Processing
- PyMuPDF (fitz)
- Text cleaning and chunking (RecursiveCharacterTextSplitter)

### Frontend
- Streamlit

### DevOps / Cloud 
- Docker
- Azure (App Service / Container Apps)
- Terraform (Infrastructure as Code)
- Azure Blob Storage
- Cosmos DB 

### Version Control
- Git
- GitHub

---

## Getting Started

### Run with Docker (recommended)

#### Build the image
```bash
docker build -t pdf-rag .
```
Run the container
```bash
docker run --rm -p 8501:8501 --env-file .env pdf-rag
```
Open:
http://localhost:8501

Run Locally
Copy environment variables
bash
Copy code
cp .env.example .env
Add your OPENAI_API_KEY to .env.

Create and activate virtual environment
Windows (PowerShell)

powershell
Copy code
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
macOS / Linux

bash
Copy code
python3.12 -m venv .venv
source .venv/bin/activate
Install dependencies
bash
Copy code
pip install -r requirements.txt
Start the application
bash
Copy code
streamlit run main.py