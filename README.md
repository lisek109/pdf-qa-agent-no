# 📄 PDF Q&A (RAG)

A lightweight application for answering questions directly from PDF documents using Retrieval-Augmented Generation (RAG).

**Pipeline:** PyMuPDF → text cleaning → chunking → embeddings → top-k retrieval → GPT-4o-mini answer generation

---

## 🎯 Focus

- **Simple and understandable architecture**
- **Traceability** (citations + page numbers)
- **Clear path toward cloud deployment** (Azure)

---

## 🧰 Requirements

- Python 3.12.x (recommended)
- pip, venv

> **💡 Windows note:** PyMuPDF==1.24.9 does not provide a wheel for Python 3.13. Use Python 3.12 or install Visual Studio C++ Build Tools.

---

## 🏗️ Tech Stack

### Backend / AI
- Python 3.12
- OpenAI API (LLM + embeddings)
- RAG pipeline (chunking, retrieval, prompt orchestration)
- ChromaDB (vector store)

### Document Processing
- PyMuPDF (fitz)
- RecursiveCharacterTextSplitter (chunking + cleaning)

### Frontend
- Streamlit

### DevOps / Cloud
- Docker
- Azure App Service / Azure Container Apps
- Terraform (Infrastructure as Code)
- Azure Blob Storage
- Cosmos DB

### Version Control
- Git
- GitHub

---

## 🚀 Getting Started

### Run with Docker (recommended)

#### 1. Build the image
```bash
docker build -t pdf-rag .
```

#### 2. Run the container
```bash
docker run --rm -p 8501:8501 --env-file .env pdf-rag
```

#### 3. Open the app
```
http://localhost:8501
```

---

### 🖥️ Run Locally

#### 1. Copy environment variables
```bash
cp .env.example .env
```
Add your `OPENAI_API_KEY` to `.env`.

#### 2. Create and activate virtual environment

**Windows (PowerShell)**
```powershell
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

#### 3. Install dependencies
```bash
pip install -r requirements.txt
```

#### 4. Start the application
```bash
streamlit run main.py
```

---

## 🐳 Example Dockerfile

```dockerfile
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
```

---

## 📦 Project Structure

```
pdf-rag/
│
├── main.py
├── rag/
│   ├── loader.py
│   ├── splitter.py
│   ├── embeddings.py
│   ├── retriever.py
│   └── answer.py
│
├── requirements.txt
├── Dockerfile
├── .env.example
└── README.md
```

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📧 Contact

For questions or feedback, please open an issue on GitHub.
