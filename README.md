RAG-based PDF Assistant (Retrieval-Augmented Generation Agent)
This project implements a robust, multi-tenant PDF Question-Answering (QA) agent utilizing the Retrieval-Augmented Generation (RAG) architectural pattern. The agent is designed to read text from uploaded PDF documents, process the data, and answer user queries in natural language, providing precise, citation-backed evidence,. The primary goal is to streamline information retrieval in large collections of unstructured documents, common in sectors like construction (e.g., contracts, plans, and cost documents),,,.
The application was developed as a final academic report, demonstrating mastery of various concepts, including AI/ML engineering, cloud infrastructure (Azure), and Infrastructure as Code (IaC),,.

--------------------------------------------------------------------------------
🎓 Note on Project Scope and Educational Value
This solution deliberately incorporates dual functionality and complexity (Versions V1-V5) to maximize educational breadth and demonstrate a deep, "under the hood" understanding of AI agent development,.
• Learning Focus: The solution includes two retrieval modes: a basic local system (NumPy cache/ChromaDB) and a scalable cloud system (Azure),. The inclusion of the local NumPy mode served a pedagogical purpose: to clearly illustrate the principles of vector search and cosine similarity without relying entirely on external database abstractions.
• Engineering Depth: Rather than relying solely on high-level libraries, the project features core implementation details, such as the use of Terraform for infrastructure and object-oriented programming (OOP) in Python,. This demonstrates the ability to control and debug complex systems from the infrastructure layer right up to the RAG pipeline.

--------------------------------------------------------------------------------
Key Features and Technical Highlights
The RAG agent implements a comprehensive pipeline: PyMuPDF → data cleaning → chunking → embeddings → context retrieval → answer generation (GPT-4o-mini),.
Cloud Architecture and Multi-Tenant Security
The production-ready version (V5) is built on a scalable multi-tenant architecture deployed in Microsoft Azure,.
• Infrastructure as Code (IaC): All cloud resources are provisioned and managed using Terraform, which ensures that the entire environment is reproducible, trackable, and version-controlled. Terraform injects necessary secrets and configuration into the application running in the Azure Container App.
• Data Isolation: Security and data separation are paramount. The system uses userId as the primary partition key in Azure Cosmos DB (where vector embeddings and chunks are stored) and in the path structure of Azure Blob Storage (where the raw PDF files reside),,. This robust filtering ensures each user only accesses their own documents.
Advanced Retrieval and Processing
The project includes advanced techniques to enhance retrieval accuracy:
• Adaptive Chunking: The agent utilizes the LangChain RecursiveCharacterTextSplitter for robust text segmentation. Furthermore, it implements Adaptive Chunking logic, which dynamically adjusts the size of the text segments based on the overall length of the PDF page, leading to improved context retrieval for diverse document types,.
• Citation and Transparency: The solution is strictly engineered to answer questions only based on the context retrieved from the uploaded documents (RAG),. Every generated answer is supported by citations, showing the specific document name and page number from which the information was sourced,.
Customization and ML Integration
• Prompt Engineering: The application's UI allows the user to dynamically adjust the System Prompt, which governs the tone and style of the LLM's output (e.g., precise scientific language or a specific creative format).
• ML Filtering: The solution integrates a simple, custom-trained Machine Learning model (TF-IDF + Logistic Regression) for automatic document classification,,. This model classifies the intent of the user's query, allowing the system to filter the vector search against only relevant document types (e.g., searching only 'contracts' when asked about payment terms), thereby increasing precision and efficiency.
Technical Stack
• Language: Python 3.12.x
• Frontend/UI: Streamlit
• RAG Components: PyMuPDF, LangChain RecursiveCharacterTextSplitter, OpenAI Embeddings (text-embedding-3-small), GPT-4o-mini,.
• Local Storage (Dev/Learning): ChromaDB, NumPy/File-cache,.
• Cloud Backend (V5): Azure Cosmos DB for NoSQL, Azure Blob Storage,.
• Infrastructure as Code (IaC): Terraform.
• Containerization: Docker (used for deployment to Azure Container Apps),.
Getting Started (Local Development)
The following steps outline how to run the application locally using the Streamlit UI.
Prerequisites
• Python 3.12.x (Recommended, due to PyMuPDF compatibility issues with 3.13).
• pip, venv
• OpenAI API Key
Setup and Running
1. Copy environment variables:
2. Create and activate a virtual environment:
    ◦ Windows PowerShell:
    ◦ macOS / Linux:
3. Install dependencies:
4. Start the Streamlit UI:
Deployment via Docker and Azure
The solution is containerized using Docker to facilitate deployment to cloud environments such as Azure Container Apps. The entire cloud infrastructure is defined using Terraform, which automatically provisions the necessary services (Cosmos DB, Blob Storage) and injects secure connection details as environment variables into the running container