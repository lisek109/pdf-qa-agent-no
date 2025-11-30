# Base image med Python (slim for mindre størrelse)
FROM python:3.12-slim

# Sett miljøvariabler for å unngå buffering og interaktive prompts
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Installer systemavhengigheter (for eksempel for PyMuPDF, etc.)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libglib2.0-0 \
        libgl1 \
        libxrender1 \
        libxext6 \
        libsm6 \
        curl && \
    rm -rf /var/lib/apt/lists/*

# Arbeidskatalog i containeren
WORKDIR /app

# Kopier requirements først for bedre caching
COPY requirements.txt /app/

# Installer Python-avhengigheter
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Kopier resten av prosjektet inn i containeren
COPY . /app/

# Streamlit standardport
ENV PORT=8501

# Streamlit-konfig for å lytte på alle interfacer i containeren
ENV STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Miljøvariabel for å styre lagrings-backend (local/cloud)  !!!!!!!!!!!!!!OOOBBBBBSSS!!!!!!!!!
# I utvikling kan du overstyre denne, i Azure settes den til "cloud"- husk på det!!!!!!!
ENV BACKEND_MODE=local

# Eksponer porten som Streamlit bruker
EXPOSE 8501

# Start kommando – kjør Streamlit-appen
CMD ["streamlit", "run", "app/main.py"]