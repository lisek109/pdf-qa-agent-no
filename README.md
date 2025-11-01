# PDF-spørsmål & svar (RAG) – norsk

En lettvektsløsning som besvarer spørsmål direkte fra et PDF-dokument ved hjelp av RAG:
**PyMuPDF → rensing → chunking → embeddings → henting av top-k kontekst → svar (GPT-4o-mini).**

## Krav
- **Python 3.12.x** (anbefalt)
- `pip`, `venv`

> 💡 Merk (Windows): `PyMuPDF==1.24.9` mangler wheel for **Python 3.13**. På 3.13 forsøker `pip` å kompilere fra kilde og feiler ofte med  
> `Exception: Unable to find Visual Studio`. Løsning: bruk **Python 3.12** (enklest), eller installer **Visual Studio C++ Build Tools**.

## Kom i gang

1. Kopier miljøvariabler:
   ```bash
   cp .env.example .env
   # legg inn OPENAI_API_KEY i .env


2. Opprett og aktiver virtuelt miljø (Windows PowerShell):


py -3.12 -m venv .venv
# Hvis du får "running scripts is disabled", se Troubleshooting under.
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
macOS / Linux:

python3.12 -m venv .venv
source .venv/bin/activate

3. Installer avhengigheter:

python -m pip install -U pip setuptools wheel
pip install -r requirements.txt

4. Start UI:

streamlit run main.py


