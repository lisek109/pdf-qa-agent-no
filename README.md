"# PDF-sporsm�l & svar (RAG) - norsk" 

En lettvekts løsning som besvarer spørsmål direkte fra et PDF-dokument ved hjelp av RAG:
PyMuPDF → rensing → chunking → embeddings → henting av top-k kontekst → svar (GPT-4o-mini).

## Kom i gang
1. Opprett `.env` fra `.env.example` og legg inn `OPENAI_API_KEY`.
2. Installer avhengigheter: `pip install -r requirements.txt`.
3. Start UI: `streamlit run app/main.py`.

## Status
MVP: ett dokument, enkel chat, kilde-sitat (fragment).
Neste: historikk, sidetall, pgvector/Azure, Terraform.