"""
Moduł do indeksacji PDF-ów do Chroma.
Zawiera logikę czytania, chunking, klasyfikacji i upsert do wektorowej bazy danych.
"""

import os
import streamlit as st
from app.parsers.pdf import extract_pages
from app.qa.chunking import split_pages_into_chunks
from app.qa.retrieval import cache_key_for_file
from app.qa.vectorstore_chroma import get_client, get_collection, upsert_chunks
from app.classifier.infer import classify_document_ml

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")


def ingest_to_chroma(
    pdf_path: str,
    adaptive_chunking: bool,
    user_id: str,
    force_reindex: bool = False,
):
    """
    Pełny ingest — czytaj, chunk, klasyfikuj, upsert do Chroma.

    Args:
        pdf_path: ścieżka do PDF na dysku
        adaptive_chunking: czy używać adaptacyjnych rozmiarów chunków
        user_id: ID użytkownika (dla metadanych)
        force_reindex: jeśli True, upsert do Chroma nawet jeśli doc key już istnieje

    Returns:
        Tuple (key, filename, chunks, chunks_meta, doc_class, doc_score)
    """
    pages = extract_pages(pdf_path)
    chunks_meta = split_pages_into_chunks(
        pages, size=1200, overlap=180, adaptive=adaptive_chunking
    )
    chunks = [c["content"] for c in chunks_meta]
    st.sidebar.markdown("---")
    st.sidebar.info(f"Antall chunks: {len(chunks)}")
    st.sidebar.code(f"Første chunk (preview):\n{chunks[0][:300]}...")
    st.sidebar.markdown("---")
    print(f"Delte dokumentet i {len(chunks)} chunks.")  # for debugging
    print(f"Første chunk preview: {chunks[0][:200]}...")  # for debugging

    # Klassifiser hele dokumentet (MIN modell)
    doc_preview = " ".join(chunks)[:8000]
    doc_class, doc_score = classify_document_ml(doc_preview)

    # Nøkkel + metadata
    key = cache_key_for_file(pdf_path, EMBED_MODEL, adaptive_chunking)
    print(f"Stabil nøkkel for dokumentet: {key} i ingest_to_chroma")  # for debugging
    filename = os.path.basename(pdf_path)
    metadatas = [
        {
            "user_id": user_id,
            "doc": key,
            "filename": filename,
            "page": c["page"],
            "start": c["start"],
            "end": c["end"],
            "class": doc_class,
            "mode": "adaptive" if adaptive_chunking else "static",
            "chunk_length": len(c["content"]),
        }
        for c in chunks_meta
    ]

    # Upsert til Chroma hvis ikke finnes
    client_ch = get_client(persist_dir="data/chroma")
    coll = get_collection(client_ch, name="pdf_chunks")
    exists = coll.get(where={"doc": key}, limit=1)
    if force_reindex or not exists.get("ids"):
        upsert_chunks(
            coll,
            doc_id=key,
            chunks=chunks,
            metadatas=metadatas,
            api_key=st.session_state.get("openai_api_key", ""),
        )
        print("Indeksering fullført (Chroma).")  # for debugging
    return key, filename, chunks, chunks_meta, doc_class, doc_score
