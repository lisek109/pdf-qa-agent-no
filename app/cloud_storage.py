"""
cloud_storage.py

Abstraksjon for lagring av PDF-er og tekst-chunks i skyen.

Mål:
- PDF-filer lagres i Azure Blob Storage (container: 'pdfs').
- Tekst-chunks + embeddings lagres i Azure Cosmos DB (container: 'chunks'),
  med fields som userId, docId, tekst, page, embedding osv.

Denne modulen leser alle nødvendige connection-verdier fra miljøvariabler
som settes av Terraform i Azure Container App:

  BLOB_CONNECTION_STRING
  COSMOS_ENDPOINT
  COSMOS_KEY
  COSMOS_DB
  COSMOS_CONTAINER
"""

from __future__ import annotations

import os
import logging
import uuid
from dataclasses import dataclass
from typing import List, Dict, Optional, Any

import numpy as np
from azure.storage.blob import BlobServiceClient, ContainerClient
from azure.cosmos import CosmosClient, PartitionKey, exceptions as cosmos_exceptions
# ContainerProxy finnes ikke i alle versjoner av azure-cosmos; bruk Any for typing


logger = logging.getLogger(__name__)


# -------------------- Datamodeller --------------------


@dataclass
class DokumentInfo:
  """
  Metadata om et dokument for visning i UI.
  """
  id: str
  navn: str
  filnavn: str
  beskrivelse: Optional[str] = None
  sideantall: Optional[int] = None
  dokumentklasse: Optional[str] = None


@dataclass
class ChunkTreff:
    """
    Representerer ett treff fra et semantisk søk i tekst-chunks.

    Dette er det som sporr_chunks() returnerer til applikasjonen.
    """
    id: str                # Chunk-ID (Cosmos-dokument-ID)
    doc_id: str            # Dokument-ID chunken tilhører
    bruker_id: str         # Eier av dokumentet / chunken
    tekst: str             # Selve tekstinnholdet
    page: Optional[int]    # Sidenummer i PDF-en (hvis tilgjengelig)
    score: float           # Relevans-score fra vector search
    filnavn: Optional[str] = None
    dokumentklasse: Optional[str] = None


# -------------------- Konfigurasjon fra miljø --------------------


# Leser connection info fra miljøvariabler (satt av Terraform i Container App)
_BLOB_CONN_STR = os.getenv("BLOB_CONNECTION_STRING", "")
_COSMOS_ENDPOINT = os.getenv("COSMOS_ENDPOINT", "")
_COSMOS_KEY = os.getenv("COSMOS_KEY", "")
_COSMOS_DB_NAME = os.getenv("COSMOS_DB", "pdfdb")
_COSMOS_CONTAINER_NAME = os.getenv("COSMOS_CONTAINER", "chunks")

_BLOB_CONTAINER_NAME = "pdfs"


def _has_blob_config() -> bool:
    """
    Sjekker om vi har nok info til å koble oss til Blob Storage.
    """
    return bool(_BLOB_CONN_STR)


def _has_cosmos_config() -> bool:
    """
    Sjekker om vi har nok info til å koble oss til Cosmos DB.
    """
    return bool(_COSMOS_ENDPOINT and _COSMOS_KEY and _COSMOS_DB_NAME and _COSMOS_CONTAINER_NAME)


# -------------------- Klient-initialisering (lazy) --------------------

_blob_container_client: Optional[ContainerClient] = None
_cosmos_container_client: Optional[Any] = None


def _get_blob_container():
    """
    Returnerer en BlobContainerClient mot containeren der PDF-er lagres.
    Opprettes første gang funksjonen kalles (lazy init).
    """
    global _blob_container_client

    if not _has_blob_config():
        raise RuntimeError("Blob-konfigurasjon mangler (BLOB_CONNECTION_STRING).")

    if _blob_container_client is None:
        service = BlobServiceClient.from_connection_string(_BLOB_CONN_STR)
        container = service.get_container_client(_BLOB_CONTAINER_NAME)
        try:
            container.create_container()
        except Exception:
            # Containeren finnes sannsynligvis fra før – det er OK
            pass
        _blob_container_client = container

    return _blob_container_client


def _get_cosmos_container():
    """
    Returnerer en Cosmos-container-klient for chunks og dokumentmetadata.
    Opprettes første gang funksjonen kalles.
    """
    global _cosmos_container_client

    if not _has_cosmos_config():
        raise RuntimeError(
            "Cosmos-konfigurasjon mangler (COSMOS_ENDPOINT / COSMOS_KEY / COSMOS_DB / COSMOS_CONTAINER)."
        )

    if _cosmos_container_client is None:
        client = CosmosClient(_COSMOS_ENDPOINT, credential=_COSMOS_KEY)

        # Opprett database hvis den ikke finnes
        try:
            db = client.create_database_if_not_exists(id=_COSMOS_DB_NAME)
        except cosmos_exceptions.CosmosHttpResponseError as e:
            logger.error("Feil ved opprettelse av Cosmos-database: %s", e)
            raise

        # Opprett container hvis den ikke finnes
        try:
            container = db.create_container_if_not_exists(
                id=_COSMOS_CONTAINER_NAME,
                partition_key=PartitionKey(path="/userId"),
            )
        except cosmos_exceptions.CosmosHttpResponseError as e:
            logger.error("Feil ved opprettelse av Cosmos-container: %s", e)
            raise

        _cosmos_container_client = container

    return _cosmos_container_client

# -------------------- API-funksjoner brukt av main.py --------------------

def lagre_pdf(bruker_id: str, filnavn: str, data: bytes) -> str:
    """
    Lagre en PDF for en gitt bruker i Blob Storage og returner et dokument-ID.

    Flyt:
      1. Generer et dokument-ID (UUID).
      2. Bygg en blob-sti: "<userId>/<documentId>.pdf".
      3. Last opp filen til Azure Blob Storage.
      4. Opprett et metadata-dokument i Cosmos (type="document").
    """
    if not data:
        raise ValueError("Tomme PDF-data kan ikke lagres.")

    dokument_id = str(uuid.uuid4())
    blob_path = f"{bruker_id}/{dokument_id}.pdf"

    # Last opp PDF til Blob
    container = _get_blob_container()
    logger.info("Lagrer PDF til Blob: bruker_id=%s, blob_path=%s", bruker_id, blob_path)
    container.upload_blob(name=blob_path, data=data, overwrite=True)

    # Lagre metadata i Cosmos (valgfritt, men nyttig for UI)
    try:
        container_cosmos = _get_cosmos_container()
        meta_doc = {
            "id": dokument_id,
            "userId": bruker_id,
            "type": "document",
            "filnavn": filnavn,
            "blobPath": blob_path,
            "dokumentklasse": None,
            "sideantall": None,
        }
        container_cosmos.upsert_item(meta_doc)
    except Exception as e:
        logger.warning("Klarte ikke å lagre dokument-metadata i Cosmos: %s", e)

    return dokument_id


def hent_pdf(bruker_id: str, dokument_id: str) -> bytes:
    """
    Hent en PDF for en gitt bruker og dokument-ID fra Blob Storage.

    Antatt blob-sti: "<userId>/<documentId>.pdf".
    """
    blob_path = f"{bruker_id}/{dokument_id}.pdf"
    container = _get_blob_container()

    logger.info("Henter PDF fra Blob: bruker_id=%s, blob_path=%s", bruker_id, blob_path)
    try:
        blob_client = container.get_blob_client(blob_path)
        data = blob_client.download_blob().readall()
        return data
    except Exception as e:
        logger.error("Klarte ikke å hente PDF fra Blob: %s", e)
        raise
      
      # -------------------- Dokumentliste (Cosmos) --------------------



def list_bruker_dokumenter(bruker_id: str) -> List[Dict]:
    """
    Returner en liste over dokumenter som tilhører en gitt bruker.

    Leser metadata fra Cosmos der:
      - userId = <bruker_id>
      - type   = "document"
    """
    container = _get_cosmos_container()

    query = """
    SELECT c.id, c.filnavn, c.dokumentklasse, c.sideantall
    FROM c
    WHERE c.userId = @userId AND c.type = "document"
    ORDER BY c.filnavn
    """

    items = list(
        container.query_items(
            query=query,
            parameters=[{"name": "@userId", "value": bruker_id}],
            enable_cross_partition_query=False,
        )
    )

    result: List[Dict] = []
    for it in items:
        result.append(
            {
                "id": it.get("id"),
                "navn": it.get("filnavn") or it.get("id"),
                "filnavn": it.get("filnavn") or "",
                "dokumentklasse": it.get("dokumentklasse"),
                "sideantall": it.get("sideantall"),
            }
        )

    return result
  
  
# -------------------- Lagring av chunks + embeddings (Cosmos) --------------------


def lagre_chunks(
    bruker_id: str,
    dokument_id: str,
    chunks_med_embeddings: List[Dict],
) -> None:
    """
    Lagre tekst-chunks + embeddings for et dokument i Cosmos.

    Forventet struktur for hver chunk i chunks_med_embeddings:
      {
        "chunk_index": int,
        "tekst": str,
        "page": int | None,
        "embedding": List[float],
        "dokumentklasse": str | None,
        "filnavn": str | None
      }
    """
    if not chunks_med_embeddings:
        logger.info("Ingen chunks å lagre for dokument_id=%s", dokument_id)
        return 0

    container = _get_cosmos_container()

    for ch in chunks_med_embeddings:
        chunk_id = str(uuid.uuid4())
        item = {
            "id": chunk_id,
            "type": "chunk",
            "userId": bruker_id,
            "docId": dokument_id,
            "chunkIndex": int(ch.get("chunk_index", 0)),
            "tekst": ch.get("tekst", ""),
            "page": ch.get("page"),
            "embedding": ch.get("embedding", []),
            "filnavn": ch.get("filnavn"),
            "dokumentklasse": ch.get("dokumentklasse"),
        }
        container.upsert_item(item)

    logger.info(
        "Lagret %d chunks i Cosmos for bruker_id=%s, dokument_id=%s",
        len(chunks_med_embeddings),
        bruker_id,
        dokument_id,
    )
    return len(chunks_med_embeddings)
    # -------------------- Semantisk søk (naiv vector search i Python) --------------------


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Enkel kosinuslikhet mellom to vektorer.
    """
    if a.shape != b.shape:
        raise ValueError("Vektorene må ha samme dimensjon.")

    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def sporr_chunks(
    bruker_id: str,
    sporsmal_embedding: List[float],
    top_k: int = 5,
    dokument_id: Optional[str] = None,
) -> List[Dict]:
    """
    Gjør et semantisk søk i chunks for en gitt bruker.

    NB: Denne varianten forventer at embedding for spørsmålet
    allerede er generert i den kallende koden (samme embed-modell).

    Flyt:
      1. Hent alle chunks for bruker_id (og ev. docId).
      2. Beregn kosinuslikhet mellom spørsmåls-vektoren og hver chunk-embedding.
      3. Sorter på score (synkende) og returner de top_k beste.

    Dette er en enkel, Python-basert løsning som fungerer fint for små datamengder,
    men som kan erstattes av innebygd vector search i Cosmos senere.
    """
    container = _get_cosmos_container()

    # Bygg query med filter på bruker og ev. dokument
    if dokument_id:
        query = """
        SELECT c.id, c.docId, c.tekst, c.page, c.embedding, c.filnavn, c.dokumentklasse
        FROM c
        WHERE c.userId = @userId AND c.type = "chunk" AND c.docId = @docId
        """
        params = [
            {"name": "@userId", "value": bruker_id},
            {"name": "@docId", "value": dokument_id},
        ]
    else:
        query = """
        SELECT c.id, c.docId, c.tekst, c.page, c.embedding, c.filnavn, c.dokumentklasse
        FROM c
        WHERE c.userId = @userId AND c.type = "chunk"
        """
        params = [
            {"name": "@userId", "value": bruker_id},
        ]

    items = list(
        container.query_items(
            query=query,
            parameters=params,
            enable_cross_partition_query=False,
        )
    )

    if not items:
        return []

    q_vec = np.array(sporsmal_embedding, dtype=float)

    treff: List[ChunkTreff] = []
    for it in items:
        emb = np.array(it.get("embedding", []), dtype=float)
        if emb.size == 0:
            score = 0.0
        else:
            score = _cosine_similarity(q_vec, emb)

        treff.append(
            ChunkTreff(
                id=it.get("id"),
                doc_id=it.get("docId"),
                bruker_id=bruker_id,
                tekst=it.get("tekst", ""),
                page=it.get("page"),
                score=score,
                filnavn=it.get("filnavn"),
                dokumentklasse=it.get("dokumentklasse"),
            )
        )

    # Sorter på score (høyest først) og klipp til top_k
    treff_sorted = sorted(treff, key=lambda t: t.score, reverse=True)[:top_k]

    # Returner som "plain dicts" til resten av appen
    return [
        {
            "id": t.id,
            "docId": t.doc_id,
            "bruker_id": t.bruker_id,
            "tekst": t.tekst,
            "page": t.page,
            "score": t.score,
            "filnavn": t.filnavn,
            "dokumentklasse": t.dokumentklasse,
        }
        for t in treff_sorted
    ]