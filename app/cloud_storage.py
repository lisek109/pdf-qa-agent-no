"""
Abstraksjon for lagring av PDF-er og embeddings i skyen.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class DokumentInfo:
    """
    Metadata om et dokument for visning i UI.

    Dette vil typisk komme fra en Cosmos-kolleksjon for dokumenter,
    eller fra en egen metadata-oversikt koblet til Blob Storage.
    """
    id: str                # Dokument-ID (f.eks. UUID)
    navn: str              # Visningsnavn (typisk originalt filnavn)
    filnavn: str           # Faktisk filnavn (som lagret i blob)
    beskrivelse: Optional[str] = None
    sideantall: Optional[int] = None
    dokumentklasse: Optional[str] = None



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



def lagre_pdf(bruker_id: str, filnavn: str, data: bytes) -> str:
    """
    Lagre en PDF for en gitt bruker og returner et dokument-ID.

    I en ekte sky-implementasjon vil dette:
    - legge filen i Azure Blob Storage (f.eks. i en container 'pdfs'),
    - bruke en sti/nøkkel som inkluderer bruker_id og dokument_id,
    - opprette eller oppdatere dokumentmetadata i Cosmos DB, f.eks.:

      {
        "id": "<docId>",
        "userId": "<bruker_id>",
        "filnavn": "<filnavn>",
        "blobPath": "pdfs/<bruker_id>/<docId>.pdf",
        "dokumentklasse": null,
        "sideantall": null,
        "type": "document"
      }

    :param bruker_id: Unik ID for innlogget bruker (fra autentisering).
    :param filnavn: Originalt filnavn som ble lastet opp.
    :param data: Rå bytes fra PDF-filen.
    :return: Generert dokument-ID (f.eks. en UUID).
    """
    raise NotImplementedError("lagre_pdf er ikke implementert ennå (planlagt Azure Blob-lagring + Cosmos-metadata).")


def hent_pdf(bruker_id: str, dokument_id: str) -> bytes:
    """
    Hent en PDF for en gitt bruker og dokument-ID.

    Typisk flyt i Azure:
    - Les dokument-metadata fra Cosmos DB for å finne blob-sti,
    - Valider at userId matcher innlogget bruker (multi-tenant sikkerhet),
    - Les filen fra Blob Storage og returner innholdet som bytes.

    :param bruker_id: Unik ID for innlogget bruker.
    :param dokument_id: ID som ble returnert fra lagre_pdf.
    :return: PDF-innhold som bytes.
    """
    raise NotImplementedError("hent_pdf er ikke implementert ennå (planlagt Azure Blob-lesing).")


def list_bruker_dokumenter(bruker_id: str) -> List[Dict]:
    """
    Returner en liste over dokumenter som tilhører en gitt bruker.

    Hver entry bør minst inneholde:
      {
        "id": "<docId>",
        "navn": "<visningsnavn i UI>",
        "filnavn": "<originalt filnavn>",
        "dokumentklasse": "<klassifisering>" (valgfritt)
      }

    I Cosmos DB kan dette f.eks. være en spørring:

      SELECT c.id, c.filnavn, c.dokumentklasse
      FROM c
      WHERE c.userId = @bruker_id AND c.type = "document"

    Partition key vil typisk være /userId, slik at alle dokumenter for en bruker
    ligger i samme partisjon.

    :param bruker_id: Unik ID for innlogget bruker.
    :return: Liste med ordbøker som beskriver dokumentene.
    """
    raise NotImplementedError("list_bruker_dokumenter er ikke implementert ennå (planlagt Cosmos-spørring på dokumenter).")


def lagre_chunks(
    bruker_id: str,
    dokument_id: str,
    chunks_med_embeddings: List[Dict],
) -> None:
    """
    Lagre tekst-chunks + embeddings for et dokument.

    For hver chunk forventes en struktur ala:

      {
        "chunk_index": 0,
        "tekst": "...",
        "page": 3,
        "embedding": [0.12, -0.03, ...],
        "dokumentklasse": "rapport",
        "filnavn": "minfil.pdf"
      }

    I Cosmos DB kan hvert chunk lagres som et eget dokument i en container
    med vector-index på feltet 'embedding', f.eks.:

      {
        "id": "<chunkId>",
        "userId": "<bruker_id>",
        "docId": "<dokument_id>",
        "chunkIndex": <int>,
        "tekst": "<tekst>",
        "page": <int>,
        "embedding": [...],
        "filnavn": "<filnavn>",
        "dokumentklasse": "<klasse>",
        "type": "chunk"
      }

    :param bruker_id: Unik bruker-ID.
    :param dokument_id: ID for dokumentet disse chunks tilhører.
    :param chunks_med_embeddings: Liste med ordbøker der hver representerer én chunk.
    """
    raise NotImplementedError("lagre_chunks er ikke implementert ennå (planlagt Cosmos-lagring av vektor-dokumenter).")


def sporr_chunks(
    bruker_id: str,
    sporsmal: str,
    top_k: int = 5,
    dokument_id: Optional[str] = None,
) -> List[Dict]:
    """
    Gjør et semantisk søk etter relevante chunks for et spørsmål.

    Forventet flyt i en endelig løsning:
    1. Generer embedding for spørsmålet (med samme embed-modell som chunkene).
    2. Kjør vector search i Cosmos DB mot feltet 'embedding', f.eks.:
         - filtrert på userId = @bruker_id
         - og ev. docId = @dokument_id hvis man bare vil søke i ett dokument.
    3. Sorter etter similarity-score og returner de top_k beste.

    Resultatet returneres som en liste med ordbøker, kompatibel med UI-et, f.eks.:

      [
        {
          "id": "<chunkId>",
          "docId": "<dokument_id>",
          "bruker_id": "<bruker_id>",
          "tekst": "<utdrag av chunk>",
          "page": 3,
          "score": 0.87,
          "filnavn": "minfil.pdf",
          "dokumentklasse": "rapport"
        },
        ...
      ]

    :param bruker_id: Unik bruker-ID (for å sikre at man kun søker i egne dokumenter).
    :param sporsmal: Naturlig språk-spørsmål fra brukeren.
    :param top_k: Hvor mange treff som ønskes.
    :param dokument_id: Hvis satt, begrens søket til ett dokument (docId = dokument_id).
    :return: Liste med ordbøker som beskriver treff (tekst, side, score, osv.).
    """
    raise NotImplementedError("sporr_chunks er ikke implementert ennå (planlagt vector search i Cosmos).")