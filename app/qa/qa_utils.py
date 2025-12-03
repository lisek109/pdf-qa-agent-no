import os
from typing import List
from openai import OpenAI

# Leser embed-modell fra miljøvariabel, med samme default
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")


def embed_sporsmal(client: OpenAI, sporsmal: str) -> List[float]:
    """
    Lager en embedding-vektor for et spørsmål.

    Bruker samme embed-modell som for dokument-chunks,
    slik at vi kan sammenligne spørsmålet med lagrede embeddings.
    """
    tekst = (sporsmal or "").strip()
    if not tekst:
        raise ValueError("Tomt spørsmål kan ikke embeddes.")

    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=tekst,
    )
    # Vi antar at modellen returnerer én vektor
    return resp.data[0].embedding