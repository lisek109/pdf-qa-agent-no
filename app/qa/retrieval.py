# app/qa/retrieval.py
from typing import List, Tuple
import hashlib, os
import re
import numpy as np
from numpy.linalg import norm
from openai import OpenAI
from app.qa.prompts import SYSTEM_PROMPT

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
MAX_CONTEXT_CHARS = 2200  # begrens prompt-lengde, kost og latens

def l2_normalize(mat: np.ndarray) -> np.ndarray:
    """
    L2-normaliserer hver vektor (lengde = 1).
    Da blir skalarproduktet lik kosinuslikhet.
    """
    return mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)

def embed_texts(client: OpenAI, texts: List[str]) -> np.ndarray:
    """
    Lager embeddings for en liste tekster og returnerer som np.array[ n x d ].
    """
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
    return l2_normalize(vecs)

def embed_query(client: OpenAI, query: str) -> np.ndarray:
    """
    Embedding for selve spørsmålet (1 x d), L2-normalisert.
    """
    v = client.embeddings.create(model=EMBED_MODEL, input=query).data[0].embedding
    v = np.array(v, dtype=np.float32)
    return v / (norm(v) + 1e-12)

def top_k_indices(query_vec: np.ndarray, chunk_vecs: np.ndarray, k: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rangerer alle chunks mot spørsmålet ved kosinuslikhet og returnerer:
    - I: indeksene til top-k
    - S: tilsvarende score-verdier
    """
    sims = chunk_vecs @ query_vec  # skalarprodukt == kosinus (pga normalisering)
    I = np.argsort(-sims)[:k]
    S = sims[I]
    return I, S

def build_context(chunks: List[str], idxs: np.ndarray, limit: int = MAX_CONTEXT_CHARS) -> str:
    """
    Slår sammen top-k biter til en begrenset KONTEKST-streng.
    """
    parts, used = [], 0
    for i in idxs:
        piece = chunks[int(i)]
        if used + len(piece) > limit:
            piece = piece[: max(0, limit - used)]
        parts.append(piece)
        used += len(piece)
        if used >= limit:
            break
    return "\n\n---\n\n".join(parts)

def answer_with_context(client: OpenAI, question: str, chunks: List[str], chunk_vecs: np.ndarray, k: int = 3):
    """
    Dettte går i stegene:
    1) embedder spørsmål
    2) finner top-k chunks
    3) bygger kontekst
    4) kaller chat-modellen med norsk systemprompt
    Returnerer (svar, [(idx, kort_sitat), ...])
    """
    q_vec = embed_query(client, question)
    idxs, scores = top_k_indices(q_vec, chunk_vecs, k=k)
    context = build_context(chunks, idxs, MAX_CONTEXT_CHARS)

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"KONTEKST:\n{context}\n\nSPØRSMÅL: {question}"}
        ],
        temperature=0
    )
    answer = resp.choices[0].message.content

    # små sitater fra hver topp-bit, greit for åha i UI
    citations = []
    for i in idxs:
        snip = re.sub(r"\s+", " ", chunks[int(i)][:200]).strip()
        citations.append((int(i), snip))
    return answer, citations

def file_sha1(path: str) -> str:
    """
    Beregn en SHA-1-hash for en fil.
    Brukes som stabil nøkkel for cache (endres når innholdet endres).
    """
    h = hashlib.sha1()  # lager en SHA-1 hash-objekt (til identifisering, ikke til sikkerhet)
    # Åpner filen i binærmodus (rb) – viktig for at bytes ikke endres av tekst-dekoding
    with open(path, "rb") as f:
        # Leser filen i biter (8192 byte) – effektivt og minnevennlig for store PDF-er
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)  # oppdaterer hash med neste bit av innholdet
    return h.hexdigest()  # heksadesimal streng (f.eks. "a94a8fe5...") som nøkkel


def load_cached_vectors(index_dir: str, key: str) -> np.ndarray | None:
    """
    Forsøk å laste embeddings fra disk-cache.
    - index_dir: katalog hvor vi lagrer cache-filer
    - key: nøkkel (f.eks. SHA-1 av fil + ev. modellnavn)
    Returnerer:
      - np.ndarray med embeddings hvis filen finnes
      - None hvis ingen cache er lagret ennå
    """
    # Lager full sti til cache-filen: <index_dir>/<key>.npy
    p = os.path.join(index_dir, f"{key}.npy")
    # Sjekker om filen finnes – hvis ja, last inn med NumPy
    if os.path.exists(p):
        # np.load leser tilbake dtype/shape eksakt som ved lagring
        return np.load(p)
    # Ingen cache – kallende kode kan da beregne embeddings og lagre dem
    return None


def save_cached_vectors(index_dir: str, key: str, arr: np.ndarray) -> None:
    """
    Lagre embeddings til disk-cache som .npy-fil.
    - index_dir: katalog for cache-filer (opprettes automatisk)
    - key: nøkkel (f.eks. SHA-1 av fil + ev. modellnavn)
    - arr: np.ndarray (typisk shape: (antall_chunks, embed_dim))
    """
    # Sørger for at katalogen finnes; gjør ingenting hvis den allerede finnes
    os.makedirs(index_dir, exist_ok=True)
    # Full sti til målfilen
    p = os.path.join(index_dir, f"{key}.npy")
    # Lagrer NumPy-array i binært .npy-format (hurtig og tapsfritt)
    np.save(p, arr)