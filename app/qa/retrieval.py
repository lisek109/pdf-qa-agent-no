# app/qa/retrieval.py
from typing import List, Tuple, Optional
import hashlib, os
import re
import numpy as np
from numpy.linalg import norm
from openai import OpenAI
from app.qa.prompts import DEFAULT_SYSTEM_PROMPT


# Konfig for modeller og Hvor mye kontekst vi maks pakker inn i promptet
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
    L2-normalisert (klar for kosinuslikhet).
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



# Man kunne laget én generell funksjon som aksepterer enten chunks + idxs eller top_chunks (liste med strenger). 
# Men det er mer lesbart – og uten behov for if-setninger – å ha to separate og enkle funksjoner

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

def build_context_from_list(top_chunks: List[str], limit: int = MAX_CONTEXT_CHARS) -> str:
    """
    Bygger KONTEKST direkte fra en liste av utvalgte tekstbiter (Chroma-path).
    """
    parts, used = [], 0
    for piece in top_chunks:
        if used + len(piece) > limit:
            piece = piece[: max(0, limit - used)]
        parts.append(piece)
        used += len(piece)
        if used >= limit:
            break
    return "\n\n---\n\n".join(parts)

# ---------- Hovedfunksjon for Q&A ----------

def answer_with_context(client: OpenAI, question: str, chunks: List[str], chunk_vecs: np.ndarray, k: int = 3,
    system_prompt: Optional[str] = None,  # <- kommer fra UI; faller tilbake på DEFAULT_SYSTEM_PROMPT
):
    """
    Steg for steg:
      1) embedder spørsmålet
      2) finner top-k relevante chunks
      3) bygger KONTEKST
      4) kaller chat-modellen med valgt systemprompt

    Returnerer:
      - answer: modellens svar (str)
      - citations: liste med (chunk_index, kort_sitat) for UI
    """
    # 1) Spørsmålets embedding
    q_vec = embed_query(client, question)

    # 2) Hent top-k
    idxs, _ = top_k_indices(q_vec, chunk_vecs, k=k)

    # 3) Bygg kontekst-streng
    context = build_context(chunks, idxs, MAX_CONTEXT_CHARS)

    # 4) Kall chat-modellen
    msgs = [
        {"role": "system", "content": system_prompt or DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": f"KONTEKST:\n{context}\n\nSPØRSMÅL: {question}"},
    ]
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=msgs,
        temperature=0
    )
    answer = resp.choices[0].message.content

    # Små sitater fra hver topp-bit – nyttig for transparens
    citations = []
    for i in idxs:
        snip = re.sub(r"\s+", " ", chunks[int(i)][:200]).strip()
        citations.append((int(i), snip))

    return answer, citations



##############  Lightweight variant for Chroma ##############

def answer_with_top_chunks(
    client: OpenAI,
    question: str,
    top_chunks: List[str],
    system_prompt: Optional[str] = None,
    examples: Optional[List[Tuple[str, str]]] = None,
):
    """
    Lettvekts variant for Chroma: vi HAR allerede topp-chunks.
    """
    context = build_context_from_list(top_chunks, MAX_CONTEXT_CHARS)

    msgs = [{"role": "system", "content": system_prompt or DEFAULT_SYSTEM_PROMPT}]
    if examples:
        for u, a in (examples or []):
            msgs.append({"role": "user", "content": u})
            msgs.append({"role": "assistant", "content": a})

    msgs.append({"role": "user", "content": f"KONTEKST:\n{context}\n\nSPØRSMÅL: {question}"})

    resp = client.chat.completions.create(model=CHAT_MODEL, messages=msgs, temperature=0)
    answer = resp.choices[0].message.content
    cites = [(i, re.sub(r"\s+", " ", ch[:200]).strip()) for i, ch in enumerate(top_chunks)]
    return answer, cites


##################   Cache embeddings   ##################

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