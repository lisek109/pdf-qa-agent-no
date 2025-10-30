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
