# app/qa/retrieval.py
from typing import List, Tuple
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
