from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv
import os
import chromadb
from chromadb.config import Settings
from langchain_openai import OpenAIEmbeddings

# Laster miljøvariabler fra .env (OpenAI-nøkkel osv.)
load_dotenv()

# Global embedding-funksjon (OpenAI) – brukes både for dokumenter og spørringer
_embeddings = OpenAIEmbeddings(model="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY"))

def get_client(persist_dir: str = "data/chroma") -> chromadb.Client:
    """
    Oppretter en Chroma-klient med vedvarende lagring (persist) i 'persist_dir'.
    Lager mappen hvis den ikke finnes.
    """
    os.makedirs(persist_dir, exist_ok=True)
    return chromadb.Client(Settings(is_persistent=True, persist_directory=persist_dir))
    # Merk: I nyere Chroma kan man også bruke chromadb.PersistentClient(path=...)

def get_collection(client: chromadb.Client, name: str = "pdf_chunks"):
    """
    Slår opp en eksisterende samling (collection) ved navn.
    Hvis den ikke finnes, opprettes en ny med kosinus-likhet (HNSW-indeks).
    """
    try:
        return client.get_collection(name=name)
    except:
        # metadata hnsw:space=cosine -> bruk kosinuslikhet for nærmeste nabo
        return client.create_collection(name=name, metadata={"hnsw:space": "cosine"})

def upsert_chunks(coll, doc_id: str, chunks: List[str], metadatas: Optional[List[Dict]] = None) -> None:
    """
    Lagrer/oppdaterer (upsert) tekst-chunks i Chroma med tilhørende embeddings.
    - coll: Chroma-samling
    - doc_id: stabil ID for dokumentet (f.eks. sha1 av fil)
    - chunks: liste av tekstbiter
    - metadatas: valgfri liste med metadata per chunk (samme lengde som chunks)
    """
    if not chunks:
        return

    # Unike id-er per chunk: <doc_id>:<løpenummer>
    ids = [f"{doc_id}:{i}" for i in range(len(chunks))]

    # Regn ut embeddings for alle chunks (batch)
    embs = _embeddings.embed_documents(chunks)

    # Standardiser metadata-liste; sørg for at 'doc' er satt til doc_id
    metas = metadatas or [{} for _ in chunks]
    for m in metas:
        m.setdefault("doc", doc_id)

    # Upsert: opprett eller oppdater poster i samlingen
    coll.upsert(ids=ids, documents=chunks, embeddings=embs, metadatas=metas)

def query_topk(coll, query: str, k: int = 3, where: Optional[Dict] = None) -> List[Tuple[str, str, Dict]]:
    """
    Kjører et vektor-søk (top-k) i Chroma for en gitt tekstspørring.
    - where: valgfritt filter på metadata (f.eks. {"doc": doc_id} eller {"page": 3})
    Returnerer liste av (id, dokumenttekst, metadata) for de beste treffene.
    """
    # Embedding av tekstspørringen
    q_emb = _embeddings.embed_query(query)

    # Hent nærmeste naboer (k resultater), med eventuelt metadata-filter
    res = coll.query(query_embeddings=[q_emb], n_results=k, where=where or {})

    # Pakk ut første spørring (vi sendte inn bare én)
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]

    # Zipp sammen til en liste av tuples: (id, tekst, metadata)
    return list(zip(ids, docs, metas))
