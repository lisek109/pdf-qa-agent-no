from typing import List, Dict, Optional, Tuple
import os
import chromadb
from chromadb.config import Settings
from langchain_openai import OpenAIEmbeddings

_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def get_client(persist_dir: str = "data/chroma") -> chromadb.Client:
    os.makedirs(persist_dir, exist_ok=True)
    return chromadb.Client(Settings(is_persistent=True, persist_directory=persist_dir))

def get_collection(client: chromadb.Client, name: str = "pdf_chunks"):
    try:
        return client.get_collection(name=name)
    except:
        return client.create_collection(name=name, metadata={"hnsw:space": "cosine"})

def upsert_chunks(coll, doc_id: str, chunks: List[str], metadatas: Optional[List[Dict]] = None) -> None:
    if not chunks:
        return
    ids = [f"{doc_id}:{i}" for i in range(len(chunks))]
    embs = _embeddings.embed_documents(chunks)
    metas = metadatas or [{} for _ in chunks]
    for m in metas:
        m.setdefault("doc", doc_id)
    coll.upsert(ids=ids, documents=chunks, embeddings=embs, metadatas=metas)

def query_topk(coll, query: str, k: int = 3, where: Optional[Dict] = None) -> List[Tuple[str, str, Dict]]:
    q_emb = _embeddings.embed_query(query)
    res = coll.query(query_embeddings=[q_emb], n_results=k, where=where or {})
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    return list(zip(ids, docs, metas))
