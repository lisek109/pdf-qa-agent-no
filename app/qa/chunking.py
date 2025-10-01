import re
from typing import List
# Her kan fikk jeg problemer med å installere pakken
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    raise RuntimeError(
        "Mangler 'langchain-text-splitters'. Installer: pip install langchain-text-splitters"
    )

def clean_text(raw: str) -> str:
    """
    Enkel rensing:
    - fjerner bindestrek + linjeskift (orddeling)
    - normaliserer mellomrom
    - begrenser mange linjeskift til maks to
    """
    s = raw.replace("-\n", "")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def split_into_chunks(text: str, size: int = 1000, overlap: int = 150) -> List[str]:
    """
    Deler tekst i biter (chunks) rundt ~size tegn, med overlap for bedre kontekst.
    Bruker heuristikker: først avsnitt, så linjer, deretter setningsslutt, til slutt mellomrom.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " "]
    )
    return splitter.split_text(text)