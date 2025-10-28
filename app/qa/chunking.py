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
    - normaliserer linjeslutt (CRLF -> LF)
    - fjerner NBSP og «soft hyphen»
    - fjerner bindestrek + linjeskift (orddeling)
    - normaliserer mellomrom
    - begrenser mange linjeskift til maks to
    """
    # Normaliser Windows-linjeslutt til bare \n (viktig for splitting på "\n\n")
    s = raw.replace("\r\n", "\n").replace("\r", "\n")
    # Bytt ut «non-breaking space» (NBSP) med vanlig mellomrom
    s = s.replace("\u00A0", " ")
    # Fjern «soft hyphen» (usynlig, kan dukke opp i PDF-tekst)
    s = s.replace("\u00AD", "")
    # Fjern bindestrek + linjeskift brukt til orddeling (nå alltid '\n' etter normalisering)
    s = s.replace("-\n", "")
    # Komprimer flere mellomrom/tab til ett (beholder linjeskift urørt)
    s = re.sub(r"[ \t]+", " ", s)
    # Komprimer 3+ tomme linjer til maks to (bevarer avsnittsskiller)
    s = re.sub(r"\n{3,}", "\n\n", s)
    # Trim ledende/etterfølgende whitespace i hele strengen
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