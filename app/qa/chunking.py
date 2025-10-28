import re
from typing import List, Dict, Tuple
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


def split_pages_into_chunks(pages: List[Tuple[int, str]], size: int = 1200, overlap: int = 150) -> List[Dict]:
    """
    Deler tekst side for side og returnerer en liste med metadata per chunk.
    Struktur per element:
      {
        "page": int,       # sidetall (1-basert)
        "content": str,    # selve tekstbiten
        "start": int,      # startposisjon innenfor side-teksten (0-basert indeks)
        "end": int         # sluttposisjon (eksklusiv) innenfor side-teksten
      }

    Parametre:
      pages   : Liste av (side_nr, tekst) fra extract_pages()
      size    : Omtrentlig maks lengde på hver chunk i tegn
      overlap : Overlapp i tegn mellom nabochunks (bedre kontekst ved retrieval)

    Merk:
      - start/end brukes senere for highlighting i UI, derfor regnes de relativt
        til den rensede side-teksten etter clean_text().
      - Offset-estimering under er en enkel heuristikk basert på første forekomst
        av part i side-teksten, med en løpende pekeren (offset) for å unngå
        å "finne tilbake" til tidligere forekomster.
    """
    chunks: List[Dict] = []

    # Itererer gjennom alle sider: page_no = 1,2,...; raw = råtekst for siden.
    for page_no, raw in pages:
        # Rens tesk
        text = clean_text(raw)

        # Initialiser tekstsplitter for denne siden
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " "]
        )

        # Deler side-teksten i deler med ønsket overlapp.
        parts = splitter.split_text(text)

        # Løpende posisjon (0-basert) i side-teksten for å estimere start/end.
        # Vi bruker denne som "hint" slik at find() leter fremover, ikke fra start.
        offset = 0

        # Gå gjennom hver del og beregn metadata.
        for part in parts:
            # Finn første forekomst av denne delen fra gjeldende offset.
            # Dette fungerer som en enkel heuristikk for å mapppe delstrengen
            # tilbake til posisjon i hele side-teksten.
            idx = text.find(part, offset)

            # Hvis delstrengen ikke ble funnet (kan skje pga små avvik/normalisering),
            # faller vi tilbake til gjeldende offset for å bevare fremdrift.
            if idx == -1:
                idx = offset

            # Legg til chunk med sideinformasjon og beregnet posisjon.
            chunks.append({
                "page": page_no,
                "content": part,
                "start": idx,
                "end": idx + len(part)
            })

            # Flytt offset for neste søk, slik at vi ikke matcher samme område igjen.
            # Viktig: selv med overlapp fra splitteren vil dette sikre monotont økende
            # posisjoner (nyttig for senere behandling).
            offset = idx + len(part)

    # Returnerer alle chunks for alle sider.
    return chunks