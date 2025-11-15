from typing import Optional, List, Tuple
import fitz  # PyMuPDF


def extract_pages(path: str, max_pages: Optional[int] = None) -> List[Tuple[int, str]]:
    """
    Leser PDF og returnerer liste av (side_nr, tekst).
    side_nr starter på 1 (slik folk forventer).
    Bruker context manager for å sikre at dokumentet lukkes automatisk.
    """
    out: List[Tuple[int, str]] = []

    try:
        # Åpner dokumentet og sørger for automatisk lukking når blokken avsluttes.
        with fitz.open(path) as doc:
            # enumerate(..., start=1) gir 1-basert sidetelling (mer naturlig i UI)
            for i, page in enumerate(doc, start=1):
                # Stopper tidlig hvis max_pages er satt (nyttig i test)
                if max_pages is not None and i > max_pages:
                    break

                # Henter ren tekst fra siden
                text = page.get_text("text")

                # Legger til tuple (side_nr, tekst)
                out.append((i, text))
    except Exception as e:
        print("⚠️ Dokumentet kunne ikke leses eller er tomt.")
        return []

    # Dokumentet er nå lukket automatisk
    return out

