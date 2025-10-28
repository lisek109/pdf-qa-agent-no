from typing import Optional, List, Tuple
import fitz  # PyMuPDF

def extract_text(path: str, max_pages: Optional[int] = None) -> str:
    """
    Leser tekst fra en PDF side for side og slår alt sammen til en streng.
    - path: stientil PDF
    - max_pages: (valgfritt) maks antall sider å lese under testing
    Merk: Skannede PDF-er (bilder) vil gi tom tekst; da trengs OCR i neste fase.
    """
    with fitz.open(path) as doc:           # automatisk lukker dokumnent
        texts = []
        for i, page in enumerate(doc, start=1):
            if max_pages is not None and i > max_pages:
                break
            texts.append(page.get_text("text"))  # "layout-aware" tekst
    return "\n".join(texts)


def extract_pages(path: str, max_pages: Optional[int] = None) -> List[Tuple[int, str]]:
    """
    Leser PDF og returnerer liste av (side_nr, tekst).
    side_nr starter på 1 (slik folk forventer).
    """
    doc = fitz.open(path)
    out: List[Tuple[int, str]] = []
    for i, page in enumerate(doc, start=1):
        if max_pages is not None and i > max_pages:
            break
        out.append((i, page.get_text("text")))
    return out


# def extract_text(path: str, max_pages: Optional[int] = None) -> str:
#     """
#     Leser tekst fra en PDF side for side og slår alt sammen til én streng.
#     Bruker context manager for sikker lukking av dokumentet.
#     """
#     texts = []
#     with fitz.open(path) as doc:
#         for i, page in enumerate(doc):
#             if max_pages is not None and i >= max_pages:
#                 break
#             texts.append(page.get_text("text") or "")
#     return "\n".join(texts)