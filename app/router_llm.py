
# LLM-as-router – velger dokumentklasse for spørsmålet (JSON)
import json, os
from typing import Tuple, List, Optional
from openai import OpenAI

CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

_SYSTEM = "Du er en kort og strukturert klassifikator. Returner KUN gyldig JSON."
_USER_TMPL = """Klassifiser følgende spørsmål til nøyaktig én av klassene:
[{labels}]

Regler:
- Velg den mest relevante klassen for kildedokumenter.
- Hvis uklart → "annet".
- Svar KUN som JSON: {{"label":"<klasse>","confidence": <0..1>}}

Eksempler:
Q: "hvor mye beatalte vi for materialler fra firma X i desember?"
→ {{"label":"faktura","confidence":0.86}}

Q: "når har vi bestilt utstyr X fra firma Y?"
→ {{"label":"bestilling","confidence":0.81}}

Q: "vis siste rapporter om prosjekt Z"
→ {{"label":"rapport","confidence":0.77}}

Nå, klassifiser dette spørsmålet:
{question}
"""

def classify_question_llm(question: str, labels: list[str], threshold: float = 0.55, client: Optional[OpenAI] = None,) -> Tuple[str, float]:
    """Returnerer (label, confidence). Under terskel -> ("annet", score)."""
    if not question or not question.strip():
        return "annet", 0.0
    
    if client is None:
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("Mangler OpenAI API-nøkkel for LLM-klassifisering.")
        client = OpenAI(api_key=api_key)
        
    user = _USER_TMPL.format(labels=", ".join(labels), question=question[:4000])
    
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[{"role":"system","content":_SYSTEM},{"role":"user","content":user}]
    )
    raw = resp.choices[0].message.content.strip()
    
    try:
        data = json.loads(raw)
        label = str(data.get("label","annet"))
        conf  = float(data.get("confidence",0.0))
        if conf < threshold:
            return "annet", conf
        return label, conf
    except Exception:
        return "annet", 0.0
