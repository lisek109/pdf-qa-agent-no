# from __future__ import annotations
from typing import Tuple
import os, json
from pathlib import Path
import joblib # For å laste inn lagret modell

# Leser miljøvariabler eller bruker standardstier
_MODEL_PATH = os.getenv("CLASSIFIER_MODEL_PATH", "models/docclf/model.joblib")
_LABELS_PATH = os.getenv("CLASSIFIER_LABELS_PATH", "models/docclf/labels.json")
_THRESHOLD = float(os.getenv("CLASSIFIER_THRESHOLD", "0.55"))

_MODEL = None  # lazy cache- laste bare engang
_LABELS = None

def _load():
    global _MODEL, _LABELS
    if _MODEL is None:
        if not Path(_MODEL_PATH).exists():
            return None, None
        _MODEL = joblib.load(_MODEL_PATH)
    if _LABELS is None and Path(_LABELS_PATH).exists():
        _LABELS = json.loads(Path(_LABELS_PATH).read_text(encoding="utf-8"))
    return _MODEL, _LABELS

def classify_document_ml(text: str) -> Tuple[str, float]:
    """
    Returnerer (label, prob). Dersom modell mangler → ('annet', 0.0).
    Bruker terskel fra miljø for å dempe usikre prediksjoner.
    """
    model, _ = _load()
    if model is None:
        return "annet", 0.0
    text = (text or "").strip()
    if not text:
        return "annet", 0.0
    proba = model.predict_proba([text])[0]  # shape [n_classes]
    # finn beste labelas
    classes = model.classes_
    idx = int(proba.argmax()) # Finner indeks for høyeste sannsynlighet
    label = str(classes[idx])
    score = float(proba[idx])
    if score < _THRESHOLD:
        # ENDRING: faller tilbake til 'annet' ved lav sikkerhet
        return "annet", score
    return label, score