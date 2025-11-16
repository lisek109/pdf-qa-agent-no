# Enkel treningsskript for dokumentklassifisering (baseline).
# - Input: CSV med kolonner ['doc_id','text','label']
# - Output: model.joblib + labels.json

import json
import argparse
import pandas as pd
from pathlib import Path


# Sklearn-moduler for tekstklassifisering
from sklearn.feature_extraction.text import TfidfVectorizer  # Tekst → numerisk vektor
from sklearn.linear_model import LogisticRegression           # Klassifikator
from sklearn.calibration import CalibratedClassifierCV        # Kalibrerer sannsynligheter
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import joblib


def build_pipeline(X_train, y_train):
    # Lager en pipeline med TF-IDF + logistisk regresjon
    vec = TfidfVectorizer(
        lowercase=True,           # Konverterer tekst til små bokstaver
        ngram_range=(1,2),        # Bruker 1- og 2-ords kombinasjoner (unigram + bigram)
        analyzer="word",          # Analyserer på ordnivå
        min_df=2,                 # Ignorerer ord som forekommer færre enn 2 ganger
        max_features=200_000      # Maksimalt antall funksjoner (ordkombinasjoner)
    )
    
    X_train_tfidf = vec.fit_transform(X_train)

    # Initialiserer logistisk regresjonsmodell
    clf = LogisticRegression(
        max_iter=2000,            # Maks antall iterasjoner for trening
        n_jobs=None,              # Bruker én CPU-kjerne
        class_weight="balanced"  # Justerer for ubalanserte klasser
    )

    # Kalibrerer klassifikatoren for bedre sannsynlighetsestimater
    cal_clf = CalibratedClassifierCV(estimator=clf, cv=3, method="sigmoid")
    #cal_clf.fit(X_train_tfidf, y_train)

    # Setter sammen pipeline
    pipe = make_pipeline(vec, cal_clf) 
    
    pipe.fit(X_train, y_train)
    return pipe                  # Returnerer den komplette pipeline-modellen


def main(args):
    # Leser CSV-fil med dokumenter
    df = pd.read_csv(args.input)
    # Ekstraherer tekst og etiketter
    X, y = df["text"].fillna(""), df["label"].astype(str)
    
    # Liste over unike etiketter (for lagring senere)
    labels = sorted(y.unique())

    # Deler data i trening og validering (80/20)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = build_pipeline(X_train, y_train) # Lager modell-pipeline
    
    y_pred = pipe.predict(X_val) # Predikerer på valideringssettet
    print(classification_report(y_val, y_pred, digits=3)) # Viser klassifiseringsrapport

    out_dir = Path(args.outdir) # Utgangskatalog
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out_dir / "model.joblib") # Lagrer den trente modellen 
    with open(out_dir / "labels.json", "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2) # Lagrer etiketter som JSON
    print(f"✅ Saved to: {out_dir/'model.joblib'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()  # Parser komend
    ap.add_argument("--input", required=True, help="Stien til input CSV-fil")  # Input CSV-fil
    ap.add_argument("--outdir", default="models/docclf", help="Katalog wyjściowy") # Utgangskatalog
    main(ap.parse_args())   # Kjører hovedfunksjonen
    
    # kjøringseksempel:
    # python train_classifier.py --input data/training/documents.csv --outdir models/docclf




