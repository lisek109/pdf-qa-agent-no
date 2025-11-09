# Enkel treningsskript for dokumentklassifisering (baseline).
# - Input: CSV med kolonner ['doc_id','text','label']
# - Output: model.joblib + labels.json




# Sklearn-moduler for tekstklassifisering
from sklearn.feature_extraction.text import TfidfVectorizer  # Tekst → numerisk vektor
from sklearn.linear_model import LogisticRegression           # Klassifikator
from sklearn.calibration import CalibratedClassifierCV        # Kalibrerer sannsynligheter
from sklearn.pipeline import Pipeline


def build_pipeline():
    # Lager en pipeline med TF-IDF + logistisk regresjon
    vec = TfidfVectorizer(
        lowercase=True,           # Konverterer tekst til små bokstaver
        ngram_range=(1,2),        # Bruker 1- og 2-ords kombinasjoner (unigram + bigram)
        analyzer="word",          # Analyserer på ordnivå
        min_df=2,                 # Ignorerer ord som forekommer færre enn 2 ganger
        max_features=200_000      # Maksimalt antall funksjoner (ordkombinasjoner)
    )

    clf = LogisticRegression(
        max_iter=2000,            # Maks antall iterasjoner for trening
        n_jobs=None,              # Bruker én CPU-kjerne
        class_weight="balanced"  # Justerer for ubalanserte klasser
    )

    pipe = Pipeline([
        ("tfidf", vec),           # Først: TF-IDF-transformasjon(Term Frequency hvor ofte ordet er i dokumentet-Inverse Document Frequency hvpot skeldent ord er i hele korpuset)
        ("clf", CalibratedClassifierCV(base_estimator=clf, cv=3, method="sigmoid"))
        # Deretter: Kalibrert klassifikator med 3-fold kryssvalidering
    ])
    return pipe                  # Returnerer den komplette pipeline-modellen
