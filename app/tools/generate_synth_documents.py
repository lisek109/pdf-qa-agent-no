import argparse           # For å lese kommandolinjeargumenter
import random             # For tilfeldig variasjon i data
from pathlib import Path  # For å håndtere filstier
import pandas as pd       # For å jobbe med tabellstrukturer (DataFrame)
from faker import Faker   # For å generere syntetiske (falske) data

# Initialiserer faker med norsk språk
fake = Faker("no_NO")
random.seed(42)  # Setter fast seed for reproduserbarhet

# Definerer dokumentklasser som skal genereres
LABELS = ["faktura", "bestilling", "kostnadsoverslag", "kontrakt", "rapport", "annet"]

def gen_faktura():
    # Lager en syntetisk faktura med linjeelementer og summer
    selskap = fake.company()
    kundenr = fake.random_number(digits=8)
    fakturanr = fake.random_number(digits=7)
    dato = fake.date()
    forfallsdato = fake.date()
    poster = []
    total = 0.0
    for _ in range(random.randint(2, 6)):
        vare = fake.bs().capitalize()
        antall = random.randint(1, 5)
        pris = round(random.uniform(100, 5000), 2)
        linje = f"{vare} x{antall} á {pris} NOK"
        poster.append(linje)
        total += antall * pris
    mva = round(total * 0.25, 2)
    total_mva = round(total + mva, 2)
    return (
        f"Faktura\nSelskap: {selskap}\nKundenummer: {kundenr}\nFakturanummer: {fakturanr}\n"
        f"Dato: {dato}\nForfallsdato: {forfallsdato}\n"
        + "\n".join(poster)
        + f"\nMVA 25%: {mva} NOK\nTOTALT: {total_mva} NOK\nBetalingsinformasjon: KID {fake.random_number(digits=10)}"
    )


def gen_bestilling():
    selskap = fake.company()
    ordrenr = fake.random_number(digits=7)
    dato = fake.date()
    leveringsdato = fake.date()
    varer = [fake.catch_phrase() for _ in range(random.randint(2, 5))]
    return (
        f"Bestilling\nLeverandør: {selskap}\nOrdrenummer: {ordrenr}\nBestillingsdato: {dato}\n"
        f"Forventet leveringsdato: {leveringsdato}\nVarer:\n- " + "\n- ".join(varer) +
        "\nBetalingsbetingelser: 14 dager netto"
    )

def gen_kostnadsoverslag():
    klient = fake.company()
    prosjekt = fake.catch_phrase()
    pos = []
    total = 0.0
    for _ in range(random.randint(3, 7)):
        arbeid = fake.bs()
        kost = round(random.uniform(2000, 20000), 2)
        pos.append(f"- {arbeid}: {kost} NOK")
        total += kost
    margin = round(total * 0.1, 2)
    return (
        f"Kostnadsoverslag\nKlient: {klient}\nProsjekt: {prosjekt}\nPoster:\n" + "\n".join(pos) +
        f"\nAdministrasjonskost: {margin} NOK\nEstimert totalsum: {round(total+margin,2)} NOK\n"
        "Gyldighet: 30 dager"
    )

def gen_kontrakt():
    part_a = fake.company()
    part_b = fake.company()
    start = fake.date()
    slutt = fake.date()
    vilkar = [
        "Parter forplikter seg til konfidensialitet.",
        "Betaling skal skje innen 30 dager.",
        "Tvister avgjøres ved Oslo tingrett.",
    ]
    return (
        f"Kontrakt\nPart A: {part_a}\nPart B: {part_b}\nGyldighetsperiode: {start} – {slutt}\n"
        "Vilkår:\n- " + "\n- ".join(vilkar) + "\nSignaturer: ____________________"
    )

def gen_rapport():
    tittel = fake.catch_phrase()
    intro = fake.paragraph(nb_sentences=3)
    metode = fake.paragraph(nb_sentences=2)
    result = fake.paragraph(nb_sentences=4)
    konkl = fake.sentence(nb_words=12)
    return (
        f"Rapport\nTittel: {tittel}\nInnledning: {intro}\nMetode: {metode}\n"
        f"Resultater: {result}\nKonklusjon: {konkl}\nVedlegg: {fake.word()}.pdf"
    )

def gen_annet():
    return fake.paragraph(nb_sentences=6)

GEN_MAP = {
    "faktura": gen_faktura,
    "bestilling": gen_bestilling,
    "kostnadsoverslag": gen_kostnadsoverslag,
    "kontrakt": gen_kontrakt,
    "rapport": gen_rapport,
    "annet": gen_annet,
}

def main(n_per_class: int, out_path: str):
    rows = []
    doc_id = 0
    for label in LABELS:
        for _ in range(n_per_class):
            text = GEN_MAP[label]()  # Kaller riktig genereringsfunksjon
            rows.append({"doc_id": f"doc_{doc_id}", "text": text, "label": label})
            doc_id += 1
    df = pd.DataFrame(rows)  # Konverterer til DataFrame
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)  # Lager mappe hvis den ikke finnes
    df.to_csv(out_path, index=False, encoding="utf-8")  # Skriver til CSV
    print(f"✅ Saved {len(df)} rows → {out_path}")
    
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_per_class", type=int, default=80, help="Antall dokumenter per klasse")
    ap.add_argument("--out", type=str, default="data/training/documents.csv", help="Sti til CSV")
    args = ap.parse_args()
    main(args.n_per_class, args.out)