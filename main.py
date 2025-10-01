import os
import re
import streamlit as st
from dotenv import load_dotenv
from app.parsers.pdf import extract_text
from app.qa.chunking import clean_text, split_into_chunks
import inspect
print("extract_text() pochodzi z pliku:", inspect.getfile(extract_text))

load_dotenv()
st.set_page_config(page_title="PDF-spørsmål (MVP)", page_icon="📄")
st.title("📄 PDF-agent (MVP) - tekstuttrekk og chunking")

# Filopplasting
uploaded = st.file_uploader("Last opp en PDF-fil", type=["pdf"])

if uploaded:
    # Lager folder hvis den ikke finnes
    os.makedirs("data/raw", exist_ok=True)
    # Lagrer filen - HUSK Å LEGGE TIL EN SKJEKK OM DET ALLEREDE EKSISTERER FIL MED SAMME NAVN
    pdf_path = os.path.join("data", "raw", uploaded.name)
    # åpner i binary mode for å unngå encoding-problemer w-write b-binary
    with open(pdf_path, "wb") as f:
        # skriver buffer direkte til fil
        f.write(uploaded.getbuffer())
    st.success(f"Lagret: {uploaded.name}")

    # Tekstuttrekk og chunking
    with st.spinner("Leser og deler opp dokumentet..."):
        raw = extract_text(pdf_path)
        text = clean_text(raw)
        chunks = split_into_chunks(text, size=1200, overlap=180)
        
    # Debug information in console
    print("DEBUG len(raw):", len(raw))    
    print("DEBUG len(text):", len(text))
    print("DEBUG antall_chunks:", len(chunks))
    print("DEBUG count('\\n'):", text.count("\n"), "count('\\r'):", text.count("\r"), "count(NBSP):", text.count("\u00A0"))
    print("DEBUG first 200:", text[:200].encode("unicode_escape"))

        
        

    st.write(f"**Lengde (tegn):** {len(text)}")
    st.write(f"**Antall chunks:** {len(chunks)}")

    # Viser noen chunker
    with st.expander("Vis de 3 første chunkene"):
        for i, ch in enumerate(chunks[:3], start=1):
            snippet = re.sub(r"\s+", " ", ch[:800]).strip()
            st.markdown(f"**Chunk {i}**  \n{snippet}…")
else:
    st.info("Last opp en PDF for å se tekstuttrekk og hvordan den deles i biter.")
