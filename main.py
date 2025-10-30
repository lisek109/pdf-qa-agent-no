import os
from pathlib import Path
import re
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from app.parsers.pdf import extract_text
from app.qa.chunking import clean_text, split_into_chunks
from app.qa.retrieval import embed_texts, answer_with_context
from app.qa.prompts import DEFAULT_SYSTEM_PROMPT

#Funksjon for å laste CSS
def load_css(path: str) -> None:
    css = Path(path).read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# Laster miljøvariabler fra .env (OpenAI-nøkkel osv.)
load_dotenv()
# Hovedprogram for Streamlit-app
st.set_page_config(page_title="PDF-spørsmål (NO)", page_icon="📄")
# Laster CSS for tilpasset styling
load_css("assets/styles.css")
# Tittel
st.title("📄 PDF-agent - flere dokumenter, norsk Q&A")



# --- Konfigurasjon (hovedkolonne) ---
with st.expander("⚙️ Konfigurasjon av systemprompt", expanded=True):
    with st.form(key="sys_prompt_form_main", border=True):
        sys_prompt_input = st.text_area(
            "Systemprompt (norsk)",
            value=DEFAULT_SYSTEM_PROMPT,
            height=120
        )
        use_prompt_btn = st.form_submit_button("Bruk denne prompten")

    if use_prompt_btn:
        st.session_state["sys_prompt"] = sys_prompt_input

current_sys_prompt = st.session_state.get("sys_prompt", DEFAULT_SYSTEM_PROMPT)


# Filopplasting
uploaded = st.file_uploader("Last opp en PDF-fil", type=["pdf"])

# Spørsmålstekst
st.markdown("### ❓ Skriv inn spørsmålet ditt til dokumentet")
spm = st.text_area("Spørsmål", placeholder="Skriv et presist spørsmål …", height=140)


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
    with st.spinner("Leser tenser og deler opp dokumentet..."):
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
    
    # --- Indeksering (embeddings) én gang per opplastet PDF ---
    # Ide: Beregn embeddings bare når vi MÅ (første gang i sesjonen eller når filen endres),
    # og legg dem i Streamlits sesjonsminne (st.session_state) for å unngå unødvendige API-kall/kostnader.
    if "chunk_vecs" not in st.session_state or st.session_state.get("pdf_path") != pdf_path:
        # Oppretter OpenAI-klient. Nøkkelen hentes fra miljøvariabel (satt via .env).
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Viser spinner i UI mens vi lager embeddings (kan ta noen sekunder for store PDF-er).
        with st.spinner("Lager embeddings for alle tekstbiter..."):
            # Kaller embed_texts(...) som:
            #  - sender alle chunks til embedding-modellen,
            #  - mottar vektorrepresentasjoner (np.ndarray, form ~ [n_chunks, dim]),
            #  - L2-normaliserer for å kunne bruke skalarprodukt som kosinuslikhet.
            chunk_vecs = embed_texts(client, chunks)

        # Husk hvilken fil som ble indeksert i denne sesjonen,
        # og legg både tekstbitene og vektorene i sesjonsminnet.
        # Dette gjør at vi kan svare på mange spørsmål uten å recompute embeddings.
        st.session_state["pdf_path"] = pdf_path
        st.session_state["chunks"] = chunks
        st.session_state["chunk_vecs"] = chunk_vecs

        # Kort bekreftelse i UI
        st.success("Indeksering fullført.")
    


    # Viser noen chunker - kommenter senere- bare for test skyld
    with st.expander("Vis de 3 første chunkene"):
        for i, ch in enumerate(chunks[:3], start=1):
            st.markdown(f"**Chunk {i}**")
            st.text_area(f"chunk_{i}", value=ch, height=200)
            #snippet = re.sub(r"\s+", " ", ch[:800]).strip()
            #st.markdown(f"**Chunk {i}**  \n{snippet}…")
else:
    st.info("Last opp en PDF for å se tekstuttrekk og hvordan den deles i biter.")
    
    
# --- Spørsmål → Svar ---  # 
if spm and "chunk_vecs" in st.session_state:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) 
    chunks = st.session_state["chunks"]
    chunk_vecs = st.session_state["chunk_vecs"]

    with st.spinner("Søker i dokumentet og genererer svar..."):
        # Vi sender med systemprompt fra sidepanelet (eller default hvis ikke endret)
        answer, cites = answer_with_context(
            client, spm, chunks, chunk_vecs, k=3,
            system_prompt=current_sys_prompt  # NEW: brukerens/standard prompt
        )

    st.markdown("### ✅ Svar")
    st.write(answer)

    with st.expander("Vis korte sitater (kildeutdrag)"):
        for i, snip in cites:
            st.markdown(f"**Chunk {i}:**\n\n> {snip} …")
elif spm:
    st.info("Last opp et dokument først, så kan du stille spørsmål.")
else:
    st.caption("Tips: Last opp dokumentet, se at det deles i biter, og prøv et presist spørsmål.")
    

