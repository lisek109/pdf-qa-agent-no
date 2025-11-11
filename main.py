import os
from pathlib import Path
import re, glob
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from app.parsers.pdf import extract_pages
from app.qa.chunking import split_pages_into_chunks
from app.qa.retrieval import embed_texts, answer_with_context, file_sha1, load_cached_vectors, save_cached_vectors, answer_with_top_chunks, cache_key_for_file
from app.qa.vectorstore_chroma import  get_client, get_collection, upsert_chunks, query_topk
from app.qa.prompts import DEFAULT_SYSTEM_PROMPT
from app.classifier.infer import classify_document_ml
from app.router_llm import classify_question_llm   

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
    
#Funksjon for å laste CSS
def load_css(path: str) -> None:
    try:
        css = Path(path).read_text(encoding="utf-8")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        #  Robust mot manglende styles.css
        st.debug("styles.css ikke funnet - fortsetter uten egen CSS.")  # trygg logg-linje


# Laster miljøvariabler fra .env (OpenAI-nøkkel osv.)
load_dotenv()

# Hovedprogram for Streamlit-app
st.set_page_config(page_title="PDF-spørsmål (NO)", page_icon="📄", layout="wide")

# Laster CSS for tilpasset styling
load_css("assets/styles.css")

# Tittel
st.title("📄 PDF Assistent ")

# --- Globalt omfang vs ett dokument (UI på hovedsiden) ---
col_a, col_b = st.columns([1, 4])
with col_a:
    # ENDRING: toggle bestemmer om vi jobber på hele korpuset
    global_mode = st.toggle("Alle dokumenter", value=False, help="Søk på tvers av alle dokumenter")
# Behold en streng-variant som resten av koden bruker
scope = "Alle dokumenter" if global_mode else "Kun valgt dokument"
st.session_state["scope"] = scope  # valgfritt: gjør tilgjengelig senere

# --- Konfigurasjon (hovedkolonne) ---
with st.expander("⚙️ Konfigurasjon av systemprompt", expanded=False):
    with st.form(key="sys_prompt_form_main", border=True):
        sys_prompt_input = st.text_area(
            "Systemprompt (norsk)",
            value=DEFAULT_SYSTEM_PROMPT,
            height=120
        )
        use_prompt_btn = st.form_submit_button("Bruk denne prompten")

    if use_prompt_btn:
        st.session_state["sys_prompt"] = sys_prompt_input
        
# Henter gjeldende systemprompt (brukerens eller default)
current_sys_prompt = st.session_state.get("sys_prompt", DEFAULT_SYSTEM_PROMPT)

# --- Sidepanel for valg av Retriever---
retriever_mode = st.sidebar.radio("Retriever", ["Lokal (NumPy)", "ChromaDB"], index=1)

# --- Sidepanel for adaptiv chunking ---
adaptive_chunking = st.sidebar.checkbox("Adaptiv chunking (prosentbasert)", value=True)
if adaptive_chunking:
    st.sidebar.markdown(
        """
        **Merk:** Ved adaptiv chunking justeres chunk-størrelsen basert på dokumentets totale lengde.
        Dette kan forbedre ytelsen for både små og store dokumenter.
        """
    )

# Sørg for at nødvendig mappe finnes
os.makedirs("data/raw", exist_ok=True) 

# Filopplasting
uploaded = st.file_uploader("Last opp en PDF-fil", type=["pdf"])

if uploaded:
    # Lager path for lagring av filen
    pdf_path = os.path.join("data", "raw", uploaded.name)
    # Sjekker om filen allerede finnes
    if os.path.exists(pdf_path):
        st.warning(f"Filen '{uploaded.name}' finnes allerede i mappen. Endre navn og prøv igjen.")
    else:
        # åpner i binary mode for å unngå encoding-problemer w-write b-binary
        with open(pdf_path, "wb") as f:
            # skriver buffer direkte til fil
            f.write(uploaded.getbuffer())
        st.success(f"Lagret: {uploaded.name}")



# --- Velg dokument fra mappe ---
st.sidebar.markdown("<br><br>", unsafe_allow_html=True)  # Legger til litt luft 
st.sidebar.markdown("### 📄 Velg dokument fra mappen") # Større overskrift

pdf_files = sorted(glob.glob("data/raw/*.pdf"))
# Valg av dokument, label_visibility="collapsed" skjuler label- dette for å unngå dobbel label og exepct når listen er tom
choice = st.sidebar.selectbox("Velg dokument fra mappen", pdf_files, index=0 if pdf_files else None, label_visibility="collapsed")
# choice = st.sidebar.selectbox("", pdf_files, index=0 if pdf_files else None)
st.sidebar.caption("Legg PDF-er i data/raw/ og oppdater listen.")



st.markdown("### ❓ Skriv inn spørsmålet ditt til dokumentet")
with st.form(key="question_form"):
    spm = st.text_area("Spørsmål", placeholder="Skriv et presist spørsmål …", height=140)
    submit_btn = st.form_submit_button("💬 Send")

if choice:
    with st.spinner("Leser og deler opp per side..."):
         pages = extract_pages(choice)
         chunks_meta = split_pages_into_chunks(pages, size=1200, overlap=180, adaptive=adaptive_chunking)
         chunks = [c["content"] for c in chunks_meta]
         
    doc_preview = " ".join(chunks)[:8000]
    doc_class, doc_score = classify_document_ml(doc_preview)  # ENDRING: ML
    st.caption(f"📄 Klassifisering: **{doc_class}** (score {doc_score:.2f})")
    
    filename = os.path.basename(choice)
    
    # Lager en stabil nøkkel for dokumentet (SHA-1 + modellnavn+ chunking)
    key = cache_key_for_file(choice, EMBED_MODEL, adaptive_chunking)

    # metadata til Chroma (doc + page/start/end)
    metadatas = [{"doc": key, "filename": filename, "page": c["page"], "start": c["start"], "end": c["end"], "class": doc_class} for c in chunks_meta]

    st.write(f"**Aktivt dokument:** {os.path.basename(choice)}")
    st.write(f"**Antall chunks:** {len(chunks)}")
    
    if retriever_mode == "ChromaDB":
        client_ch = get_client(persist_dir="data/chroma")
        coll = get_collection(client_ch, name="pdf_chunks")
        
        # Indekser kun hvis det ikke finnes eksisterende poster for dette dokumentet
        exists = coll.get(where={"doc": key}, limit=1)
        if not exists.get("ids"):
            upsert_chunks(coll, doc_id=key, chunks=chunks, metadatas=metadatas)
            st.success("Indeksering fullført (Chroma).")
            
        LABELS = ["faktura", "bestilling", "rapport", "annet", "kostnadsoverslag", "kontrakt"]
        
        if submit_btn and spm:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            where = {}
            if scope == "Kun valgt dokument":
                where["doc"] = key
            else:
                # ENDRING: LLM-router snevrer søket til klasse
                label, conf = classify_question_llm(spm, LABELS, threshold=0.55)
                st.caption(f"🧭 Intent (LLM): **{label}** (conf {conf:.2f})")
                if label != "annet":
                    where["class"] = {"$in": [label]}
           
            hits = query_topk(coll, spm, k=3, where=where)
            
            # Fallback: hvis ingen treff i snevret søk (kun for 'Alle dokumenter')
            if not hits and scope == "Alle dokumenter":
                hits = query_topk(coll, spm, k=3, where={})
            
            
            top_chunks = [h[1] for h in hits]  # chunk-tekster fra treffene
            answer, cites = answer_with_top_chunks(
                client, spm, top_chunks,
                system_prompt=current_sys_prompt # NEW: brukerens/standard prompt
            )

            st.markdown("### ✅ Svar")
            st.write(answer)
            with st.expander("Vis sitater (med side)"):
                for i, (hid, text, meta) in enumerate(hits):
                    st.markdown(f"**Treff {i+1} - side {meta.get('page')}**  \n> {text[:200]} …")
    
    else:
        # --- Embeddings cache pr. fil ---
        vecs = load_cached_vectors("indexes", key)

        if vecs is None:
            with st.spinner("Lager embeddings (første gang for dette dokumentet)..."):
                client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                vecs = embed_texts(client, chunks)
                save_cached_vectors("indexes", key, vecs)
            st.success("Indeksering fullført (cache lagret).")

        # --- Spørsmål → svar ---
        if submit_btn and spm:
            with st.spinner("Søker i dokumentet og genererer svar..."):
                client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                answer, cites = answer_with_context(client, spm, chunks, vecs, k=3,
                system_prompt=current_sys_prompt  # brukerens/standard prompt
                )
                

            st.markdown("### ✅ Svar")
            st.write(answer)

            with st.expander("Vis sitater (med side)"):
                for i, snip in cites:
                    page = chunks_meta[i]["page"]
                    st.markdown(f"**Chunk {i} – side {page}:**\n\n> {snip} …")
else:
        st.info("Legg inn PDF-er i `data/raw/`, velg ett i venstremenyen og still et spørsmål.")
    

