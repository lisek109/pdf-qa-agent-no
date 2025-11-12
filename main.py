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
        
        
# Funksjon for å prioritere chunks basert på nøkkelord
def prioritize_chunks_by_keywords(query: str, hits, topk: int = 3):
    # hits: liste av (id, text, meta)
    toks = [t for t in re.split(r"[\W_]+", query.lower()) if len(t) > 2]
    def score(text: str) -> int:
        tl = text.lower()
        return sum(1 for t in toks if t in tl)
    ranked = sorted(hits, key=lambda h: score(h[1]), reverse=True)
    return ranked[:topk]


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
    
    
# --- INITIALISERING AV SESSION_STATE ---
if "active_file" not in st.session_state:
    st.session_state["active_file"] = None
    
if "upload_reset" not in st.session_state:
    # Denne telleren brukes til å generere en unik 'key' for filopplasteren.
    # Ved å øke tallet tvinges widgeten til å nullstille seg.
    st.session_state["upload_reset"] = 0
    
# --- WIDGET MED DYNAMISK NØKKEL ---
uploaded = st.file_uploader(
    "Last opp en PDF-fil",
    type=["pdf"],
    # Nøkkelen er dynamisk, f.eks. "uploader_0", "uploader_1", osv.
    key=f"uploader_{st.session_state['upload_reset']}",
)
    
if uploaded:
    # Definerer basekatalogen og sikrer at den eksisterer
    base_dir = os.path.join("data", "raw")
    os.makedirs(base_dir, exist_ok=True)
    
    # Bygger fullstendig filsti
    pdf_path = os.path.join(base_dir, os.path.basename(uploaded.name))

    if os.path.exists(pdf_path):
        # Hvis filen allerede finnes, bruk eksisterende
        st.info(f"Bruker eksisterende fil: {uploaded.name}")
        st.session_state["active_file"] = pdf_path
    else:
        # Hvis filen er ny, skriv den til disk
        with open(pdf_path, "wb") as f:
            f.write(uploaded.getbuffer())
        st.success(f"Lagret: {uploaded.name}")
        st.session_state["active_file"] = pdf_path
    
    # NULLSTILLER WIDGETEN FOR FILOPPLASTING:
    # Øker telleren, noe som endrer 'key' for neste kjøring.
    st.session_state["upload_reset"] += 1
    
    # Start appen på nytt for å laste widgeten med den nye nøkkelen/statusen
    st.rerun()



# --- Velg dokument fra mappe ---
st.sidebar.markdown("<br><br>", unsafe_allow_html=True)  # Legger til litt luft 
st.sidebar.markdown("### 📄 Velg dokument fra mappen") # Større overskrift

pdf_files = sorted(glob.glob("data/raw/*.pdf"))
# Valg av dokument, label_visibility="collapsed" skjuler label- dette for å unngå dobbel label og exepct når listen er tom
choice = st.sidebar.selectbox("Velg dokument fra mappen", pdf_files, index=0 if pdf_files else None, label_visibility="collapsed")
st.sidebar.caption("Legg PDF-er i data/raw/ og oppdater listen.")



st.markdown("### ❓ Skriv inn spørsmålet ditt til dokumentet")
with st.form(key="question_form"):
    spm = st.text_area("Spørsmål", placeholder="Skriv et presist spørsmål …", height=140)
    submit_btn = st.form_submit_button("💬 Send")


###############  Valg av omfang  ####################
if scope == "Kun valgt dokument" and choice:
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

    # metadata 
    metadatas = [{"doc": key, "filename": filename, "page": c["page"], "start": c["start"], "end": c["end"], "class": doc_class}
                 for c in chunks_meta]
    
    st.write(f"**Aktivt dokument:** {os.path.basename(choice)}")
    st.write(f"**Antall chunks:** {len(chunks)}")
    
    # Hvis user velger ChromaDB som retriever
    if retriever_mode == "ChromaDB":
        client_ch = get_client(persist_dir="data/chroma")
        coll = get_collection(client_ch, name="pdf_chunks")
        
        # Indekser kun hvis det ikke finnes eksisterende poster for dette dokumentet
        exists = coll.get(where={"doc": key}, limit=1)
        if not exists.get("ids"):
            upsert_chunks(coll, doc_id=key, chunks=chunks, metadatas=metadatas)
            st.success("Indeksering fullført (Chroma).")
            
        ###################LABELS = ["faktura", "bestilling", "rapport", "annet", "kostnadsoverslag", "kontrakt"]
        
        if submit_btn and spm:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            where = {"doc": key}  # NB: alltid kun valgt dokument i denne grenen
            hits = query_topk(coll, spm, k=8, where=where)
            hits = prioritize_chunks_by_keywords(spm, hits, topk=3)
            if not hits:
                st.warning("Ingen treff i valgt dokument.")
            top_chunks = [h[1] for h in hits]
            answer, cites = answer_with_top_chunks(client, spm, top_chunks, system_prompt=current_sys_prompt)
            st.markdown("### ✅ Svar"); st.write(answer)
            with st.expander("Vis sitater (med side)"):
                for i, (hid, text, meta) in enumerate(hits):
                    st.markdown(f"**Treff {i+1} – side {meta.get('page')}**  \n> {text[:200]} …")
        
    else:
        # Lokal (NumPy) – jak masz teraz
        vecs = load_cached_vectors("indexes", key)
        if vecs is None:
            with st.spinner("Lager embeddings (første gang for dette dokumentet)..."):
                client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                vecs = embed_texts(client, chunks)
                save_cached_vectors("indexes", key, vecs)
            st.success("Indeksering fullført (cache lagret).")
        if submit_btn and spm:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            answer, cites = answer_with_context(client, spm, chunks, vecs, k=3, system_prompt=current_sys_prompt)
            st.markdown("### ✅ Svar"); st.write(answer)
            with st.expander("Vis sitater (med side)"):
                for i, snip in cites:
                    page = chunks_meta[i]["page"]
                    st.markdown(f"**Chunk {i} – side {page}:**\n\n> {snip} …")
                    
###############  Globalt omfang  ####################
elif scope == "Alle dokumenter":
    client_ch = get_client(persist_dir="data/chroma")
    coll = get_collection(client_ch, name="pdf_chunks")
    
    if submit_btn and spm:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        LABELS = ["faktura","bestilling","rapport","annet","kostnadsoverslag","kontrakt"]

        # ENDRING: LLM som router for hele korpuset
        label, conf = classify_question_llm(spm, LABELS, threshold=0.55)
        st.caption(f"🧭 Intent (LLM): **{label}** (conf {conf:.2f})")
        where = {"class": {"$in": [label]}} if label != "annet" else {}

        hits = query_topk(coll, spm, k=8, where=where)
        hits = prioritize_chunks_by_keywords(spm, hits, topk=3)
        if not hits:
            # robust fallback til hele korpuset
            hits = query_topk(coll, spm, k=3, where={})

        top_chunks = [h[1] for h in hits]
        answer, cites = answer_with_top_chunks(client, spm, top_chunks, system_prompt=current_sys_prompt)
        st.markdown("### ✅ Svar"); st.write(answer)
        with st.expander("Vis sitater (fil/side)"):
            for i, (hid, text, meta) in enumerate(hits):
                st.markdown(f"**Treff {i+1} – {meta.get('filename','?')} – side {meta.get('page')}**  \n> {text[:200]} …")

# Mangler valg av dokument
else:
    st.info("Legg inn PDF-er i `data/raw/`, velg ett i venstremenyen og still et spørsmål.")
