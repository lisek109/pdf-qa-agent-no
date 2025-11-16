import os
from pathlib import Path
import re, glob
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from app.parsers.pdf import extract_pages
from app.qa.chunking import split_pages_into_chunks
from app.qa.retrieval import embed_texts, answer_with_context, load_cached_vectors, save_cached_vectors, answer_with_top_chunks, cache_key_for_file
from app.qa.vectorstore_chroma import  get_client, get_collection, upsert_chunks, query_topk
from app.qa.prompts import DEFAULT_SYSTEM_PROMPT
from app.classifier.infer import classify_document_ml
from app.router_llm import classify_question_llm   

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

# Laster miljøvariabler fra .env (OpenAI-nøkkel osv.)
load_dotenv()
    
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


# --- INIT av session state ---
if "active_file" not in st.session_state:
    st.session_state["active_file"] = None
if "upload_reset" not in st.session_state:
    st.session_state["upload_reset"] = 0
if "global_mode" not in st.session_state:
    st.session_state["global_mode"] = False  # start i "Kun valgt dokument"
if "file_query" not in st.session_state:
    st.session_state["file_query"] = ""



# Hovedprogram for Streamlit-app
st.set_page_config(page_title="PDF-spørsmål (NO)", page_icon="📄", layout="wide")
# Laster CSS for tilpasset styling
load_css("assets/styles.css")
# Tittel
st.title("📄 PDF Assistent ")


# --- Sidepanel: OpenAI API-nøkkel ---
with st.sidebar:
    st.markdown("### 🔑 OpenAI API key")

    use_user_key = st.checkbox(
        "Bruk min egen nøkkel",
        value=True,
        help="Anbefalt for cluod eller delte miljøer.",
    )

    api_key = None
    if use_user_key:
        api_key = st.text_input(
            "Din OpenAI API Key",
            type="password",
            placeholder="sk-...",
        )
    else:
        # fallback til miljøvariabel
        api_key = os.getenv("OPENAI_API_KEY", "")

    if not api_key:
        st.info("Oppgi OpenAI API-nøkkel for å bruke appen.")

# Lagrer nøkkelen i session_state for gjenbruk ikke i disken
st.session_state["openai_api_key"] = api_key


def get_openai_client() -> OpenAI:
    """Returnerer OpenAI-klient med riktig API-nøkkel."""
    key = st.session_state.get("openai_api_key") or ""
    if not key:
        raise RuntimeError("Mangler OpenAI API-nøkkel. Vennligst oppgi en gyldig nøkkel i sidepanelet.")
    return OpenAI(api_key=key)



# --- Toggle: omfang på hovedsiden ---
global_mode = st.toggle(
    "Alle dokumenter",
    value=st.session_state["global_mode"],
    help="Søk på tvers av alle dokumenter",
    key="global_mode",
)
scope = "Alle dokumenter" if global_mode else "Kun valgt dokument"



# --- Konfigurasjon av systemprompt (hovedkolonne) ---
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
    
    
# --- WIDGET MED DYNAMISK NØKKEL ---
uploaded = st.file_uploader(
    "Last opp en PDF-fil",
    type=["pdf"],
    # Nøkkelen er dynamisk, f.eks. "uploader_0", "uploader_1", osv.
    key=f"uploader_{st.session_state['upload_reset']}",
)



# --- Funksjon: ingest til Chroma umiddelbart etter opplasting ---
def ingest_to_chroma(pdf_path: str, adaptive_chunking: bool ):
    """ENDRING: full ingest – les, chunk, klassifiser, upsert til Chroma."""
    pages = extract_pages(pdf_path)
    chunks_meta = split_pages_into_chunks(
        pages, size=1200, overlap=180, adaptive=adaptive_chunking
    )
    chunks = [c["content"] for c in chunks_meta]
    st.sidebar.markdown("---") 
    st.sidebar.info(f"Liczba chunków: {len(chunks)}")
    st.sidebar.code(f"Pierwszy chunk (preview):\n{chunks[0][:300]}...")
    st.sidebar.markdown("---")
    print(f"Delte dokumentet i {len(chunks)} chunks.")  # for debugging
    print(f"Første chunk preview: {chunks[0][:200]}...")  # for debugging

    # Klassifiser hele dokumentet (DIN modell)
    doc_preview = " ".join(chunks)[:8000]
    doc_class, doc_score = classify_document_ml(doc_preview)

    # Nøkkel + metadata
    key = cache_key_for_file(pdf_path, EMBED_MODEL, adaptive_chunking)
    print(f"Stabil nøkkel for dokumentet: {key} i ingest_to_chroma")  # for debugging
    filename = os.path.basename(pdf_path)
    metadatas = [
        {"doc": key, "filename": filename, "page": c["page"], "start": c["start"], "end": c["end"], "class": doc_class}
        for c in chunks_meta
    ]

    # Upsert til Chroma hvis ikke finnes
    client_ch = get_client(persist_dir="data/chroma")
    coll = get_collection(client_ch, name="pdf_chunks")
    exists = coll.get(where={"doc": key}, limit=1)
    if not exists.get("ids"):
        upsert_chunks(coll, doc_id=key, chunks=chunks, metadatas=metadatas, api_key=st.session_state.get("openai_api_key", ""),)
        print("Indeksering fullført (Chroma).") # for debugging
    return key, filename, chunks, chunks_meta, doc_class, doc_score




# --- Håndtering av opplasting ---    
if uploaded:
    # Definerer basekatalogen og sikrer at den eksisterer
    base_dir = os.path.join("data", "raw")
    os.makedirs(base_dir, exist_ok=True)
    
    # Bygger fullstendig filsti
    pdf_path = os.path.join(base_dir, os.path.basename(uploaded.name))
    pdf_path = pdf_path.replace("\\", "/") 
    print(f"Opplastet fil: {pdf_path}")  # for debugging

    if os.path.exists(pdf_path):
        # Hvis filen allerede finnes, bruk eksisterende
        st.info(f"Bruker eksisterende fil: {uploaded.name}")
    else:
        # Hvis filen er ny, skriv den til disk
        with open(pdf_path, "wb") as f:
            f.write(uploaded.getbuffer())
        st.success(f"Lagret: {uploaded.name}")
        
      # Direkte ingest til Chroma slik at filen er med i 'Alle dokumenter'
    try:
        key, filename, chunks, chunks_meta, doc_class, doc_score = ingest_to_chroma(pdf_path, adaptive_chunking)
        st.caption(f"📄 Klassifisering: **{doc_class}** (score {doc_score:.2f})")
    except Exception as e:
        st.warning(f"Ingest feilet: {e}")
    
    # NULLSTILLER WIDGETEN FOR FILOPPLASTING:
    # Øker telleren, noe som endrer 'key' for neste kjøring.
    st.session_state["upload_reset"] += 1
    # Sett som aktivt dokument og tvang 'Kun valgt dokument' for å jobbe direkte
    st.session_state["active_file"] = pdf_path
    #st.session_state["global_mode"] = False
    
    # Start appen på nytt for å laste widgeten med den nye nøkkelen/statusen
    st.rerun()
    
    
###############  Sidepanel: Velg dokument  ####################
st.sidebar.subheader("Dokumenter")
file_query = st.sidebar.text_input("🔎 Søk i filnavn", key="file_query", placeholder="f.eks. 'examp' eller 'fil.pdf'")
    
st.sidebar.markdown("<br><br>", unsafe_allow_html=True) # Legger til litt luft 
st.sidebar.markdown("### 📄 Velg dokument fra mappen") # Større overskrift

all_pdfs = sorted(glob.glob("data/raw/**/*.pdf", recursive=True))
all_pdfs = [p.replace("\\", "/") for p in all_pdfs]

if file_query:
    q = file_query.lower()
    # Filtrer PDF-liste basert på søkestrengen
    pdf_list_paths = [p for p in all_pdfs if os.path.basename(p).lower().find(q) != -1]
else:
    pdf_list_paths = all_pdfs
    
# Gjør om til bare filnavn for visning i selectbox
pdf_list_names = [os.path.basename(p) for p in pdf_list_paths]

# --- Widget SelectBox ---

# Hvis listen ikke er tom, prøv å finne indeksen til den aktive filen
if st.session_state.get("active_file") and st.session_state["active_file"] in pdf_list_paths:
    # Finn index til filen fra st.session_state["active_file"]
    default_index = pdf_list_names.index(os.path.basename(st.session_state["active_file"]))
else:
    default_index = 0 if pdf_list_names else None

choice_name = st.sidebar.selectbox(
    "Velg dokument fra mappen", 
    options=pdf_list_names, 
    index=default_index,
    # Skjuler label for å unngå dobbel label og expect når listen er tom
    label_visibility="collapsed", 
    key="selectbox_choice_name" # Ny nøkkel for selectbox for å unngå caching-problemer
)

# Endelig synkronsiering:
# Mappe valgt navn tilbake til full sti og lagre i session_state
if choice_name and choice_name != st.session_state.get("last_choice_name"):
    # Finn full sti basert på valgt navn
    selected_full_path = next((p for p in pdf_list_paths if os.path.basename(p) == choice_name), None)
    print(selected_full_path)  # for debugging
    
    if selected_full_path:
        st.session_state["active_file"] = selected_full_path
    
    # Lagre det siste valgte navnet for å unngå unødvendige oppdateringer
    st.session_state["last_choice_name"] = choice_name

choice = st.session_state.get("active_file")
print("Valgt dokument:", choice)  # for debugging
st.sidebar.caption("Legg PDF-er i data/raw/ og oppdater listen.")
    


###############  Spørsmål  ####################
st.markdown("### ❓ Skriv inn spørsmålet ditt til dokumentet")
with st.form(key="question_form"):
    spm = st.text_area("Spørsmål", placeholder="Skriv et presist spørsmål …", height=140)
    submit_btn = st.form_submit_button("💬 Send")


###############  Valg av omfang  ####################
if scope == "Kun valgt dokument" and choice:
    
    filename = os.path.basename(choice)
    # Lager en stabil nøkkel for dokumentet (SHA-1 + modellnavn+ chunking)
    key = cache_key_for_file(choice, EMBED_MODEL, adaptive_chunking)
    print(f"Stabil nøkkel for dokumentet: {key} i Kun valgt dokument")  # for debugging
    
    st.write(f"**Aktivt dokument:** {os.path.basename(choice)}")
    
    # Hvis user velger ChromaDB som retriever
    if retriever_mode == "ChromaDB":
        client_ch = get_client(persist_dir="data/chroma")
        coll = get_collection(client_ch, name="pdf_chunks")
        
        if submit_btn and spm:
            client = get_openai_client()
            where = {"doc": key}  # NB: alltid kun valgt dokument i denne grenen
            
            hits = query_topk(coll, spm, k=8, where=where, api_key=st.session_state.get("openai_api_key", ""),)
            st.info(f"Hits z query_topk: {len(hits)}") # <-- SPRAWDŹ!
            hits = prioritize_chunks_by_keywords(spm, hits, topk=3)
            st.info(f"Hits po priorytetyzacji: {len(hits)}") # <-- SPRAWDŹ!
            
            if not hits:
                st.warning("Ingen treff i valgt dokument.")
                top_chunks = []
            else:
                top_chunks = [h[1] for h in hits]
            answer, cites = answer_with_top_chunks(client, spm, top_chunks, system_prompt=current_sys_prompt)
            st.markdown("### ✅ Svar"); st.write(answer)
            with st.expander("Vis sitater (med side)"):
                for i, (hid, text, meta) in enumerate(hits):
                    st.markdown(f"**Treff {i+1} – side {meta.get('page')}**  \n> {text[:200]} …")
        
    else:
        # Lokal (NumPy) 
        client = get_openai_client()
        pages = extract_pages(choice)
        chunks_meta = split_pages_into_chunks(pages, size=1200, overlap=180, adaptive=adaptive_chunking)
        chunks = [c["content"] for c in chunks_meta]
        vecs = load_cached_vectors("indexes", key)
        if vecs is None:
            with st.spinner("Lager embeddings (første gang for dette dokumentet)..."):
                vecs = embed_texts(client, chunks)
                save_cached_vectors("indexes", key, vecs)
            st.success("Indeksering fullført (cache lagret).")
        if submit_btn and spm:
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
        client = get_openai_client()
        LABELS = ["faktura","bestilling","rapport","annet","kostnadsoverslag","kontrakt"]

        # LLM som router for hele korpuset
        label, conf = classify_question_llm(spm, LABELS, threshold=0.55)
        st.caption(f"🧭 Intent (LLM): **{label}** (conf {conf:.2f})")
        where = {"class": {"$in": [label]}} if label != "annet" else {}

        hits = query_topk(coll, spm, k=8, where=where, api_key=st.session_state.get("openai_api_key", ""),)
        print(f"Hits z query_topk (global): {len(hits)}")# <-- SPRAWDŹ!
        print(hits[0])  # for debugging
        hits = prioritize_chunks_by_keywords(spm, hits, topk=3)
        print(f"Hits po priorytetyzacji (global): {len(hits)}")# <-- SPRAWDŹ!
        print(hits[0])  # for debugging
        if not hits:
            # robust fallback til hele korpuset
            hits = query_topk(coll, spm, k=3, where={}, api_key=st.session_state.get("openai_api_key", ""),)

        top_chunks = [h[1] for h in hits]
        answer, cites = answer_with_top_chunks(client, spm, top_chunks, system_prompt=current_sys_prompt)
        st.markdown("### ✅ Svar"); st.write(answer)
        with st.expander("Vis sitater (fil/side)"):
            for i, (hid, text, meta) in enumerate(hits):
                st.markdown(f"**Treff {i+1} – {meta.get('filename','?')} – side {meta.get('page')}**  \n> {text[:200]} …")

# Mangler valg av dokument
else:
    st.info("Legg inn PDF-er i `data/raw/`, velg ett i venstremenyen og still et spørsmål.")
