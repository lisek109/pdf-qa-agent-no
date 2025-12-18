# main.py
import os
from pathlib import Path
import re, glob
import numpy as np
import tempfile
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from app.parsers.pdf import extract_pages
from app.qa.chunking import split_pages_into_chunks
from app.qa.retrieval import embed_texts, answer_with_context, load_cached_vectors, save_cached_vectors, answer_with_top_chunks, cache_key_for_file, file_sha1, prioritize_chunks_by_keywords
from app.qa.vectorstore_chroma import  get_client, get_collection, upsert_chunks, query_topk
from app.qa.ingest import ingest_to_chroma
from app.qa.prompts import DEFAULT_SYSTEM_PROMPT
from app.router_llm import classify_question_llm   
from enum import Enum
from app.cloud_storage import list_bruker_dokumenter, sporr_chunks, lagre_pdf, lagre_chunks
from app.qa.qa_utils import embed_sporsmal


class StorageBackend(Enum):
    LOCAL = "local"
    CLOUD = "cloud"


# Velg lagrings-backend (kan styres via miljøvariabel i container)
_BACKEND_MODE = os.getenv("BACKEND_MODE", "local")
try:
    STORAGE_BACKEND = StorageBackend(_BACKEND_MODE)
except ValueError:
    STORAGE_BACKEND = StorageBackend.LOCAL

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
        pass  # Fortsetter uten egen CSS hvis fil mangler
        
        

def get_openai_client() -> OpenAI:
    """Returnerer OpenAI-klient med riktig API-nøkkel."""
    key = st.session_state.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
    if not key:
        # Viser en vennlig beskjed i stedet for rød feilmelding
        st.warning("Oppgi OpenAI API-nøkkel i sidepanelet før du stiller spørsmål.")
        st.stop()  # Avbryter resten av skriptet uten stack trace
    return OpenAI(api_key=key)


def get_current_user_id() -> str:
    """
    Placeholder for user_id
    """
    return st.session_state.get("user_id", "demo_user")

user_id = get_current_user_id()


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
        value=False,
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



# --- Håndtering av opplasting ---    
if uploaded:
    
    if STORAGE_BACKEND is StorageBackend.LOCAL:
        # Definerer basekatalogen og sikrer at den eksisterer
        base_dir = os.path.join("data", "raw")
        os.makedirs(base_dir, exist_ok=True)
        
        # Bygger fullstendig filsti
        pdf_path = os.path.join(base_dir, os.path.basename(uploaded.name))
        pdf_path = pdf_path.replace("\\", "/") 
        print(f"Opplastet fil (lokal): {pdf_path}")  # for debugging

        if os.path.exists(pdf_path):
            # Hvis filen allerede finnes, bruk eksisterende
            st.info(f"Bruker eksisterende fil: {uploaded.name}")
        else:
            # Hvis filen er ny, skriv den til disk
            with open(pdf_path, "wb") as f:
                f.write(uploaded.getbuffer())
            st.success(f"Lagret: {uploaded.name}")
            print(f"Lagret opplastet fil til: {pdf_path}")  # for debugging
            
        # Direkte ingest til Chroma slik at filen er med i 'Alle dokumenter'
        try:
            key, filename, chunks, chunks_meta, doc_class, doc_score = ingest_to_chroma(pdf_path, adaptive_chunking, user_id)
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
    else:
        # ---- SKY-MODUS: lagre PDF i chunks i embeddings i Cosmos/Blob ----
        data = uploaded.getvalue()
        try:
            # 1) Lagrer selve PDF-en i Blob + metadata i Cosmos
            dokument_id = lagre_pdf(user_id, uploaded.name, data)

            # 2) Ekstraher sider og chunks lokalt (midlertidig), og lag embeddings
            tmp_dir = os.path.join("data", "tmp_cloud")
            os.makedirs(tmp_dir, exist_ok=True)
            tmp_path = os.path.join(tmp_dir, f"{dokument_id}.pdf")
            
            # Skriver PDF til en midlertidig fil slik at extract_pages kan lese den
            with open(tmp_path, "wb") as f:
                f.write(data)

            pages = extract_pages(tmp_path)
            chunks_meta = split_pages_into_chunks(
                pages,
                size=1200,
                overlap=180,
                adaptive=adaptive_chunking,
            )
            chunks = [c["content"] for c in chunks_meta]

            # Lager OpenAI-klient og embeddings for alle chunks
            client = get_openai_client()
            embeddings = embed_texts(client, chunks)

            # 3) Bygg struktur for lagring i Cosmos
            chunks_for_cloud = []
            for meta, emb, tekst in zip(chunks_meta, embeddings, chunks):
            # Sørger for at embedding er vanlig Python-liste (JSON-serialiserbar)
                if hasattr(emb, "tolist"):
                    emb_list = emb.tolist()
                else:
                    emb_list = emb

                chunks_for_cloud.append(
                    {
                        "tekst": tekst,
                        "page": meta.get("page"),
                        "embedding": emb_list,
                        "filnavn": uploaded.name,
                        # "dokumentklasse": <kan settes senere hvis du vil>
                    }
                )

            from app.cloud_storage import lagre_chunks as cloud_lagre_chunks

            antall = cloud_lagre_chunks(user_id, dokument_id, chunks_for_cloud)
            st.success(
                f"PDF lagret i sky for bruker {user_id} med {antall} chunks i Cosmos."
            )

            st.session_state["active_file"] = dokument_id

        except Exception as e:
            st.error(f"Uventet feil ved cloud-opplasting: {e}")

        st.session_state["upload_reset"] += 1
        st.rerun()
        
        
    
###############################################################
###############  Sidepanel: Velg dokument  ####################
st.sidebar.subheader("Dokumenter")
file_query = st.sidebar.text_input("🔎 Søk i filnavn", key="file_query", placeholder="f.eks. 'examp' eller 'fil.pdf'")
    
st.sidebar.markdown("<br><br>", unsafe_allow_html=True) # Legger til litt luft 
st.sidebar.markdown("### 📄 Velg dokument fra mappen") # Større overskrift


if STORAGE_BACKEND is StorageBackend.LOCAL:
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
    
else:
    # --- Sky-modus: dokumentliste fra cloud storage (multi-user) ---
    try:
        dokumenter = list_bruker_dokumenter(user_id)
    except NotImplementedError:
        dokumenter = []
        st.sidebar.warning("Cloud storage er ikke implementert ennå. (list_bruker_dokumenter)")

    if file_query and dokumenter:
        q = file_query.lower()
        dokumenter = [
            d for d in dokumenter
            if q in d.get("navn", "").lower()
        ]

    navn_liste = [d.get("navn", d.get("filnavn", "ukjent")) for d in dokumenter]

    if st.session_state.get("active_file"):
        aktiv_id = st.session_state["active_file"]
        aktiv_dok = next((d for d in dokumenter if d.get("id") == aktiv_id), None)
        if aktiv_dok:
            try:
                default_index = navn_liste.index(
                    aktiv_dok.get("navn", aktiv_dok.get("filnavn", "ukjent"))
                )
            except ValueError:
                default_index = 0 if navn_liste else None
        else:
            default_index = 0 if navn_liste else None
    else:
        default_index = 0 if navn_liste else None

    choice_name = st.sidebar.selectbox(
        "Velg dokument",
        options=navn_liste,
        index=default_index,
        label_visibility="collapsed",
        key="selectbox_choice_name",
    )

    if choice_name and choice_name != st.session_state.get("last_choice_name") and dokumenter:
        valgt = next(
            (d for d in dokumenter
             if d.get("navn", d.get("filnavn", "ukjent")) == choice_name),
            None,
        )
        if valgt:
            # I sky-modus lagrer vi dokument-ID, ikke filsti
            st.session_state["active_file"] = valgt.get("id")

        st.session_state["last_choice_name"] = choice_name

    choice = st.session_state.get("active_file")
    st.sidebar.caption("Du ser kun dokumentene som tilhører din bruker i sky-modus.")

print("Valgt dokument:", choice)  # for debugging




###############  Spørsmål  ####################
st.markdown("### ❓ Skriv inn spørsmålet ditt til dokumentet")
with st.form(key="question_form"):
    spm = st.text_area("Spørsmål", placeholder="Skriv et presist spørsmål …", height=140)
    submit_btn = st.form_submit_button("💬 Send")




###############################################################
###############  Valg av omfang og svar ######################
if scope == "Kun valgt dokument" and choice:
    
    client_ch = get_client(persist_dir="data/chroma")
    client = get_openai_client()
    
    if STORAGE_BACKEND is StorageBackend.LOCAL:
        filename = os.path.basename(choice)
        # Lager en stabil nøkkel for dokumentet (SHA-1 + modellnavn+ chunking)
        key = cache_key_for_file(choice, EMBED_MODEL, adaptive_chunking)
        print(f"Stabil nøkkel for dokumentet: {key} i Kun valgt dokument")  # for debugging
        
        st.write(f"**Aktivt dokument:** {filename}")
        
        # Hvis user velger ChromaDB som retriever
        if retriever_mode == "ChromaDB":
            coll = get_collection(client_ch, name="pdf_chunks")
            if submit_btn and spm:
                where = {"doc": key}  # NB: alltid kun valgt dokument i denne grenen
                hits = query_topk(coll, spm, k=3, where=where, api_key=st.session_state.get("openai_api_key", ""),)

                if not hits:
                    # Fallback: kanskje dokument er indeksert med annen chunking (adaptive/static)
                    # Prøv å finne treff basert på filename (uavhengig av doc-key)
                    fallback_where = {"filename": filename}
                    fallback_hits = query_topk(coll, spm, k=8, where=fallback_where, api_key=st.session_state.get("openai_api_key", ""),)
                    if fallback_hits:
                        st.info("Fant treff via filename-fallback — mulig annen chunking brukt ved indeksering.")
                        hits = fallback_hits
                        top_chunks = [h[1] for h in hits]
                    else:
                        st.warning("Ingen treff i valgt dokument.")
                        top_chunks = []
                else:
                    top_chunks = [h[1] for h in hits]
                answer, cites = answer_with_top_chunks(client, spm, top_chunks, system_prompt=current_sys_prompt)
                st.markdown("### ✅ Svar"); st.write(answer)
                if not answer.lower().strip().startswith("mangler"):
                    with st.expander("Vis sitater (med side)"):
                        for i, (hid, text, meta) in enumerate(hits):
                            st.markdown(f"**Treff {i+1} – side {meta.get('page')}**  \n> {text[:200]} …")
            
        else:
            # Lokal (NumPy) 
            # Merk: kunne optimalisert ved å cache både chunks og metadata sammen med vektorene,
            # slik at vi slipper å kjøre extract_pages og split_pages_into_chunks hver gang.

            pages = extract_pages(choice)
            chunks_meta = split_pages_into_chunks(pages, size=1200, overlap=180, adaptive=adaptive_chunking)
            chunks = [c["content"] for c in chunks_meta]
            vecs = load_cached_vectors("indexes", key)
            if vecs is None:
                # Fallback: hvis dokumentet tidligere ble indeksert med annen chunking (adaptive/static),
                # prøv å finne en eksisterende cachefil basert på filens SHA1 uavhengig av chunking-flag.
                sha = file_sha1(choice)
                pattern = os.path.join("indexes", f"{sha}__{EMBED_MODEL}_*.npy")
                matches = glob.glob(pattern)
                if matches:
                    vec_path = matches[0]
                    try:
                        vecs = np.load(vec_path)
                        st.info(f"Bruker eksisterende vektor-cache ({os.path.basename(vec_path)}) som fallback — mulig annen chunking enn valgt.")
                    except Exception:
                        vecs = None

                if vecs is None:
                    with st.spinner("Lager embeddings (første gang for dette dokumentet)..."):
                        vecs = embed_texts(client, chunks)
                        save_cached_vectors("indexes", key, vecs)
                    st.success("Indeksering fullført (cache lagret).")
                    
            if vecs is not None and len(vecs) != len(chunks):
                st.warning(
                    "Eksisterende vektor-cache passer ikke til nåværende chunking. "
                    "Lager nye embeddings for dette dokumentet."
                )
                with st.spinner("Lager embeddings på nytt..."):
                    vecs = embed_texts(client, chunks)
                    save_cached_vectors("indexes", key, vecs)
                st.success("Ny vektor-cache lagret.")
                
            if submit_btn and spm:
                answer, cites = answer_with_context(client, spm, chunks, vecs, k=3, system_prompt=current_sys_prompt)
                st.markdown("### ✅ Svar"); st.write(answer)
                if not answer.lower().strip().startswith("mangler"):
                    with st.expander("Vis sitater (med side)"):
                        for i, snip in cites:
                            page = chunks_meta[i]["page"]
                            st.markdown(f"**Chunk {i} – side {page}:**\n\n> {snip} …")
    else:
                # ---- SKY-MODUS: choice er dokument-ID, ikke filsti ----
        aktivt_navn = st.session_state.get("last_choice_name", "ukjent dokument")
        st.write(f"**Aktivt dokument (sky):** {aktivt_navn}")

        if submit_btn and spm:
            from app.cloud_storage import sporr_chunks
            sporsmal_emb = embed_sporsmal(client, spm)
            
            active_doc_id = st.session_state.get("active_document_id")
            
            try:
                treff = sporr_chunks(
                    bruker_id=user_id,
                    sporsmal_embedding=sporsmal_emb,
                    top_k=5,
                    dokument_id=choice,
                )
            except Exception as e:
                st.error(f"Feil ved cloud-søk: {e}")
                treff = []

            if not treff:
                st.warning("Ingen treff i sky-backend for dette dokumentet.")
            else:
                 # Mappee treff til samme struktur som lokal 'hits'-liste
                hits = []
                for t in treff:
                    meta = {
                        "docId": t.get("docId"),
                        "page": t.get("page") or t.get("side"),
                        "filnavn": t.get("filnavn"),
                        "dokumentklasse": t.get("dokumentklasse"),
                        "score": t.get("score"),
                    }
                    hits.append((t.get("id"), t.get("tekst", ""), meta))

                # Bruk samme svarfunksjon som lokalt
                top_chunks = [h[1] for h in hits]
                answer, cites = answer_with_top_chunks(
                    client,
                    spm,
                    top_chunks,
                    system_prompt=current_sys_prompt,
                )
                st.markdown("### ✅ Svar")
                st.write(answer)

                #  Vis sitater i samme stil- nydeliggg
                if not answer.lower().strip().startswith("mangler"):
                    with st.expander("Vis sitater (fra cloud)"):
                    
                        for i, (hid, text, meta) in enumerate(hits, start=1):
                            side = meta.get("page")
                            st.markdown(f"**Treff {i} – side {side}**  \n> {text[:200]} …")
                        
                    
###############  Globalt omfang  ####################
elif scope == "Alle dokumenter":
    client = get_openai_client()
    if STORAGE_BACKEND is StorageBackend.LOCAL:
        client_ch = get_client(persist_dir="data/chroma")
        coll = get_collection(client_ch, name="pdf_chunks")
        
        if submit_btn and spm:
            LABELS = ["faktura","bestilling","rapport","annet","kostnadsoverslag","kontrakt"]

            # LLM som router for hele korpuset
            label, conf = classify_question_llm(spm, LABELS, threshold=0.55, client=client,)
            st.caption(f"🧭 Intent (LLM): **{label}** (conf {conf:.2f})")
            where = {"class": {"$in": [label]}, "user_id": user_id} if label != "annet" else {}

            hits = query_topk(coll, spm, k=8, where=where, api_key=st.session_state.get("openai_api_key", ""),)
            hits = prioritize_chunks_by_keywords(spm, hits, topk=3)

            # Forsøk å sørge for at de endelige sitatene kommer fra ÉN fil når mulig.
            # 1) Hvis brukeren eksplisitt nevner et filnavn i spørsmålet, filtrer til den filen.
            # 2) Ellers, hvis treffene kommer fra flere filer, velg majoritetsfilen.
            if hits:
                filenames = [ (h[2].get('filename') or h[2].get('filnavn') or '') for h in hits ]
                unique_files = [f for f in set(filenames) if f]
                target_file = None
                qlow = spm.lower()
                # 1) eksplisitt nevnt fil i spørsmål?
                for f in unique_files:
                    base = os.path.splitext(os.path.basename(f))[0].lower()
                    if base in qlow or f.lower() in qlow:
                        target_file = f
                        break
                # 2) velg majoritetsfil hvis flere filer representert
                if target_file is None and len(unique_files) > 1:
                    from collections import Counter
                    cnt = Counter(filenames)
                    most_common = cnt.most_common(1)[0][0]
                    if most_common:
                        target_file = most_common
                # Filtrer hits til target_file hvis vi fant en og filtreringen ikke tømmer resultatet
                if target_file:
                    filtered = [h for h in hits if (h[2].get('filename') or h[2].get('filnavn') or '') == target_file]
                    if filtered:
                        hits = filtered

            if not hits:
                # robust fallback til hele korpuset
                hits = query_topk(coll, spm, k=3, where={"user_id": user_id}, api_key=st.session_state.get("openai_api_key", ""),)

            top_chunks = [h[1] for h in hits]
            answer, cites = answer_with_top_chunks(client, spm, top_chunks, system_prompt=current_sys_prompt)
            st.markdown("### ✅ Svar"); st.write(answer)
            if not answer.lower().strip().startswith("mangler"):
                with st.expander("Vis sitater (fil/side)"):
                    for i, (hid, text, meta) in enumerate(hits):
                        st.markdown(f"**Treff {i+1} – {meta.get('filename','?')} – side {meta.get('page')}**  \n> {text[:200]} …")
    else:
        # ---- SKY-MODUS: globalt søk på tvers av alle dokumenter ----
        if submit_btn and spm:
            # 1) Lag embedding for spørsmålet
            sporsmal_emb = embed_sporsmal(client, spm)

            try:
                # 2) Søk i alle dokumenter for brukeren (dokument_id=None)
                treff = sporr_chunks(
                    bruker_id=user_id,
                    sporsmal_embedding=sporsmal_emb,
                    top_k=5,
                    dokument_id=None,  # alle dokumenter for brukeren
                )
            except Exception as e:
                st.error(f"Feil ved cloud-globalt søk: {e}")
                treff = []

            if not treff:
                st.warning("Ingen treff i sky-backend for globalt søk.")
            else:
                # 3) Mapp treff til 'hits'-struktur
                hits = []
                for t in treff:
                    meta = {
                        "docId": t.get("docId"),
                        "page": t.get("page") or t.get("side"),
                        "filnavn": t.get("filnavn"),
                        "dokumentklasse": t.get("dokumentklasse"),
                        "score": t.get("score"),
                    }
                    hits.append((t.get("id"), t.get("tekst", ""), meta))

                top_chunks = [h[1] for h in hits]
                answer, cites = answer_with_top_chunks(
                    client,
                    spm,
                    top_chunks,
                    system_prompt=current_sys_prompt,
                )
                st.markdown("### ✅ Svar")
                st.write(answer)

                if not answer.lower().strip().startswith("mangler"):
                    with st.expander("Vis sitater (fra cloud, alle dokumenter)"):
                        for i, (hid, text, meta) in enumerate(hits, start=1):
                            side = meta.get("page")
                            filnavn = meta.get("filnavn", "?")
                            st.markdown(f"**Treff {i} – {filnavn} – side {side}**  \n> {text[:200]} …")

# Mangler valg av dokument
else:
    st.info("Legg inn PDF-er i `data/raw/`, velg ett i venstremenyen og still et spørsmål.")
