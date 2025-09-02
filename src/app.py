import os, shutil, requests
import streamlit as st
from dotenv import load_dotenv
from utils import (
    list_index_rows, delete_by_sources, delete_all_index,
    list_data_files, delete_data_files
)

# =========================
# PAGE CONFIG (primero SIEMPRE)
# =========================
if not st.session_state.get("_page_config_done"):
    st.set_page_config(page_title="RAG Chatbot", page_icon="📄", layout="wide")
    st.session_state["_page_config_done"] = True

import sqlite3  # CHANGES: sin volver a importar streamlit

load_dotenv()

# -------------------------
# Validación de configuración
# -------------------------
def validate_config():
    provider = os.getenv("LLM_PROVIDER", "hf").lower()  # CHANGES: "hf" por defecto
    errs, warns = [], []

    if provider == "hf":
        if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
            errs.append("HUGGINGFACEHUB_API_TOKEN no está configurado")
        if not os.getenv("HF_LLM_REPO"):
            warns.append("HF_LLM_REPO no está configurado, se usará un modelo por defecto")
    elif provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            errs.append("Falta OPENAI_API_KEY")
    else:
        errs.append("LLM_PROVIDER debe ser 'hf' u 'openai'")

    try:
        t = int(os.getenv("HF_TIMEOUT", "120"))
        if t < 10:
            warns.append("`HF_TIMEOUT` es muy bajo (<10s).")
    except Exception:
        warns.append("`HF_TIMEOUT` inválido; usando 120s.")

    return errs, warns

errs, warns = validate_config()
for w in warns:
    st.sidebar.warning(w, icon="⚠️")
for e in errs:
    st.sidebar.error(e, icon="❌")
if errs:
    st.stop()

# -------------------------
# Métricas de latencia UI
# -------------------------
import time as _t
_t0 = _t.perf_counter()

def _mark(msg: str):
    st.sidebar.caption(f"⏱️ {msg}: {(_t.perf_counter()-_t0):.3f}s")

# -------------------------
# Título
# -------------------------
st.title("AI RAG CHATBOT")
st.caption("Sube documentos (PDF/DOCX/TXT), ingesta en base vectorial y chatea con su contenido.")

# Estado global
if "qa_chain" not in st.session_state or st.session_state.qa_chain is None:
    st.session_state.qa_chain = None
if "k" not in st.session_state:
    st.session_state.k = 4

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.text("Analisis de rendimiento:")
    _mark("post-imports")
    _mark("sidebar-rendered")
    _mark("chat-header")
    st.markdown("---")

    st.header("Inserta documentos")
    uploaded_files = st.file_uploader(
        "Sube uno o varios archivos (PDF, DOCX, TXT)",
        accept_multiple_files=True,
        type=["pdf", "docx", "txt"]
    )

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(data_dir, exist_ok=True)

    if uploaded_files:
        for uf in uploaded_files:
            save_path = os.path.join(data_dir, uf.name)
            with open(save_path, "wb") as f:
                f.write(uf.read())
        st.success(f"{len(uploaded_files)} archivo(s) guardado(s) en /data")
        st.session_state._data_files = list_data_files()

    if st.button("Reconstruir índice", key="rebuild_index_btn"):
        def run_ingest():
            from ingest import ingest as _ingest  # import diferido
            return _ingest()
        with st.spinner("Ingestando documentos y construyendo índice..."):
            run_ingest()
            st.session_state.qa_chain = None
            sources, _, _ = list_index_rows()
            st.session_state._indexed_sources = sorted(set(sources))
        st.success("Índice actualizado.")

    # CHANGES: este subheader estaba dentro del if ↑
    st.subheader("Gestión del índice / archivos")

    st.markdown("---")
    # Índice vectorial (lazy refresh)
    if "_indexed_sources" not in st.session_state:
        st.session_state._indexed_sources = []

    cols = st.columns([1, 1])
    with cols[0]:
        if st.button("Refrescar índice", key="refresh_sources_btn"):
            with st.spinner("Leyendo índice..."):
                sources, _, _ = list_index_rows()
                sources = sorted(set(sources))
                st.session_state._indexed_sources = sources
            st.success(f"Encontrados {len(st.session_state._indexed_sources)} documentos")

    with cols[1]:
        if st.button("Vaciar índice", key="clear_index_btn"):
            delete_all_index()
            st.session_state._indexed_sources = []
            st.session_state.qa_chain = None
            st.success("Índice vectorial vaciado.")

    sources = st.session_state._indexed_sources
    sel_sources = st.multiselect(
        "Documentos en el índice (por 'source')",
        options=sources,
        key="select_index_sources",
    )

    if st.button("Borrar seleccionados del índice", key="delete_selected_index_btn"):
        if sel_sources:
            delete_by_sources(sel_sources)
            left = [s for s in st.session_state._indexed_sources if s not in sel_sources]
            st.session_state._indexed_sources = left
            st.session_state.qa_chain = None
            st.success(f"Eliminados del índice: {len(sel_sources)} documento(s).")
        else:
            st.info("Selecciona al menos un documento.")

    st.markdown("—")

    # Archivos físicos en /data (lazy también)
    if "_data_files" not in st.session_state:
        st.session_state._data_files = []

    cols2 = st.columns([1, 1])
    with cols2[0]:
        if st.button("Refrescar /data", key="refresh_data_btn"):
            with st.spinner("Listando /data..."):
                st.session_state._data_files = list_data_files()
            st.success(f"{len(st.session_state._data_files)} archivo(s) en /data")

    data_files = st.session_state._data_files
    sel_files = st.multiselect("Archivos en /data", options=data_files, key="select_data_files")

    if st.button("Borrar seleccionados de /data", key="delete_selected_data_btn"):
        if sel_files:
            n = delete_data_files(sel_files)
            left = [s for s in st.session_state._data_files if s not in sel_files]
            st.session_state._data_files = left
            st.success(f"Eliminados de /data: {n} archivo(s).")
        else:
            st.info("Selecciona al menos un archivo de /data.")

    st.markdown("---")
    st.subheader("Ajustes")
    k = st.slider("Número de fragmentos a recuperar (top-k)", 2, 8, st.session_state.get("k", 4))
    if st.button("Aplicar k", key="apply_k_btn"):
        st.session_state.k = k
        st.session_state.qa_chain = None
        st.success(f"Now retrieving top-{k} chunks")

# =========================
# Chat
# =========================
st.subheader("Chat")

def get_qa_chain(k: int = 4):
    from qa_chain import build_qa_chain  # import diferido
    return build_qa_chain(k=k)

with st.form("chat_form", clear_on_submit=False):
    user_input = st.text_input("Escribe tu pregunta sobre los documentos:", key="chat_input")
    submitted = st.form_submit_button("Enviar", use_container_width=True)

if submitted and user_input.strip():
    if "qa_chain" not in st.session_state or st.session_state.qa_chain is None:
        with st.spinner("Inicializando motor de QA..."):
            st.session_state.qa_chain = get_qa_chain(k=st.session_state.get("k", 4))

    with st.spinner("Pensando..."):
        t0 = _t.time()
        result = st.session_state.qa_chain({"question": user_input, "chat_history": []})
        t1 = _t.time()

    answer = result["answer"]
    sources = result.get("source_documents", [])

    st.markdown("### Respuesta")
    st.write(answer)

    if sources:
        st.markdown("### Fuentes")
        for i, doc in enumerate(sources, start=1):
            meta = doc.metadata or {}
            st.write(f"{i}. **{meta.get('source','desconocido')}** (chunk {meta.get('chunk_id','?')})")

    st.caption(f"⏱️ Total: {(t1-t0):.2f}s")
