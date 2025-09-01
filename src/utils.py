# src/utils.py
import os
from typing import List, Dict, Any, Tuple
import docx2txt
from pypdf import PdfReader

from config import PERSIST_DIR, DATA_DIR

# -------- Lectura de archivos --------
def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def read_pdf(path: str) -> str:
    reader = PdfReader(path)
    texts = []
    for page in reader.pages:
        t = page.extract_text() or ""
        texts.append(t)
    return "\n".join(texts)

def read_docx(path: str) -> str:
    return docx2txt.process(path) or ""

def load_text_from_path(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".txt":
        return read_txt(path)
    if ext == ".pdf":
        return read_pdf(path)
    if ext == ".docx":
        return read_docx(path)
    raise ValueError(f"Formato no soportado: {ext}")

def clean_text(text: str) -> str:
    return " ".join(text.replace("\xa0", " ").split())

def chunk_stats(chunks: List[str]) -> dict:
    lengths = [len(c) for c in chunks]
    if not lengths:
        return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
    return {
        "count": len(chunks),
        "avg_len": sum(lengths) / len(lengths),
        "min_len": min(lengths),
        "max_len": max(lengths),
    }

# -------- Gestión de índice con chromadb nativo (sin LangChain) --------
import chromadb
from chromadb.config import Settings

def _open_collection():
    os.makedirs(PERSIST_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=PERSIST_DIR, settings=Settings(allow_reset=True))
    try:
        col = client.get_collection("docs")
    except Exception:
        col = client.create_collection("docs")
    return client, col

def list_index_rows() -> Tuple[list, list, list]:
    """Devuelve (sources_unicos, ids, filas) para pintar la UI."""
    client, col = _open_collection()
    try:
        raw = col.get(include=["metadatas", "ids"])
    except Exception:
        return [], [], []
    ids: list[str] = raw.get("ids", [])
    metas: list[Dict[str, Any]] = raw.get("metadatas", [])
    rows = []
    for i in range(len(ids)):
        md = metas[i] or {}
        md["id"] = ids[i]
        rows.append(md)
    sources = sorted({r.get("source", "desconocido") for r in rows})
    return sources, ids, rows

def delete_by_sources(sources: List[str]) -> int:
    if not sources:
        return 0
    client, col = _open_collection()
    col.delete(where={"source": {"$in": sources}})
    return 1

def delete_all_index() -> None:
    client, _ = _open_collection()
    try:
        client.delete_collection("docs")
    except Exception:
        # fallback: recrear
        try:
            client.get_collection("docs").delete(where={})
        except Exception:
            pass
    os.makedirs(PERSIST_DIR, exist_ok=True)

# -------- /data --------
def list_data_files() -> list[str]:
    os.makedirs(DATA_DIR, exist_ok=True)
    return sorted([f for f in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, f))])

def delete_data_files(files: List[str]) -> int:
    count = 0
    for f in files:
        p = os.path.join(DATA_DIR, f)
        if os.path.isfile(p):
            try:
                os.remove(p); count += 1
            except Exception:
                pass
    return count
