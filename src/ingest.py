# src/ingest.py
import os
from typing import List, Tuple
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# Usamos SIEMPRE los embeddings y el PERSIST_DIR definidos en qa_chain
from qa_chain import get_embeddings, PERSIST_DIR
from utils import load_text_from_path, clean_text, chunk_stats

load_dotenv(override=True)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120


def open_vectorstore() -> Chroma:
    """Abre Chroma garantizando que tenga embedding_function asociado."""
    os.makedirs(PERSIST_DIR, exist_ok=True)
    embeddings = get_embeddings()
    if embeddings is None:
        raise RuntimeError(
            "get_embeddings() devolvió None. Revisa tu .env: "
            "EMBEDDINGS_PROVIDER=hf (o openai) y modelos configurados."
        )
    return Chroma(
        collection_name="docs",
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )


def build_docs_from_folder(folder: str) -> List[Tuple[str, str]]:
    """(filename, cleaned_text) para cada archivo legible en la carpeta."""
    docs: List[Tuple[str, str]] = []
    if not os.path.isdir(folder):
        return docs

    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            continue
        try:
            text = clean_text(load_text_from_path(path))
            if text.strip():
                docs.append((name, text))
        except Exception as e:
            print(f"[WARN] No se pudo cargar {name}: {e}")
    return docs


def split_text(text: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_text(text)


def ingest():
    """Ingesta todo lo que haya en /data a la colección 'docs'."""
    vectorstore = open_vectorstore()

    pairs = build_docs_from_folder(DATA_DIR)
    if not pairs:
        print(f"[INFO] No se encontraron documentos en {DATA_DIR}.")
        return

    total_chunks = 0
    for fname, text in pairs:
        chunks = split_text(text)
        stats = chunk_stats(chunks)
        print(
            f"[INFO] {fname}: {stats['count']} chunks "
            f"(avg={stats['avg_len']:.1f}, min={stats['min_len']}, max={stats['max_len']})"
        )

        metadatas = [{"source": fname, "chunk_id": i} for i in range(len(chunks))]
        # add_texts → Chroma calculará embeddings usando embedding_function
        vectorstore.add_texts(texts=chunks, metadatas=metadatas)
        total_chunks += len(chunks)

    print(f"[OK] Ingesta completa. Chunks totales: {total_chunks}. Índice: {PERSIST_DIR}")


if __name__ == "__main__":
    ingest()
