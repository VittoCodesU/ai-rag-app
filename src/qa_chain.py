# src/qa_chain.py
import os
import streamlit as st
from dotenv import load_dotenv

# LLMs
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_openai import ChatOpenAI

# Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

# Vector store & chain
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv(override=True)

# Rutas
PERSIST_DIR = os.path.join(os.path.dirname(__file__), "..", "vectordb")

# ===== Prompts estrictos (anti-alucinación) =====
QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Eres un asistente de QA sobre documentos. "
     "Responde EXCLUSIVAMENTE con la información contenida en el CONTEXTO. "
     "Si no encuentras la respuesta en el contexto, di exactamente: "
     "'No encuentro esa información en los documentos.' "
     "Responde en ESPAÑOL, claro, sin inventar."),
    ("human",
     "Pregunta: {question}\n\n"
     "CONTEXTO (fragmentos relevantes):\n{context}")
])

CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Convierte la consulta de seguimiento en una pregunta independiente, clara y en español."),
    ("human", "Historial:\n{chat_history}\n\nConsulta actual: {question}")
])

# ===== Embeddings =====
def _embeddings_from_env():
    provider = os.getenv("EMBEDDINGS_PROVIDER", "hf").lower()
    if provider == "hf":
        model = os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        return HuggingFaceEmbeddings(
            model_name=model,
            encode_kwargs={"normalize_embeddings": True},
        )
    elif provider == "openai":
        model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
        return OpenAIEmbeddings(model=model)
    else:
        raise ValueError("EMBEDDINGS_PROVIDER debe ser 'hf' u 'openai'")

@st.cache_resource(show_spinner=False)
def cached_embeddings():
    return _embeddings_from_env()

# ===== LLM =====
def _llm_from_env():
    provider = os.getenv("LLM_PROVIDER", "hf").lower()
    if provider == "openai":
        model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
        return ChatOpenAI(model=model, temperature=0.2)

    # Hugging Face Hosted Inference API (featherless -> conversational)
    repo_id = os.getenv("HF_LLM_REPO", "mistralai/Mistral-7B-Instruct-v0.2")
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        raise RuntimeError("Falta HUGGINGFACEHUB_API_TOKEN en .env")

    base_llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        task="conversational",  # CLAVE: estos modelos no soportan text-generation
        huggingfacehub_api_token=token,
        temperature=float(os.getenv("HF_TEMPERATURE", "0.1")),
        max_new_tokens=int(os.getenv("HF_MAX_NEW_TOKENS", "160")),
        top_p=float(os.getenv("HF_TOP_P", "0.9")),
        timeout=int(os.getenv("HF_TIMEOUT", "120")),
    )
    return ChatHuggingFace(llm=base_llm)

@st.cache_resource(show_spinner=False)
def cached_llm():
    return _llm_from_env()

# ===== Vector store =====
@st.cache_resource(show_spinner=False)
def cached_vectorstore():
    return Chroma(
        collection_name="docs",
        embedding_function=cached_embeddings(),
        persist_directory=PERSIST_DIR,
    )

def _make_retriever(vectordb: Chroma, k: int):
    """Crea un retriever; intenta usar umbral si la versión lo soporta."""
    search_type = os.getenv("RETRIEVAL_MODE", "similarity").lower()
    if search_type == "threshold":
        try:
            return vectordb.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": k,
                    "score_threshold": float(os.getenv("SCORE_THRESHOLD", "0.2")),
                },
            )
        except Exception:
            pass  # cae a similarity si tu versión no lo soporta
    return vectordb.as_retriever(search_kwargs={"k": k})

# ===== API pública que usa la app =====
def get_embeddings():
    """Export para otros módulos (p.ej. ingest.py)"""
    return cached_embeddings()

def get_llm():
    """Export para otros módulos si lo necesitas."""
    return cached_llm()

def build_qa_chain(k: int = 4):
    vectordb = cached_vectorstore()
    retriever = _make_retriever(vectordb, k=min(k, 4))  # mantener k pequeño acelera

    # Memoria: indicar input/output para evitar conflictos de claves
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key="question",
        output_key="answer",
    )

    llm = cached_llm()

    # ¿Condensar preguntas? (2 llamadas al LLM) — se puede desactivar para rendimiento
    use_condense = os.getenv("USE_CONDENSE", "false").strip().lower() in ("1", "true", "yes")

    kwargs = {
        "llm": llm,
        "retriever": retriever,
        "memory": memory,
        "return_source_documents": True,
        "verbose": False,
        "output_key": "answer",
        "combine_docs_chain_kwargs": {"prompt": QA_PROMPT},
    }
    if use_condense:
        kwargs["question_generator_kwargs"] = {"prompt": CONDENSE_PROMPT}

    return ConversationalRetrievalChain.from_llm(**kwargs)
