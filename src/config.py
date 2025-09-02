# src/config.py
import os
BASE_DIR   = os.path.dirname(__file__)
PERSIST_DIR = os.path.join(BASE_DIR, "..", "vectordb")
DATA_DIR    = os.path.join(BASE_DIR, "..", "data")
os.environ["CHROMA_DB_IMPL"] = "duckdb+parquet"