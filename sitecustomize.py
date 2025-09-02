# sitecustomize.py
# Reemplaza el módulo stdlib sqlite3 por pysqlite3 (SQLite moderno) si está disponible.
import os,sys

try:
    import pysqlite3  # noqa: F401
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    # Si no está, no rompemos el arranque; Chroma seguirá mostrando un error claro
    pass
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")