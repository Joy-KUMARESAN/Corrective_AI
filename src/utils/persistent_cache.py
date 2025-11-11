# persistent_cache.py
import sqlite3, time, json, threading, hashlib
from pathlib import Path

_cache_lock = threading.Lock()

def _norm(s: str) -> str:
    return " ".join((s or "").strip().lower().split())

def _key(addr: str) -> str:
    return hashlib.sha1(_norm(addr).encode("utf-8")).hexdigest()

def _ensure_schema(con: sqlite3.Connection):
    con.execute("""
        CREATE TABLE IF NOT EXISTS geocache (
            k  TEXT PRIMARY KEY,
            v  TEXT NOT NULL,
            ts INTEGER NOT NULL
        )
    """)

def cache_get(db_path: str, addr: str, ttl_secs: int = 86400):
    """Return cached json (dict) or None if missing/expired."""
    if not addr:
        return None
    k = _key(addr)
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with _cache_lock, sqlite3.connect(db_path) as con:
        _ensure_schema(con)
        row = con.execute("SELECT v, ts FROM geocache WHERE k=?", (k,)).fetchone()
        if not row:
            return None
        v, ts = row
        if int(time.time()) - int(ts) > int(ttl_secs):
            con.execute("DELETE FROM geocache WHERE k=?", (k,))
            return None
        try:
            return json.loads(v)
        except Exception:
            con.execute("DELETE FROM geocache WHERE k=?", (k,))
            return None

def cache_set(db_path: str, addr: str, value: dict):
    """Persist dict value under normalized address key."""
    if not addr:
        return
    k = _key(addr)
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with _cache_lock, sqlite3.connect(db_path) as con:
        _ensure_schema(con)
        con.execute(
            "INSERT OR REPLACE INTO geocache (k, v, ts) VALUES (?, ?, ?)",
            (k, json.dumps(value), int(time.time()))
        )
