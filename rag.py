"""Local vector-store RAG for Gainsay.

Example:
    import rag
    rag.index_path("./docs")                       # one-time or when files change
    for chunk in rag.search("what's the refund policy?", k=5):
        print(chunk["path"], chunk["text"])

Stores chunks + nomic-embed-text vectors in a sqlite db. The default db lives at
``./gainsay_rag.db`` (override with the GAINSAY_RAG_DB env var, or pass an
explicit ``instance`` to target a separate, isolated db file). The corpus is
user-supplied and starts empty — index your own documents to populate it.

Embeddings are produced by a local Ollama server (set OLLAMA_BASE /
GAINSAY_EMBED_MODEL to point elsewhere). Nothing leaves the machine.
"""
from __future__ import annotations

import hashlib
import html as _html
import json
import os
import re as _re
import sqlite3
import struct
import threading
import urllib.request
from pathlib import Path
from typing import Iterable

# numpy turns the per-query cosine scan from a pure-Python loop over every chunk
# (O(N) Python) into a single cached matmul (<1s after first load). Optional: if
# numpy is missing we fall back to the original loop so nothing breaks.
try:
    import numpy as _np
    _NUMPY = True
except ImportError:
    _np = None
    _NUMPY = False

# In-process vector cache, keyed by db path. Holds a normalized float32 matrix plus
# compact metadata so a query is one matmul. Invalidated when the chunk count or max
# rowid changes (i.e. the index was updated). A long-running server loads it once and
# reuses it across queries.
_VEC_CACHE: dict = {}
_VEC_LOCK = threading.Lock()

# HTML→text strip for indexing .html / .htm files. Without this, saved
# webpages get indexed as raw HTML — DOCTYPE, CSS, span-tag soup —
# producing many semantically-empty chunks (lots of boilerplate vs real content).
_HTML_SCRIPT_RE = _re.compile(r"<(script|style)[^>]*>.*?</\1>",
                              _re.DOTALL | _re.IGNORECASE)
_HTML_TAG_RE = _re.compile(r"<[^>]+>")


def _strip_html(s: str) -> str:
    s = _HTML_SCRIPT_RE.sub("", s)
    s = _HTML_TAG_RE.sub("", s)
    s = _html.unescape(s)
    s = _re.sub(r"[ \t]+", " ", s)
    s = _re.sub(r"\n\s*\n\s*\n+", "\n\n", s)
    return s.strip()

OLLAMA_BASE = os.environ.get("OLLAMA_BASE", "http://localhost:11434")
EMBED_MODEL = os.environ.get("GAINSAY_EMBED_MODEL", "nomic-embed-text")
CHUNK_CHARS = 1400      # ~350 tokens — comfortable for nomic's 8k ctx
CHUNK_OVERLAP = 160
MAX_FILE_BYTES = 2_000_000
# Prose/notes corpus only — never code. A broad allowlist lets helper scripts and
# source files pollute the index, so `def __init__` can outrank actual documents for
# natural-language queries. If you ever need to RAG over code, do it in a separate
# db/instance — not the same store as your knowledge corpus.
TEXT_EXTS = {".txt", ".md", ".rst"}

# Default db path. Generic and relative so the public build writes next to the
# working directory rather than any user-specific location. Override with
# GAINSAY_RAG_DB to point at an absolute path.
_DEFAULT_DB = os.environ.get("GAINSAY_RAG_DB", "gainsay_rag.db")


def _db_path(instance: str | None = None) -> str:
    """Resolve the rag.db path.

    ``instance=None`` (default) uses the default db (``./gainsay_rag.db`` or
    GAINSAY_RAG_DB). Pass an explicit ``instance`` name to target a separate,
    isolated db file — e.g. a quarantine store for untrusted content — so a single
    process can keep several corpora apart. An instance name that already looks like
    a path (contains a separator or a ``.db`` suffix) is used verbatim; a bare name
    becomes ``<dir-of-default>/<instance>.db``.
    """
    if instance is None:
        return _DEFAULT_DB
    name = instance.strip()
    if not name:
        return _DEFAULT_DB
    if os.sep in name or (os.altsep and os.altsep in name) or name.lower().endswith(".db"):
        return name
    return os.path.join(os.path.dirname(os.path.abspath(_DEFAULT_DB)), f"{name}.db")


def _conn(instance: str | None = None) -> sqlite3.Connection:
    path = _db_path(instance)
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    c = sqlite3.connect(path, timeout=60.0)
    # WAL mode lets one writer run concurrent with many readers — so a background bulk
    # index doesn't lock out a concurrent stats() / search() call.
    c.execute("PRAGMA journal_mode=WAL")
    # index_path() holds ONE write transaction open for an entire bulk run (every chunk
    # embedded before the single commit) — minutes for a big corpus. A generous
    # busy_timeout lets a transient writer wait it out instead of failing with
    # "database is locked".
    c.execute("PRAGMA busy_timeout=60000")
    c.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id      INTEGER PRIMARY KEY,
            path    TEXT    NOT NULL,
            mtime   REAL    NOT NULL,
            hash    TEXT    NOT NULL,
            ord     INTEGER NOT NULL,
            text    TEXT    NOT NULL,
            vec     BLOB    NOT NULL
        )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(path)")
    return c


def _embed(text: str) -> list[float]:
    body = json.dumps({"model": EMBED_MODEL, "prompt": text}).encode()
    req = urllib.request.Request(
        f"{OLLAMA_BASE}/api/embeddings",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.loads(r.read())
    vec = data.get("embedding") or []
    if not vec:
        raise RuntimeError(f"empty embedding from {EMBED_MODEL}")
    return vec


def _pack(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _unpack(blob: bytes) -> list[float]:
    return list(struct.unpack(f"{len(blob)//4}f", blob))


def _chunk(text: str) -> list[str]:
    out = []
    i = 0
    n = len(text)
    while i < n:
        j = min(i + CHUNK_CHARS, n)
        out.append(text[i:j])
        if j >= n:
            break
        i = j - CHUNK_OVERLAP
    return out


def _iter_files(root: str) -> Iterable[Path]:
    p = Path(root)
    if p.is_file():
        yield p
        return
    for entry in p.rglob("*"):
        if not entry.is_file():
            continue
        if entry.suffix.lower() not in TEXT_EXTS:
            continue
        try:
            if entry.stat().st_size > MAX_FILE_BYTES:
                continue
        except OSError:
            continue
        yield entry


def index_path(root: str, verbose: bool = True, instance: str | None = None) -> dict:
    """Index a file or directory. Skips files unchanged since last index.
    `instance` targets a separate, isolated db (default: the main db)."""
    c = _conn(instance)
    added = updated = skipped = 0
    for f in _iter_files(root):
        fpath = str(f.resolve())
        try:
            mtime = f.stat().st_mtime
        except OSError:
            continue
        row = c.execute(
            "SELECT mtime FROM chunks WHERE path = ? LIMIT 1", (fpath,)
        ).fetchone()
        if row and abs(row[0] - mtime) < 1.0:
            skipped += 1
            continue
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        # HTML files: strip tags + scripts + entities so chunks contain
        # the actual prose, not boilerplate (DOCTYPE, CSS, span tags).
        if f.suffix.lower() in (".html", ".htm"):
            text = _strip_html(text)
        if not text.strip():
            continue
        digest = hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()
        # Wipe old rows for this path
        is_update = bool(row)
        c.execute("DELETE FROM chunks WHERE path = ?", (fpath,))
        for ord_, piece in enumerate(_chunk(text)):
            try:
                vec = _embed(piece)
            except Exception as e:
                if verbose:
                    print(f"[rag] embed failed for {fpath}#{ord_}: {e}")
                continue
            c.execute(
                "INSERT INTO chunks(path, mtime, hash, ord, text, vec) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (fpath, mtime, digest, ord_, piece, _pack(vec)),
            )
        if is_update:
            updated += 1
        else:
            added += 1
        if verbose:
            print(f"[rag] {'updated' if is_update else 'indexed'} {fpath}")
    c.commit()
    c.close()
    return {"added": added, "updated": updated, "skipped": skipped}


def _cos(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    da = sum(x * x for x in a) ** 0.5
    db = sum(y * y for y in b) ** 0.5
    if da == 0 or db == 0:
        return 0.0
    return num / (da * db)


def _cache_sig(c: sqlite3.Connection) -> tuple[int, int]:
    """Cheap fingerprint of the index: (row count, max rowid). Changes whenever
    chunks are added/updated/removed, so the cached matrix auto-invalidates."""
    row = c.execute("SELECT COUNT(*), COALESCE(MAX(id), 0) FROM chunks").fetchone()
    return (int(row[0]), int(row[1]))


def _load_vec_cache(c: sqlite3.Connection, db_path: str) -> dict | None:
    """Load every chunk vector into a normalized float32 matrix + compact metadata
    (path factorized to int ids; texts fetched lazily per query, not cached). One
    sequential pass over the store. Returns the cache entry or None if empty."""
    n = _cache_sig(c)[0]
    if n == 0:
        return None
    cur = c.execute("SELECT id, path, ord, vec FROM chunks ORDER BY id")
    first = cur.fetchone()
    dim = len(first[3]) // 4
    mat = _np.empty((n, dim), dtype=_np.float32)
    ids = _np.empty(n, dtype=_np.int64)
    ords = _np.empty(n, dtype=_np.int32)
    path_idx = _np.empty(n, dtype=_np.int32)
    paths: list[str] = []
    path_to_i: dict[str, int] = {}

    def _put(i, rid, path, ordn, blob):
        mat[i] = _np.frombuffer(blob, dtype=_np.float32, count=dim)
        ids[i] = rid
        ords[i] = ordn
        pi = path_to_i.get(path)
        if pi is None:
            pi = len(paths)
            path_to_i[path] = pi
            paths.append(path)
        path_idx[i] = pi

    _put(0, first[0], first[1], first[2], first[3])
    i = 1
    for rid, path, ordn, blob in cur:
        if i >= n:        # row added mid-load; sig recheck on next query handles it
            break
        _put(i, rid, path, ordn, blob)
        i += 1
    mat = mat[:i]; ids = ids[:i]; ords = ords[:i]; path_idx = path_idx[:i]
    norms = _np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat /= norms                      # row-normalized → dot product == cosine
    entry = {"sig": _cache_sig(c), "mat": mat, "ids": ids, "ords": ords,
             "path_idx": path_idx, "paths": paths,
             "id_to_row": {int(ids[j]): j for j in range(len(ids))}}
    _VEC_CACHE[db_path] = entry
    return entry


# ===== Hybrid search: FTS5 (BM25 keyword) fused with embeddings via RRF =====
# Embeddings miss exact terms (proper names, exact identifiers, API symbols); BM25
# catches them. Reciprocal Rank Fusion combines both rankings. The FTS5 table is
# EXTERNAL-CONTENT (content='chunks') so it stores only the index, not a copy of the
# text — minimal disk. Triggers keep it synced with future writes.
# Opt-in (search(hybrid=True)); default behavior is unchanged.
_RRF_K = 60          # standard RRF damping constant
_HYBRID_POOL = 60    # candidates pulled from each ranker before fusion
_FTS_WORD_RE = _re.compile(r"[0-9A-Za-z_]+")


def _ensure_fts(c: sqlite3.Connection) -> bool:
    """Create the FTS5 mirror + sync triggers if missing and backfill once from the
    existing chunks. Returns True if FTS is usable; False (→ caller falls back to pure
    embeddings) if this sqlite build lacks FTS5."""
    try:
        have = c.execute("SELECT name FROM sqlite_master WHERE type='table' "
                         "AND name='chunks_fts'").fetchone()
        if have:
            return True
        c.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5("
                  "text, content='chunks', content_rowid='id')")
        c.execute("CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN "
                  "INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text); END")
        c.execute("CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN "
                  "INSERT INTO chunks_fts(chunks_fts, rowid, text) "
                  "VALUES('delete', old.id, old.text); END")
        c.execute("CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN "
                  "INSERT INTO chunks_fts(chunks_fts, rowid, text) "
                  "VALUES('delete', old.id, old.text); "
                  "INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text); END")
        c.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")  # one-time backfill
        c.commit()
        return True
    except sqlite3.OperationalError:
        return False


def _fts_match(query: str) -> str:
    """Natural query → safe FTS5 MATCH string: alnum tokens OR-joined (recall-first),
    each quoted so FTS5 can't misparse punctuation/operators."""
    words = [w for w in _FTS_WORD_RE.findall(query.lower()) if len(w) > 1][:24]
    return " OR ".join(f'"{w}"' for w in words)


def _fts_search(c: sqlite3.Connection, query: str, limit: int) -> list[int]:
    """Chunk ids ranked by BM25 (best first). Empty on no match / FTS5 absent."""
    match = _fts_match(query)
    if not match:
        return []
    try:
        rows = c.execute("SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? "
                         "ORDER BY rank LIMIT ?", (match, limit)).fetchall()
        return [int(r[0]) for r in rows]
    except sqlite3.OperationalError:
        return []


def _rrf_fuse(*ranked_lists: list[int]) -> list[int]:
    """Reciprocal Rank Fusion of several id-rankings (each best-first)."""
    score: dict[int, float] = {}
    for lst in ranked_lists:
        for rank, _id in enumerate(lst):
            score[_id] = score.get(_id, 0.0) + 1.0 / (_RRF_K + rank + 1)
    return sorted(score, key=lambda i: score[i], reverse=True)


def search(query: str, k: int = 5, unique_paths: bool = True,
           instance: str | None = None, hybrid: bool = False) -> list[dict]:
    """Return top-k chunks for a query, ranked by cosine similarity.

    Each result is a dict: ``{"path", "ord", "text", "score"}``.

    `unique_paths=True` (default): each result comes from a different file.
    Otherwise a long file with many semantically-similar chunks (e.g. one
    HTML chapter at 1400-char granularity producing hundreds of chunks) can
    fill every slot for a generic query, drowning out shorter relevant docs.
    Pass unique_paths=False to allow multi-chunk results from the same
    file when you specifically want depth on one source.

    `hybrid=True` fuses the embedding ranking with a BM25 keyword ranking via
    Reciprocal Rank Fusion (helps with exact terms embeddings miss); it falls
    back to pure embeddings if FTS5 is unavailable or the query matched no
    keywords. `instance` selects an isolated db.

    Uses a cached, row-normalized numpy matrix (one matmul per query) when numpy
    is available; falls back to the pure-Python scan otherwise. Results are
    identical either way (exact cosine, same ordering)."""
    if not _NUMPY:
        return _search_python(query, k, unique_paths, instance)  # hybrid n/a without numpy
    db_path = _db_path(instance)
    c = _conn(instance)
    try:
        sig = _cache_sig(c)
        entry = _VEC_CACHE.get(db_path)
        if entry is None or entry.get("sig") != sig:
            with _VEC_LOCK:
                entry = _VEC_CACHE.get(db_path)        # re-check under lock
                if entry is None or entry.get("sig") != sig:
                    entry = _load_vec_cache(c, db_path)
        if entry is None:
            return []
        try:
            qv = _np.asarray(_embed(query), dtype=_np.float32)
        except Exception as e:
            raise RuntimeError(f"failed to embed query: {e}")
        qn = _np.linalg.norm(qv)
        if qn:
            qv = qv / qn
        scores = entry["mat"] @ qv                     # cosine (rows pre-normalized)
        emb_order = _np.argsort(-scores)               # best-first (embedding)
        mat_paths, path_idx = entry["paths"], entry["path_idx"]
        ids, ords = entry["ids"], entry["ords"]
        # Hybrid: fuse embedding ranking with BM25 keyword ranking via RRF. Falls back
        # to pure embeddings if FTS5 is absent or the query matched no keywords.
        order = emb_order
        if hybrid:
            fts_ids = _fts_search(c, query, _HYBRID_POOL) if _ensure_fts(c) else []
            if fts_ids:
                id_to_row = entry["id_to_row"]
                emb_ids = [int(ids[int(ri)]) for ri in emb_order[:_HYBRID_POOL]]
                fused = _rrf_fuse(emb_ids, fts_ids)
                rows = [id_to_row[i] for i in fused if i in id_to_row]
                seen_r = set(rows)
                # backfill from embedding order so unique_paths can always reach k
                for ri in emb_order[:max(_HYBRID_POOL * 4, k * 20)]:
                    r = int(ri)
                    if r not in seen_r:
                        rows.append(r)
                        seen_r.add(r)
                order = rows
        chosen: list[int] = []
        if unique_paths:
            seen: set[int] = set()
            for ri in order:
                pi = int(path_idx[ri])
                if pi in seen:
                    continue
                seen.add(pi)
                chosen.append(int(ri))
                if len(chosen) >= k:
                    break
        else:
            chosen = [int(ri) for ri in order[:k]]
        # Fetch texts only for the chosen rows (lazy — texts aren't held in cache).
        id_list = [int(ids[ri]) for ri in chosen]
        text_map: dict[int, str] = {}
        if id_list:
            ph = ",".join("?" * len(id_list))
            for rid, txt in c.execute(
                    f"SELECT id, text FROM chunks WHERE id IN ({ph})", id_list):
                text_map[rid] = txt
        out = []
        for ri in chosen:
            out.append({"path": mat_paths[int(path_idx[ri])], "ord": int(ords[ri]),
                        "text": text_map.get(int(ids[ri]), ""),
                        "score": float(scores[ri])})
        return out
    finally:
        c.close()


def _search_python(query: str, k: int = 5, unique_paths: bool = True,
                   instance: str | None = None) -> list[dict]:
    """Original pure-Python cosine scan — fallback when numpy is unavailable."""
    c = _conn(instance)
    rows = c.execute("SELECT path, ord, text, vec FROM chunks").fetchall()
    c.close()
    if not rows:
        return []
    try:
        qv = _embed(query)
    except Exception as e:
        raise RuntimeError(f"failed to embed query: {e}")
    scored: list[tuple[float, dict]] = []
    for path, ord_, text, vec_blob in rows:
        sim = _cos(qv, _unpack(vec_blob))
        scored.append((sim, {"path": path, "ord": ord_, "text": text, "score": sim}))
    scored.sort(key=lambda x: x[0], reverse=True)
    if not unique_paths:
        return [item for _, item in scored[:k]]
    seen_paths: set[str] = set()
    out: list[dict] = []
    for _, item in scored:
        if item["path"] in seen_paths:
            continue
        seen_paths.add(item["path"])
        out.append(item)
        if len(out) >= k:
            break
    return out


def stats(instance: str | None = None) -> dict:
    c = _conn(instance)
    total = c.execute("SELECT COUNT(*), COUNT(DISTINCT path) FROM chunks").fetchone()
    c.close()
    return {"chunks": total[0], "files": total[1], "db": _db_path(instance)}


def forget(path: str, instance: str | None = None) -> int:
    fpath = str(Path(path).resolve())
    c = _conn(instance)
    n = c.execute("DELETE FROM chunks WHERE path = ? OR path LIKE ?",
                  (fpath, fpath + os.sep + "%")).rowcount
    c.commit()
    c.close()
    return n


def prune_denied(instance: str | None = None) -> dict:
    """Delete chunks whose file extension is not in TEXT_EXTS.

    Useful after narrowing the allowlist — existing rows from previously-allowed
    extensions stay until pruned. Returns counts of paths and chunks removed.
    """
    c = _conn(instance)
    rows = c.execute("SELECT DISTINCT path FROM chunks").fetchall()
    denied = [p for (p,) in rows if Path(p).suffix.lower() not in TEXT_EXTS]
    if not denied:
        c.close()
        return {"paths_pruned": 0, "chunks_pruned": 0}
    chunks_pruned = 0
    for p in denied:
        chunks_pruned += c.execute("DELETE FROM chunks WHERE path = ?", (p,)).rowcount
    c.commit()
    c.close()
    return {"paths_pruned": len(denied), "chunks_pruned": chunks_pruned}
