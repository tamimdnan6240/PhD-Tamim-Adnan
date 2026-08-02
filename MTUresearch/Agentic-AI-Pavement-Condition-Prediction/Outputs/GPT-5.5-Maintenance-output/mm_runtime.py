import os, re, json, time, glob, hashlib, sqlite3
from typing import Optional, List, Dict, Any, Tuple, Iterable
import numpy as np

# -------------------------
# Optional KG dependency
# -------------------------
try:
    import networkx as nx  # type: ignore
except Exception:
    nx = None

# -------------------------
# Workspace paths
# -------------------------
WORK_DIR = os.getenv("WORK_DIR", "/content/pavement_agentic_workspace")
os.makedirs(WORK_DIR, exist_ok=True)

MEM_PATH      = os.path.join(WORK_DIR, "memory.jsonl")
DB_PATH       = os.path.join(WORK_DIR, "memory_hybrid.sqlite")
VEC_PATH      = os.path.join(WORK_DIR, "memory_vectors.npz")
KG_PATH       = os.path.join(WORK_DIR, "knowledge_graph.graphml")
REGISTRY_PATH = os.path.join(WORK_DIR, "file_path_registry.json")

_ENFORCE_LIMITS = os.getenv("ENFORCE_LIMITS", "1").strip() != "0"
_EPS = 1e-12

# ============================================================
# BASIC HELPERS
# ============================================================
def _now_ts() -> str:
    """Return local timestamp string for audit logging."""
    return time.strftime("%Y-%m-%d %H:%M:%S")

def _sha24(s: str) -> str:
    """Short stable-ish ID from sha256, used for memory record IDs."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:24]

def _stable_file_fingerprint(file_path: str) -> str:
    """
    Create a small dataset_id fingerprint using:
    abs_path + file_size + mtime.
    """
    st = os.stat(file_path)
    raw = f"{os.path.abspath(file_path)}|{st.st_size}|{int(st.st_mtime)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]

def _ensure_files():
    """Ensure workspace + base memory file exist."""
    os.makedirs(WORK_DIR, exist_ok=True)
    if not os.path.exists(MEM_PATH):
        open(MEM_PATH, "a", encoding="utf-8").close()

def validate_registry_or_fail(work_dir: str) -> Dict[str, Any]:
    """
    Governance gate for dataset access.
    Ensures:
    - registry exists
    - latest_path exists on disk
    - registry does NOT point inside WORK_DIR
    - if PINNED_FILE_PATH env var is set, it must match latest_path exactly

    Returns the registry dict if all checks pass.
    """
    reg = registry_load()
    if reg is None:
        raise RuntimeError("Registry not found. Run mm.registry_build(...) or mm.registry_pin_to_file(...).")

    FILE_PATH = reg.get("latest_path", "")
    if not FILE_PATH:
        raise RuntimeError("Registry missing latest_path.")

    if not os.path.exists(FILE_PATH):
        raise FileNotFoundError(FILE_PATH)

    # Guard: registry must not point inside WORK_DIR
    if os.path.commonpath([os.path.abspath(FILE_PATH), os.path.abspath(work_dir)]) == os.path.abspath(work_dir):
        raise RuntimeError("Registry points inside WORK_DIR; fix registry pinning.")

    # Guard: pinned env var must match exactly if set
    pinned = os.environ.get("PINNED_FILE_PATH", "").strip()
    if pinned:
        if os.path.abspath(FILE_PATH) != os.path.abspath(pinned):
            raise AssertionError(f"Pinned file mismatch. registry={FILE_PATH} env={pinned}")

    return reg

# ============================================================
# KNOWLEDGE GRAPH (OPTIONAL)
# ============================================================
def _kg_enabled() -> bool:
    """Return True if networkx is available."""
    return nx is not None

def _kg_load():
    """Load KG from GraphML if available; otherwise return new MultiDiGraph."""
    if not _kg_enabled():
        return None
    if os.path.exists(KG_PATH):
        try:
            g = nx.read_graphml(KG_PATH)
            mg = nx.MultiDiGraph()
            mg.add_nodes_from(g.nodes(data=True))
            mg.add_edges_from(g.edges(data=True))
            return mg
        except Exception:
            pass
    return nx.MultiDiGraph()

_KG = _kg_load()

def _kg_save():
    """Persist KG to GraphML (lossy for multiedges but keeps core relations)."""
    if not _kg_enabled() or _KG is None:
        return
    g = nx.DiGraph()
    g.add_nodes_from(_KG.nodes(data=True))
    for u, v, data in _KG.edges(data=True):
        g.add_edge(u, v, **data)
    nx.write_graphml(g, KG_PATH)

def kg_add_fact(subj: str, pred: str, obj: str, confidence: float = 1.0, meta: Optional[dict] = None):
    """Add a simple subject-predicate-object fact to KG and save."""
    if not _kg_enabled() or _KG is None:
        return
    meta = meta or {}
    subj = str(subj); pred = str(pred); obj = str(obj)
    _KG.add_node(subj)
    _KG.add_node(obj)
    _KG.add_edge(subj, obj, relation=pred, confidence=float(confidence), **meta)
    _kg_save()

# ============================================================
# JSONL MEMORY — APPEND ONLY
# ============================================================
def _enforce_limits(kind: str, text: str):
    """
    Prevent storing huge text blobs in memory (audit logs should be light).
    """
    if not _ENFORCE_LIMITS:
        return
    if len(text) > 5000 and kind not in {"pipeline_state", "results"}:
        raise RuntimeError("Memory text too large; store only summaries and artifact paths.")

def mem_add_jsonl(kind: str, text: str, meta: Optional[dict] = None) -> str:
    """
    Append one JSON record per line to memory.jsonl.
    Returns memory_id.
    """
    meta = meta or {}
    ts = _now_ts()
    text = str(text).strip()
    _enforce_limits(kind, text)
    raw = f"{ts}|{kind}|{text}|{json.dumps(meta, sort_keys=True)}"
    memory_id = _sha24(raw)
    rec = {"id": memory_id, "ts": ts, "kind": str(kind), "text": text, "meta": meta}
    _ensure_files()
    with open(MEM_PATH, "a", encoding="utf-8") as f:
        # IMPORTANT: real newline, not literal backslash+n
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return memory_id

# ============================================================
# SQLITE STRUCT + FTS
# ============================================================
def _db_connect() -> sqlite3.Connection:
    """
    Connect to SQLite and ensure both:
    - memory_struct: canonical storage
    - memory_fts: FTS5 index for fast keyword search
    """
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS memory_struct ("
        "id TEXT PRIMARY KEY, ts TEXT, kind TEXT, text TEXT, meta_json TEXT);"
    )
    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts "
        "USING fts5(id UNINDEXED, kind, text, content='', tokenize='porter');"
    )
    conn.commit()
    return conn

def keyword_index_add(memory_id: str, ts: str, kind: str, text: str, meta: dict):
    """
    Store record in memory_struct AND index it in memory_fts.
    """
    conn = _db_connect()
    conn.execute(
        "INSERT OR REPLACE INTO memory_struct(id, ts, kind, text, meta_json) VALUES (?, ?, ?, ?, ?)",
        (memory_id, ts, kind, text, json.dumps(meta, ensure_ascii=False))
    )
    conn.execute("DELETE FROM memory_fts WHERE id = ?", (memory_id,))
    conn.execute("INSERT INTO memory_fts(id, kind, text) VALUES (?, ?, ?)", (memory_id, kind, text))
    conn.commit()
    conn.close()

def _fts_safe_query(q: str) -> str:
    """
    Convert a user query to a safe FTS5 query using AND over tokens.
    """
    q = (q or "").strip()
    if not q:
        return '""'
    toks = re.findall(r"[a-zA-Z0-9_]+", q)
    if not toks:
        return '""'
    return " AND ".join([f'"{t}"' for t in toks])

def keyword_search(query: str, k: int = 5, kind: Optional[str] = None):
    """
    Keyword search using SQLite FTS5; falls back to LIKE if needed.
    Returns list of dicts: {id, kind, text, score}
    """
    conn = _db_connect()
    where_kind = " AND kind = ? " if kind else " "
    params = [_fts_safe_query(query)]
    if kind:
        params.append(kind)
    params.append(k)

    sql = (
        "SELECT id, kind, text, bm25(memory_fts) AS score "
        "FROM memory_fts WHERE memory_fts MATCH ? "
        + where_kind +
        "ORDER BY score LIMIT ?;"
    )
    try:
        rows = conn.execute(sql, params).fetchall()
    except Exception:
        like = f"%{(query or '').strip()}%"
        if kind:
            rows = conn.execute(
                "SELECT id, kind, text, 0.0 AS score FROM memory_struct WHERE kind = ? AND text LIKE ? LIMIT ?;",
                (kind, like, k)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, kind, text, 0.0 AS score FROM memory_struct WHERE text LIKE ? LIMIT ?;",
                (like, k)
            ).fetchall()
    conn.close()

    # FTS bm25 lower is better; we return inverted sign as "higher is better"
    return [{"id": rid, "kind": rkind, "text": rtext, "score": float(-score)} for rid, rkind, rtext, score in rows]

def db_fetch_by_ids(ids: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    """
    Fetch structured records by IDs from memory_struct.
    """
    ids = [i for i in ids if i]
    if not ids:
        return {}
    conn = _db_connect()
    qmarks = ",".join(["?"] * len(ids))
    rows = conn.execute(
        f"SELECT id, ts, kind, text, meta_json FROM memory_struct WHERE id IN ({qmarks});",
        ids
    ).fetchall()
    conn.close()

    out: Dict[str, Dict[str, Any]] = {}
    for rid, ts, kind, text, meta_json in rows:
        try:
            meta = json.loads(meta_json) if meta_json else {}
        except Exception:
            meta = {}
        out[rid] = {"id": rid, "ts": ts, "kind": kind, "text": text, "meta": meta}
    return out

# ============================================================
# SEMANTIC STORE (NPZ)
# ============================================================
def _load_vec_store() -> Tuple[List[str], np.ndarray]:
    """Load vector store from NPZ (ids + vecs)."""
    if not os.path.exists(VEC_PATH):
        return [], np.zeros((0, 0), dtype=np.float32)
    data = np.load(VEC_PATH, allow_pickle=True)
    return data["ids"].tolist(), data["vecs"].astype(np.float32)

def _save_vec_store(ids: List[str], vecs: np.ndarray):
    """Save vector store to NPZ."""
    np.savez(VEC_PATH, ids=np.array(ids, dtype=object), vecs=vecs.astype(np.float32))

def _normalize_rows(X: np.ndarray) -> np.ndarray:
    """Row-normalize vectors for cosine similarity."""
    if X.size == 0:
        return X
    norms = np.linalg.norm(X, axis=1, keepdims=True) + _EPS
    return X / norms

def embed_text(text: str) -> np.ndarray:
    """
    Embed text using SentenceTransformers if available.
    Fallback: deterministic hashing into 512-d sparse-ish vector.
    """
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        global _ST_MODEL
        if "_ST_MODEL" not in globals():
            _ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        v = _ST_MODEL.encode([text], normalize_embeddings=True)[0]
        return np.array(v, dtype=np.float32)
    except Exception:
        dim = 512
        v = np.zeros(dim, dtype=np.float32)
        for token in re.findall(r"[a-z0-9_]+", (text or "").lower()):
            h = int(hashlib.sha256(token.encode()).hexdigest(), 16)
            v[h % dim] += 1.0
        v = v / (np.linalg.norm(v) + _EPS)
        return v

def semantic_add(item_id: str, text: str):
    """Add or update embedding for an item_id in NPZ store."""
    ids, vecs = _load_vec_store()
    v = embed_text(text).reshape(1, -1)
    if vecs.size == 0 or (v.shape[1] != vecs.shape[1]):
        _save_vec_store([item_id], v)
        return
    try:
        idx = ids.index(item_id)
        vecs[idx:idx+1, :] = v
        _save_vec_store(ids, vecs)
    except ValueError:
        ids.append(item_id)
        vecs = np.vstack([vecs, v])
        _save_vec_store(ids, vecs)

def semantic_search_ids(query: str, k: int = 5) -> List[Tuple[str, float]]:
    """
    Return top-k (id, similarity) by cosine similarity.
    """
    ids, vecs = _load_vec_store()
    if vecs.size == 0:
        return []
    q = embed_text(query).reshape(1, -1).astype(np.float32)
    sims = (_normalize_rows(vecs) @ _normalize_rows(q).T).ravel()
    top = np.argsort(-sims)[: min(k, len(ids))]
    return [(ids[i], float(sims[i])) for i in top]

# ============================================================
# REGISTRY
# ============================================================
def registry_build(
    dataset_dir: str,
    allow_globs: Optional[List[str]] = None,
    exclude_dirs: Optional[List[str]] = None,
    prefer_regex: Optional[str] = None
) -> Dict[str, Any]:
    """
    Build a registry by scanning dataset_dir for CSVs and selecting the latest by mtime.
    Also stores mapping: path -> dataset_id fingerprint.
    """
    dataset_dir = os.path.abspath(dataset_dir)
    allow_globs = allow_globs or ["**/*.csv"]
    exclude_dirs = exclude_dirs or []
    if os.path.abspath(WORK_DIR) not in [os.path.abspath(d) for d in exclude_dirs]:
        exclude_dirs = exclude_dirs + [WORK_DIR]

    csvs: List[str] = []
    for g in allow_globs:
        csvs.extend(glob.glob(os.path.join(dataset_dir, g), recursive=True))
    csvs = sorted(set(csvs))

    def _keep(p: str) -> bool:
        for d in exclude_dirs or []:
            try:
                if os.path.commonpath([os.path.abspath(p), os.path.abspath(d)]) == os.path.abspath(d):
                    return False
            except Exception:
                pass
        return True

    csvs = [p for p in csvs if _keep(p)]
    if not csvs:
        raise FileNotFoundError(f"No CSV found under dataset_dir={dataset_dir}")

    if prefer_regex:
        try:
            rx = re.compile(prefer_regex)
            preferred = [p for p in csvs if rx.search(os.path.basename(p)) or rx.search(p)]
            if preferred:
                csvs = preferred
        except Exception:
            pass

    mapping = {p: _stable_file_fingerprint(p) for p in csvs}
    latest_path = max(mapping.keys(), key=lambda p: os.path.getmtime(p))

    reg = {
        "created_ts": _now_ts(),
        "dataset_dir": dataset_dir,
        "allow_globs": allow_globs,
        "exclude_dirs": [os.path.abspath(d) for d in exclude_dirs],
        "prefer_regex": prefer_regex,
        "path_to_dataset_id": mapping,
        "latest_path": latest_path,
        "latest_dataset_id": mapping[latest_path],
        "pinned": False,  # important for downstream logic
    }
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(reg, f, ensure_ascii=False, indent=2)
    return reg

def registry_load() -> Optional[Dict[str, Any]]:
    """Load registry json from disk."""
    if not os.path.exists(REGISTRY_PATH):
        return None
    try:
        with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def registry_pin_to_file(pinned_path: str) -> Dict[str, Any]:
    """
    Pin the registry to a single CSV file (federally auditable pinning).
    """
    pinned_path = os.path.abspath(pinned_path)
    if not os.path.exists(pinned_path):
        raise FileNotFoundError(f"Pinned file not found: {pinned_path}")

    dsid = _stable_file_fingerprint(pinned_path)
    reg = {
        "created_ts": _now_ts(),
        "path_to_dataset_id": {pinned_path: dsid},
        "latest_path": pinned_path,
        "latest_dataset_id": dsid,
        "pinned": True,
    }
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(reg, f, ensure_ascii=False, indent=2)
    return reg

# ============================================================
# CONDITION CANDIDATE DISCOVERY (NO TIME-SIDE INFERENCE)
# ============================================================
def _norm(s: str) -> str:
    """Normalize column name for matching."""
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def _family_match(col_lower: str) -> Optional[Dict[str, Any]]:
    """
    Classify column into: iri, pci, cracking(_percent), rutting
    Using explicit patterns (interpretable, auditable).
    """
    reasons: List[str] = []

    if re.search(r"(^|[^a-z0-9])iri([^a-z0-9]|$)", col_lower) or "roughness" in col_lower:
        reasons.append("Matched IRI/Roughness via token 'iri' or substring 'roughness'.")
        return {"family": "iri", "confidence": "high", "reasons": reasons}

    if re.search(r"(^|[^a-z0-9])pci([^a-z0-9]|$)", col_lower):
        reasons.append("Matched PCI via token 'pci'.")
        return {"family": "pci", "confidence": "high", "reasons": reasons}

    if "rutting" in col_lower or re.search(r"(^|[^a-z0-9])rut([^a-z0-9]|$)", col_lower):
        reasons.append("Matched Rutting via substring 'rutting' or token 'rut'.")
        return {"family": "rutting", "confidence": "medium", "reasons": reasons}

    if "cracking" in col_lower or re.search(r"(^|[^a-z0-9])crack([^a-z0-9]|$)", col_lower):
        reasons.append("Matched Cracking via substring 'cracking' or token 'crack'.")
        if ("%" in col_lower) or ("percent" in col_lower) or ("pct" in col_lower):
            reasons.append("Name suggests percent measure (%, percent, or pct).")
            return {"family": "cracking_percent", "confidence": "high", "reasons": reasons}
        return {"family": "cracking", "confidence": "medium", "reasons": reasons}

    return None

def discover_condition_candidates(columns: List[str]) -> List[Dict[str, Any]]:
    """
    Return ranked list of candidate condition metric columns.
    No time-side inference; downstream agents handle leakage rules.
    """
    cols = list(columns or [])
    out: List[Dict[str, Any]] = []

    for col in cols:
        fam = _family_match(_norm(col))
        if not fam:
            continue
        out.append({
            "column": col,
            "family": fam["family"],
            "family_confidence": fam["confidence"],
            "reasons": fam["reasons"],
            "leakage_guard": (
                "Potential condition metric. Downstream agents must prevent leakage by excluding "
                "any post-period / outcome / future condition columns from features if applicable."
            ),
        })

    family_rank = {"iri": 1, "pci": 2, "cracking_percent": 3, "cracking": 4, "rutting": 5}
    conf_rank = {"high": 1, "medium": 2, "low": 3}

    out.sort(key=lambda r: (
        family_rank.get(r["family"], 99),
        conf_rank.get(r["family_confidence"], 99),
        _norm(r["column"]),
    ))
    return out

# ============================================================
# META FILTER
# ============================================================
def _meta_matches_filters(meta: dict, filters: Optional[Dict[str, Any]]) -> bool:
    """Return True if meta satisfies all key=value filters."""
    if not filters:
        return True
    for k, v in filters.items():
        if meta.get(k, None) != v:
            return False
    return True

# ============================================================
# MM CLASS (PUBLIC INTERFACE)
# ============================================================
class MM:
    def health_check(self) -> bool:
        """Initialize storage and verify DB connectivity."""
        _ensure_files()
        conn = _db_connect()
        conn.close()
        return True

    def rag_add(self, kind: str, text: str, meta: Optional[dict] = None) -> str:
        """Add an auditable record to JSONL + SQLite + vector store."""
        meta = meta or {}
        ts = _now_ts()
        mid = mem_add_jsonl(kind, text, meta)
        keyword_index_add(mid, ts, kind, text, meta)
        semantic_add(mid, text)
        return mid

    # Convenience alias used by your agents (prevents AttributeError)
    def record_artifact(self, kind: str, meta: Optional[dict] = None, text: str = "") -> str:
        """Alias for rag_add for artifact logging."""
        meta = meta or {}
        if not text:
            text = f"artifact kind={kind}"
        return self.rag_add(kind=kind, text=text, meta=meta)

    def rag_search(
        self,
        query: str,
        k: int = 5,
        kind: Optional[str] = None,
        alpha: float = 0.65,
        meta_filters: Optional[Dict[str, Any]] = None
    ):
        """
        Hybrid search: keyword (FTS) + semantic (vectors), then meta-filter.
        Returns list of (score, record_dict).
        """
        kw = keyword_search(query, k=max(10, k * 4), kind=kind)
        sem_ids = semantic_search_ids(query, k=max(10, k * 4))
        sem_map = {rid: sc for rid, sc in sem_ids}

        scores: Dict[str, Dict[str, float]] = {}
        for r in kw:
            scores.setdefault(r["id"], {"kw": 0.0, "sem": 0.0})
            scores[r["id"]]["kw"] = max(scores[r["id"]]["kw"], r["score"])
        for rid, sc in sem_map.items():
            scores.setdefault(rid, {"kw": 0.0, "sem": 0.0})
            scores[rid]["sem"] = max(scores[rid]["sem"], sc)

        if not scores:
            return []

        ids = list(scores.keys())
        kw_vals = np.array([scores[rid]["kw"] for rid in ids], dtype=np.float32)
        sem_vals = np.array([scores[rid]["sem"] for rid in ids], dtype=np.float32)

        def _minmax(v):
            if v.size == 0:
                return v
            vmin = float(v.min()); vmax = float(v.max())
            if abs(vmax - vmin) < _EPS:
                return np.full_like(v, 0.5, dtype=np.float32)
            return (v - vmin) / (vmax - vmin + _EPS)

        kw_norm = _minmax(kw_vals)
        sem_norm = _minmax(sem_vals)

        for i, rid in enumerate(ids):
            scores[rid]["hybrid"] = float(alpha * kw_norm[i] + (1 - alpha) * sem_norm[i])

        id_to_row = db_fetch_by_ids(ids)

        filtered: List[Tuple[str, float]] = []
        for rid in ids:
            row = id_to_row.get(rid)
            if not row:
                continue
            if _meta_matches_filters(row.get("meta", {}), meta_filters):
                filtered.append((rid, scores[rid]["hybrid"]))

        filtered.sort(key=lambda x: x[1], reverse=True)
        filtered = filtered[:k]
        return [(sc, id_to_row[rid]) for rid, sc in filtered]

    def rag_get_latest(
        self,
        kind: str,
        meta_filters: Optional[Dict[str, Any]] = None,
        contains_text: Optional[str] = None
    ):
        """
        Return the most recent memory record of a given kind, optionally filtered by meta and text.
        """
        conn = _db_connect()
        params = [kind]
        sql = "SELECT id, ts, kind, text, meta_json FROM memory_struct WHERE kind = ? "
        if contains_text:
            sql += " AND text LIKE ? "
            params.append(f"%{contains_text}%")
        sql += " ORDER BY ts DESC LIMIT 200;"
        rows = conn.execute(sql, params).fetchall()
        conn.close()

        for rid, ts, knd, text, meta_json in rows:
            try:
                meta = json.loads(meta_json) if meta_json else {}
            except Exception:
                meta = {}
            if _meta_matches_filters(meta, meta_filters):
                return {"id": rid, "ts": ts, "kind": knd, "text": text, "meta": meta}
        return None

    # -------------------------
    # Registry APIs
    # -------------------------
    def registry_build(self, dataset_dir: str, allow_globs=None, exclude_dirs=None, prefer_regex=None):
        return registry_build(dataset_dir, allow_globs, exclude_dirs, prefer_regex)

    def registry_pin_to_file(self, pinned_path: str, extra_meta: Optional[dict] = None, write_mm_record: bool = True) -> Dict[str, Any]:
        reg = registry_pin_to_file(pinned_path)
        if write_mm_record:
            meta = {
                "dataset_id": reg["latest_dataset_id"],
                "file_path": reg["latest_path"],
                "state": "REGISTRY_PINNED",
                "pinned": True,
            }
            if extra_meta:
                meta.update(extra_meta)
            self.rag_add(
                kind="pinned_dataset",
                text=f"Registry pinned to {reg['latest_path']} (dataset_id={reg['latest_dataset_id']})",
                meta=meta
            )
        return reg

    def validate_registry(self, work_dir: Optional[str] = None) -> Dict[str, Any]:
        """Public wrapper around validate_registry_or_fail."""
        work_dir = work_dir or WORK_DIR
        return validate_registry_or_fail(work_dir)

    def rag_get_latest_registry(self) -> Dict[str, Any]:
        """Load registry (no extra validation)."""
        reg = registry_load()
        if reg is None:
            raise RuntimeError("Registry not found. Run mm.registry_build(...) or mm.registry_pin_to_file(...).")
        return reg

    # -------------------------
    # Condition candidate discovery API
    # -------------------------
    def discover_condition_candidates(self, columns: List[str]) -> List[Dict[str, Any]]:
        return discover_condition_candidates(columns)

    # -------------------------
    # KG API
    # -------------------------
    def kg_add_fact(self, subj: str, pred: str, obj: str, confidence: float = 1.0, meta: Optional[dict] = None) -> bool:
        kg_add_fact(subj, pred, obj, confidence=confidence, meta=meta)
        return True

mm = MM()
