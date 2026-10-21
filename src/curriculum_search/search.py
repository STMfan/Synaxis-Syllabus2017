"""SQLite lexical search + pluggable local cosine-vector search.

No cloud API, paid service, or model is loaded on import or lexical queries.
"""

from __future__ import annotations

import json
import re
import sqlite3
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Protocol

import numpy as np

from .extract import CATEGORIES, digest_file

DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"
MODEL_REVISION = "46fbe35fd4374a00fee7de77dfddaeb6dd6a2c59"
MODEL_FILES = ("model_optimized.onnx", "config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


class FastEmbedder:
    def __init__(self, model: str = DEFAULT_MODEL, cache_dir: str | Path = "models",
                 local_only: bool = True, threads: int = 4):
        from fastembed import TextEmbedding

        self.model_name = model
        cache_dir = Path(cache_dir)
        local = cache_dir
        if model == DEFAULT_MODEL and all((local / name).is_file() for name in MODEL_FILES) and (local.parent / "import.json").is_file():
            receipt = json.loads((local.parent / "import.json").read_text(encoding="utf-8"))
            if any(digest_file(local / name) != receipt["files"][f"models/{name}"] for name in MODEL_FILES):
                raise ValueError("Imported model checksum mismatch")
        if model == DEFAULT_MODEL and not all((local / name).is_file() for name in MODEL_FILES):
            local = cache_dir / "models--Qdrant--bge-small-zh-v1.5" / "snapshots" / MODEL_REVISION
            if not all((local / name).is_file() for name in MODEL_FILES):
                if local_only:
                    raise FileNotFoundError("Pinned local model missing; explicitly run embed --allow-download first")
                from huggingface_hub import snapshot_download

                local = Path(snapshot_download(repo_id="Qdrant/bge-small-zh-v1.5", revision=MODEL_REVISION,
                                              cache_dir=str(cache_dir), allow_patterns=[*MODEL_FILES, "README.md", "LICENSE*"]))
        options = {"specific_model_path": str(local)} if all((local / name).is_file() for name in MODEL_FILES) else {}
        self.engine = TextEmbedding(model_name=model, cache_dir=str(cache_dir),
                                    local_files_only=True if model == DEFAULT_MODEL else local_only,
                                    threads=threads, **options)

    def embed_documents(self, texts):
        return np.asarray(list(self.engine.passage_embed(texts, batch_size=32)), dtype=np.float32)

    def embed_query(self, text):
        return np.asarray(list(self.engine.query_embed([text]))[0], dtype=np.float32)

    def token_counts(self, texts):
        # Expose the actual ONNX tokenizer for auditing model-window truncation.
        tokenizer = self.engine.model.tokenizer
        tokenizer.no_truncation()
        try:
            return [len(x.ids) for x in tokenizer.encode_batch(texts)]
        finally:
            tokenizer.enable_truncation(max_length=512)


def tokens(text: str) -> list[str]:
    result = []
    for segment in re.findall(r"[\u3400-\u9fff]+|[a-zA-Z0-9]+", text.lower()):
        if re.fullmatch(r"[\u3400-\u9fff]+", segment):
            result.extend(segment)
            result.extend(segment[i:i + 2] for i in range(len(segment) - 1))
        else:
            result.append(segment)
    return result


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_index(data_dir: Path) -> dict:
    chunks_path = data_dir / "chunks.jsonl"
    chunks = read_jsonl(chunks_path)
    temporary = data_dir / "curriculum.building.sqlite3"
    with closing(sqlite3.connect(temporary)) as db, db:
        db.executescript("""
            DROP TABLE IF EXISTS chunks;
            DROP TABLE IF EXISTS search_fts;
            DROP TABLE IF EXISTS metadata;
            CREATE TABLE chunks (id TEXT PRIMARY KEY, subject TEXT NOT NULL,
                                 categories TEXT NOT NULL, payload TEXT NOT NULL);
            CREATE VIRTUAL TABLE search_fts USING fts5(id UNINDEXED, tokens);
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        """)
        for c in chunks:
            db.execute("INSERT INTO chunks VALUES(?,?,?,?)", (
                c["id"], c["subject"], "|" + "|".join(c["course_categories"]) + "|",
                json.dumps(c, ensure_ascii=False)))
            db.execute("INSERT INTO search_fts VALUES(?,?)", (c["id"], " ".join(tokens(c["embedding_text"]))))
        db.execute("INSERT INTO metadata VALUES('chunks_sha256',?)", (digest_file(chunks_path),))
        db.execute("INSERT INTO search_fts(search_fts) VALUES('optimize')")
    temporary.replace(data_dir / "curriculum.sqlite3")
    return {"chunks": len(chunks), "database": str(data_dir / "curriculum.sqlite3")}


def unit_rows(vectors):
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ValueError("Embedding matrix must be finite and 2-dimensional")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Embedding contains zero vectors")
    return vectors / norms


def build_vectors(data_dir: Path, embedder: Embedder, model_name: str,
                  cache_dir: Path | None = None) -> dict:
    chunks = read_jsonl(data_dir / "chunks.jsonl")
    texts = [c["embedding_text"] for c in chunks]
    max_tokens = None
    if hasattr(embedder, "token_counts"):
        counts = embedder.token_counts(texts)
        max_tokens = max(counts)
        if max_tokens > 512:
            raise ValueError(f"Embedding text has {max_tokens} tokens; shorten extract --max-chars/--overlap. No silent truncation.")
    vectors = unit_rows(embedder.embed_documents(texts))
    if len(vectors) != len(chunks):
        raise ValueError("Embedding count does not match corpus")
    tmp = data_dir / "vectors.building.npy"
    np.save(tmp, vectors, allow_pickle=False)
    tmp.replace(data_dir / "vectors.npy")
    meta = {
        "model": model_name, "dimension": int(vectors.shape[1]), "count": len(chunks),
        "ids": [c["id"] for c in chunks], "chunks_sha256": digest_file(data_dir / "chunks.jsonl"),
        "vectors_sha256": digest_file(data_dir / "vectors.npy"), "normalization": "l2",
        "max_input_tokens": max_tokens,
        "cache_dir_hint": cache_dir.name if cache_dir else None,
        "query_encoding": "FastEmbed.query_embed", "document_encoding": "FastEmbed.passage_embed",
    }
    (data_dir / "vectors.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {k: v for k, v in meta.items() if k != "ids"}


class CurriculumSearch:
    def __init__(self, data_dir: str | Path, embedder: Embedder | None = None,
                 model_cache: str | Path | None = None):
        self.data_dir = Path(data_dir).resolve()
        self.db_path = self.data_dir / "curriculum.sqlite3"
        if not self.db_path.is_file():
            raise FileNotFoundError(f"Build the index first: {self.db_path}")
        self.embedder = embedder
        self.model_cache = Path(model_cache).resolve() if model_cache else None
        self._vectors = self._vector_meta = None
        self._sections = None
        with self._db() as db:
            stored = db.execute("SELECT value FROM metadata WHERE key='chunks_sha256'").fetchone()[0]
        self.chunks_sha256 = digest_file(self.data_dir / "chunks.jsonl")
        if self.chunks_sha256 != stored:
            raise ValueError("SQLite index is stale; run index after extraction")

    @contextmanager
    def _db(self):
        db = sqlite3.connect(self.db_path.as_uri() + "?mode=ro", uri=True)
        try:
            yield db
        finally:
            db.close()

    @property
    def subjects(self):
        with self._db() as db:
            return [r[0] for r in db.execute("SELECT DISTINCT subject FROM chunks ORDER BY subject")]

    def get(self, chunk_id: str) -> dict | None:
        with self._db() as db:
            row = db.execute("SELECT payload FROM chunks WHERE id=?", (chunk_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def get_section(self, section_id: str) -> dict | None:
        """Retrieve the parent section, including fragments and classification evidence."""
        if self._sections is None:
            self._sections = {s["id"]: s for s in read_jsonl(self.data_dir / "sections.jsonl")}
        return self._sections.get(section_id)

    def _filters(self, subject, category):
        clauses, params = [], []
        if subject is not None:
            if subject not in self.subjects:
                raise ValueError(f"Unknown subject: {subject}")
            clauses.append("c.subject=?")
            params.append(subject)
        if category is not None:
            if category not in CATEGORIES:
                raise ValueError(f"Unknown course category: {category}")
            clauses.append("instr(c.categories,?) > 0")
            params.append("|" + category + "|")
        return (" AND " + " AND ".join(clauses) if clauses else ""), params

    def _lexical(self, query, k, subject, category):
        terms = list(dict.fromkeys(tokens(query)))[:128]
        if not terms:
            return []
        match = " OR ".join('"' + term + '"' for term in terms)
        where, params = self._filters(subject, category)
        with self._db() as db:
            rows = db.execute("""
                SELECT c.payload, bm25(search_fts) AS rank
                FROM search_fts JOIN chunks c ON c.id=search_fts.id
                WHERE search_fts MATCH ?
                """ + where + " ORDER BY rank, c.id LIMIT ?", [match, *params, k]).fetchall()
        return [{**json.loads(payload), "score": -rank, "retrieval_method": "lexical"}
                for payload, rank in rows]

    def _load_vectors(self):
        if self._vectors is not None:
            return
        metadata_path = self.data_dir / "vectors.json"
        if not metadata_path.is_file():
            raise FileNotFoundError("Semantic index missing; run embed first (lexical mode is available)")
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        if meta["chunks_sha256"] != self.chunks_sha256:
            raise ValueError("Vector index is stale; run embed after extraction")
        if digest_file(self.data_dir / "vectors.npy") != meta["vectors_sha256"]:
            raise ValueError("Vector file checksum mismatch")
        # This corpus is small; an in-memory matrix releases the file handle so
        # Windows can replace/migrate data without an uncloseable memmap lease.
        vectors = np.load(self.data_dir / "vectors.npy", allow_pickle=False)
        if vectors.shape != (len(meta["ids"]), meta["dimension"]):
            raise ValueError("Vector shape mismatch")
        vectors.flags.writeable = False
        if self.embedder is None:
            cache = self.model_cache or (self.data_dir.parent / "models")
            self.embedder = FastEmbedder(meta["model"], cache_dir=cache, local_only=True)
        elif getattr(self.embedder, "model_name", meta["model"]) != meta["model"]:
            raise ValueError("Query embedding model differs from index model")
        self._vector_meta, self._vectors = meta, vectors

    def _semantic(self, query, k, subject, category):
        self._load_vectors()
        if hasattr(self.embedder, "token_counts") and self.embedder.token_counts([query])[0] > 512:
            raise ValueError("Query exceeds the embedding model's 512-token window; shorten the query")
        q = np.asarray(self.embedder.embed_query(query), dtype=np.float32).reshape(-1)
        if q.shape != (self._vectors.shape[1],) or not np.isfinite(q).all() or np.linalg.norm(q) == 0:
            raise ValueError("Invalid query embedding")
        scores = self._vectors @ (q / np.linalg.norm(q))
        where, params = self._filters(subject, category)
        with self._db() as db:
            allowed = {r[0]: r[1] for r in db.execute("SELECT c.id,c.payload FROM chunks c WHERE 1=1" + where, params)}
        candidates = [(i, chunk_id) for i, chunk_id in enumerate(self._vector_meta["ids"]) if chunk_id in allowed]
        candidates.sort(key=lambda pair: (-float(scores[pair[0]]), pair[1]))
        return [{**json.loads(allowed[chunk_id]), "score": float(scores[i]), "retrieval_method": "semantic"}
                for i, chunk_id in candidates[:k]]

    def search(self, query: str, *, top_k: int = 5, subject: str | None = None,
               course_category: str | None = None, mode: str = "hybrid") -> list[dict]:
        if not isinstance(query, str) or not query.strip() or len(query) > 4096:
            raise ValueError("query must contain 1..4096 characters")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= 100:
            raise ValueError("top_k must be 1..100")
        if mode not in ("lexical", "semantic", "hybrid"):
            raise ValueError("mode must be lexical, semantic or hybrid")
        self._filters(subject, course_category)
        if mode == "lexical":
            return self._lexical(query, top_k, subject, course_category)
        if mode == "semantic":
            return self._semantic(query, top_k, subject, course_category)
        # Reciprocal rank fusion: raw BM25 and cosine scores are not comparable.
        count = max(40, top_k * 4)
        lexical = self._lexical(query, count, subject, course_category)
        semantic = self._semantic(query, count, subject, course_category)
        combined, scores = {}, {}
        for results in (lexical, semantic):
            for rank, item in enumerate(results, 1):
                combined[item["id"]] = item
                scores[item["id"]] = scores.get(item["id"], 0) + 1 / (60 + rank)
        order = sorted(combined, key=lambda key: (-scores[key], key))[:top_k]
        return [{**combined[key], "score": scores[key], "retrieval_method": "hybrid"} for key in order]
