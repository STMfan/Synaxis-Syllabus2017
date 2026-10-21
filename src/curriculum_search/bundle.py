"""Create a private portable data/model bundle for Synaxis; never a source release."""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from importlib.resources import files
from pathlib import Path

import numpy as np

from .extract import CATEGORIES, digest_file
from .search import DEFAULT_MODEL, MODEL_REVISION, CurriculumSearch, read_jsonl

DATA_FILES = ("chunks.jsonl", "sections.jsonl", "curriculum.sqlite3", "manifest.json", "vectors.npy", "vectors.json")
MODEL_FILES = ("model_optimized.onnx", "config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")


def validate_data(directory: Path) -> dict:
    """Check the wire format consumed by the MIT Synaxis backend, without loading ONNX."""
    directory = Path(directory).resolve()
    if not all((directory / name).is_file() for name in DATA_FILES):
        raise ValueError("Backend bundles require extracted data, lexical index and vectors")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    meta = json.loads((directory / "vectors.json").read_text(encoding="utf-8"))
    chunks = read_jsonl(directory / "chunks.jsonl")
    sections = read_jsonl(directory / "sections.jsonl")
    ids = [chunk["id"] for chunk in chunks]
    parents = {section["id"]: section for section in sections}
    sources = {d["source_file"]: d["source_sha256"] for d in manifest["documents"] if d["status"] == "extracted"}
    if (
        manifest["schema_version"] != "1.0" or "2017" not in manifest["edition"]
        or not chunks or not sections or len(set(ids)) != len(ids) or len(parents) != len(sections)
        or manifest["total_chunks"] != len(chunks) or manifest["total_sections"] != len(sections)
        or meta["model"] != DEFAULT_MODEL or meta["dimension"] != 512 or meta["count"] != len(chunks)
        or meta["normalization"] != "l2" or meta["query_encoding"] != "FastEmbed.query_embed"
        or meta["document_encoding"] != "FastEmbed.passage_embed"
        or meta["chunks_sha256"] != digest_file(directory / "chunks.jsonl")
        or meta["vectors_sha256"] != digest_file(directory / "vectors.npy")
        or len(set(meta["ids"])) != len(meta["ids"]) or set(meta["ids"]) != set(ids)
    ):
        raise ValueError("Corpus edition, identifiers, counts or vector metadata mismatch")
    for item in [*chunks, *sections]:
        source = item["source"]
        if (
            not re.fullmatch(r"[0-9a-f]{64}", source["sha256"])
            or sources.get(source["file"]) != source["sha256"]
            or not source["pdf_pages"] or any(not isinstance(page, int) or page < 1 for page in source["pdf_pages"])
            or not item["course_categories"] or not set(item["course_categories"]) <= set(CATEGORIES)
        ):
            raise ValueError("Corpus provenance or category mismatch")
    for chunk in chunks:
        if chunk["section_id"] not in parents or parents[chunk["section_id"]]["subject"] != chunk["subject"]:
            raise ValueError("Chunk parent section mismatch")
    vectors = np.load(directory / "vectors.npy", allow_pickle=False)
    if (
        vectors.shape != (len(chunks), 512) or vectors.dtype != np.float32
        or not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=0.001)
    ):
        raise ValueError("Invalid backend vectors")
    search = CurriculumSearch(directory)
    for chunk in chunks:
        if search.get(chunk["id"]) != chunk:
            raise ValueError("Lexical index payload differs from source chunk")
    return {"edition": manifest["edition"], "chunks": len(chunks), "sections": len(sections)}


def _model_snapshot(cache: Path) -> tuple[Path, str]:
    """Accept a Hugging Face cache or an already imported flat bundle."""
    if all((cache / name).is_file() for name in MODEL_FILES):
        receipt = json.loads((cache.parent / "import.json").read_text(encoding="utf-8"))
        revision = receipt["model_revision"]
        for name in MODEL_FILES:
            if digest_file(cache / name) != receipt["files"][f"models/{name}"]:
                raise ValueError("Imported model checksum mismatch")
        snapshot = cache
    else:
        model = cache / "models--Qdrant--bge-small-zh-v1.5"
        pinned = model / "snapshots" / MODEL_REVISION
        revision = MODEL_REVISION if all((pinned / name).is_file() for name in MODEL_FILES) else (model / "refs/main").read_text(encoding="utf-8").strip()
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Invalid cached model revision")
        snapshot = model / "snapshots" / revision
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Invalid cached model revision")
    if not all((snapshot / name).is_file() and (snapshot / name).stat().st_size for name in MODEL_FILES):
        raise ValueError("The complete local ONNX/tokenizer snapshot is required")
    return snapshot, revision


def export_bundle(data: Path, model_cache: Path, destination: Path) -> dict:
    """Export only declared local assets, atomically; never overwrite an existing path."""
    data, model_cache, destination = Path(data).resolve(), Path(model_cache).resolve(), Path(destination).resolve()
    if destination.exists():
        raise ValueError("Destination already exists; select a new bundle directory")
    if destination.is_relative_to(data) or destination.is_relative_to(model_cache):
        raise ValueError("Bundle destination must be outside its input directories")
    info = validate_data(data)
    snapshot, revision = _model_snapshot(model_cache)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".syllabus-bundle-", dir=destination.parent) as temporary:
        staged = Path(temporary) / "bundle"
        (staged / "data").mkdir(parents=True)
        (staged / "models").mkdir()
        for name in DATA_FILES:
            shutil.copyfile(data / name, staged / "data" / name)
        if (data / "source_pages.jsonl").is_file():
            shutil.copyfile(data / "source_pages.jsonl", staged / "data/source_pages.jsonl")
        for name in MODEL_FILES:
            # Dereference cache links so deployment does not depend on the original cache.
            shutil.copyfile(snapshot / name, staged / "models" / name)
        # Keep the model's upstream MIT text separate from this application's AGPL.
        notice = files("curriculum_search").joinpath("model_license.txt").read_bytes()
        (staged / "models/MODEL-LICENSE.txt").write_bytes(notice)
        validate_data(staged / "data")
        receipt = {
            "schema_version": "1.0", **info, "model_revision": revision,
            "pdf_hashes_verified": 0,
            "files": {path.relative_to(staged).as_posix(): digest_file(path)
                      for path in sorted(staged.rglob("*")) if path.is_file()},
        }
        (staged / "import.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
        staged.rename(destination)
    return receipt
