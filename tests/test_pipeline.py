"""Authored synthetic fixtures only: no original curriculum text, PDF or model."""

import json
import importlib
import importlib.util
import sqlite3
import subprocess
import sys
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from http.server import HTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np

from curriculum_search import CurriculumSearch, RELEASE, __version__
from curriculum_search.bundle import MODEL_FILES, export_bundle, validate_data
from curriculum_search.extract import chunk_sections, digest_file, extract, split_text, write_jsonl
from curriculum_search.search import DEFAULT_MODEL, MODEL_REVISION, FastEmbedder, build_index, build_vectors
from curriculum_search.server import make_handler

ROOT = Path(__file__).resolve().parents[1]


class FakeEmbedder:
    model_name = DEFAULT_MODEL

    def embed_documents(self, texts):
        result = np.zeros((len(texts), 512), dtype=np.float32)
        for i in range(len(texts)):
            result[i, i] = 1
        return result

    def embed_query(self, text):
        result = np.zeros(512, dtype=np.float32)
        result[1] = 1
        return result


def synthetic_corpus(root):
    data = root / "data"
    data.mkdir()
    chunks, sections = [], []
    for i, (subject, category, text) in enumerate([
        ("数学", "必修课程", "合成函数样例"),
        ("数学", "选择性必修课程", "合成圆锥曲线样例"),
        ("生物学", "必修课程", "合成遗传信息样例"),
    ]):
        common = {
            "schema_version": "1.0", "document_id": f"document-{i}",
            "subject": subject, "course_categories": [category],
            "original_course_categories": [category, "选修课程"],
            "scope": "shared_with_elective", "heading_path": ["课程内容", category, text],
            "text": text, "quality_warnings": ["synthetic-warning"],
            "source": {"file": "synthetic.pdf", "sha256": "a" * 64, "pdf_pages": [20 + i],
                       "pdf_page_start": 20 + i, "pdf_page_end": 20 + i, "printed_pages": [10 + i]},
        }
        chunks.append({**common, "id": f"chunk-{i}", "section_id": f"section-{i}",
                       "context_before": "", "context_pdf_pages": [], "embedding_text": text})
        sections.append({**common, "id": f"section-{i}", "title": text,
                         "fragments": [{"pdf_page": 20 + i, "text": text}]})
    write_jsonl(data / "chunks.jsonl", chunks)
    write_jsonl(data / "sections.jsonl", sections)
    write_jsonl(data / "source_pages.jsonl", [])
    manifest = {"schema_version": "1.0", "edition": "2017年版（合成测试）", "total_chunks": 3,
                "total_sections": 3, "documents": [{"source_file": "synthetic.pdf", "source_sha256": "a" * 64,
                                                     "status": "extracted"}]}
    (data / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    build_index(data)
    build_vectors(data, FakeEmbedder(), DEFAULT_MODEL)
    cache = root / "models"
    model = cache / "models--Qdrant--bge-small-zh-v1.5"
    (model / "refs").mkdir(parents=True)
    (model / "refs/main").write_text("b" * 40, encoding="utf-8")
    snapshot = model / "snapshots" / ("b" * 40)
    snapshot.mkdir(parents=True)
    for name in MODEL_FILES:
        (snapshot / name).write_bytes(b"synthetic-not-a-real-model")
    return data, cache, chunks, sections


class PipelineTests(unittest.TestCase):
    def setUp(self):
        output = ROOT / "test-results"
        output.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="synthetic-", dir=output)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data, self.cache, self.chunks, self.sections = synthetic_corpus(self.root)
        self.search = CurriculumSearch(self.data, embedder=FakeEmbedder())

    def test_modes_preserve_provenance_categories_and_parent_section(self):
        before = digest_file(self.data / "curriculum.sqlite3")
        for mode in ("lexical", "semantic", "hybrid"):
            hits = self.search.search('圆锥曲线" OR *', subject="数学", course_category="选择性必修课程", mode=mode)
            self.assertEqual([hit["id"] for hit in hits], ["chunk-1"])
            self.assertEqual(hits[0]["source"]["pdf_pages"], [21])
            self.assertEqual(hits[0]["original_course_categories"], ["选择性必修课程", "选修课程"])
            self.assertEqual(hits[0]["quality_warnings"], ["synthetic-warning"])
            self.assertEqual(self.search.get_section(hits[0]["section_id"])["id"], "section-1")
        self.assertEqual(digest_file(self.data / "curriculum.sqlite3"), before)
        with self.search._db() as db, self.assertRaises(sqlite3.OperationalError):
            db.execute("DELETE FROM chunks")

    def test_invalid_queries_are_rejected_and_missing_ids_are_explicit(self):
        for args in ({"query": ""}, {"query": "x", "top_k": True}, {"query": "x", "top_k": 0},
                     {"query": "x", "mode": "invalid"}, {"query": "x", "subject": "不存在"},
                     {"query": "x", "course_category": "选修课程"}):
            with self.assertRaises(ValueError):
                self.search.search(**args)
        self.assertIsNone(self.search.get("missing"))
        self.assertIsNone(self.search.get_section("missing"))
        self.assertEqual(self.search.search("!!!", mode="lexical"), [])

    def test_rebuilding_index_closes_connections_and_stale_index_is_rejected(self):
        build_index(self.data)
        self.assertEqual(CurriculumSearch(self.data).get("chunk-0")["id"], "chunk-0")
        with (self.data / "chunks.jsonl").open("a", encoding="utf-8") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "stale"):
            CurriculumSearch(self.data)

    def test_missing_or_corrupt_vectors_never_fall_back_silently(self):
        (self.data / "vectors.json").unlink()
        with self.assertRaises(FileNotFoundError):
            self.search.search("合成", mode="hybrid")
        self.assertTrue(self.search.search("合成", mode="lexical"))
        build_vectors(self.data, FakeEmbedder(), DEFAULT_MODEL)
        (self.data / "vectors.npy").write_bytes(b"corrupted")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.search.search("合成", mode="semantic")

    def test_query_token_window_is_checked_before_inference(self):
        class Oversized(FakeEmbedder):
            def token_counts(self, texts):
                return [513] * len(texts)

            def embed_query(self, text):
                raise AssertionError("Inference must not run after token overflow")

        search = CurriculumSearch(self.data, embedder=Oversized())
        with self.assertRaisesRegex(ValueError, "512-token"):
            search.search("合成", mode="semantic")

    def test_default_model_download_is_explicit_and_revision_is_pinned(self):
        empty_cache = self.root / "empty-cache"
        calls = []

        def download(**kwargs):
            calls.append(kwargs)
            snapshot = empty_cache / "models--Qdrant--bge-small-zh-v1.5/snapshots" / MODEL_REVISION
            snapshot.mkdir(parents=True)
            for name in MODEL_FILES:
                (snapshot / name).write_bytes(b"synthetic-model")
            return str(snapshot)

        engines = []
        with patch.dict(sys.modules, {"fastembed": SimpleNamespace(TextEmbedding=lambda **kwargs: engines.append(kwargs)),
                                      "huggingface_hub": SimpleNamespace(snapshot_download=download)}):
            with self.assertRaises(FileNotFoundError):
                FastEmbedder(cache_dir=empty_cache, local_only=True)
            self.assertFalse(calls)
            FastEmbedder(cache_dir=empty_cache, local_only=False)
            self.assertEqual(calls[0]["revision"], MODEL_REVISION)
            self.assertEqual(calls[0]["repo_id"], "Qdrant/bge-small-zh-v1.5")
            self.assertTrue(engines[-1]["local_files_only"])
            self.assertTrue(engines[-1]["specific_model_path"].endswith(MODEL_REVISION))
            FastEmbedder(cache_dir=empty_cache, local_only=True)
            self.assertEqual(len(calls), 1)

    def test_chunking_preserves_text_and_keeps_overlap_outside_body(self):
        text = "甲" * 150 + "。" + "乙" * 120
        section = {**self.sections[0], "text": text, "fragments": [{"pdf_page": 20, "text": text}]}
        chunks = chunk_sections([section], 100, 30)
        self.assertEqual("".join(chunk["text"] for chunk in chunks), text)
        self.assertTrue(all(len(chunk["text"]) <= 100 for chunk in chunks))
        self.assertTrue(chunks[1]["context_before"])
        self.assertEqual("".join(split_text(text, 100)), text)

    def test_mismatching_pdf_is_rejected_before_parsing_or_output(self):
        inputs = self.root / "sources"
        inputs.mkdir()
        (inputs / "unsupported.pdf").write_bytes(b"not a real PDF")
        output = self.root / "extracted"
        with self.assertRaisesRegex(ValueError, "reviewed 2017"):
            extract(inputs, output)
        self.assertFalse(output.exists())

    def test_bundle_is_flat_portable_and_does_not_copy_reports_or_caches(self):
        (self.data / "validation.json").write_text("private report", encoding="utf-8")
        destination = self.root / "bundle"
        receipt = export_bundle(self.data, self.cache, destination)
        self.assertEqual(receipt["schema_version"], "1.0")
        self.assertEqual((receipt["chunks"], receipt["sections"]), (3, 3))
        self.assertEqual(receipt["model_revision"], "b" * 40)
        self.assertEqual({p.name for p in destination.iterdir()}, {"data", "models", "import.json"})
        self.assertFalse((destination / "data/validation.json").exists())
        self.assertTrue(all((destination / "models" / name).is_file() for name in MODEL_FILES))
        for name, expected in receipt["files"].items():
            self.assertEqual(digest_file(destination / name), expected)
        self.assertEqual(validate_data(destination / "data")["chunks"], 3)
        second = self.root / "second-bundle"
        self.assertEqual(export_bundle(destination / "data", destination / "models", second)["files"], receipt["files"])
        with self.assertRaisesRegex(ValueError, "exists"):
            export_bundle(self.data, self.cache, destination)
        with self.assertRaisesRegex(ValueError, "outside"):
            export_bundle(self.data, self.cache, self.data / "nested-bundle")

    def test_invalid_vector_and_provenance_block_bundle_before_output(self):
        vectors = np.zeros((3, 512), dtype=np.float32)
        np.save(self.data / "vectors.npy", vectors)
        meta = json.loads((self.data / "vectors.json").read_text(encoding="utf-8"))
        meta["vectors_sha256"] = digest_file(self.data / "vectors.npy")
        (self.data / "vectors.json").write_text(json.dumps(meta), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Invalid backend vectors"):
            export_bundle(self.data, self.cache, self.root / "invalid")
        self.assertFalse((self.root / "invalid").exists())

    def test_inconsistent_source_hash_blocks_bundle_before_output(self):
        path = self.data / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["documents"][0]["source_sha256"] = "c" * 64
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "provenance"):
            export_bundle(self.data, self.cache, self.root / "inconsistent")
        self.assertFalse((self.root / "inconsistent").exists())

    def test_imported_model_checksum_is_verified_before_reexport(self):
        destination = self.root / "bundle"
        export_bundle(self.data, self.cache, destination)
        (destination / "models/tokenizer.json").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "checksum"):
            export_bundle(destination / "data", destination / "models", self.root / "rejected")
        self.assertFalse((self.root / "rejected").exists())

    def test_cli_help_version_and_sources_do_not_need_pdf_parser_or_local_assets(self):
        for args in (["--help"], ["--version"], ["sources"]):
            result = subprocess.run([sys.executable, "-m", "curriculum_search.cli", *args],
                                    cwd=self.root, capture_output=True, text=True, encoding="utf-8")
            self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(__version__, "1.0.0.post1")
        self.assertEqual(RELEASE, "1.0.0-hotfix1")

    def test_runtime_source_catalog_matches_public_provenance(self):
        runtime = json.loads((ROOT / "src/curriculum_search/sources.json").read_text(encoding="utf-8"))
        provenance = json.loads((ROOT / "provenance/sources.json").read_text(encoding="utf-8"))
        expected = {item["source_file"]: (item["source_sha256"], item["pdf_pages_total"]) for item in provenance["documents"]}
        actual = {item["file"]: (item["sha256"], item["pages"]) for item in runtime["documents"]}
        self.assertEqual(actual, expected)

    def test_http_search_uses_synthetic_data_and_exposes_clear_errors(self):
        server = HTTPServer(("127.0.0.1", 0), make_handler(self.search))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 5)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        url = f"http://127.0.0.1:{server.server_port}"
        with urlopen(url + "/health") as response:
            self.assertEqual(json.load(response)["status"], "ok")
        body = json.dumps({"query": "圆锥曲线", "subject": "数学", "mode": "hybrid"}).encode()
        with urlopen(Request(url + "/search", data=body)) as response:
            self.assertTrue(json.load(response)["results"])
        for path, payload, status in [("/search", b'{"query":""}', 400), ("/search", b"[]", 400),
                                      ("/missing", b"{}", 404)]:
            with self.assertRaises(HTTPError) as error:
                urlopen(Request(url + path, data=payload))
            self.assertEqual(error.exception.code, status)


@unittest.skipUnless(importlib.util.find_spec("pymupdf"), "Optional extract extra (PyMuPDF) is not installed")
class PDFExtractionTests(unittest.TestCase):
    """Parse only freshly generated synthetic PDFs, including actual detected tables."""

    def test_real_pdf_extraction_exercises_math_and_english_table_geometry(self):
        import pymupdf

        extraction = importlib.import_module("curriculum_search.extract")
        output = ROOT / "test-results"
        output.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="synthetic-pdf-", dir=output) as temporary:
            root = Path(temporary)

            def text(page, value, x=100, y=120):
                page.insert_text((x, y), value, fontname="china-s", fontsize=12)

            def table(page, headers, values):
                xs = [100 + 140 * n for n in range(len(headers) + 1)]
                for x in xs:
                    page.draw_line((x, 180), (x, 240))
                for y in (180, 210, 240):
                    page.draw_line((xs[0], y), (xs[-1], y))
                for n, value in enumerate(headers):
                    text(page, value, xs[n] + 10, 200)
                for n, value in enumerate(values):
                    text(page, value, xs[n] + 10, 230)

            for subject in ("数学", "英语"):
                with self.subTest(subject=subject):
                    inputs = root / subject
                    inputs.mkdir()
                    path = inputs / ("ZW18-72_synthetic.pdf" if subject == "数学" else "ZW18-133_synthetic.pdf")
                    with pymupdf.open() as document:
                        for _ in range(25 if subject == "数学" else 55):
                            document.new_page()
                        if subject == "数学":
                            text(document[8], "必修课程 选择性必修课程 选修课程")
                            text(document[22], "（一）必修课程")
                            table(document[22], ["主题", "单元", "课时"], ["合成主题", "合成单元", "4"])
                            text(document[23], "（二）选择性必修课程")
                            text(document[23], "合成选择性内容", y=150)
                            text(document[24], "（三）选修课程")
                            profiles = {"ZW18-72_": ("数学", 0, 23, 24, 25)}
                        else:
                            text(document[8], "课程内容 主题语境 语篇类型 学习策略")
                            text(document[21], "（一）主题语境")
                            table(document[21], ["主题", "内容"], ["合成主题", "合成共享内容"])
                            text(document[22], "不分课程类别进行描述")
                            profiles = {"ZW18-133_": ("英语", 0, 22, None, 55)}
                        document.save(path)
                    catalog = root / (subject + "-catalog")
                    catalog.mkdir()
                    (catalog / "sources.json").write_text(json.dumps({"documents": [{
                        "file": path.name, "sha256": digest_file(path),
                    }]}), encoding="utf-8")
                    destination = root / (subject + "-output")
                    with patch.dict(extraction.PROFILES, profiles, clear=True), patch.object(extraction, "files", return_value=catalog):
                        manifest = extraction.extract(inputs, destination)
                    self.assertGreater(manifest["total_sections"], 0)
                    self.assertGreater(manifest["total_chunks"], 0)
                    self.assertLess(manifest["documents"][0]["pages_read_count"], manifest["documents"][0]["pdf_pages_total"])
                    sections = [json.loads(line) for line in (destination / "sections.jsonl").read_text(encoding="utf-8").splitlines()]
                    self.assertTrue(any("合成" in item["text"] for item in sections))
                    self.assertTrue(all(item["source"]["sha256"] == digest_file(path) for item in sections))
                    if subject == "数学":
                        self.assertTrue(any("表格行" in item["text"] for item in sections))
                    else:
                        self.assertTrue(any(item.get("extraction_method") == "table_cells" for item in sections))


if __name__ == "__main__":
    unittest.main(verbosity=2)
