"""Synthetic inputs exercise the no-corpus/no-credential release boundary."""

import runpy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

MODULE = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/publication.py"))


class PublicationTests(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)
        for name in MODULE["FILES"]:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("Synthetic source fixture\n", encoding="utf-8")
        (root / "pyproject.toml").write_text(
            '[project]\nversion="1.0.0.post1"\nlicense="AGPL-3.0-or-later"\n', encoding="utf-8"
        )
        for name in MODULE["DIRECTORIES"]:
            (root / name).mkdir(exist_ok=True)
        return root

    def test_local_data_and_models_never_enter_public_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.fixture(directory)
            for name in ("data/chunks.jsonl", "models/private.onnx", "sources/source.pdf", ".env"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("Private synthetic sentinel", encoding="utf-8")
            with patch.dict(MODULE["collect"].__globals__, ROOT=root):
                self.assertEqual(set(MODULE["collect"]()), set(MODULE["FILES"]))

    def test_pdf_or_database_hidden_under_source_is_rejected(self):
        for name in ("src/unreviewed.pdf", "tests/runtime.sqlite3"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = self.fixture(directory)
                (root / name).write_bytes(b"synthetic private input")
                with patch.dict(MODULE["collect"].__globals__, ROOT=root):
                    with self.assertRaisesRegex(ValueError, "Unsupported publication input"):
                        MODULE["collect"]()

    def test_credential_failure_reports_only_its_location(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.fixture(directory)
            secret = "sk-" + "synthetic" * 6
            (root / "src/accidental.py").write_text(secret, encoding="utf-8")
            with patch.dict(MODULE["collect"].__globals__, ROOT=root):
                with self.assertRaises(ValueError) as caught:
                    MODULE["collect"]()
            self.assertIn("src/accidental.py", str(caught.exception))
            self.assertNotIn(secret, str(caught.exception))


if __name__ == "__main__":
    unittest.main()
