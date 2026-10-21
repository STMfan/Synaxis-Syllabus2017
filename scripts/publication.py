"""Check/export source only; locally generated curriculum and model files stay private."""

import argparse
import re
import tempfile
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = (
    "README.md", "LICENSE", "NOTICE", "pyproject.toml", "uv.lock", ".gitignore",
    ".gitattributes", "provenance/sources.json", ".github/workflows/checks.yml",
)
DIRECTORIES = ("src", "tests", "scripts")
SKIP = {"__pycache__", ".pytest_cache", ".ruff_cache", ".venv", "test-results"}
SUFFIXES = {".py", ".json", ".md", ".toml", ".lock", ".yml", ".txt"}
TOKENS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|AKIA[A-Z0-9]{16})\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{24,}"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}"),
)


def collect():
    result = {}

    def add(path):
        name = path.relative_to(ROOT).as_posix()
        if any(
            part in {".", ".."} or any(char in "\\:" or ord(char) < 32 for char in part)
            for part in path.relative_to(ROOT).parts
        ):
            raise ValueError("Nonportable publication source name")
        if path.is_symlink() or getattr(path, "is_junction", lambda: False)():
            raise ValueError(f"Linked publication input: {name}")
        if path.is_dir():
            if path.name not in SKIP:
                for child in sorted(path.iterdir()):
                    add(child)
            return
        if not path.is_file() or (path.suffix not in SUFFIXES and name not in FILES):
            raise ValueError(f"Unsupported publication input: {name}")
        if path.stat().st_size > 2 * 1024 * 1024:
            raise ValueError(f"Oversized source: {name}")
        data = path.read_bytes()
        text = data.decode("utf-8")
        if "\0" in text or any(pattern.search(text) for pattern in TOKENS):
            raise ValueError(f"Binary content or potential credential in {name}; value withheld")
        if re.search(r"[A-Z]:[\\/]+(?:Users|Projects|AndroidSDK)[\\/]", text):
            raise ValueError(f"Personal machine path in {name}")
        result[name] = data

    for name in FILES + DIRECTORIES:
        add(ROOT / name)
    metadata = tomllib.loads(result["pyproject.toml"].decode())["project"]
    if metadata["version"] != "1.0.0.post1" or metadata["license"] != "AGPL-3.0-or-later":
        raise ValueError("Version/license metadata differs from the reviewed release")
    if len(result) != len({name.casefold() for name in result}):
        raise ValueError("Source names collide on case-insensitive filesystems")
    return dict(sorted(result.items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--check", action="store_true")
    modes.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        sources = collect()
        if args.output:
            destination = args.output.absolute()
            if destination.suffix != ".zip" or ".." in args.output.parts:
                raise ValueError("Output must be a new ZIP without parent traversal")
            if destination.exists() or destination.is_symlink():
                raise ValueError("Output already exists")
            for parent in destination.parents:
                if parent.is_symlink():
                    raise ValueError("Output cannot pass through links")
            if destination.is_relative_to(ROOT) and not destination.is_relative_to(ROOT / "dist"):
                raise ValueError("Output inside source must be under dist/")
            destination.parent.mkdir(parents=True, exist_ok=True)
            # Build privately, then copy with exclusive creation; never overwrite an existing artifact.
            with tempfile.TemporaryFile() as staged:
                with zipfile.ZipFile(staged, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                    for name, data in sources.items():
                        info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                        info.external_attr = 0o100644 << 16
                        archive.writestr(info, data)
                staged.seek(0)
                created = False
                try:
                    with destination.open("xb") as output:
                        created = True
                        while block := staged.read(1024 * 1024):
                            output.write(block)
                except OSError:
                    if created:
                        destination.unlink(missing_ok=True)
                    raise
        print(f"PASS: {len(sources)} source files; no curriculum corpus, models or installed dependencies included")
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Publication blocked: {error}\n")


if __name__ == "__main__":
    main()
