import argparse
import json
import sys
from importlib.resources import files
from pathlib import Path

from . import RELEASE
from .bundle import export_bundle
from .extract import extract
from .search import DEFAULT_MODEL, CurriculumSearch, FastEmbedder, build_index, build_vectors
from .server import serve


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="2017课标定向提取及本地检索")
    parser.add_argument("--version", action="version", version=RELEASE)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("sources", help="显示支持的2017原始PDF文件名、页数和SHA-256；不下载文件")
    p = commands.add_parser("extract", help="仅提取配置的目标页；不会扫描整份PDF")
    p.add_argument("--input", type=Path, default=Path("."))
    p.add_argument("--output", type=Path, default=Path("data"))
    p.add_argument("--max-chars", type=int, default=340)
    p.add_argument("--overlap", type=int, default=30)
    p = commands.add_parser("index", help="建立SQLite中文词项索引")
    p.add_argument("--data", type=Path, default=Path("data"))
    p = commands.add_parser("embed", help="建立本地ONNX语义索引；默认只使用本地模型")
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--cache", type=Path, default=Path("models"))
    p.add_argument("--allow-download", action="store_true", help="明确允许首次下载模型到--cache")
    p = commands.add_parser("bundle", help="从本地索引和模型导出供Synaxis导入的私有便携包")
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--model-cache", type=Path, default=Path("models"))
    p.add_argument("--output", type=Path, required=True)
    for name in ("search", "serve"):
        p = commands.add_parser(name)
        p.add_argument("--data", type=Path, default=Path("data"))
        p.add_argument("--model-cache", type=Path, default=None)
        if name == "search":
            p.add_argument("query")
            p.add_argument("--subject")
            p.add_argument("--category", dest="course_category")
            p.add_argument("--top-k", type=int, default=5)
            p.add_argument("--mode", choices=["lexical", "semantic", "hybrid"], default="hybrid")
        else:
            p.add_argument("--host", default="127.0.0.1")
            p.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    try:
        if args.command == "sources":
            result = json.loads(files("curriculum_search").joinpath("sources.json").read_text(encoding="utf-8"))
        elif args.command == "bundle":
            receipt = export_bundle(args.data, args.model_cache, args.output)
            result = {key: receipt[key] for key in ("edition", "chunks", "sections", "model_revision")}
        elif args.command == "extract":
            result = extract(args.input, args.output, args.max_chars, args.overlap)
            result = {"sections": result["total_sections"], "chunks": result["total_chunks"],
                      "subjects": result["subject_chunk_counts"],
                      "pages_read": {d["source_file"]: d["pages_read_count"] for d in result["documents"]}}
        elif args.command == "index":
            result = build_index(args.data)
        elif args.command == "embed":
            embedder = FastEmbedder(args.model, args.cache, local_only=not args.allow_download)
            result = build_vectors(args.data, embedder, args.model, args.cache)
        else:
            search = CurriculumSearch(args.data, model_cache=args.model_cache)
            if args.command == "serve":
                return serve(search, args.host, args.port)
            result = search.search(args.query, top_k=args.top_k, subject=args.subject,
                                   course_category=args.course_category, mode=args.mode)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as exc:
        parser.exit(1, f"{type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    main()
