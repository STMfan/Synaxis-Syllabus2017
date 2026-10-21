"""Read designated TOC/boundary/target pages only; never scan whole PDFs.

The profiles describe this exact 2017 corpus. Unknown files fail explicitly.
"""

from __future__ import annotations

import hashlib
from importlib.resources import files
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


COMPULSORY = "必修课程"
SELECTIVE = "选择性必修课程"
CATEGORIES = [COMPULSORY, SELECTIVE]
SCHEMA_VERSION = "1.0"

# Printed page numbers come from the two TOC pages, not a full-document scan.
PROFILES = {
    "ZW18-101_": ("地理", 10, 8, 11, 16),
    "ZW18-102_": ("生物学", 12, 11, 19, 31),
    "ZW18-103_": ("化学", 10, 11, 27, 53),
    "ZW18-131_": ("物理", 10, 11, 24, 36),
    "ZW18-133_": ("英语", 10, 12, None, 46),
    "ZW18-144_": ("思想政治", 10, 11, 23, 31),
    "ZW18-310_": ("语文", 10, 11, None, 35),
    "ZW18-72_": ("数学", 10, 13, 36, 50),
    "ZW18-86_": ("历史", 10, 12, 22, 33),
}


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable_id(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode()).hexdigest()[:24]


def compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


def clean(text: str) -> str:
    # Do NOT NFKC mathematical symbols, superscripts, circled numbers or units.
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\u00ad]", "", text)
    text = text.replace("\u3000", " ").replace("\u2002", " ").replace("\u00a0", " ")
    return re.sub(r"[ \t]+", " ", text).strip()


def join_lines(lines: list[str]) -> str:
    result = ""
    for line in lines:
        line = clean(line)
        if not line:
            continue
        if result and re.search(r"[A-Za-z0-9]$", result) and re.match(r"[A-Za-z0-9]", line):
            result += " "
        result += line
    return result


@dataclass
class Line:
    text: str
    bbox: tuple
    size: float


class Source:
    def __init__(self, path: Path, subject: str, offset: int):
        self.path, self.subject, self.offset = path, subject, offset
        import pymupdf

        self.doc = pymupdf.open(path)
        self.sha256 = digest_file(path)
        self.id = stable_id(path.name, self.sha256)
        self.reads: dict[int, set[str]] = {}
        self.raw_pages: dict[int, dict] = {}
        self._lines: dict[int, list[Line]] = {}

    def page(self, n: int, reason: str):
        if not 1 <= n <= len(self.doc):
            raise ValueError(f"{self.path.name}: invalid page {n}")
        self.reads.setdefault(n, set()).add(reason)
        return self.doc[n - 1]

    def lines(self, n: int, reason: str = "target") -> list[Line]:
        page = self.page(n, reason)
        if n not in self._lines:
            items = []
            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    text = clean("".join(s["text"] for s in line["spans"]))
                    if not text:
                        continue
                    bbox = tuple(line["bbox"])
                    # Running heads and page numbers are outside the body box.
                    if bbox[1] < 75 or bbox[1] > page.rect.height - 80:
                        continue
                    items.append(Line(text, bbox, max(s["size"] for s in line["spans"])))
            self._lines[n] = sorted(items, key=lambda x: (round(x.bbox[1], 1), x.bbox[0]))
        if reason == "target":
            self.raw_pages[n] = {
                "document_id": self.id, "subject": self.subject,
                "source_file": self.path.name, "pdf_page": n,
                "printed_page": n - self.offset,
                "raw_text": page.get_text(sort=True),
                "body_lines": [{"text": x.text, "bbox": [round(v, 2) for v in x.bbox]}
                               for x in self._lines[n]],
            }
        return self._lines[n]

    def summary(self) -> dict:
        return {
            "document_id": self.id, "subject": self.subject,
            "source_file": self.path.name, "source_sha256": self.sha256,
            "pdf_pages_total": len(self.doc), "printed_page_offset": self.offset,
            "pages_read": [{"pdf_page": n, "reasons": sorted(r)} for n, r in sorted(self.reads.items())],
            "pages_read_count": len(self.reads), "status": "extracted",
        }


def verify_toc(source: Source, required: list[str]):
    text = "".join(source.page(n, "toc").get_text() for n in (9, 10))
    missing = [s for s in required if s not in compact(text)]
    if missing:
        raise ValueError(f"{source.path.name}: TOC profile mismatch: {missing}")


def find_anchor(lines: list[Line], anchor: str) -> int:
    for i, line in enumerate(lines):
        if compact(line.text) == compact(anchor):
            return i
    raise ValueError(f"Boundary heading not found: {anchor}")


def fragment(n: int, text: str, bbox=None) -> dict:
    result = {"pdf_page": n, "text": text}
    if bbox:
        result["bbox"] = [round(x, 2) for x in bbox]
    return result


def make_section(source: Source, title: str, cats: list[str], fragments: list[dict],
                 path: list[str], scope: str = "course_specific", **extra) -> dict:
    fragments = [f for f in fragments if f["text"].strip()]
    if not fragments:
        raise ValueError(f"Empty section: {source.subject} {title}")
    pages = sorted({f["pdf_page"] for f in fragments})
    text = "\n\n".join(f["text"] for f in fragments)
    warnings = []
    if source.subject == "数学":
        warnings.append("源PDF部分公式和特殊字形编码异常；保留提取文本，精确公式须核对原页。")
    if re.search(r"[\ue000-\uf8ff\ufffd]", text):
        warnings.append("含私用区或替代字符，请核对原PDF。")
    original = extra.pop("original_course_categories", cats)
    section = {
        "schema_version": SCHEMA_VERSION,
        "id": stable_id(source.id, "/".join(path), str(pages), text),
        "document_id": source.id, "subject": source.subject,
        "course_categories": cats, "original_course_categories": original,
        "scope": scope, "title": title, "heading_path": path,
        "text": text, "fragments": fragments,
        "source": {"file": source.path.name, "sha256": source.sha256,
                   "pdf_pages": pages, "pdf_page_start": min(pages), "pdf_page_end": max(pages),
                   "printed_pages": [n - source.offset for n in pages]},
        "quality_warnings": warnings,
    }
    section.update(extra)
    return section


def paragraphs(lines: list[Line]) -> list[str]:
    """Join printed wraps; keep list starts and visible paragraph indents."""
    result, pending = [], []
    for line in lines:
        is_start = bool(re.match(r"^(?:[（(][0-9０-９]+[)）]|[0-9０-９]+[.．、]|[①-⑳])", line.text))
        if pending and (is_start or (line.bbox[0] >= 112 and re.search(r"[。！？；]$", pending[-1]))):
            result.append(join_lines(pending))
            pending = []
        pending.append(line.text)
    if pending:
        result.append(join_lines(pending))
    return result


def math_schedule_lines(source: Source, n: int, lines: list[Line]) -> list[Line]:
    """Preserve the four course-hours tables with their merged-cell relationships."""
    import pymupdf

    tables = source.page(n, "target").find_tables().tables
    result = [line for line in lines if not any(
        pymupdf.Rect(line.bbox).intersects(pymupdf.Rect(t.bbox)) for t in tables)]
    for table in tables:
        rows = table.extract()
        if table.col_count != 3 or compact(rows[0][0] or "") != "主题":
            raise ValueError(f"Unexpected math schedule table on page {n}")
        theme, hours = "", ""
        for ri, row in enumerate(rows[1:], 1):
            if row[0]:
                theme = join_lines(row[0].splitlines())
            if row[2]:
                hours = join_lines(row[2].splitlines())
            unit = join_lines((row[1] or "").splitlines())
            text = f"表格行：主题：{theme}；单元：{unit}；该主题建议课时（合并单元格）：{hours}。"
            bbox = tuple(table.rows[ri].bbox)
            result.append(Line(text, bbox, 12.0))
    return sorted(result, key=lambda x: (x.bbox[1], x.bbox[0]))


def standard_sections(source: Source, start: int, selective: int, end: int) -> list[dict]:
    verify_toc(source, ["必修课程", "选择性必修课程", "选修课程"])
    output = []
    for cat, a, b, begin, finish in [
        (COMPULSORY, start, selective, "（一）必修课程", "（二）选择性必修课程"),
        (SELECTIVE, selective, end, "（二）选择性必修课程", "（三）选修课程"),
    ]:
        title, current_path, parts = cat, ["课程内容", cat], []
        module = None

        def flush():
            nonlocal parts
            if parts:
                output.append(make_section(source, title, [cat], parts, current_path.copy()))
                parts = []

        for n in range(a + source.offset, b + source.offset + 1):
            lines = source.lines(n)
            if source.subject == "数学" and n in (23, 24, 46, 47):
                lines = math_schedule_lines(source, n, lines)
            if n == a + source.offset:
                lines = lines[find_anchor(lines, begin) + 1:]
            if n == b + source.offset:
                lines = lines[:find_anchor(lines, finish)]
            pending = []

            def drain():
                nonlocal pending
                for p in paragraphs(pending):
                    parts.append(fragment(n, p))
                pending = []

            for line in lines:
                text = line.text
                top = re.match(r"^(?:主题[一二三四五六七八九十0-9０-９]+[ ：:、]+.+|模块\s*[一二三四五六七八九十0-9０-９]+(?:[ ：:、].*|$)|(?:选择性)?必修\s*[123１２３](?:[ ：:、].*|$))", text)
                label = re.fullmatch(r"【[^】]{2,18}】", text)
                if top or label:
                    drain()
                    flush()
                    title = text
                    if top:
                        module = text
                    current_path = ["课程内容", cat] + ([module] if module else [])
                    if label:
                        current_path.append(text.strip("【】"))
                else:
                    pending.append(line)
            drain()
        flush()
    return output


def chinese_sections(source: Source) -> list[dict]:
    verify_toc(source, ["课程内容", "学习任务群", "学习要求"])
    # The course structure on printed page 9 is the authoritative task assignment.
    structure = compact(source.page(19, "classification_evidence").get_text())
    if "必修课程7个" not in structure or "选择性必修课程9个" not in structure:
        raise ValueError("Chinese task assignment profile mismatch")
    output, parts, task, title = [], [], None, ""

    def flush():
        nonlocal parts
        if not parts:
            return
        if task is not None and task <= 12:
            cats = CATEGORIES.copy() if task <= 4 else ([COMPULSORY] if task <= 7 else [SELECTIVE])
            output.append(make_section(
                source, title, cats, parts, ["课程内容", "学习任务群", title],
                scope="shared_task" if len(cats) > 1 else "course_specific",
                original_course_categories=(CATEGORIES + ["选修课程"]) if task <= 3 else cats,
                classification_evidence={"pdf_page": 19, "printed_page": 9,
                                         "rule": "必修任务群1-7；选择性必修任务群1-4及8-12"},
            ))
        elif task is None and title in ("必修课程学习要求", "选择性必修和选修课程学习要求"):
            shared = title.startswith("选择性")
            output.append(make_section(
                source, title, [SELECTIVE] if shared else [COMPULSORY], parts,
                ["课程内容", "学习要求", title],
                scope="shared_with_elective" if shared else "course_specific",
                original_course_categories=[SELECTIVE, "选修课程"] if shared else [COMPULSORY],
            ))
        parts = []

    # Task 13 starts on PDF page 36. Read that boundary, skip elective-only 37-41.
    for n in list(range(21, 37)) + [42, 43, 44]:
        lines = source.lines(n)
        pending = []

        def drain():
            nonlocal pending
            for p in paragraphs(pending):
                parts.append(fragment(n, p))
            pending = []

        for line in lines:
            text = line.text
            match = re.match(r"^学习任务群\s*(\d+)\s*(.*)", text)
            req = compact(text) in ("必修课程学习要求", "选择性必修和选修课程学习要求")
            if match or req:
                drain()
                flush()
                task = int(match[1]) if match else None
                title = text
            elif compact(text) in ("四、课程内容", "（一）学习任务群", "（二）学习要求"):
                drain()
                flush()
                task, title = None, ""
            elif task is not None and task >= 13:
                continue
            elif title:
                pending.append(line)
        drain()
    flush()
    return output


def english_sections(source: Source) -> list[dict]:
    import pymupdf

    verify_toc(source, ["课程内容", "主题语境", "语篇类型", "学习策略"])
    evidence = compact(source.page(23, "classification_evidence").get_text())
    if "不分课程类别进行描述" not in evidence:
        raise ValueError("English shared-theme classification evidence missing")
    output = []
    heading, subheading, content_kind = "课程内容概述", None, "共用说明"
    last_category = None
    last_caption, last_subcategory = "", ""
    for n in range(22, 56):
        page = source.page(n, "target")
        lines = source.lines(n)
        tables = page.find_tables().tables
        events = []
        for table in tables:
            events.append((table.bbox[1], "table", table))
        for line in lines:
            if n == 23 and 220 < line.bbox[1] < 360:
                # Figure 2 uses a corrupt font map. Keep its caption and raw page,
                # not nonsensical glyphs in semantic input. Do not invent an OCR gloss.
                continue
            if not any(pymupdf.Rect(line.bbox).intersects(pymupdf.Rect(t.bbox)) for t in tables):
                events.append((line.bbox[1], "line", line))
        events.sort(key=lambda item: item[0])
        pending = []

        def shared():
            nonlocal pending
            if pending:
                fs = [fragment(n, p) for p in paragraphs(pending)]
                path = ["课程内容", heading] + ([subheading] if subheading else []) + [content_kind]
                output.append(make_section(
                    source, path[-2] + " / " + content_kind, CATEGORIES.copy(), fs, path,
                    scope="shared_across_courses", original_course_categories=CATEGORIES + ["选修课程"],
                    classification_evidence={"pdf_page": 23, "printed_page": 13,
                        "rule": "主题语境不分课程类别；正文说明与教学提示为共用内容，表格按课程行区分"},
                ))
                pending = []

        for _, kind, value in events:
            if kind == "line":
                text = value.text
                if compact(text) == "续表":
                    continue
                if re.match(r"^表\s*\d+", text):
                    last_caption = text
                    continue
                main = re.fullmatch(r"（[一二三四五六]）\s*(.+)", text)
                sub = re.fullmatch(r"[1-5][.．]\s*(语音知识|词汇知识|语法知识|语篇知识|语用知识)", text)
                if main or sub or re.fullmatch(r"【[^】]+】", text):
                    shared()
                    if main:
                        heading, subheading, content_kind = main[1], None, "共用说明"
                        last_category = None
                    elif sub:
                        subheading, content_kind = sub[1], "共用说明"
                        last_category = None
                    else:
                        content_kind = text.strip("【】")
                elif compact(text) != "四、课程内容":
                    pending.append(value)
                continue
            shared()
            table = value
            rows = table.extract()
            header = [compact(x or "") for x in rows[0]]
            is_course = "课程类别" in header
            if not is_course:
                # Theme rows preserve columns, including merged cells, rather than flattening columns.
                previous = [""] * table.col_count
                for ri, row in enumerate(rows[1:], 1):
                    cells = []
                    for ci, cell in enumerate(row):
                        if cell:
                            previous[ci] = join_lines(cell.splitlines())
                        cells.append(previous[ci])
                    text = "\n".join(f"{header[i] or f'列{i+1}'}：{cell}" for i, cell in enumerate(cells) if cell)
                    if text:
                        output.append(make_section(
                            source, heading + " / " + (cells[0] or "主题"), CATEGORIES.copy(),
                            [fragment(n, text, table.rows[ri].bbox)], ["课程内容", heading, cells[0]],
                            scope="shared_across_courses", original_course_categories=CATEGORIES + ["选修课程"],
                            extraction_method="table_cells", table_row=ri, table_caption=last_caption,
                        ))
                continue
            for ri, row in enumerate(rows[1:], 1):
                label = compact(row[0] or "")
                if label:
                    if label == "必修":
                        last_category = COMPULSORY
                    elif label == "选择性必修":
                        last_category = SELECTIVE
                    elif label.startswith("选修"):
                        last_category = "选修课程"
                    else:
                        raise ValueError(f"Unknown English table category: {n} {label}")
                if last_category is None:
                    raise ValueError(f"Unassigned English table continuation: {n} {ri}")
                if last_category not in CATEGORIES:
                    continue
                cells = list(row)
                if table.col_count == 3:
                    if cells[1]:
                        last_subcategory = join_lines(cells[1].splitlines())
                    cells[1] = last_subcategory
                text = "\n".join(f"{header[i] or f'列{i+1}'}：{join_lines(cell.splitlines())}"
                                 for i, cell in enumerate(cells) if i > 0 and cell)
                if not text:
                    raise ValueError(f"Empty English requirement: {n} {ri}")
                path = ["课程内容", heading] + ([subheading] if subheading else []) + ["内容要求", last_category]
                output.append(make_section(
                    source, (subheading or heading) + " / " + last_category, [last_category],
                    [fragment(n, text, table.rows[ri].bbox)], path,
                    extraction_method="table_cells", table_row=ri, table_caption=last_caption,
                ))
        shared()
    return output


def split_text(text: str, limit: int) -> list[str]:
    """Keep exact source substrings. No tail characters are silently discarded."""
    pieces = []
    while len(text) > limit:
        options = [m.end() for m in re.finditer(r"[。；！？\n]", text[:limit]) if m.end() >= limit // 2]
        cut = options[-1] if options else limit
        pieces.append(text[:cut])
        text = text[cut:]
    if text:
        pieces.append(text)
    return pieces


def chunk_sections(sections: list[dict], max_chars: int = 340, overlap: int = 30) -> list[dict]:
    if not 80 <= max_chars <= 1500 or not 0 <= overlap < max_chars // 2:
        raise ValueError("max_chars must be 80..1500; overlap < max_chars/2")
    chunks = []
    for section in sections:
        pieces = []
        for f in section["fragments"]:
            pieces.extend((p, f["pdf_page"]) for p in split_text(f["text"], max_chars))
        groups, texts, pages, size = [], [], set(), 0
        for text, n in pieces:
            if texts and size + len(text) + 2 > max_chars:
                groups.append(("\n\n".join(texts), sorted(pages)))
                texts, pages, size = [], set(), 0
            texts.append(text)
            pages.add(n)
            size += len(text) + 2
        if texts:
            groups.append(("\n\n".join(texts), sorted(pages)))
        previous, previous_pages = "", []
        for i, (text, pages) in enumerate(groups):
            context = previous[-overlap:] if overlap else ""
            prefix = " / ".join([section["subject"], "、".join(section["course_categories"])] + section["heading_path"])
            embedding_text = prefix + "\n" + (context + "\n" if context else "") + text
            offset = section["source"]["pdf_page_start"] - section["source"]["printed_pages"][0]
            chunks.append({
                "schema_version": SCHEMA_VERSION,
                "id": stable_id(section["id"], str(i), text), "section_id": section["id"],
                "document_id": section["document_id"], "subject": section["subject"],
                "course_categories": section["course_categories"],
                "original_course_categories": section["original_course_categories"],
                "scope": section["scope"], "heading_path": section["heading_path"],
                "text": text, "context_before": context,
                "context_pdf_pages": previous_pages if context else [],
                "embedding_text": embedding_text,
                "source": {**section["source"], "pdf_pages": pages,
                           "pdf_page_start": min(pages), "pdf_page_end": max(pages),
                           "printed_pages": [n - offset for n in pages]},
                "quality_warnings": section["quality_warnings"],
            })
            previous, previous_pages = text, pages
    return chunks


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract(input_dir: Path, output_dir: Path, max_chars: int = 340, overlap: int = 30) -> dict:
    catalog = json.loads(files("curriculum_search").joinpath("sources.json").read_text(encoding="utf-8"))
    expected = {item["file"]: item["sha256"] for item in catalog["documents"]}
    inputs = sorted(input_dir.glob("*.pdf"))
    if not inputs:
        raise ValueError(f"No PDF files in {input_dir}")
    # Reject a different edition/layout before a single PDF page is parsed.
    for path in inputs:
        if path.name not in expected or digest_file(path) != expected[path.name]:
            raise ValueError(f"Source PDF is not the reviewed 2017 edition: {path.name}; see sources command")
    if not 1 <= max_chars <= 1000 or not 0 <= overlap <= max_chars:
        raise ValueError("max-chars must be 1..1000 and overlap must be 0..max-chars")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Extraction output must be a new or empty directory")
    import pymupdf

    output_dir.mkdir(parents=True, exist_ok=True)
    sections, pages, summaries, unknown = [], [], [], []
    for path in inputs:
        if path.name.startswith("ZW18-61_"):
            # No need to read its body: the TOC has principles, not disciplinary requirements.
            with pymupdf.open(path) as d:
                toc = compact(d[8].get_text())
                if "课程内容确定的原则" not in toc:
                    raise ValueError("Curriculum-plan TOC profile mismatch")
                summaries.append({"source_file": path.name, "source_sha256": digest_file(path),
                                  "pdf_pages_total": len(d), "pages_read_count": 1,
                                  "pages_read": [{"pdf_page": 9, "reasons": ["toc"]}],
                                  "status": "not_applicable",
                                  "reason": "课程方案仅列课程内容确定原则，没有学科课程内容的必修/选择性必修正文。"})
            continue
        profile = next((v for k, v in PROFILES.items() if path.name.startswith(k)), None)
        if not profile:
            unknown.append(path.name)
            continue
        subject, offset, a, b, c = profile
        source = Source(path, subject, offset)
        try:
            if subject == "语文":
                extracted = chinese_sections(source)
            elif subject == "英语":
                extracted = english_sections(source)
            else:
                extracted = standard_sections(source, a, b, c)
            if not extracted:
                raise ValueError(f"No sections for {subject}")
            sections.extend(extracted)
            pages.extend(source.raw_pages.values())
            summary = source.summary()
            summary["sections"] = len(extracted)
            summary["quality_warnings"] = sorted({w for s in extracted for w in s["quality_warnings"]})
            summaries.append(summary)
        finally:
            source.doc.close()
    if unknown:
        raise ValueError(f"Unprofiled PDFs (not silently skipped): {unknown}")
    chunks = chunk_sections(sections, max_chars, overlap)
    write_jsonl(output_dir / "sections.jsonl", sections)
    write_jsonl(output_dir / "chunks.jsonl", chunks)
    write_jsonl(output_dir / "source_pages.jsonl", pages)
    md_dir = output_dir / "markdown"
    md_dir.mkdir(exist_ok=True)
    for subject in sorted({s["subject"] for s in sections}):
        content = [f"# {subject} · 课程内容（2017年版）", "", "仅收录必修、选择性必修及其适用的共用内容。"]
        for section in (s for s in sections if s["subject"] == subject):
            src = section["source"]
            content += ["", "## " + " / ".join(section["heading_path"][1:]), "",
                        f"课程类别：{'、'.join(section['course_categories'])}；适用范围：{section['scope']}",
                        f"来源：{src['file']}；PDF页 {src['pdf_page_start']}-{src['pdf_page_end']}；印刷页 {src['printed_pages']}", ""]
            if section["quality_warnings"]:
                content += ["> " + "；".join(section["quality_warnings"]), ""]
            content.append(section["text"])
        (md_dir / f"{subject}.md").write_text("\n".join(content) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": SCHEMA_VERSION, "edition": "2017年版（本地文件）",
        "documents": summaries, "total_sections": len(sections), "total_chunks": len(chunks),
        "subject_chunk_counts": dict(Counter(c["subject"] for c in chunks)),
        "chunking": {"max_text_chars": max_chars, "context_before_chars": overlap,
                     "note": "text为原文片段；overlap仅放context_before，不重复污染text；embedding_text包含章节路径。"},
        "files": {n: digest_file(output_dir / n) for n in ("sections.jsonl", "chunks.jsonl", "source_pages.jsonl")},
        "limitations": ["未OCR；数学部分公式/特殊字体编码有损，精确公式以PDF为准。",
                         "图示不生成替代文字；正文及表格文本保留，图片本身未进入embedding。",
                         "语文共享任务群及共同学习要求保留原文适用范围，未把选修专属任务群13-18混入。",
                         "英语按表格课程类别逐行抽取，跨行合并单元格延续类别；通用主题及教学提示单独标注。"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest
