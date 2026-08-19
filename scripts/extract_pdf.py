"""Extract text from a PDF into a page-delimited .txt file.

Usage:
    python scripts/extract_pdf.py data/2026-Record-and-Fact-Book.pdf
    python scripts/extract_pdf.py <pdf> --out data/extracted/book.txt --engine pdfplumber

Two engines:
  pypdf       - fast, good for flowing prose. Default.
  pdfplumber  - slower, but `layout=True` preserves column spacing, which
                matters for the statistical tables in the NFL Fact Book.

Pages are separated by a marker line so downstream chunking can keep a
page number as metadata (useful for citations in a RAG answer).
"""
import argparse
import sys
import time
from pathlib import Path

PAGE_MARKER = "=== PAGE {n} ==="


def extract_pypdf(pdf_path: Path, first: int, last: int):
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    total = len(reader.pages)
    last = min(last, total) if last else total
    for i in range(first - 1, last):
        yield i + 1, reader.pages[i].extract_text() or ""


def extract_pdfplumber(pdf_path: Path, first: int, last: int):
    import pdfplumber

    with pdfplumber.open(str(pdf_path)) as pdf:
        total = len(pdf.pages)
        last = min(last, total) if last else total
        for i in range(first - 1, last):
            page = pdf.pages[i]
            # layout=True keeps horizontal whitespace, so table columns stay
            # visually aligned instead of collapsing into one run-on line.
            text = page.extract_text(layout=True, x_density=4.5) or ""
            yield i + 1, text
            page.flush_cache()  # 22 MB file: don't hold every page in memory


ENGINES = {"pypdf": extract_pypdf, "pdfplumber": extract_pdfplumber}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--engine", choices=sorted(ENGINES), default="pypdf")
    ap.add_argument("--first", type=int, default=1, help="first page (1-based)")
    ap.add_argument("--last", type=int, default=0, help="last page, 0 = end")
    args = ap.parse_args()

    if not args.pdf.exists():
        print(f"error: {args.pdf} not found", file=sys.stderr)
        return 1

    out = args.out or args.pdf.with_suffix(f".{args.engine}.txt")
    out.parent.mkdir(parents=True, exist_ok=True)

    started = time.time()
    pages = chars = 0
    empty: list[int] = []

    with out.open("w", encoding="utf-8") as fh:
        for page_no, text in ENGINES[args.engine](args.pdf, args.first, args.last):
            fh.write(PAGE_MARKER.format(n=page_no) + "\n")
            fh.write(text.rstrip() + "\n\n")
            pages += 1
            chars += len(text)
            if not text.strip():
                empty.append(page_no)
            if pages % 50 == 0:
                print(f"  ...{pages} pages ({time.time() - started:.0f}s)", flush=True)

    print(f"\nwrote {out}")
    print(f"  engine     : {args.engine}")
    print(f"  pages      : {pages}")
    print(f"  characters : {chars:,}  (avg {chars // max(pages, 1):,}/page)")
    print(f"  empty pages: {len(empty)}" + (f" -> {empty[:20]}" if empty else ""))
    print(f"  elapsed    : {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
