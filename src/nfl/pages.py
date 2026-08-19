"""Load the extracted Fact Book text as a {pdf_page_number: text} map.

Every parser works off page numbers because the book's layout is
positional - club blocks are six pages long, the statistics matrices sit
on four known pages - so keeping the page boundaries is what makes the
structure recoverable.
"""
import re
from pathlib import Path

from .. import config

_PAGE_SPLIT = re.compile(r"=== PAGE (\d+) ===")


def load_pages(path: Path | None = None) -> dict[int, str]:
    """Read the page-delimited text file produced by scripts/extract_pdf.py."""
    path = path or config.FACTBOOK_TXT
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Extract the PDF first:\n"
            f"  python scripts/extract_pdf.py data/2026-Record-and-Fact-Book.pdf "
            f"--out {path}"
        )
    parts = _PAGE_SPLIT.split(path.read_text(encoding="utf-8"))
    return {int(parts[i]): parts[i + 1] for i in range(1, len(parts), 2)}


def find_pages(pages: dict[int, str], pattern: str) -> list[int]:
    """Page numbers whose text matches `pattern`, in order."""
    rx = re.compile(pattern)
    return sorted(n for n, text in pages.items() if rx.search(text))
