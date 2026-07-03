"""Stage 1 - Load: read raw source files into plain text.

For now we handle plain-text and markdown. To support a new file type
later (PDFs, say), you only add a branch here; nothing downstream changes.
"""
from pathlib import Path

# File extensions we currently know how to read as plain text.
SUPPORTED_SUFFIXES = {".txt", ".md", ".markdown"}


def load_documents(data_dir: Path) -> list[dict]:
    """Read every supported file under `data_dir` into memory.

    Returns one dict per file:
        {"source": "notes.md", "text": "<file contents>"}
    """
    documents = []
    for path in sorted(data_dir.rglob("*")):
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue  # skip empty files
        documents.append({"source": path.name, "text": text})
    return documents
