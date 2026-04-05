from __future__ import annotations

from io import BytesIO
import os
import re
from typing import BinaryIO

try:
    import pdfplumber
except Exception:
    pdfplumber = None


PARAGRAPH_BREAK = re.compile(r"\n\s*\n+")
PAGE_MARKER = re.compile(r"(?:^|\s*)\[Page\s+\d+\]\s*", re.IGNORECASE)


def ensure_directory(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def extract_text_from_pdf(uploaded_file: BinaryIO) -> str:
    if pdfplumber is None:
        raise ValueError(
            "PDF support is unavailable because 'pdfplumber' is not installed. "
            "Install dependencies from requirements.txt to enable PDF uploads."
        )
    try:
        file_bytes = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
        if not file_bytes:
            raise ValueError("The uploaded PDF file is empty.")

        with pdfplumber.open(BytesIO(file_bytes)) as pdf:
            if not pdf.pages:
                raise ValueError("The uploaded PDF has no pages.")

            extracted_pages = []
            for page_number, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text() or ""
                cleaned = page_text.strip()
                if cleaned:
                    extracted_pages.append(f"[Page {page_number}]\n{cleaned}")

            if not extracted_pages:
                raise ValueError("No readable text was found in the uploaded PDF.")
            return "\n\n".join(extracted_pages)
    except Exception as exc:
        raise ValueError(f"Could not extract text from the PDF: {exc}") from exc


def strip_page_markers(text: str) -> str:
    cleaned = PAGE_MARKER.sub("\n", text or "")
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def chunk_text(text: str, length: int = 1800, overlap: int = 250) -> list[str]:
    normalized = text.replace("\r\n", "\n")
    paragraphs = [part.strip() for part in PARAGRAPH_BREAK.split(normalized) if part.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = paragraph if not current else f"{current}\n\n{paragraph}"
        if len(candidate) <= length:
            current = candidate
            continue

        if current:
            chunks.append(current)
        if len(paragraph) <= length:
            current = paragraph
            continue

        start = 0
        while start < len(paragraph):
            end = min(start + length, len(paragraph))
            chunk = paragraph[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(paragraph):
                break
            start = max(end - overlap, start + 1)
        current = ""

    if current:
        chunks.append(current)

    return chunks


def clean_token(token: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "", token).lower().strip()
