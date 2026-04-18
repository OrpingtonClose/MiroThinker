# Copyright (c) 2025 MiroMind
# This source code is licensed under the Apache 2.0 License.

"""
Enhanced document extraction backends for the research pipeline.

Provides pluggable extraction that augments the baseline pdfplumber/ebooklib
with higher-quality alternatives when available:

  1. Docling  — IBM's document AI (tables, multi-column, scanned PDFs, math)
  2. Mathpix  — STEM-optimised OCR, LaTeX equation extraction
  3. Fallback — pdfplumber (already in book_pipeline.py / document_tools.py)

The pipeline tries the best available backend and falls back gracefully.
No external service is *required* — everything degrades to pdfplumber.

Environment variables:
    MATHPIX_APP_ID         — Mathpix application ID
    MATHPIX_APP_KEY        — Mathpix application key
    DOCLING_ENDPOINT       — Docling MCP / API endpoint (if remote)
    EXTRACTION_BACKEND     — Force a backend: "docling", "mathpix", "pdfplumber"
"""

from __future__ import annotations

import io
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────

MATHPIX_APP_ID = os.getenv("MATHPIX_APP_ID", "")
MATHPIX_APP_KEY = os.getenv("MATHPIX_APP_KEY", "")
DOCLING_ENDPOINT = os.getenv("DOCLING_ENDPOINT", "")
EXTRACTION_BACKEND = os.getenv("EXTRACTION_BACKEND", "auto")


# ── Backend availability checks ───────────────────────────────────────


def _docling_available() -> bool:
    """Check if Docling is importable."""
    try:
        from docling.document_converter import DocumentConverter  # noqa: F401
        return True
    except ImportError:
        return False


def _mathpix_configured() -> bool:
    """Check if Mathpix credentials are set."""
    return bool(MATHPIX_APP_ID and MATHPIX_APP_KEY)


def _pdfplumber_available() -> bool:
    """Check if pdfplumber is importable."""
    try:
        import pdfplumber  # noqa: F401
        return True
    except ImportError:
        return False


# ── Docling backend ───────────────────────────────────────────────────


def extract_with_docling(
    pdf_bytes: bytes,
    max_pages: int = 100,
    enable_ocr: bool = False,
) -> Optional[str]:
    """Extract text from PDF using Docling (IBM Document AI).

    Handles complex layouts: multi-column, tables, math formulas,
    scanned content (with OCR enabled).

    Returns None if Docling is not available.
    """
    if not _docling_available():
        return None

    try:
        import tempfile
        from docling.document_converter import DocumentConverter

        # Docling needs a file path
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(pdf_bytes)
            tmp_path = f.name

        try:
            converter = DocumentConverter()
            result = converter.convert(tmp_path)

            # Export as markdown (preserves tables and structure)
            text = result.document.export_to_markdown()

            if not text or len(text.strip()) < 50:
                return None

            return text
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    except Exception as exc:
        logger.debug("Docling extraction failed: %s", exc)
        return None


# ── Mathpix backend ───────────────────────────────────────────────────


def extract_with_mathpix(
    pdf_bytes: bytes,
    max_pages: int = 100,
    output_format: str = "text",
) -> Optional[str]:
    """Extract text from PDF using Mathpix OCR API.

    Excellent for STEM content — converts math equations to LaTeX,
    preserves table structure, handles scanned documents.

    Args:
        pdf_bytes: Raw PDF content.
        max_pages: Maximum pages to process.
        output_format: "text" for plain text, "mmd" for Mathpix Markdown,
                      "latex" for LaTeX.

    Returns:
        Extracted text, or None if Mathpix is not configured/available.
    """
    if not _mathpix_configured():
        return None

    try:
        import httpx
        import base64
        import time

        # Mathpix PDF API: POST the PDF and poll for results
        b64_pdf = base64.b64encode(pdf_bytes).decode("ascii")

        headers = {
            "app_id": MATHPIX_APP_ID,
            "app_key": MATHPIX_APP_KEY,
            "Content-Type": "application/json",
        }

        # Submit PDF for processing
        resp = httpx.post(
            "https://api.mathpix.com/v3/pdf",
            headers=headers,
            json={
                "pdf": b64_pdf,
                "options_json": {
                    "conversion_formats": {output_format: True},
                    "math_inline_delimiters": ["$", "$"],
                    "math_display_delimiters": ["$$", "$$"],
                },
            },
            timeout=60,
        )

        if resp.status_code != 200:
            logger.debug("Mathpix PDF submit failed: HTTP %d", resp.status_code)
            return None

        pdf_id = resp.json().get("pdf_id")
        if not pdf_id:
            return None

        # Poll for completion (Mathpix processes async)
        for _ in range(60):  # max 5 minutes
            time.sleep(5)
            status_resp = httpx.get(
                f"https://api.mathpix.com/v3/pdf/{pdf_id}",
                headers=headers,
                timeout=30,
            )
            if status_resp.status_code != 200:
                continue

            status = status_resp.json().get("status", "")
            if status == "completed":
                # Fetch the output
                out_resp = httpx.get(
                    f"https://api.mathpix.com/v3/pdf/{pdf_id}.{output_format}",
                    headers=headers,
                    timeout=60,
                )
                if out_resp.status_code == 200:
                    return out_resp.text
                return None
            elif status in ("error", "failed"):
                logger.debug("Mathpix processing failed for pdf_id=%s", pdf_id)
                return None

        logger.debug("Mathpix processing timed out for pdf_id=%s", pdf_id)
        return None

    except Exception as exc:
        logger.debug("Mathpix extraction failed: %s", exc)
        return None


# ── pdfplumber fallback ───────────────────────────────────────────────


def extract_with_pdfplumber(pdf_bytes: bytes, max_pages: int = 100) -> str:
    """Extract text from PDF using pdfplumber (baseline)."""
    try:
        import pdfplumber
    except ImportError:
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(io.BytesIO(pdf_bytes))
            pages = []
            for i, page in enumerate(reader.pages[:max_pages]):
                text = page.extract_text()
                if text:
                    pages.append(f"--- Page {i + 1} ---\n{text}")
            return "\n\n".join(pages) if pages else "[No text extracted from PDF]"
        except ImportError:
            return "[ERROR] No PDF library. Install: pip install pdfplumber"

    pages = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for i, page in enumerate(pdf.pages[:max_pages]):
            text = page.extract_text()
            if text:
                pages.append(f"--- Page {i + 1} ---\n{text}")

    return "\n\n".join(pages) if pages else "[No text extracted from PDF]"


# ── Unified extraction entry point ────────────────────────────────────


def extract_pdf(
    pdf_bytes: bytes,
    max_pages: int = 100,
    enable_ocr: bool = False,
    prefer_latex: bool = False,
) -> str:
    """Extract text from PDF using the best available backend.

    Backend selection (unless overridden by EXTRACTION_BACKEND):
      1. Docling — if installed (best for complex layouts)
      2. Mathpix — if API keys configured (best for STEM/math)
      3. pdfplumber — always available (baseline)

    Args:
        pdf_bytes: Raw PDF content.
        max_pages: Maximum pages to extract.
        enable_ocr: Enable OCR for scanned documents (Docling only).
        prefer_latex: If True and Mathpix available, output equations as LaTeX.

    Returns:
        Extracted text content.
    """
    backend = EXTRACTION_BACKEND.lower().strip()

    # Force specific backend if configured
    if backend == "docling":
        result = extract_with_docling(pdf_bytes, max_pages, enable_ocr)
        if result:
            return f"[Extracted via Docling]\n{result}"
        logger.warning("Docling forced but unavailable, falling back to pdfplumber")
        return extract_with_pdfplumber(pdf_bytes, max_pages)

    if backend == "mathpix":
        fmt = "latex" if prefer_latex else "text"
        result = extract_with_mathpix(pdf_bytes, max_pages, fmt)
        if result:
            return f"[Extracted via Mathpix]\n{result}"
        logger.warning("Mathpix forced but unavailable, falling back to pdfplumber")
        return extract_with_pdfplumber(pdf_bytes, max_pages)

    if backend == "pdfplumber":
        return extract_with_pdfplumber(pdf_bytes, max_pages)

    # Auto mode: try best available
    if _docling_available():
        result = extract_with_docling(pdf_bytes, max_pages, enable_ocr)
        if result:
            return f"[Extracted via Docling]\n{result}"

    if _mathpix_configured():
        fmt = "latex" if prefer_latex else "text"
        result = extract_with_mathpix(pdf_bytes, max_pages, fmt)
        if result:
            return f"[Extracted via Mathpix]\n{result}"

    return extract_with_pdfplumber(pdf_bytes, max_pages)


def get_extraction_status() -> dict:
    """Return which extraction backends are available."""
    return {
        "docling": _docling_available(),
        "mathpix": _mathpix_configured(),
        "pdfplumber": _pdfplumber_available(),
        "active_backend": EXTRACTION_BACKEND,
    }
