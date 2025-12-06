# parse_pdf.py
"""
Refactored PDF parsing utilities: clearer structure, better error handling,
and improved readability while preserving original behavior.
"""

from pathlib import Path
import io
import json
import logging
import os
import re
from typing import List, Dict, Optional, Tuple, Set, Any

import pdfplumber
import pytesseract
from PIL import Image

# Optional dependency for table extraction
try:
    import camelot  # type: ignore

    _HAS_CAMELOT = True
except Exception:
    camelot = None  # type: ignore
    _HAS_CAMELOT = False

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Dates like "12 March 2021" or "12-03-2021" / "12/03/21"
DATE_RE = re.compile(
    r"(\d{1,2}\s+"
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{4})"
    r"|(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
    re.IGNORECASE,
)

# Very rough citation / case-name pattern: "X v/s Y", "X vs Y", "X v. Y", etc.
CITATION_RE = re.compile(
    r"\b([A-Z][A-Za-z.\-& ]+\s+v(?:s\.?|ersus)?\s+[A-Z][A-Za-z.\-& ]+)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_tables(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract tables from a PDF using Camelot, if available.

    Returns a list of:
    {
      "page": int,
      "index": int,
      "data": List[List[str]]
    }
    """
    if not _HAS_CAMELOT:
        return []

    tables_out: List[Dict[str, Any]] = []
    try:
        # 'lattice' works better on ruling-based PDFs; you can try 'stream' if needed.
        camelot_tables = camelot.read_pdf(pdf_path, pages="all")
    except Exception as e:
        logger.warning("Camelot failed for %s: %s", pdf_path, e)
        return []

    for i, t in enumerate(camelot_tables):
        try:
            # t.df is a pandas DataFrame
            df = t.df.fillna("")
            tables_out.append(
                {
                    "page": int(getattr(t, "page", i + 1)),
                    "index": i,
                    "data": df.values.tolist(),
                }
            )
        except Exception as e:
            logger.warning("Failed to process Camelot table %d: %s", i, e)

    return tables_out


def _extract_section(full_text: str, name: str) -> str:
    """
    Heuristic: extract a section body that starts with a heading like 'Facts', 'Issues', etc.

    It looks for:
        <name>[:.-]? <body> ... until the next capitalized word with ':' or end of text.

    This is intentionally fuzzy; it will not be perfect but works decently for many judgments.
    """
    # Example pattern:
    # Facts: <body ...>
    # NextHeading: ...
    pattern = re.compile(
        rf"{name}\s*[:.-]?\s*(?P<body>.*?)\n\s*(?:[A-Z][a-z]+\s*:|$)",
        re.IGNORECASE | re.DOTALL,
    )
    match = pattern.search(full_text)
    if match:
        return match.group("body").strip()
    return ""


# ---------------------------------------------------------------------------
# Core parser
# ---------------------------------------------------------------------------


def parse_pdf(path: str) -> Dict[str, Any]:
    """
    Parse a single judgment PDF and return structured metadata.

    Output structure (JSON-serializable):
    {
      "title": str,
      "court": str,
      "date": str,
      "facts": str,
      "issues": List[str],
      "arguments": {
          "petitioner": str,
          "respondent": str
      },
      "ratio": str,
      "holding": str,
      "citations": List[str],
      "paragraphs": [
          {"id": "p1", "text": "..."},
          ...
      ],
      "tables": [
          {"page": int, "index": int, "data": [[...], ...]},
          ...
      ]
    }
    """
    pdf_path = str(path)

    title: str = ""
    court: str = ""
    date: str = ""
    citations: Set[str] = set()
    facts: str = ""
    issues: List[str] = []
    arguments: Dict[str, str] = {"petitioner": "", "respondent": ""}
    ratio: str = ""
    holding: str = ""
    page_texts: List[str] = []

    # ---------- Extract page-wise text (with OCR fallback) ----------
    logger.info("Opening PDF: %s", pdf_path)

    with pdfplumber.open(pdf_path) as pdf:
        for page in tqdm(pdf.pages, desc=f"Parsing {os.path.basename(pdf_path)}"):
            text = page.extract_text() or ""
            if not text.strip():
                # OCR fallback when no text layer present
                try:
                    # pdfplumber page -> PIL image
                    img = page.to_image(resolution=150)
                    pil: Image.Image = img.original
                    text = pytesseract.image_to_string(pil)
                except Exception as e:
                    logger.warning("OCR failed on page: %s", e)
                    text = ""
            page_texts.append(text)

    full_text = "\n\n".join(page_texts).strip()

    # ---------- Basic header info: title, court, date, citations ----------
    lines = [ln.strip() for ln in full_text.splitlines() if ln.strip()]

    if lines:
        # Naive title: first non-empty line
        title = lines[0]

        # Try to find a line that looks like the court name in the top ~10 lines
        for ln in lines[:10]:
            lower = ln.lower()
            if (
                "high court" in lower
                or "supreme court" in lower
                or lower.startswith("in the")
            ):
                court = ln
                break

    # Date anywhere in the document
    m_date = DATE_RE.search(full_text)
    if m_date:
        date = m_date.group(0)

    # Citations / case names (heuristic)
    for c in CITATION_RE.findall(full_text):
        # CITATION_RE returns a single group; keep flattening logic in case of future changes
        if isinstance(c, tuple):
            c = next((x for x in c if x), "")
        if c:
            citations.add(c.strip())

    # ---------- Paragraphs ----------
    # Split by blank lines (2+ newlines)
    raw_paras = [p.strip() for p in re.split(r"\n{2,}", full_text) if p.strip()]
    paragraphs = [
        {"id": f"p{i + 1}", "text": raw_paras[i]} for i in range(len(raw_paras))
    ]

    # ---------- Heuristic sections: Facts, Issues, Arguments, Ratio, Holding ----------
    facts = _extract_section(full_text, "Facts") or _extract_section(full_text, "FACTS")

    issues_text = _extract_section(full_text, "Issues") or _extract_section(
        full_text, "ISSUES"
    )
    if issues_text:
        # Split into bullet-like items by newlines / semicolons / bullet characters / dashes
        raw_issues = re.split(r"[\n;•\-]+", issues_text)
        issues = [i.strip() for i in raw_issues if i.strip()]

    # For now, treat the generic "Arguments" block as petitioner's arguments
    arguments["petitioner"] = _extract_section(full_text, "Arguments") or ""

    ratio = _extract_section(full_text, "Ratio") or _extract_section(full_text, "RATIO")
    holding = _extract_section(full_text, "Holding") or _extract_section(
        full_text, "HOLDING"
    )

    # ---------- Tables ----------
    tables = _extract_tables(pdf_path)

    # ---------- Output ----------
    out: Dict[str, Any] = {
        "title": title or Path(pdf_path).stem,
        "court": court or "",
        "date": date or "",
        "facts": facts,
        "issues": issues,
        "arguments": arguments,
        "ratio": ratio,
        "holding": holding,
        "citations": sorted(citations),
        "paragraphs": paragraphs,
        "tables": tables,
    }

    return out


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path", help="Path to a single PDF to parse")
    parser.add_argument(
        "--out_dir",
        default="./output",
        help="Output directory (default: ./output)",
    )
    args = parser.parse_args()

    metadata = parse_pdf(args.pdf_path)
    case_id = Path(args.pdf_path).stem

    out_dir = Path(args.out_dir) / case_id
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "metadata.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("Wrote", out_path)
