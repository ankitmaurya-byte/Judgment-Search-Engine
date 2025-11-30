# parse_pdf.py
import pdfplumber
import re
import json
import os
from pathlib import Path
import pytesseract
from PIL import Image
import io
import camelot
from typing import List, Dict
from tqdm import tqdm

DATE_RE = re.compile(r'(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})|(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})', re.I)
CITATION_RE = re.compile(r'\b[A-Z][A-Za-z.\-& ]+ v(?:s|ersus|s\.) [A-Z][A-Za-z.\-& ]+', re.I)

def ocr_page_image(img_bytes):
    img = Image.open(io.BytesIO(img_bytes))
    return pytesseract.image_to_string(img)

def extract_tables_from_pdf(path: str):
    tables = []
    try:
        # Camelot works best on machine PDFs with table borders/stream
        mats = camelot.read_pdf(path, pages='all', flavor='stream')
        for i, t in enumerate(mats):
            tables.append({"table_id": i+1, "rows": t.df.values.tolist()})
    except Exception:
        pass
    return tables

def parse_pdf(path: str) -> Dict:
    paragraphs = []
    tables = extract_tables_from_pdf(path)
    title = None
    court = None
    date = None
    citations = set()
    facts = ""
    issues = []
    arguments = {"petitioner": "", "respondent": ""}
    ratio = ""
    holding = ""
    page_texts = []
    with pdfplumber.open(path) as pdf:
        for page in tqdm(pdf.pages, desc=f"Parsing {os.path.basename(path)}"):
            text = page.extract_text() or ""
            if not text.strip():
                # OCR fallback
                try:
                    im = page.to_image(resolution=150)
                    pil = im.original
                    text = pytesseract.image_to_string(pil)
                except Exception:
                    text = ""
            page_texts.append(text)
    full_text = "\n\n".join(page_texts).strip()
    # quick heuristics:
    lines = [ln.strip() for ln in full_text.splitlines() if ln.strip()]
    if lines:
        title = lines[0]
        # find lines that look like court
        for ln in lines[:10]:
            if 'high court' in ln.lower() or 'supreme court' in ln.lower() or 'in the' in ln.lower():
                court = ln
                break
        # find date
        m = DATE_RE.search(full_text)
        if m:
            date = m.group(0)
        # citations (simple heuristic)
        for c in CITATION_RE.findall(full_text):
            # CITATION_RE can return tuples when alternatives used; flatten
            if isinstance(c, tuple):
                c = next(filter(None, c))
            citations.add(c.strip())
    # break into paragraphs by double newline
    paras = [p.strip() for p in re.split(r'\n{2,}', full_text) if p.strip()]
    paragraphs = [{"id": f"p{i+1}", "text": paras[i]} for i in range(len(paras))]
    # Heuristics: try find 'Facts', 'Issues', 'Arguments', 'Ratio', 'Holding' sections
    def extract_section(name):
        pat = re.compile(rf'{name}\s*[:.-]?\s*(?P<body>.*?)\n\s*(?:[A-Z][a-z]+\s*:|$)', re.S | re.I)
        m = pat.search(full_text)
        if m:
            return m.group('body').strip()
        return ""
    facts = extract_section('Facts') or extract_section('FACTS')
    issues_text = extract_section('Issues') or extract_section('ISSUES')
    if issues_text:
        # split into bullets by lines or semicolons
        issues = [i.strip() for i in re.split(r'[\n;•-]+', issues_text) if i.strip()]
    arguments['petitioner'] = extract_section('Arguments') or ""
    ratio = extract_section('Ratio') or ""
    holding = extract_section('Holding') or ""
    out = {
        "title": title or Path(path).stem,
        "court": court or "",
        "date": date or "",
        "facts": facts,
        "issues": issues,
        "arguments": arguments,
        "ratio": ratio,
        "holding": holding,
        "citations": list(citations),
        "paragraphs": paragraphs,
        "tables": tables
    }
    return out

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("pdf_path", help="Path to a single PDF to parse")
    p.add_argument("--out_dir", default="./output", help="Output directory")
    args = p.parse_args()
    meta = parse_pdf(args.pdf_path)
    case_id = Path(args.pdf_path).stem
    od = Path(args.out_dir) / case_id
    od.mkdir(parents=True, exist_ok=True)
    with open(od / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("Wrote", od / "metadata.json")
