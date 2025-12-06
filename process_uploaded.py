# process_uploaded.py

import glob
import os
import subprocess
from pathlib import Path
from typing import List

PDF_DIR = "./mnt/data"
OUT_DIR = "./output"


def _find_pdfs(pdf_dir: str) -> List[str]:
    """
    Find all PDF-like files to process.

    Logic preserved from original:
      - glob over "<PDF_DIR>/*"
      - glob over "<PDF_DIR>/*.pdf"
      - concatenate both lists (no filtering)
    """
    pattern_all = os.path.join(pdf_dir, "*")
    pattern_pdf = os.path.join(pdf_dir, "*.pdf")

    pdfs = glob.glob(pattern_all) + glob.glob(pattern_pdf)
    print("Found PDFs patterns:", pattern_all, pattern_pdf)
    return pdfs


def _run_subprocess(cmd: list) -> None:
    """
    Helper to run a subprocess with check=True.
    """
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def process_all() -> None:
    """
    Process all PDFs found in PDF_DIR:

      1. Run parse_pdf.py on each file to generate metadata.json.
      2. Run chunk_and_embed.py on that metadata.json to build FAISS index and embeddings.
      3. Run bm25_index.py on chunks.json to build BM25 index.

    Outputs are stored under OUT_DIR/<case_id>/.
    """
    pdfs = _find_pdfs(PDF_DIR)

    if not pdfs:
        print("No PDFs found in", PDF_DIR)
        return

    for p in pdfs:
        print("Processing", p)

        # 1. Parse metadata
        _run_subprocess(
            [
                "python",
                "parse_pdf.py",
                p,
                "--out_dir",
                OUT_DIR,
            ]
        )

        case_id = Path(p).stem
        case_dir = Path(OUT_DIR) / case_id

        # 2. Chunk + embed
        metadata_path = case_dir / "metadata.json"
        _run_subprocess(
            [
                "python",
                "chunk_and_embed.py",
                str(metadata_path),
                "--out_dir",
                OUT_DIR,
            ]
        )

        # 3. BM25 build
        chunks_path = case_dir / "chunks.json"
        _run_subprocess(
            [
                "python",
                "bm25_index.py",
                str(chunks_path),
                "--build",
            ]
        )

    print("Done. Outputs in", OUT_DIR)


if __name__ == "__main__":
    process_all()
