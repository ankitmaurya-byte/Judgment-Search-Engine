# process_uploaded.py
import glob, os
from pathlib import Path
import subprocess

PDF_DIR = "./mnt/data"
OUT_DIR = "./output"

def process_all():
    pdfs = glob.glob(os.path.join(PDF_DIR, "*")) + glob.glob(os.path.join(PDF_DIR, "*.pdf"))
    print("Found PDFs:", os.path.join(PDF_DIR, "*"), os.path.join(PDF_DIR, "*.pdf"))
    if not pdfs:
        print("No PDFs found in", PDF_DIR)
        return
    for p in pdfs:
        print("Processing", p)
        # parse metadata
        subprocess.run(["python", "parse_pdf.py", p, "--out_dir", OUT_DIR], check=True)
        case_id = Path(p).stem
        # chunk + embed
        subprocess.run(["python", "chunk_and_embed.py", str(Path(OUT_DIR)/case_id/"metadata.json"), "--out_dir", OUT_DIR], check=True)
        # bm25 build
        subprocess.run(["python", "bm25_index.py", str(Path(OUT_DIR)/case_id/"chunks.json"), "--build"], check=True)
    print("Done. Outputs in", OUT_DIR)

if __name__ == "__main__":
    process_all()
