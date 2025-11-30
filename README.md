# LegalTech Judgment Pipeline

## Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# (Optional) Install system deps:
# sudo apt-get install tesseract-ocr ghostscript

## Process all PDFs placed in /mnt/data
python process_uploaded.py

## Start API server
uvicorn api_server:app --reload --port 8000

## Endpoints
GET /case/{case_id}
GET /search/bm25?q=...&case_id=...
GET /search/semantic?q=...&case_id=...

## Notes
- FAISS used as local vector DB. To use Qdrant, see Qdrant comments in chunk_and_embed.py and replace FAISS calls with qdrant-client payload inserts.
- Tokenization is approximated; replace with tiktoken or exact tokenizer for strict token counts.
