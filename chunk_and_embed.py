# chunk_and_embed.py
import json
from sentence_transformers import SentenceTransformer
from pathlib import Path
import faiss
import numpy as np
import math
import os
import pickle
from tqdm import tqdm
import re

MODEL_NAME = "all-MiniLM-L6-v2"  # fast and good

def approx_tokens(text):
    # rough tokens ≈ words * 1.3? we want 200-400 tokens. We'll approximate: 1 token ≈ 0.75 words
    words = len(re.findall(r'\w+', text))
    return int(words / 0.75 + 0.5)

def chunk_paragraphs(paragraphs, min_tok=200, max_tok=400):
    chunks = []
    cur = ""
    cur_id = None
    cur_tok = 0
    idx = 0
    for p in paragraphs:
        text = p["text"].strip()
        t = approx_tokens(text)
        if cur_tok + t > max_tok and cur:
            chunks.append({"chunk_id": f"c{len(chunks)+1}", "text": cur.strip()})
            cur = text
            cur_tok = t
        else:
            if cur:
                cur += "\n\n" + text
            else:
                cur = text
            cur_tok += t
    if cur:
        chunks.append({"chunk_id": f"c{len(chunks)+1}", "text": cur.strip()})
    return chunks

def embed_and_store(metadata_path, out_dir="./output"):
    meta = json.load(open(metadata_path, "r", encoding="utf-8"))
    paras = meta["paragraphs"]
    chunks = chunk_paragraphs(paras)
    model = SentenceTransformer(MODEL_NAME)
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine similarity via normalized vectors
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    case_id = Path(metadata_path).parent.name
    od = Path(out_dir) / case_id
    od.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(od / "faiss_index.bin"))
    # save mapping
    mapping = {"chunks": chunks, "meta": meta}
    with open(od / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    # save embedding raw matrix for semantic search retrieval (optional)
    np.save(od / "embeddings.npy", embeddings)
    print("Wrote FAISS index and chunks.json to", od)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("metadata_json")
    p.add_argument("--out_dir", default="./output")
    args = p.parse_args()
    embed_and_store(args.metadata_json, args.out_dir)
