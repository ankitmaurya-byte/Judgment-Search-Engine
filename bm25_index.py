# bm25_index.py
import json
from rank_bm25 import BM25Okapi
import pickle
from pathlib import Path
import re

def tokenize(text):
    return re.findall(r"\w+", text.lower())

def build_bm25(chunks_json_path):
    j = json.load(open(chunks_json_path, "r", encoding="utf-8"))
    chunks = j["chunks"]
    docs = [c["text"] for c in chunks]
    tokenized = [tokenize(d) for d in docs]
    bm25 = BM25Okapi(tokenized)
    outdir = Path(chunks_json_path).parent
    with open(outdir / "bm25_index.pkl", "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)
    print("BM25 built at", outdir / "bm25_index.pkl")

def bm25_search(bm25_pkl_path, query, top_k=5):
    data = pickle.load(open(bm25_pkl_path, "rb"))
    bm25 = data["bm25"]
    chunks = data["chunks"]
    qtok = tokenize(query)
    scores = bm25.get_scores(qtok)
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    results = []
    for i in top_idx:
        results.append({
            "chunk_id": chunks[i]["chunk_id"],
            "score": float(scores[i]),
            "preview": chunks[i]["text"][:300].replace("\n", " ")
        })
    return results

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("chunks_json")
    p.add_argument("--build", action="store_true")
    p.add_argument("--query", help="If provided, run a query")
    args = p.parse_args()
    if args.build:
        build_bm25(args.chunks_json)
    if args.query:
        bm25_pkl = Path(args.chunks_json).parent / "bm25_index.pkl"
        res = bm25_search(bm25_pkl, args.query)
        import pprint; pprint.pprint(res)
