# bm25_index.py

import json
import pickle
import re
from pathlib import Path
from typing import List, Dict, Any

from rank_bm25 import BM25Okapi


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------


def tokenize(text: str) -> List[str]:
    """
    Simple tokenizer used for BM25:

    - Lowercases the text.
    - Splits on word characters: \w+ (letters, digits, underscore).

    This logic is intentionally the same as the original implementation.
    """
    return re.findall(r"\w+", text.lower())


# ---------------------------------------------------------------------------
# BM25 build + search
# ---------------------------------------------------------------------------


def build_bm25(chunks_json_path: str) -> None:
    """
    Build a BM25 index from a chunks.json file.

    chunks.json format (as produced by chunk_and_embed.py):
    {
      "chunks": [
        { "chunk_id": "c1", "text": "..." },
        { "chunk_id": "c2", "text": "..." },
        ...
      ],
      "meta": { ... }
    }

    This function:
      - Reads the chunks,
      - Tokenizes them,
      - Builds BM25Okapi,
      - Writes bm25_index.pkl next to chunks.json containing:
        { "bm25": <BM25Okapi>, "chunks": <chunks_list> }
    """
    chunks_path = Path(chunks_json_path)

    with chunks_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    chunks: List[Dict[str, Any]] = data["chunks"]
    docs = [c["text"] for c in chunks]
    tokenized_docs = [tokenize(d) for d in docs]

    bm25 = BM25Okapi(tokenized_docs)

    outdir = chunks_path.parent
    out_path = outdir / "bm25_index.pkl"

    with out_path.open("wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)

    print("BM25 built at", out_path)


def bm25_search(
    bm25_pkl_path: str | Path,
    query: str,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Run a BM25 search against a pre-built bm25_index.pkl.

    Returns a list of:
    [
      {
        "chunk_id": str,
        "score": float,
        "preview": str
      },
      ...
    ]

    - `preview` is the first 300 characters of the chunk text with newlines removed.
    """
    pkl_path = Path(bm25_pkl_path)

    with pkl_path.open("rb") as f:
        data = pickle.load(f)

    bm25: BM25Okapi = data["bm25"]
    chunks: List[Dict[str, Any]] = data["chunks"]

    q_tokens = tokenize(query)
    scores = bm25.get_scores(q_tokens)

    # Get indices of top_k scores, sorted descending
    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )[:top_k]

    results: List[Dict[str, Any]] = []
    for i in top_indices:
        chunk = chunks[i]
        preview = chunk["text"][:300].replace("\n", " ")
        results.append(
            {
                "chunk_id": chunk["chunk_id"],
                "score": float(scores[i]),
                "preview": preview,
            }
        )

    return results


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import pprint

    parser = argparse.ArgumentParser()
    parser.add_argument("chunks_json", help="Path to chunks.json")
    parser.add_argument(
        "--build",
        action="store_true",
        help="Build BM25 index from chunks.json",
    )
    parser.add_argument(
        "--query",
        help="If provided, run a query against the built index",
    )
    args = parser.parse_args()

    if args.build:
        build_bm25(args.chunks_json)

    if args.query:
        bm25_pkl = Path(args.chunks_json).parent / "bm25_index.pkl"
        results = bm25_search(bm25_pkl, args.query)
        pprint.pprint(results)
