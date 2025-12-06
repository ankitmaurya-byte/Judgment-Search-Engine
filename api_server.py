# api_server.py

from pathlib import Path
import json
import pickle
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, Query

app = FastAPI(title="LegalTech Judgment Pipeline API")

# ------------------------
# Constants & Globals
# ------------------------

DATA_ROOT = Path("./output")

_MODEL_NAME = "all-MiniLM-L6-v2"
_sentence_model = None  # lazy-loaded SentenceTransformer instance

# FAISS availability
try:
    import faiss  # type: ignore

    FAISS_AVAILABLE = True
except Exception as e:
    faiss = None  # type: ignore
    FAISS_AVAILABLE = False
    print("Warning: faiss import failed. Semantic search via FAISS disabled. Error:", repr(e))

# In-memory caches
_case_faiss_index_cache: Dict[str, Any] = {}
_case_embeddings_cache: Dict[str, np.ndarray] = {}
_case_chunks_cache: Dict[str, List[Dict[str, Any]]] = {}
_case_bm25_cache: Dict[str, Any] = {}

# ------------------------
# Utility / lazy loaders
# ------------------------


def get_sentence_model():
    """
    Lazily load and cache the SentenceTransformer model.
    """
    global _sentence_model

    if _sentence_model is None:
        try:
            from sentence_transformers import SentenceTransformer  # local import
        except Exception as e:
            raise RuntimeError(f"SentenceTransformer import failed: {e}")

        _sentence_model = SentenceTransformer(_MODEL_NAME)

    return _sentence_model


def load_chunks(case_id: str) -> List[Dict[str, Any]]:
    """
    Load chunks.json for a case and cache it.
    """
    if case_id in _case_chunks_cache:
        return _case_chunks_cache[case_id]

    path = DATA_ROOT / case_id / "chunks.json"
    if not path.exists():
        raise FileNotFoundError(f"chunks.json not found for case {case_id}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    _case_chunks_cache[case_id] = data["chunks"]
    return _case_chunks_cache[case_id]


def load_bm25(case_id: str) -> Dict[str, Any]:
    """
    Load bm25_index.pkl for a case and cache it.
    """
    if case_id in _case_bm25_cache:
        return _case_bm25_cache[case_id]

    path = DATA_ROOT / case_id / "bm25_index.pkl"
    if not path.exists():
        raise FileNotFoundError(f"bm25_index.pkl not found for case {case_id}")

    with path.open("rb") as f:
        data = pickle.load(f)

    _case_bm25_cache[case_id] = data
    return _case_bm25_cache[case_id]


def load_faiss_index(case_id: str):
    """
    Load a FAISS index from disk and cache it.
    """
    if case_id in _case_faiss_index_cache:
        return _case_faiss_index_cache[case_id]

    idx_file = DATA_ROOT / case_id / "faiss_index.bin"
    if not idx_file.exists():
        raise FileNotFoundError(f"faiss_index.bin not found for case {case_id}")

    if not FAISS_AVAILABLE:
        raise RuntimeError("faiss is not available in this environment.")

    index = faiss.read_index(str(idx_file))
    _case_faiss_index_cache[case_id] = index
    return index


def load_embeddings(case_id: str) -> np.ndarray:
    """
    Load embeddings.npy for a case and cache it.
    Normalizes once to speed up cosine similarity.
    """
    if case_id in _case_embeddings_cache:
        return _case_embeddings_cache[case_id]

    emb_file = DATA_ROOT / case_id / "embeddings.npy"
    if not emb_file.exists():
        raise FileNotFoundError(f"embeddings.npy not found for case {case_id}")

    arr = np.load(emb_file)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    arr = arr / norms

    _case_embeddings_cache[case_id] = arr
    return arr


# ------------------------
# Basic endpoints
# ------------------------


@app.get("/health")
def health() -> Dict[str, Any]:
    """
    Simple health check endpoint.
    """
    return {
        "status": "ok",
        "faiss_available": FAISS_AVAILABLE,
        "model_loaded": _sentence_model is not None,
    }


@app.get("/case/{case_id}")
def get_case(case_id: str) -> Dict[str, Any]:
    """
    Return metadata.json for a given case, if it exists.
    """
    md = DATA_ROOT / case_id / "metadata.json"
    if not md.exists():
        return {"error": "case not found"}

    with md.open("r", encoding="utf-8") as f:
        return json.load(f)


# ------------------------
# BM25 search endpoint
# ------------------------


@app.get("/search/bm25")
def search_bm25(
    q: str = Query(...),
    case_id: Optional[str] = Query(None),
) -> Dict[str, Any]:
    """
    BM25 search.

    - If `case_id` is provided, search within that case only.
    - Otherwise, iterate over all cases and return top results aggregated.
    """
    # Per-case search
    if case_id:
        try:
            data = load_bm25(case_id)
        except FileNotFoundError:
            return {"error": "bm25 not built for case"}

        bm25 = data["bm25"]
        chunks = data["chunks"]

        # Try rank_bm25's get_top_n
        res = bm25.get_top_n(q.split(), chunks, n=10)

        try:
            # Map back to chunk objects
            output = []
            for text in res:
                for c in chunks:
                    if c["text"] == text:
                        output.append(
                            {
                                "chunk_id": c["chunk_id"],
                                "preview": c["text"][:300],
                            }
                        )
                        break
            return {"results": output}
        except Exception:
            # Fallback: use custom scoring
            from bm25_index import bm25_search

            return {
                "results": bm25_search(
                    DATA_ROOT / case_id / "bm25_index.pkl",
                    q,
                    top_k=10,
                )
            }

    # Aggregate search across all cases
    aggregated_results: List[Dict[str, Any]] = []

    for d in DATA_ROOT.iterdir():
        if not d.is_dir():
            continue

        p = d / "bm25_index.pkl"
        if not p.exists():
            continue

        try:
            from bm25_index import bm25_search

            res = bm25_search(p, q, top_k=3)
            for r in res:
                r["case_id"] = d.name
            aggregated_results.extend(res)
        except Exception:
            continue

    aggregated_results = sorted(
        aggregated_results,
        key=lambda x: x.get("score", 0),
        reverse=True,
    )[:20]

    return {"results": aggregated_results}


# ------------------------
# Semantic search helpers
# ------------------------


def semantic_search_with_faiss(
    case_id: str,
    qvec: np.ndarray,
    top_k: int,
) -> List[Dict[str, Any]]:
    """
    Use FAISS index to perform semantic search.
    Expects `qvec` to be normalized.
    """
    index = load_faiss_index(case_id)
    D, I = index.search(qvec, top_k)

    chunks = load_chunks(case_id)
    results: List[Dict[str, Any]] = []

    for dist, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(chunks):
            continue

        chunk = chunks[idx]
        results.append(
            {
                "chunk_id": chunk["chunk_id"],
                "score": float(dist),
                "preview": chunk["text"][:300],
            }
        )

    return results


def semantic_search_with_numpy(
    case_id: str,
    qvec: np.ndarray,
    top_k: int,
) -> List[Dict[str, Any]]:
    """
    Fallback semantic search using numpy dot product (cosine)
    with stored embeddings.npy.

    `qvec` should be a (1, D) array.
    """
    embeddings = load_embeddings(case_id)  # already normalized

    # Normalize query vector
    qn = qvec / (np.linalg.norm(qvec, axis=1, keepdims=True) + 1e-12)

    scores = (embeddings @ qn.T).squeeze()  # shape (N,)
    top_idx = np.argsort(scores)[-top_k:][::-1]

    chunks = load_chunks(case_id)
    results: List[Dict[str, Any]] = []

    for idx in top_idx:
        i = int(idx)
        chunk = chunks[i]
        results.append(
            {
                "chunk_id": chunk["chunk_id"],
                "score": float(scores[i]),
                "preview": chunk["text"][:300],
            }
        )

    return results


# ------------------------
# Semantic search endpoint
# ------------------------


@app.get("/search/semantic")
def search_semantic(
    q: str = Query(...),
    case_id: Optional[str] = Query(None),
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Semantic search.

    - If FAISS is available and `faiss_index.bin` exists for the case, use FAISS.
    - Otherwise fall back to numpy-based cosine search over `embeddings.npy`.
    - If `case_id` is not provided, search across all available cases and
      return aggregated top-k results.
    """
    # Ensure model is available
    try:
        model = get_sentence_model()
    except RuntimeError as e:
        return {"error": f"SentenceTransformer model unavailable: {e}"}

    # Encode and normalize query
    qvec = model.encode([q], convert_to_numpy=True)
    qvec = qvec / (np.linalg.norm(qvec, axis=1, keepdims=True) + 1e-12)

    # Per-case semantic search
    if case_id:
        idx_file = DATA_ROOT / case_id / "faiss_index.bin"
        emb_file = DATA_ROOT / case_id / "embeddings.npy"

        # Prefer FAISS when available
        if idx_file.exists() and FAISS_AVAILABLE:
            try:
                results = semantic_search_with_faiss(case_id, qvec, top_k)
            except Exception as e:
                # Fallback to numpy
                try:
                    results = semantic_search_with_numpy(case_id, qvec, top_k)
                except Exception as e2:
                    return {
                        "error": f"semantic search failed: {e}; fallback failed: {e2}"
                    }
        elif emb_file.exists():
            try:
                results = semantic_search_with_numpy(case_id, qvec, top_k)
            except Exception as e:
                return {"error": f"numpy semantic search failed: {e}"}
        else:
            return {
                "error": "no semantic index found for this case (no faiss_index.bin or embeddings.npy)"
            }

        return {"results": results}

    # Cross-case semantic search (no case_id)
    aggregated_results: List[Dict[str, Any]] = []

    for d in DATA_ROOT.iterdir():
        if not d.is_dir():
            continue

        case = d.name
        idx_file = d / "faiss_index.bin"
        emb_file = d / "embeddings.npy"

        try:
            if idx_file.exists() and FAISS_AVAILABLE:
                res = semantic_search_with_faiss(case, qvec, top_k)
            elif emb_file.exists():
                res = semantic_search_with_numpy(case, qvec, top_k)
            else:
                continue

            for r in res:
                r["case_id"] = case

            aggregated_results.extend(res)
        except Exception:
            continue

    aggregated_results = sorted(
        aggregated_results,
        key=lambda x: x.get("score", 0),
        reverse=True,
    )[:20]

    return {"results": aggregated_results}
