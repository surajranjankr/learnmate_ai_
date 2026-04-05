from __future__ import annotations

import json
import os
from pathlib import Path
from importlib import import_module

import numpy as np

from modules.utils import clean_token, ensure_directory

_embed_model = None
_embed_model_error = None
_faiss_module = None


def _get_faiss():
    global _faiss_module
    if _faiss_module is not None:
        return _faiss_module
    try:
        _faiss_module = import_module("faiss")
    except Exception:
        _faiss_module = None
    return _faiss_module


def _get_embed_model():
    global _embed_model, _embed_model_error
    if _embed_model is not None:
        return _embed_model
    try:
        sentence_transformers = import_module("sentence_transformers")
        SentenceTransformer = sentence_transformers.SentenceTransformer
    except Exception:
        _embed_model_error = "sentence-transformers is not installed."
        return None
    local_only = os.getenv("VECTORSTORE_LOCAL_ONLY", "true").lower() == "true"
    try:
        _embed_model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1", local_files_only=local_only)
        _embed_model_error = None
        return _embed_model
    except Exception as exc:
        if local_only:
            _embed_model_error = str(exc)
            return None
        try:
            _embed_model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1", local_files_only=False)
            _embed_model_error = None
            return _embed_model
        except Exception as inner_exc:
            _embed_model_error = str(inner_exc)
            return None


def _index_paths(index_path: str) -> tuple[Path, Path]:
    return Path(f"{index_path}.index"), Path(f"{index_path}_texts.json")


def _token_overlap_score(query: str, text: str) -> float:
    query_tokens = [clean_token(token) for token in query.split() if clean_token(token)]
    text_tokens = [clean_token(token) for token in text.split() if clean_token(token)]
    if not query_tokens or not text_tokens:
        return 0.0
    overlap = sum(text_tokens.count(token) for token in set(query_tokens))
    phrase_bonus = 2.5 if query.strip().lower() in text.lower() else 0.0
    coverage_bonus = min(len(set(query_tokens) & set(text_tokens)), len(set(query_tokens))) * 1.2
    ordered_hits = 0.0
    lowered_text = text.lower()
    for token in query_tokens:
        if token and token in lowered_text:
            ordered_hits += 0.35
    return float(overlap) + phrase_bonus + coverage_bonus + ordered_hits


def build_vectorstore(texts: list[str], index_path: str = "embeddings/vectordb"):
    ensure_directory("embeddings")
    index_file, text_file = _index_paths(index_path)
    with text_file.open("w", encoding="utf-8") as file:
        json.dump(list(texts), file, ensure_ascii=False, indent=2)
    embed_model = _get_embed_model()
    faiss = _get_faiss()
    if faiss is None or embed_model is None or not texts:
        return
    embeddings = np.asarray(embed_model.encode(texts), dtype="float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(index_file))


def retrieve_relevant_chunks_with_scores(
    query: str,
    k: int = 5,
    index_path: str = "embeddings/vectordb",
    score_threshold: float = 3.5,
) -> list[dict[str, float | str]]:
    index_file, text_file = _index_paths(index_path)
    if not text_file.exists():
        return []
    with text_file.open("r", encoding="utf-8") as file:
        texts = json.load(file)

    ranked_pairs = sorted(((text, _token_overlap_score(query, text)) for text in texts), key=lambda item: item[1], reverse=True)
    strong_lexical = []
    lexical_threshold = max(2.5, min(6.0, len([token for token in query.split() if clean_token(token)]) * 1.4))
    for text, score in ranked_pairs[: max(k * 2, k)]:
        if score >= lexical_threshold:
            strong_lexical.append(
                {
                    "text": text,
                    "score": round(float(score), 3),
                    "confidence": round(min(0.98, 0.45 + score / 12), 2),
                }
            )
        if len(strong_lexical) >= k:
            break
    if strong_lexical:
        return strong_lexical

    embed_model = _get_embed_model()
    faiss = _get_faiss()
    if faiss is not None and embed_model is not None and index_file.exists():
        query_embedding = np.asarray(embed_model.encode([query]), dtype="float32")
        index = faiss.read_index(str(index_file))
        distances, indices = index.search(query_embedding, min(max(k * 3, k), len(texts)))
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx >= 0 and distance <= score_threshold and texts[idx] not in {item['text'] for item in results}:
                confidence = round(max(0.05, 1 - (float(distance) / max(score_threshold, 0.1))), 2)
                results.append({"text": texts[idx], "score": round(float(distance), 3), "confidence": confidence})
            if len(results) >= k:
                break
        if results:
            return results

    output = []
    for text, score in ranked_pairs[: max(k * 2, k)]:
        if score > 0:
            output.append({"text": text, "score": round(float(score), 3), "confidence": round(min(0.95, 0.35 + score / 10), 2)})
        if len(output) >= k:
            break
    return output


def retrieve_relevant_chunks(query: str, k: int = 5, index_path: str = "embeddings/vectordb", score_threshold: float = 3.5):
    return [item["text"] for item in retrieve_relevant_chunks_with_scores(query, k=k, index_path=index_path, score_threshold=score_threshold)]
