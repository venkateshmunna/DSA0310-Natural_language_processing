import json
import os
import re
from pathlib import Path

import numpy as np

import media_processor

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

_clip_processor = None
_clip_model = None


def _index_paths():
    index_directory = media_processor.DATA_DIR / "faiss"
    return index_directory, index_directory / "clip.index", index_directory / "clip.json"


def _get_faiss():
    try:
        import faiss
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "FAISS is unavailable. Install the project dependencies with "
            "'python -m pip install -r requirements.txt'."
        ) from error
    return faiss


def _get_clip_model():
    global _clip_processor, _clip_model
    if _clip_processor is None or _clip_model is None:
        os.environ["USE_TF"] = "0"
        try:
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as error:
            raise RuntimeError(
                "CLIP dependencies are not installed. Install the project "
                "dependencies with 'python -m pip install -r requirements.txt'."
            ) from error
        _clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        _clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
        _clip_model.eval()
    return _clip_processor, _clip_model


def _normalise_vector(features):
    vector = np.asarray(features, dtype=np.float32).reshape(-1)
    if not np.all(np.isfinite(vector)):
        raise ValueError("CLIP returned non-finite embedding values.")
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        raise ValueError("CLIP returned an empty embedding.")
    return vector / norm


def _encode_image(image_path):
    import torch
    from PIL import Image

    processor, model = _get_clip_model()
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.inference_mode():
        features = model.get_image_features(pixel_values=inputs["pixel_values"])
    return _normalise_vector(features.detach().cpu().numpy())


def _encode_text(text):
    import torch

    processor, model = _get_clip_model()
    inputs = processor(
        text=[text], return_tensors="pt", padding=True, truncation=True
    )
    with torch.inference_mode():
        features = model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
    return _normalise_vector(features.detach().cpu().numpy())


def _read_index():
    _directory, index_path, metadata_path = _index_paths()
    if not index_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("The CLIP/FAISS index has not been built.")

    try:
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        index = _get_faiss().read_index(str(index_path))
    except (OSError, ValueError, RuntimeError) as error:
        raise RuntimeError(f"Could not read the saved FAISS index: {error}") from error

    if not isinstance(metadata, dict):
        raise RuntimeError("The saved FAISS index metadata is invalid.")
    if metadata.get("model") != CLIP_MODEL_NAME:
        raise RuntimeError("The saved index uses a different CLIP model.")
    asset_ids = metadata.get("asset_ids")
    dimension = metadata.get("dimension")
    if (
        not isinstance(asset_ids, list)
        or not asset_ids
        or not all(isinstance(asset_id, str) for asset_id in asset_ids)
        or index.ntotal != len(asset_ids)
        or index.d != dimension
    ):
        raise RuntimeError("The saved FAISS index metadata is inconsistent.")
    return index, asset_ids


def get_index_status():
    assets = media_processor.list_media_assets()
    try:
        index, indexed_ids = _read_index()
    except (FileNotFoundError, RuntimeError) as error:
        return {
            "indexed_assets": 0,
            "available_assets": len(assets),
            "is_current": False,
            "message": str(error),
        }

    available_ids = [str(asset["id"]) for asset in assets]
    is_current = set(indexed_ids) == set(available_ids)
    message = (
        "Index is current."
        if is_current
        else "Media changed since indexing. Rebuild the CLIP/FAISS index."
    )
    return {
        "indexed_assets": index.ntotal,
        "available_assets": len(assets),
        "is_current": is_current,
        "message": message,
    }


def build_index():
    assets = media_processor.list_media_assets()
    if not assets:
        raise ValueError("Add and process at least one image or video before indexing.")

    faiss = _get_faiss()
    embeddings = []
    asset_ids = []
    for asset in assets:
        image_path = Path(asset["path"])
        if not image_path.is_file():
            raise FileNotFoundError(
                f"Cannot index asset {asset['id']}: image file is missing at {image_path}."
            )
        embeddings.append(_encode_image(image_path))
        asset_ids.append(str(asset["id"]))

    vectors = np.vstack(embeddings).astype(np.float32, copy=False)
    if vectors.ndim != 2 or vectors.shape[0] != len(asset_ids):
        raise ValueError("CLIP returned an invalid embedding matrix.")
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    index_directory, index_path, metadata_path = _index_paths()
    index_directory.mkdir(parents=True, exist_ok=True)
    temporary_index = index_path.with_suffix(".index.tmp")
    temporary_metadata = metadata_path.with_suffix(".json.tmp")
    metadata = {
        "model": CLIP_MODEL_NAME,
        "dimension": int(vectors.shape[1]),
        "asset_ids": asset_ids,
    }
    try:
        faiss.write_index(index, str(temporary_index))
        with temporary_metadata.open("w", encoding="utf-8") as file:
            json.dump(metadata, file)
        os.replace(temporary_index, index_path)
        os.replace(temporary_metadata, metadata_path)
    finally:
        temporary_index.unlink(missing_ok=True)
        temporary_metadata.unlink(missing_ok=True)

    return {"indexed_assets": len(asset_ids), "dimension": int(vectors.shape[1])}


def _tokenize(text):
    return re.findall(r"\w+", text.lower())


def _document_text(asset):
    timestamp = asset["timestamp_seconds"]
    timestamp_text = ""
    if timestamp is not None:
        total_seconds = int(timestamp)
        minutes, seconds = divmod(total_seconds, 60)
        timestamp_text = f"{minutes:02d}:{seconds:02d}"
    return " ".join(
        (
            asset["filename"] or "",
            timestamp_text,
            asset["caption"] or "",
            asset["ocr_text"] or "",
        )
    )


def _normalise_scores(scores):
    values = np.asarray(scores, dtype=np.float32)
    if values.size == 0:
        return values
    minimum = float(values.min())
    maximum = float(values.max())
    if maximum - minimum <= 1e-12:
        return np.ones_like(values)
    return (values - minimum) / (maximum - minimum)


def _sparse_scores(query_tokens, tokenized_documents):
    if not query_tokens or not any(tokenized_documents):
        return np.zeros(len(tokenized_documents), dtype=np.float32)

    try:
        from rank_bm25 import BM25Plus
    except ImportError as error:
        raise RuntimeError(
            "BM25 is unavailable. Install the project dependencies with "
            "'python -m pip install -r requirements.txt'."
        ) from error

    model = BM25Plus(tokenized_documents)
    return np.asarray(model.get_scores(query_tokens), dtype=np.float32)


def _dense_scores(query, assets):
    index, indexed_ids = _read_index()
    available_ids = [str(asset["id"]) for asset in assets]
    if set(indexed_ids) != set(available_ids):
        raise RuntimeError(
            "The CLIP/FAISS index is out of date. Rebuild it from the Knowledge Base."
        )

    query_vector = _encode_text(query).reshape(1, -1).astype(np.float32, copy=False)
    if query_vector.shape[1] != index.d:
        raise ValueError("The query embedding does not match the saved FAISS index.")
    faiss = _get_faiss()
    faiss.normalize_L2(query_vector)
    scores, positions = index.search(query_vector, index.ntotal)
    return {
        indexed_ids[int(position)]: float(score)
        for score, position in zip(scores[0], positions[0])
        if position >= 0
    }


def _format_result(asset, score, dense_score=0.0, sparse_score=0.0, similarity=None):
    timestamp = asset["timestamp_seconds"]
    source = asset["filename"]
    if timestamp is not None:
        total_seconds = int(timestamp)
        minutes, seconds = divmod(total_seconds, 60)
        source += f" · {minutes:02d}:{seconds:02d}"
    return {
        "asset_id": str(asset["id"]),
        "filename": asset["filename"],
        "media_type": asset["media_type"],
        "path": asset["path"],
        "timestamp_seconds": timestamp,
        "source": source,
        "caption": asset["caption"],
        "ocr_text": asset["ocr_text"],
        "score": float(score),
        "dense_score": float(dense_score),
        "sparse_score": float(sparse_score),
        "similarity": similarity,
    }


def search(
    query,
    method="Hybrid",
    top_k=5,
    dense_weight=0.5,
    sparse_weight=0.5,
    media_ids=None,
):
    query = query.strip()
    if not query:
        raise ValueError("Enter a question or search phrase.")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError("The number of results must be at least one.")

    method = method.strip().lower()
    if method not in {"bm25", "dense", "hybrid"}:
        raise ValueError("Choose BM25, Dense or Hybrid retrieval.")

    if method == "hybrid" and (
        not np.isfinite(dense_weight)
        or not np.isfinite(sparse_weight)
        or dense_weight < 0
        or sparse_weight < 0
        or dense_weight + sparse_weight == 0
    ):
        raise ValueError("Hybrid weights must be non-negative and cannot both be zero.")

    all_assets = media_processor.list_media_assets()
    if not all_assets:
        return []
    selected_media_ids = {str(media_id) for media_id in media_ids or []}
    assets = (
        [
            asset
            for asset in all_assets
            if str(asset["media_id"]) in selected_media_ids
        ]
        if selected_media_ids
        else all_assets
    )
    if not assets:
        return []

    if method == "bm25":
        query_tokens = _tokenize(query)
        documents = [_tokenize(_document_text(asset)) for asset in assets]
        raw_scores = _sparse_scores(query_tokens, documents)
        matching_positions = [
            position
            for position, document in enumerate(documents)
            if any(token in document for token in query_tokens)
        ]
        if not matching_positions:
            return []
        matching_scores = _normalise_scores(raw_scores[matching_positions])
        results = [
            _format_result(
                assets[position],
                score,
                sparse_score=score,
            )
            for position, score in zip(matching_positions, matching_scores)
        ]
        results.sort(key=lambda result: result["score"], reverse=True)
        return results[:top_k]

    if method == "dense":
        dense_by_id = _dense_scores(query, all_assets)
        ordered_scores = np.asarray(
            [dense_by_id[str(asset["id"])] for asset in assets], dtype=np.float32
        )
        normalized_scores = _normalise_scores(ordered_scores)
        results = [
            _format_result(
                asset,
                normalized_scores[position],
                dense_score=normalized_scores[position],
                similarity=ordered_scores[position],
            )
            for position, asset in enumerate(assets)
        ]
        results.sort(key=lambda result: result["score"], reverse=True)
        return results[:top_k]

    query_tokens = _tokenize(query)
    documents = [_tokenize(_document_text(asset)) for asset in assets]
    normalized_sparse_scores = np.zeros(len(assets), dtype=np.float32)
    if sparse_weight > 0:
        raw_sparse_scores = _sparse_scores(query_tokens, documents)
        matching = np.asarray(
            [
                any(token in document for token in query_tokens)
                for document in documents
            ],
            dtype=bool,
        )
        if np.any(matching):
            normalized_sparse_scores[matching] = _normalise_scores(
                raw_sparse_scores[matching]
            )

    similarities = None
    normalized_dense_scores = np.zeros(len(assets), dtype=np.float32)
    if dense_weight > 0:
        dense_by_id = _dense_scores(query, all_assets)
        similarities = np.asarray(
            [dense_by_id[str(asset["id"])] for asset in assets], dtype=np.float32
        )
        normalized_dense_scores = _normalise_scores(similarities)

    weight_total = dense_weight + sparse_weight
    normalized_dense_weight = dense_weight / weight_total
    normalized_sparse_weight = sparse_weight / weight_total
    combined_scores = (
        normalized_dense_scores * normalized_dense_weight
        + normalized_sparse_scores * normalized_sparse_weight
    )
    results = [
        _format_result(
            asset,
            combined_scores[position],
            dense_score=normalized_dense_scores[position],
            sparse_score=normalized_sparse_scores[position],
            similarity=similarities[position] if similarities is not None else None,
        )
        for position, asset in enumerate(assets)
    ]
    results.sort(key=lambda result: result["score"], reverse=True)
    return results[:top_k]
