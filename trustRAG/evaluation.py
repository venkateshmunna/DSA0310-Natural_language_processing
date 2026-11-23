import csv
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

import answer_generator
import media_processor
import retrieval

EVALUATION_DIR = media_processor.DATA_DIR / "evaluation"
DATASET_PATH = EVALUATION_DIR / "dataset.csv"
RESULTS_PATH = EVALUATION_DIR / "results.json"
DEFAULT_K = 5


def save_dataset(rows):
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = ["query", "relevant_asset_ids", "reference_answer"]
    with DATASET_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "query": str(row.get("query", "")).strip(),
                    "relevant_asset_ids": str(
                        row.get("relevant_asset_ids", "")
                    ).strip(),
                    "reference_answer": str(row.get("reference_answer", "")).strip(),
                }
            )


def load_dataset():
    if not DATASET_PATH.is_file():
        return []
    frame = pd.read_csv(DATASET_PATH)
    required = {"query", "relevant_asset_ids"}
    if not required.issubset(frame.columns):
        raise ValueError(
            "Evaluation CSV must contain 'query' and 'relevant_asset_ids' columns."
        )
    if "reference_answer" not in frame.columns:
        frame["reference_answer"] = ""
    frame = frame.fillna("")
    return frame.to_dict("records")


def save_results(results):
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)


def load_results():
    if not RESULTS_PATH.is_file():
        return None
    with RESULTS_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def _relevant_ids(value):
    if isinstance(value, (list, tuple, set)):
        raw_ids = value
    else:
        raw_ids = re.split(r"[|;,]", str(value))
    return {str(asset_id).strip() for asset_id in raw_ids if str(asset_id).strip()}


def precision_at_k(ranked_ids, relevant_ids, k):
    top_ids = ranked_ids[:k]
    if not top_ids:
        return 0.0
    return len(set(top_ids) & relevant_ids) / len(top_ids)


def recall_at_k(ranked_ids, relevant_ids, k):
    if not relevant_ids:
        return 0.0
    top_ids = ranked_ids[:k]
    return len(set(top_ids) & relevant_ids) / len(relevant_ids)


def reciprocal_rank(ranked_ids, relevant_ids):
    for position, asset_id in enumerate(ranked_ids, start=1):
        if asset_id in relevant_ids:
            return 1.0 / position
    return 0.0


def _tokens(text):
    return set(re.findall(r"[a-z0-9]+", str(text).lower()))


def faithfulness_from_text(answer, evidence_results):
    answer_tokens = _tokens(answer)
    evidence_tokens = set()
    for result in evidence_results:
        evidence_tokens |= _tokens(result.get("caption", ""))
        evidence_tokens |= _tokens(result.get("ocr_text", ""))
    if not answer_tokens:
        return 0.0
    if not evidence_tokens:
        return 0.0
    return len(answer_tokens & evidence_tokens) / len(answer_tokens)


def _ollama_faithfulness(answer, evidence_results, model):
    evidence = []
    for index, result in enumerate(evidence_results, start=1):
        evidence.append(
            f"[{index}] {result.get('source', result.get('filename', 'Unknown'))}\n"
            f"Description: {result.get('caption', '')}\n"
            f"Visible text: {result.get('ocr_text', '')}"
        )
    prompt = (
        "Determine whether every factual claim in the ANSWER is supported by the "
        "EVIDENCE. Reply with exactly one label: FAITHFUL or NOT_FAITHFUL.\n"
        "If the answer contains unsupported facts, speculation, or outside knowledge "
        "not stated in the evidence, reply NOT_FAITHFUL.\n\n"
        f"EVIDENCE:\n{chr(10).join(evidence)}\n\nANSWER:\n{answer}"
    )
    response = answer_generator.requests.post(
        f"{answer_generator.OLLAMA_BASE_URL}/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a strict factuality classifier."},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": 0},
        },
        timeout=answer_generator.REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    label = str(response.json().get("message", {}).get("content", "")).strip()
    if "NOT_FAITHFUL" in label.upper():
        return 0.0
    if "FAITHFUL" in label.upper():
        return 1.0
    raise RuntimeError("Ollama returned an invalid faithfulness label.")


def evaluate(
    k=DEFAULT_K,
    dense_weight=0.5,
    sparse_weight=0.5,
    model=None,
    faithfulness_method="Text overlap",
):
    rows = load_dataset()
    if not rows:
        raise ValueError("Upload or create an evaluation dataset first.")
    if k < 1:
        raise ValueError("K must be at least one.")

    methods = {
        "BM25": {"dense_weight": 0.0, "sparse_weight": 1.0},
        "Dense": {"dense_weight": 1.0, "sparse_weight": 0.0},
        "Hybrid": {
            "dense_weight": dense_weight,
            "sparse_weight": sparse_weight,
        },
    }
    per_method = {}
    detail_rows = []

    for method_name, weights in methods.items():
        precision_scores = []
        recall_scores = []
        reciprocal_scores = []
        faithfulness_scores = []

        for row in rows:
            query = str(row["query"]).strip()
            relevant_ids = _relevant_ids(row["relevant_asset_ids"])
            if not query:
                continue
            results = retrieval.search(
                query,
                method=method_name,
                top_k=k,
                dense_weight=weights["dense_weight"],
                sparse_weight=weights["sparse_weight"],
            )
            ranked_ids = [str(result["asset_id"]) for result in results]
            precision_scores.append(
                precision_at_k(ranked_ids, relevant_ids, k)
            )
            recall_scores.append(recall_at_k(ranked_ids, relevant_ids, k))
            reciprocal_scores.append(reciprocal_rank(ranked_ids, relevant_ids))

            if faithfulness_method == "Ollama classifier" and model:
                try:
                    answer, _ = answer_generator.generate_answer(
                        query,
                        results,
                        model=model,
                    )
                    if answer.startswith("Insufficient evidence"):
                        faithfulness_scores.append(0.0)
                    else:
                        faithfulness_scores.append(
                            _ollama_faithfulness(answer, results, model)
                        )
                except Exception as error:
                    raise RuntimeError(
                        f"Faithfulness evaluation failed for '{query}': {error}"
                    ) from error
            else:
                reference_answer = str(row.get("reference_answer", "")).strip()
                if reference_answer:
                    faithfulness_scores.append(
                        faithfulness_from_text(reference_answer, results)
                    )
                else:
                    faithfulness_scores.append(0.0)

            detail_rows.append(
                {
                    "method": method_name,
                    "query": query,
                    "relevant_asset_ids": "|".join(sorted(relevant_ids)),
                    "ranked_asset_ids": "|".join(ranked_ids),
                    "precision_at_k": precision_scores[-1],
                    "recall_at_k": recall_scores[-1],
                    "reciprocal_rank": reciprocal_scores[-1],
                    "faithfulness": faithfulness_scores[-1],
                }
            )

        valid_count = len(precision_scores)
        per_method[method_name] = {
            f"Precision@{k}": (
                float(np.mean(precision_scores)) if precision_scores else 0.0
            ),
            f"Recall@{k}": (
                float(np.mean(recall_scores)) if recall_scores else 0.0
            ),
            "MRR": (
                float(np.mean(reciprocal_scores)) if reciprocal_scores else 0.0
            ),
            "Faithfulness": (
                float(np.mean(faithfulness_scores)) if faithfulness_scores else 0.0
            ),
            "Query count": valid_count,
        }

    results = {
        "k": k,
        "dense_weight": dense_weight,
        "sparse_weight": sparse_weight,
        "faithfulness_method": faithfulness_method,
        "methods": per_method,
        "details": detail_rows,
    }
    save_results(results)
    return results
