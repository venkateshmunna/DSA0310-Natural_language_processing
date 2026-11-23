import re

import requests

OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "gemma2:2b"
REQUEST_TIMEOUT_SECONDS = 120
MINIMUM_EVIDENCE_LENGTH = 20


def list_models(timeout_seconds=5):
    try:
        response = requests.get(
            f"{OLLAMA_BASE_URL}/api/tags",
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as error:
        raise RuntimeError(
            "Cannot reach Ollama. Start Ollama and confirm it is available at "
            f"{OLLAMA_BASE_URL}."
        ) from error

    names = []
    for model in payload.get("models", []):
        name = model.get("name")
        if name:
            names.append(name)
    if not names:
        raise RuntimeError(
            "Ollama is running but has no downloaded models. "
            "Use 'ollama pull llama3.2:3b' and try again."
        )
    return names


def has_sufficient_evidence(results):
    if not results:
        return False

    useful_evidence = []
    for result in results:
        parts = [
            str(result.get("caption") or "").strip(),
            str(result.get("ocr_text") or "").strip(),
        ]
        evidence_text = "\n".join(part for part in parts if part)
        if len(evidence_text) >= MINIMUM_EVIDENCE_LENGTH:
            useful_evidence.append(evidence_text)
    return bool(useful_evidence)


def _source_label(index, result):
    source = str(result.get("source") or result.get("filename") or "Unknown source")
    timestamp = result.get("timestamp_seconds")
    if timestamp is not None:
        total_seconds = int(timestamp)
        minutes, seconds = divmod(total_seconds, 60)
        source = f"{source} (timestamp {minutes:02d}:{seconds:02d})"
    return f"[{index}] {source}"


def build_prompt(question, results):
    evidence_lines = []
    source_lines = []
    for index, result in enumerate(results, start=1):
        caption = str(result.get("caption") or "").strip()
        ocr_text = str(result.get("ocr_text") or "").strip()
        evidence = "\n".join(
            line
            for line in (
                f"Description: {caption}" if caption else "",
                f"Visible text: {ocr_text}" if ocr_text else "",
            )
            if line
        )
        evidence_lines.append(
            f"SOURCE [{index}]\n"
            f"Filename: {result.get('filename', 'Unknown')}\n"
            f"Timestamp: "
            f"{result.get('timestamp_seconds') if result.get('timestamp_seconds') is not None else 'Not applicable'}\n"
            f"{evidence or 'No textual evidence extracted.'}"
        )
        source_lines.append(_source_label(index, result))

    prompt = (
        "You are trustRAG, an evidence-grounded assistant for uploaded images "
        "and video frames.\n\n"
        "Answer the user's question using only the evidence below. Do not use "
        "outside knowledge or invent facts. Cite evidence using [1], [2], and "
        "so on. If the evidence is insufficient or unrelated, respond exactly "
        "with: 'Insufficient evidence: the uploaded media does not provide enough "
        "information to answer this question.'\n"
        "Keep the answer concise. Include the relevant filename and video "
        "timestamp for every factual claim when a timestamp is available.\n\n"
        f"QUESTION:\n{question.strip()}\n\n"
        "RETRIEVED EVIDENCE:\n"
        f"{chr(10).join(evidence_lines)}\n\n"
        "SOURCE LABELS:\n"
        f"{chr(10).join(source_lines)}"
    )
    return prompt


def generate_answer(question, results, model=DEFAULT_MODEL):
    question = question.strip()
    if not question:
        raise ValueError("Enter a question before generating an answer.")
    if not has_sufficient_evidence(results):
        return (
            "Insufficient evidence: the uploaded media does not provide enough "
            "information to answer this question.",
            [],
        )

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Follow the evidence-grounding rules exactly. Never invent facts "
                    "or claim knowledge outside the provided evidence."
                ),
            },
            {"role": "user", "content": build_prompt(question, results)},
        ],
        "stream": False,
        "options": {"temperature": 0.1},
    }

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        response_payload = response.json()
    except requests.Timeout as error:
        raise RuntimeError(
            f"Ollama timed out after {REQUEST_TIMEOUT_SECONDS} seconds. "
            "Try a smaller model or a shorter question."
        ) from error
    except (requests.RequestException, ValueError) as error:
        raise RuntimeError(
            "Could not generate an answer with Ollama. Confirm the Ollama server "
            "is running and the selected model is installed."
        ) from error

    message = response_payload.get("message", {})
    answer = str(message.get("content") or "").strip()
    if not answer:
        raise RuntimeError("Ollama returned an empty answer.")

    cited_numbers = {
        int(number) for number in re.findall(r"\[(\d+)\]", answer)
    }
    cited_sources = [
        result
        for index, result in enumerate(results, start=1)
        if index in cited_numbers
    ]
    return answer, cited_sources
