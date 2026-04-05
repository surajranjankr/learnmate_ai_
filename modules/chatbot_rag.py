from __future__ import annotations

import re
from statistics import mean
from typing import Any

from modules.llama_model import generate_llm_response, llm_is_available
from modules.utils import clean_token, strip_page_markers
from modules.vectorstore import retrieve_relevant_chunks_with_scores


ANSWER_MODE_INSTRUCTIONS = {
    "teacher": "Explain like a helpful teacher with clear concepts and simple reasoning.",
    "short": "Give a short direct answer in 2 to 4 lines.",
    "step_by_step": "Answer step by step with numbered reasoning.",
}

GENERIC_QUERY_TOKENS = {
    "the", "a", "an", "of", "for", "about", "project", "document", "pdf", "file",
    "explain", "tell", "me", "what", "is", "are", "how", "used",
}
NOISE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\broll\s*no\b",
        r"\bteam\s+members?\b",
        r"\bcourse\b",
        r"\bbtech\b",
        r"\bsubmitted\s+by\b",
        r"\bdepartment\b",
        r"\bstudent\b",
        r"\bguide\b",
        r"\bmentor\b",
        r"\bname\s*:",
    ]
]
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
PAGE_PATTERN = re.compile(r"(?:^|\n)\[Page\s+(\d+)\]\s*\n", re.IGNORECASE)
PAGE_QUERY_PATTERN = re.compile(r"\bpage\s+(\d+)\b", re.IGNORECASE)
KEYWORD_EXPANSIONS = {
    "workflow": {"workflow", "process", "pipeline", "steps", "architecture", "flow"},
    "tech": {"tech", "technology", "tools", "framework", "stack", "backend", "frontend", "database", "rfid", "api"},
    "stack": {"stack", "technology", "tools", "framework", "backend", "frontend", "database", "rfid", "api"},
    "evaluation": {"evaluation", "criteria", "assessment", "judging", "marks", "weightage"},
    "criteria": {"criteria", "evaluation", "assessment", "judging", "marks", "weightage"},
}


def _extract_pages(document_text: str) -> list[dict[str, Any]]:
    matches = list(PAGE_PATTERN.finditer(document_text))
    if not matches:
        cleaned = strip_page_markers(document_text)
        return [{"page_number": 1, "text": cleaned}] if cleaned else []

    pages: list[dict[str, Any]] = []
    for index, match in enumerate(matches):
        page_number = int(match.group(1))
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(document_text)
        page_text = document_text[start:end].strip()
        cleaned = strip_page_markers(page_text)
        if cleaned:
            pages.append({"page_number": page_number, "text": cleaned})
    return pages


def _query_terms(query: str) -> set[str]:
    raw_tokens = [clean_token(token) for token in re.findall(r"[A-Za-z0-9-]+", query)]
    expanded = set(raw_tokens)
    for token in list(raw_tokens):
        expanded.update(KEYWORD_EXPANSIONS.get(token, set()))
    return {token for token in expanded if token and token not in GENERIC_QUERY_TOKENS}


def _is_noise_text(text: str) -> bool:
    lowered = text.lower()
    if not lowered.strip():
        return True
    if any(pattern.search(text) for pattern in NOISE_PATTERNS):
        return True
    if text.count("(") >= 2 and any(char.isdigit() for char in text):
        return True
    return False


def _split_units(text: str) -> list[str]:
    cleaned_text = strip_page_markers(text)
    units: list[str] = []
    for line in cleaned_text.splitlines():
        line = line.strip(" -\t")
        if not line:
            continue
        sentence_parts = SENTENCE_SPLIT.split(line)
        for part in sentence_parts:
            candidate = part.strip(" -\t")
            if len(candidate.split()) >= 4 and not _is_noise_text(candidate):
                units.append(candidate)
    return units


def _unit_score(query: str, unit: str) -> float:
    query_terms = _query_terms(query)
    unit_terms = {clean_token(token) for token in re.findall(r"[A-Za-z0-9-]+", unit)}
    if not query_terms or not unit_terms:
        return 0.0

    overlap = len(query_terms & unit_terms)
    if overlap == 0:
        return 0.0

    exact_phrase_bonus = 2.5 if query.strip().lower() in unit.lower() else 0.0
    definition_bonus = 1.5 if re.search(r"\b(?:is|refers to|defined as|means)\b", unit, re.IGNORECASE) else 0.0
    heading_bonus = 1.0 if len(unit.split()) <= 10 and unit.istitle() else 0.0
    return overlap * 2.2 + exact_phrase_bonus + definition_bonus + heading_bonus


def _page_requested(query: str) -> int | None:
    match = PAGE_QUERY_PATTERN.search(query)
    return int(match.group(1)) if match else None


def _best_units_from_pages(query: str, pages: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    requested_page = _page_requested(query)
    page_pool = pages
    if requested_page is not None:
        page_pool = [page for page in pages if page["page_number"] == requested_page]
        if not page_pool:
            return []

    scored_units: list[dict[str, Any]] = []
    for page in page_pool:
        for unit in _split_units(page["text"]):
            score = _unit_score(query, unit)
            if score > 0:
                scored_units.append({"text": unit, "page_number": page["page_number"], "score": score})

    scored_units.sort(key=lambda item: item["score"], reverse=True)
    if scored_units:
        return scored_units[:limit]

    if requested_page is not None:
        fallback_units = _split_units(page_pool[0]["text"])
        return [{"text": unit, "page_number": requested_page, "score": 0.5} for unit in fallback_units[:limit]]
    return []


def _vectorstore_units(query: str, limit: int = 5) -> list[dict[str, Any]]:
    chunks = retrieve_relevant_chunks_with_scores(query, k=4, score_threshold=4.0)
    output: list[dict[str, Any]] = []
    for chunk in chunks:
        for unit in _split_units(str(chunk["text"])):
            score = _unit_score(query, unit)
            if score > 0:
                output.append({"text": unit, "page_number": None, "score": score})
    output.sort(key=lambda item: item["score"], reverse=True)
    return output[:limit]


def _format_answer(units: list[dict[str, Any]], answer_mode: str) -> str:
    unique_lines: list[str] = []
    seen: set[str] = set()
    for item in units:
        text = strip_page_markers(item["text"]).strip()
        if not text:
            continue
        normalized = text.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        if not text.endswith((".", "!", "?")):
            text += "."
        unique_lines.append(text)

    if not unique_lines:
        return "I could not find a clear answer for that question in the uploaded document."

    if answer_mode == "step_by_step":
        return "\n".join(f"{index}. {line}" for index, line in enumerate(unique_lines[:4], start=1))
    if answer_mode == "short":
        return " ".join(unique_lines[:2])
    return " ".join(unique_lines[:3])


def _history_needed(question: str) -> bool:
    lowered = question.lower()
    return any(token in lowered for token in {"this", "that", "it", "them", "these", "those", "continue", "previous"})


def chatbot_respond(
    question: str,
    history: list[dict[str, str]] | None = None,
    *,
    answer_mode: str = "teacher",
    document_text: str | None = None,
) -> dict[str, Any]:
    """Answer from the active uploaded document, using direct keyword search before fallback retrieval."""
    cleaned_question = question.strip()
    if not cleaned_question:
        return {"answer": "Please enter a question about the uploaded document.", "confidence": 0.0, "sources": []}

    history = history or []
    pages = _extract_pages(document_text or "")
    best_units = _best_units_from_pages(cleaned_question, pages, limit=6) if pages else []
    if not best_units:
        best_units = _vectorstore_units(cleaned_question, limit=6)
    if not best_units:
        return {
            "answer": "I could not find a clear answer for that exact question in the uploaded document.",
            "confidence": 0.05,
            "sources": [],
        }

    avg_confidence = round(min(0.98, max(0.25, mean(item["score"] for item in best_units) / 8)), 2)
    lexical_answer = _format_answer(best_units, answer_mode)

    if not llm_is_available():
        return {"answer": lexical_answer, "confidence": avg_confidence, "sources": best_units}

    context = "\n".join(item["text"] for item in best_units[:4])
    history_text = ""
    if _history_needed(cleaned_question):
        recent_history = history[-2:]
        history_text = "\n".join(f"{item['role'].upper()}: {item['content']}" for item in recent_history)

    prompt = f"""
You are a document-grounded tutor.
Use only the provided CONTEXT to answer the QUESTION.
Do not mention page numbers.
Do not mention unrelated sections, member names, roll numbers, or front matter.
If the exact answer is not present, say that clearly instead of guessing.
Answer style instruction: {ANSWER_MODE_INSTRUCTIONS.get(answer_mode, ANSWER_MODE_INSTRUCTIONS['teacher'])}

RECENT CONVERSATION:
{history_text or 'No prior context needed.'}

CONTEXT:
{context}

QUESTION:
{cleaned_question}

Return only a direct answer to the question in 2 to 4 sentences.
"""
    answer = generate_llm_response(prompt, max_tokens=220, temperature=0.15).strip()
    answer = strip_page_markers(answer)
    if not answer:
        answer = lexical_answer
    return {"answer": answer, "confidence": avg_confidence, "sources": best_units}
