from __future__ import annotations

from collections import Counter
import json
import math
import re
from typing import Any

from database.database_manager import get_cached_summary, store_summary
from modules.llama_model import generate_llm_response, llm_is_available
from modules.utils import chunk_text


STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "in", "is", "it",
    "of", "on", "or", "that", "the", "to", "was", "were", "with", "this", "these", "those", "we", "you",
    "their", "there", "into", "than", "then", "such", "also", "have", "been", "being", "can", "will",
}
MODE_ALIASES = {
    "bullet_summary": "bullet_points",
    "revision": "brief",
    "exam_notes": "detailed",
    "concept_explanation": "detailed",
}
MODE_PROMPTS = {
    "brief": "Summarize each relevant page in 1 to 2 clear lines.",
    "detailed": "Summarize each relevant page in 4 to 5 clear lines, focusing only on the important ideas.",
    "bullet_points": "Create crisp bullet points covering the most important parts of the document.",
}
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
STAT_PATTERN = re.compile(r"\b\d+(?:\.\d+)?%?\b")
DEFINITION_PATTERN = re.compile(r"\b(?:is|refers to|defined as|means)\b", re.IGNORECASE)
TOPIC_PATTERN = re.compile(r"\b[A-Z][A-Za-z0-9+-]{2,}(?:\s+[A-Z][A-Za-z0-9+-]{2,})*\b")
PAGE_PATTERN = re.compile(r"(?:^|\n)\[Page\s+(\d+)\]\s*\n", re.IGNORECASE)


def normalize_mode(mode: str) -> str:
    return MODE_ALIASES.get(mode, mode)


def _analysis_ready_content(content: str) -> str:
    return PAGE_PATTERN.sub("\n", content).strip()


def _extract_pages(content: str) -> list[dict[str, Any]]:
    matches = list(PAGE_PATTERN.finditer(content))
    if not matches:
        stripped = content.strip()
        return [{"page_number": 1, "text": stripped}] if stripped else []

    pages: list[dict[str, Any]] = []
    for index, match in enumerate(matches):
        page_number = int(match.group(1))
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
        page_text = content[start:end].strip()
        if page_text:
            pages.append({"page_number": page_number, "text": page_text})
    return pages


def _is_relevant_page(page_text: str) -> bool:
    words = page_text.split()
    sentences = _split_sentences(page_text)
    if len(words) < 35:
        return False
    if len(sentences) < 2 and len(words) < 60:
        return False
    alnum_ratio = sum(char.isalnum() for char in page_text) / max(len(page_text), 1)
    return alnum_ratio >= 0.55


def _select_relevant_pages(content: str) -> list[dict[str, Any]]:
    pages = _extract_pages(content)
    relevant = [page for page in pages if _is_relevant_page(page["text"])]
    if relevant:
        return relevant
    return sorted(pages, key=lambda page: len(page["text"].split()), reverse=True)[: max(1, min(5, len(pages)))]


def _page_lines(page_text: str, count: int) -> list[str]:
    tfidf_lines = extractive_tfidf_summary(page_text, count + 2)
    textrank_lines = extractive_textrank_summary(page_text, count + 2)
    merged: list[str] = []
    for line in tfidf_lines + textrank_lines:
        if line not in merged:
            merged.append(line)
        if len(merged) >= count:
            break
    return merged


def _page_topic_label(page_text: str, fallback_number: int, used_labels: set[str]) -> str:
    topics = extract_topics(page_text, limit=3)
    for topic in topics:
        cleaned = topic.strip()
        if not cleaned:
            continue
        label = cleaned[:1].upper() + cleaned[1:]
        normalized = label.lower()
        if normalized not in used_labels:
            used_labels.add(normalized)
            return label

    lines = [line.strip(" -•\t") for line in page_text.splitlines() if line.strip()]
    for line in lines[:6]:
        words = line.split()
        if 2 <= len(words) <= 8:
            label = re.sub(r"\s+", " ", line).strip(" :.-")
            normalized = label.lower()
            if label and normalized not in used_labels:
                used_labels.add(normalized)
                return label

    fallback_label = f"Topic {fallback_number}"
    used_labels.add(fallback_label.lower())
    return fallback_label


def _pagewise_summary(content: str, detail_level: str) -> tuple[str, dict[str, Any]]:
    relevant_pages = _select_relevant_pages(content)
    line_count = 2 if detail_level == "brief" else 5
    rendered_pages: list[dict[str, Any]] = []
    blocks: list[str] = []
    used_labels: set[str] = set()
    for page in relevant_pages:
        lines = _page_lines(page["text"], line_count)
        if not lines:
            continue
        topic_label = _page_topic_label(page["text"], page["page_number"], used_labels)
        rendered_pages.append({"topic": topic_label, "page": page["page_number"], "summary": lines})
        blocks.append(f"### {topic_label}\n" + "\n".join(f"- {line}" for line in lines))
    return "\n\n".join(blocks), {"page_summaries": rendered_pages, "page_count": len(rendered_pages)}


def detect_language(text: str) -> str:
    sample = text[:3000]
    ascii_ratio = sum(1 for char in sample if ord(char) < 128) / max(len(sample), 1)
    if ascii_ratio > 0.97:
        return "en"
    if any("\u0900" <= char <= "\u097F" for char in sample):
        return "hi"
    return "unknown"


def _split_sentences(content: str) -> list[str]:
    prepared = _analysis_ready_content(content)
    return [sentence.strip() for sentence in SENTENCE_SPLIT.split(prepared) if len(sentence.split()) >= 5]


def _tokenize(sentence: str) -> list[str]:
    return [token.lower() for token in re.findall(r"[A-Za-z0-9-]+", sentence) if token.lower() not in STOP_WORDS]


def extract_topics(content: str, limit: int = 8) -> list[str]:
    sentences = _split_sentences(content)
    candidate_counts: Counter[str] = Counter()
    for sentence in sentences:
        for match in TOPIC_PATTERN.findall(sentence):
            cleaned = match.strip()
            if len(cleaned.split()) <= 5 and cleaned.lower() not in STOP_WORDS:
                candidate_counts[cleaned] += 2
        for token in _tokenize(sentence):
            if len(token) > 5:
                candidate_counts[token.title()] += 1
    ordered_topics = [topic for topic, _ in candidate_counts.most_common(limit * 3)]
    filtered: list[str] = []
    for topic in ordered_topics:
        normalized = topic.strip()
        if len(normalized) < 4:
            continue
        if normalized.lower() in STOP_WORDS:
            continue
        if any(normalized.lower() == existing.lower() for existing in filtered):
            continue
        if len(normalized.split()) == 1 and any(normalized.lower() in existing.lower().split() for existing in ordered_topics if len(existing.split()) > 1):
            continue
        if any(
            normalized.lower() in existing.lower() or existing.lower() in normalized.lower()
            for existing in filtered
        ):
            if len(normalized.split()) <= 1:
                continue
        filtered.append(normalized)
        if len(filtered) >= limit:
            break
    return filtered


def _topic_sentences(content: str, topic: str, limit: int = 4) -> list[str]:
    related = [sentence for sentence in _split_sentences(content) if topic.lower() in sentence.lower()]
    if len(related) >= limit:
        return related[:limit]
    if not related:
        topic_tokens = {token.lower() for token in topic.split() if token}
        for sentence in _split_sentences(content):
            sentence_tokens = set(_tokenize(sentence))
            if topic_tokens & sentence_tokens:
                related.append(sentence)
            if len(related) >= limit:
                break
    return related[:limit]


def extractive_tfidf_summary(content: str, sentence_limit: int = 10) -> list[str]:
    sentences = _split_sentences(content)
    if not sentences:
        return []
    tokenized = [_tokenize(sentence) for sentence in sentences]
    doc_freq = Counter()
    for tokens in tokenized:
        doc_freq.update(set(tokens))
    total_docs = max(len(sentences), 1)
    scored: list[tuple[float, int, str]] = []
    for index, tokens in enumerate(tokenized):
        if not tokens:
            continue
        term_freq = Counter(tokens)
        score = 0.0
        for token, freq in term_freq.items():
            idf = math.log((1 + total_docs) / (1 + doc_freq[token])) + 1
            score += freq * idf
        scored.append((score / len(tokens), index, sentences[index]))
    top = sorted(scored, key=lambda item: item[0], reverse=True)[:sentence_limit]
    return [sentence for _, _, sentence in sorted(top, key=lambda item: item[1])]


def extractive_textrank_summary(content: str, sentence_limit: int = 8) -> list[str]:
    sentences = _split_sentences(content)
    if not sentences:
        return []
    tokenized = [set(_tokenize(sentence)) for sentence in sentences]
    scores: list[tuple[float, int, str]] = []
    for index, tokens in enumerate(tokenized):
        score = 0.0
        for other_index, other_tokens in enumerate(tokenized):
            if index == other_index or not tokens or not other_tokens:
                continue
            overlap = len(tokens & other_tokens)
            denom = math.log(len(tokens) + 1) + math.log(len(other_tokens) + 1)
            score += overlap / denom if denom else 0.0
        scores.append((score, index, sentences[index]))
    top = sorted(scores, key=lambda item: item[0], reverse=True)[:sentence_limit]
    return [sentence for _, _, sentence in sorted(top, key=lambda item: item[1])]


def important_sentences(content: str, limit: int = 5) -> list[str]:
    tfidf_sentences = extractive_tfidf_summary(content, limit)
    textrank_sentences = extractive_textrank_summary(content, limit)
    merged: list[str] = []
    for sentence in tfidf_sentences + textrank_sentences:
        if sentence not in merged:
            merged.append(sentence)
        if len(merged) >= limit:
            break
    return merged


def extract_key_insights(content: str) -> list[dict[str, str]]:
    sentences = _split_sentences(content)
    insights: list[dict[str, str]] = []
    seen = set()
    for sentence in sentences:
        if DEFINITION_PATTERN.search(sentence) and sentence not in seen:
            insights.append({"type": "definition", "text": sentence})
            seen.add(sentence)
        if STAT_PATTERN.search(sentence) and sentence not in seen:
            insights.append({"type": "statistic", "text": sentence})
            seen.add(sentence)
        if len(insights) >= 6:
            break
    for sentence in important_sentences(content, limit=4):
        if sentence not in seen:
            insights.append({"type": "important_line", "text": sentence})
            seen.add(sentence)
    return insights[:10]


def build_hierarchical_summary(content: str, method: str) -> dict[str, Any]:
    chunks = chunk_text(content, length=2200, overlap=250)
    section_level = []
    for index, chunk in enumerate(chunks, start=1):
        lines = extractive_textrank_summary(chunk, 3) if method == "textrank" else extractive_tfidf_summary(chunk, 3)
        section_level.append({"section": index, "summary": lines})
    topics = extract_topics(content)
    topic_sections = []
    for topic in topics[:6]:
        related = _topic_sentences(content, topic, 4)
        if related:
            topic_sections.append({"topic": topic, "summary": related[:3]})
    document_level = important_sentences(content, 10 if method != "textrank" else 8)
    return {
        "section_level": section_level,
        "topic_level": topic_sections,
        "document_level": document_level,
    }


def _compose_prompt(mode: str, content: str) -> str:
    instruction = MODE_PROMPTS.get(mode, MODE_PROMPTS["bullet_summary"])
    return f"{instruction}\nFocus on the full document, not just the beginning.\n\nTEXT:\n{content}\n\nOUTPUT:"


def _abstractive_summary(content: str, mode: str) -> str:
    if not llm_is_available():
        if mode in {"brief", "detailed"}:
            summary_text, _ = _pagewise_summary(content, mode)
            return summary_text
        hierarchy = build_hierarchical_summary(content, "tfidf")
        return "\n".join(f"- {bullet}" for bullet in hierarchy["document_level"][:10])

    chunk_summaries = []
    for chunk in chunk_text(content, length=2200, overlap=300):
        prompt = _compose_prompt(mode, chunk)
        chunk_summaries.append(generate_llm_response(prompt, max_tokens=420, temperature=0.25))
    combined = "\n\n".join(chunk_summaries)
    final_prompt = (
        f"Combine these chunk summaries into a polished {mode} output for the full document. "
        "Keep structure, cover beginning/middle/end, and avoid repetition.\n\n"
        f"{combined}"
    )
    return generate_llm_response(final_prompt, max_tokens=900, temperature=0.25)


def _format_topic_sections(topic_sections: list[dict[str, Any]], mode: str) -> str:
    blocks: list[str] = []
    for item in topic_sections:
        topic = item["topic"]
        lines = list(dict.fromkeys(item["summary"]))
        if mode == "revision":
            blocks.append(f"### {topic}\n- " + "\n- ".join(lines[:2]))
        elif mode == "exam_notes":
            blocks.append(f"### {topic}\n- Key point: " + "\n- Key point: ".join(lines[:3]))
        elif mode == "concept_explanation":
            blocks.append(f"### {topic}\n- " + "\n- ".join(lines[:3]))
        else:
            blocks.append(f"### {topic}\n- " + "\n- ".join(lines[:2]))
    return "\n\n".join(blocks)


def _format_summary_text(hierarchy: dict[str, Any], mode: str, method: str) -> str:
    document_lines = hierarchy.get("document_level", [])
    if mode in {"brief", "detailed"}:
        return ""
    return "\n".join(f"- {line}" for line in document_lines[:10])


def _hybrid_summary_text(content: str, mode: str) -> tuple[str, dict[str, Any]]:
    if mode in {"brief", "detailed"}:
        return _pagewise_summary(content, mode)
    hierarchy = build_hierarchical_summary(content, "tfidf")
    tfidf_lines = extractive_tfidf_summary(content, 10 if mode == "concept_explanation" else 8)
    textrank_lines = extractive_textrank_summary(content, 8)
    merged_lines: list[str] = []
    for line in tfidf_lines + textrank_lines:
        if line not in merged_lines:
            merged_lines.append(line)
    if merged_lines:
        hierarchy["document_level"] = merged_lines[:10]
    return _format_summary_text(hierarchy, mode, "tfidf"), hierarchy


def _translate_if_needed(text: str, source_language: str, target_language: str) -> str:
    if not text or target_language in {"", source_language, "same"}:
        return text
    if not llm_is_available():
        return text
    prompt = f"Translate the following summary from {source_language} to {target_language} while preserving bullets and structure.\n\n{text}"
    return generate_llm_response(prompt, max_tokens=900, temperature=0.2)


def summarize_document(user_id: int | str, document_id: int | str, content: str, *, mode: str = "bullet_summary", method: str = "abstractive", target_language: str = "en", config=None) -> dict[str, Any]:
    mode = normalize_mode(mode)
    if method == "auto":
        method = "abstractive" if llm_is_available() else "hybrid"
    cached = get_cached_summary(user_id, document_id, method, mode, target_language, config)
    if cached is not None:
        hierarchy = json.loads(cached.get("hierarchy_json") or "{}")
        return {
            "summary_text": cached["summary_text"],
            "key_insights": json.loads(cached.get("key_insights") or "[]"),
            "hierarchy": hierarchy,
            "topics": [item["topic"] for item in hierarchy.get("topic_level", [])],
            "important_sentences": important_sentences(content, 5),
            "method": method,
            "mode": mode,
            "language": target_language,
            "cached": True,
        }
    source_language = detect_language(content)
    if method == "tfidf":
        hierarchy = build_hierarchical_summary(content, method)
        summary_text = _format_summary_text(hierarchy, mode, method)
    elif method == "textrank":
        hierarchy = build_hierarchical_summary(content, method)
        summary_text = _format_summary_text(hierarchy, mode, method)
    elif method == "hybrid":
        summary_text, hierarchy = _hybrid_summary_text(content, mode)
    else:
        hierarchy = build_hierarchical_summary(content, "tfidf")
        summary_text = _abstractive_summary(content, mode)
    key_insights = extract_key_insights(content)
    summary_text = _translate_if_needed(summary_text, source_language, target_language)
    record = store_summary(user_id, document_id, method, mode, target_language, summary_text, key_insights, hierarchy, config)
    return {
        "summary_text": summary_text,
        "key_insights": key_insights,
        "hierarchy": hierarchy,
        "topics": [item["topic"] for item in hierarchy.get("topic_level", [])],
        "important_sentences": important_sentences(content, 5),
        "method": method,
        "mode": mode,
        "language": target_language,
        "cached": False,
        "summary_id": record["id"],
    }


def summarize_text(content: str, mode: str = "bullet_summary") -> str:
    mode = normalize_mode(mode)
    if not content.strip():
        return "No text was found to summarize."
    if mode in {"brief", "detailed"}:
        summary_text, _ = _pagewise_summary(content, mode)
        return summary_text
    hierarchy = build_hierarchical_summary(content, "tfidf")
    return _format_summary_text(hierarchy, mode, "tfidf")
