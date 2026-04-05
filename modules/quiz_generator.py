from __future__ import annotations

import hashlib
import json
import random
import re
from typing import Any

from database.database_manager import get_cached_questions, get_user_performance_summary, store_quiz_questions
from modules.llama_model import generate_llm_response, llm_is_available
from modules.utils import chunk_text, strip_page_markers
from modules.summarizer import extract_topics


QUESTION_TYPES = ["multiple_choice", "true_false", "fill_blank", "short_answer"]
STOP_WORDS = {"a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "in", "is", "it", "of", "on", "or", "that", "the", "to", "was", "were", "with"}


def _split_sentences(content: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", strip_page_markers(content).strip())
    return [part.strip() for part in parts if len(part.split()) >= 8]


def _sanitize_text(value: str) -> str:
    cleaned = strip_page_markers(value)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _important_words(sentence: str) -> list[str]:
    cleaned_words = [re.sub(r"[^A-Za-z0-9-]", "", word).strip() for word in sentence.split()]
    return [word for word in cleaned_words if len(word) > 4 and word.lower() not in STOP_WORDS]


def _sentence_clauses(sentence: str) -> list[str]:
    clauses = [part.strip(" ,;:-") for part in re.split(r",|;|:|\band\b", sentence) if part.strip()]
    return [clause for clause in clauses if len(clause.split()) >= 3]


def _concise_fact(sentence: str) -> str:
    clauses = _sentence_clauses(sentence)
    if clauses:
        return clauses[0].rstrip(".") + "."
    return sentence.strip().rstrip(".") + "."


def _infer_difficulty(user_id: int | str | None, config=None) -> str:
    if not user_id:
        return "medium"
    return get_user_performance_summary(user_id, config)["recommended_difficulty"]


def _sample_sentences(sentences: list[str], count: int) -> list[str]:
    if len(sentences) <= count:
        return sentences
    step = max(1, len(sentences) // count)
    return [sentences[index] for index in range(0, len(sentences), step)][:count]


def _topic_focus_sentences(content: str, topics: list[str], count: int) -> list[str]:
    all_sentences = _split_sentences(content)
    if not topics:
        return _sample_sentences(all_sentences, count)
    selected: list[str] = []
    for topic in topics:
        topic_words = {word.lower() for word in topic.split() if word}
        for sentence in all_sentences:
            sentence_words = {word.lower() for word in _important_words(sentence)}
            if topic.lower() in sentence.lower() or topic_words & sentence_words:
                if sentence not in selected:
                    selected.append(sentence)
            if len(selected) >= count:
                return selected[:count]
    if len(selected) < count:
        for sentence in _sample_sentences(all_sentences, count):
            if sentence not in selected:
                selected.append(sentence)
    return selected[:count]


def _question_seed(content: str) -> int:
    return int(hashlib.sha256(content.encode("utf-8")).hexdigest(), 16)


def _fallback_question(sentence: str, question_type: str, rng: random.Random, keyword_pool: list[str], difficulty: str) -> dict[str, Any] | None:
    sentence = _sanitize_text(sentence)
    keywords = _important_words(sentence)
    if not keywords:
        return None
    answer_word = max(keywords, key=len)
    distractors = [word for word in keyword_pool if word.lower() != answer_word.lower()]
    rng.shuffle(distractors)
    concise_fact = _concise_fact(sentence)

    if question_type == "true_false":
        statement = sentence
        is_true = rng.choice([True, False])
        if not is_true and distractors:
            statement = re.sub(rf"\b{re.escape(answer_word)}\b", distractors[0], sentence, count=1)
        return {
            "type": "true_false",
            "question": _sanitize_text(statement),
            "options": ["True", "False"],
            "answer": "True" if is_true else "False",
            "difficulty": difficulty,
            "skill_level": "intermediate" if difficulty == "medium" else difficulty,
            "explanation": _sanitize_text(sentence),
            "quality_score": 0.72,
        }

    if question_type == "fill_blank":
        return {
            "type": "fill_blank",
            "question": _sanitize_text(re.sub(rf"\b{re.escape(answer_word)}\b", "_____", sentence, count=1)),
            "options": [],
            "answer": _sanitize_text(answer_word),
            "difficulty": difficulty,
            "skill_level": "intermediate" if difficulty == "medium" else difficulty,
            "explanation": _sanitize_text(sentence),
            "quality_score": 0.74,
        }

    if question_type == "short_answer":
        return {
            "type": "short_answer",
            "question": _sanitize_text(f"Explain how '{answer_word}' is used in the document."),
            "options": [],
            "answer": _sanitize_text(concise_fact),
            "difficulty": difficulty,
            "skill_level": "advanced" if difficulty == "hard" else "intermediate",
            "explanation": _sanitize_text(concise_fact),
            "quality_score": 0.7,
        }

    correct_option = concise_fact
    option_pool = [correct_option]
    for distractor in distractors[:3]:
        option_pool.append(f"It is mainly related to {distractor}.")
    while len(option_pool) < 4:
        option_pool.append(f"It is mainly related to Concept {len(option_pool)}.")
    options = option_pool[:4]
    while len(options) < 4:
        options.append(f"Concept{len(options) + 1}")
    rng.shuffle(options)
    answer_index = options.index(correct_option)
    return {
        "type": "multiple_choice",
        "question": _sanitize_text(f"What does the document state about {answer_word}?"),
        "options": [_sanitize_text(option) for option in options],
        "answer": _sanitize_text(options[answer_index]),
        "difficulty": difficulty,
        "skill_level": "intermediate" if difficulty == "medium" else difficulty,
        "explanation": _sanitize_text(concise_fact),
        "quality_score": 0.78,
    }


def _fallback_quiz(content: str, count: int, difficulty: str) -> list[dict[str, Any]]:
    topics = extract_topics(content, limit=max(4, count))
    sentences = _topic_focus_sentences(content, topics, max(count * 3, 8))
    if not sentences:
        return []
    rng = random.Random(_question_seed(content))
    keyword_pool: list[str] = []
    for sentence in sentences:
        keyword_pool.extend(_important_words(sentence))
    keyword_pool = list(dict.fromkeys(keyword_pool))
    output: list[dict[str, Any]] = []
    used_questions: set[str] = set()
    for index in range(count):
        sentence = sentences[index % len(sentences)]
        question_type = QUESTION_TYPES[index % len(QUESTION_TYPES)]
        question = _fallback_question(sentence, question_type, rng, keyword_pool, difficulty)
        if question is not None:
            if question["question"] in used_questions:
                alt_sentence = sentences[(index + 1) % len(sentences)]
                question = _fallback_question(alt_sentence, question_type, rng, keyword_pool, difficulty)
                if question is None or question["question"] in used_questions:
                    continue
            used_questions.add(question["question"])
            question["topic"] = topics[index % len(topics)] if topics else "general_document"
            output.append(question)
    return output


def _strip_fences(raw_text: str) -> str:
    raw_text = raw_text.strip()
    if raw_text.startswith("```"):
        raw_text = re.sub(r"^```(?:json)?", "", raw_text).strip()
        raw_text = re.sub(r"```$", "", raw_text).strip()
    return raw_text


def _validate_question(item: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    question_type = item.get("type", "multiple_choice")
    question = _sanitize_text(str(item.get("question", "")).strip())
    answer = _sanitize_text(str(item.get("answer", "")).strip())
    if not question or not answer:
        return None
    options = [_sanitize_text(str(option)) for option in (item.get("options", []) or [])]
    if question_type in {"multiple_choice", "true_false"}:
        if question_type == "true_false":
            options = ["True", "False"]
            if answer not in {"True", "False"}:
                return None
        elif len(options) != 4:
            return None
    else:
        options = []
    return {
        "type": question_type,
        "question": question,
        "options": options,
        "answer": answer,
        "difficulty": item.get("difficulty", "medium"),
        "skill_level": item.get("skill_level", "intermediate"),
        "explanation": _sanitize_text(str(item.get("explanation", "")).strip()),
        "quality_score": float(item.get("quality_score", 0.75)),
    }


def _llm_quiz(content: str, count: int, difficulty: str) -> list[dict[str, Any]]:
    cleaned_content = strip_page_markers(content)
    topics = extract_topics(cleaned_content, limit=max(4, count))
    topic_text = ", ".join(topics[: min(len(topics), count)]) or "the most important concepts"
    prompt = f"""
You are an AI instructor creating a quiz from study material.
Return valid JSON only.
Generate exactly {count} questions.
Mix these types when possible: multiple_choice, true_false, fill_blank, short_answer.
Difficulty should be {difficulty}.
Focus on these document topics: {topic_text}.
Questions should test understanding, not surface memorization.
For multiple_choice, options must contain exactly 4 choices and answer must be the full correct option text.
For true_false, options must be ["True", "False"].
Every question must include a short explanation describing why the answer is correct.
Avoid repeating the same concept.

JSON format:
[
  {{
    "type": "multiple_choice",
    "question": "...",
    "options": ["...", "...", "...", "..."],
    "answer": "...",
    "difficulty": "medium",
    "skill_level": "intermediate",
    "explanation": "...",
    "quality_score": 0.84
  }}
]

CONTENT:
{cleaned_content}
"""
    raw = generate_llm_response(prompt, max_tokens=2200, temperature=0.45)
    cleaned = _strip_fences(raw)
    parsed = json.loads(cleaned)
    if not isinstance(parsed, list):
        raise ValueError("Quiz output is not a JSON list.")
    validated = []
    for item in parsed:
        question = _validate_question(item)
        if question is not None:
            validated.append(question)
    if len(validated) < max(2, count // 2):
        raise ValueError("Quiz JSON did not contain enough valid questions.")
    return validated[:count]


def generate_quiz_package(
    content: str,
    count: int = 5,
    *,
    user_id: int | str | None = None,
    topic: str = "general_document",
    document_id: int | None = None,
    difficulty_override: str | None = None,
    config=None,
) -> dict[str, Any]:
    content = strip_page_markers(content)
    difficulty = difficulty_override or _infer_difficulty(user_id, config)
    extracted_topics = extract_topics(content, limit=max(4, count))
    cache_topic = topic if topic != "general_document" else (extracted_topics[0] if extracted_topics else topic)
    cached = get_cached_questions(document_id, cache_topic, difficulty, count, config)
    if len(cached) >= count:
        return {"questions": cached[:count], "difficulty": difficulty, "cached": True, "topics": extracted_topics}

    questions: list[dict[str, Any]] = []
    if llm_is_available():
        retry_prompts = 2
        for _ in range(retry_prompts):
            try:
                questions = _llm_quiz(content, count, difficulty)
                break
            except Exception:
                questions = []
    if not questions:
        questions = _fallback_quiz(content, count, difficulty)

    question_ids = store_quiz_questions(document_id, cache_topic, questions, config)
    for question, question_id in zip(questions, question_ids, strict=False):
        question["question_id"] = question_id
    return {"questions": questions, "difficulty": difficulty, "cached": False, "topics": extracted_topics}


def generate_quiz_questions(content: str, count: int = 5) -> list[dict[str, Any]]:
    return generate_quiz_package(content, count)["questions"]
