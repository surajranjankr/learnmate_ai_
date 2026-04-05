from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
import re
import secrets
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from learnmate_ai.config import AppConfig, get_config


EMAIL_PATTERN = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")


TABLE_STATEMENTS = {
    "users": """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """,
    "documents": """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            file_hash TEXT NOT NULL UNIQUE,
            file_type TEXT NOT NULL,
            topic TEXT NOT NULL,
            language TEXT,
            text_content TEXT NOT NULL,
            usage_count INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """,
    "summaries": """
        CREATE TABLE IF NOT EXISTS summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            document_id INTEGER NOT NULL,
            summary_hash TEXT NOT NULL UNIQUE,
            method TEXT NOT NULL,
            mode TEXT NOT NULL,
            target_language TEXT NOT NULL,
            summary_text TEXT NOT NULL,
            key_insights TEXT,
            hierarchy_json TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (document_id) REFERENCES documents(id)
        )
    """,
    "study_sessions": """
        CREATE TABLE IF NOT EXISTS study_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            document_id INTEGER,
            subject TEXT NOT NULL,
            topic TEXT NOT NULL,
            time_spent INTEGER NOT NULL,
            engagement_score REAL DEFAULT 0,
            completion_percentage REAL DEFAULT 0,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (document_id) REFERENCES documents(id)
        )
    """,
    "quiz_results": """
        CREATE TABLE IF NOT EXISTS quiz_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            document_id INTEGER,
            subject TEXT NOT NULL,
            topic TEXT NOT NULL,
            score REAL NOT NULL,
            total_questions INTEGER NOT NULL,
            score_percent REAL DEFAULT 0,
            difficulty_level TEXT,
            question_types TEXT,
            question_set_json TEXT,
            summary_cache TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (document_id) REFERENCES documents(id)
        )
    """,
    "quiz_questions": """
        CREATE TABLE IF NOT EXISTS quiz_questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER,
            topic TEXT NOT NULL,
            question_type TEXT NOT NULL,
            difficulty_level TEXT,
            skill_level TEXT,
            question_text TEXT NOT NULL,
            options_json TEXT,
            answer_text TEXT NOT NULL,
            explanation TEXT,
            quality_score REAL DEFAULT 0,
            struggle_count INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            last_used_at TEXT,
            FOREIGN KEY (document_id) REFERENCES documents(id)
        )
    """,
    "chat_sessions": """
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            document_id INTEGER,
            title TEXT NOT NULL,
            topic TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (document_id) REFERENCES documents(id)
        )
    """,
    "chat_messages": """
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            message_text TEXT NOT NULL,
            confidence_score REAL,
            retrieval_metadata TEXT,
            feedback_rating INTEGER,
            created_at TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES chat_sessions(id),
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """,
    "events": """
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            activity_type TEXT NOT NULL,
            resource_id TEXT,
            metadata_json TEXT NOT NULL,
            duration_seconds INTEGER DEFAULT 0,
            engagement_score REAL DEFAULT 0,
            topics_json TEXT,
            skill_level TEXT,
            completion_percentage REAL DEFAULT 0,
            session_id TEXT,
            event_type TEXT NOT NULL,
            event_data TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """,
}


EVENT_COLUMN_MIGRATIONS = {
    "activity_type": "TEXT",
    "resource_id": "TEXT",
    "metadata_json": "TEXT",
    "duration_seconds": "INTEGER DEFAULT 0",
    "engagement_score": "REAL DEFAULT 0",
    "topics_json": "TEXT",
    "skill_level": "TEXT",
    "completion_percentage": "REAL DEFAULT 0",
    "session_id": "TEXT",
    "event_type": "TEXT",
    "event_data": "TEXT",
}


QUIZ_RESULT_MIGRATIONS = {
    "document_id": "INTEGER",
    "score_percent": "REAL DEFAULT 0",
    "difficulty_level": "TEXT",
    "question_types": "TEXT",
    "question_set_json": "TEXT",
    "summary_cache": "TEXT",
}


STUDY_SESSION_MIGRATIONS = {
    "document_id": "INTEGER",
    "engagement_score": "REAL DEFAULT 0",
    "completion_percentage": "REAL DEFAULT 0",
}


def _db_path(config: AppConfig | None = None) -> Path:
    app_config = config or get_config()
    db_path = Path(app_config.sqlite_db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


def _connect(config: AppConfig | None = None) -> sqlite3.Connection:
    connection = sqlite3.connect(_db_path(config), detect_types=sqlite3.PARSE_DECLTYPES)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _table_columns(connection: sqlite3.Connection, table_name: str) -> set[str]:
    rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {row[1] for row in rows}


def _ensure_columns(connection: sqlite3.Connection, table_name: str, columns: dict[str, str]) -> None:
    existing = _table_columns(connection, table_name)
    for column, definition in columns.items():
        if column not in existing:
            connection.execute(f"ALTER TABLE {table_name} ADD COLUMN {column} {definition}")


def _hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    derived_key = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=2**14,
        r=8,
        p=1,
        dklen=32,
    )
    return f"{salt.hex()}${derived_key.hex()}"


def _verify_password(password: str, stored_hash: str) -> bool:
    salt_hex, hash_hex = stored_hash.split("$", 1)
    derived_key = hashlib.scrypt(
        password.encode("utf-8"),
        salt=bytes.fromhex(salt_hex),
        n=2**14,
        r=8,
        p=1,
        dklen=32,
    )
    return secrets.compare_digest(derived_key.hex(), hash_hex)


def initialize_database_schema(config: AppConfig | None = None) -> None:
    with _connect(config) as connection:
        for statement in TABLE_STATEMENTS.values():
            connection.execute(statement)
        _ensure_columns(connection, "events", EVENT_COLUMN_MIGRATIONS)
        _ensure_columns(connection, "quiz_results", QUIZ_RESULT_MIGRATIONS)
        _ensure_columns(connection, "study_sessions", STUDY_SESSION_MIGRATIONS)
        connection.commit()


def register_user(full_name: str, email: str, password: str, config: AppConfig | None = None) -> dict[str, Any]:
    full_name = full_name.strip()
    email = email.strip().lower()
    password = password.strip()

    if len(full_name) < 2:
        raise ValueError("Full name must be at least 2 characters long.")
    if not EMAIL_PATTERN.match(email):
        raise ValueError("Enter a valid email address.")
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters long.")

    initialize_database_schema(config)
    with _connect(config) as connection:
        existing = connection.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
        if existing is not None:
            raise ValueError("A user with this email already exists.")
        cursor = connection.execute(
            "INSERT INTO users (full_name, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
            (full_name, email, _hash_password(password), _now()),
        )
        connection.commit()
        user_id = int(cursor.lastrowid)
    log_event(user_id, "user_registered", {"email": email}, config=config, activity_type="account")
    return {"user_id": user_id, "full_name": full_name, "email": email, "stored": True}


def authenticate_user(email: str, password: str, config: AppConfig | None = None) -> dict[str, Any]:
    email = email.strip().lower()
    password = password.strip()
    if not EMAIL_PATTERN.match(email):
        raise ValueError("Enter a valid email address.")
    if not password:
        raise ValueError("Password is required.")

    initialize_database_schema(config)
    with _connect(config) as connection:
        row = connection.execute(
            "SELECT id, full_name, email, password_hash FROM users WHERE email = ?",
            (email,),
        ).fetchone()
        if row is None or not _verify_password(password, row["password_hash"]):
            raise ValueError("Invalid email or password.")
    log_event(int(row["id"]), "user_signed_in", {"email": row["email"]}, config=config, activity_type="account")
    return {"user_id": int(row["id"]), "full_name": row["full_name"], "email": row["email"], "signed_in": True}


def get_user(user_id: int | str, config: AppConfig | None = None) -> dict[str, Any] | None:
    initialize_database_schema(config)
    with _connect(config) as connection:
        row = connection.execute(
            "SELECT id, full_name, email, created_at FROM users WHERE id = ?",
            (int(user_id),),
        ).fetchone()
    return dict(row) if row else None


def list_registered_users(config: AppConfig | None = None) -> list[dict[str, Any]]:
    initialize_database_schema(config)
    with _connect(config) as connection:
        rows = connection.execute("SELECT id, full_name, email, created_at FROM users ORDER BY created_at DESC").fetchall()
    return [dict(row) for row in rows]


def get_or_create_document(
    user_id: int | str,
    filename: str,
    file_type: str,
    topic: str,
    text_content: str,
    language: str,
    config: AppConfig | None = None,
) -> dict[str, Any]:
    initialize_database_schema(config)
    file_hash = hashlib.sha256(text_content.encode("utf-8")).hexdigest()
    now = _now()
    with _connect(config) as connection:
        existing = connection.execute("SELECT * FROM documents WHERE file_hash = ?", (file_hash,)).fetchone()
        if existing is not None:
            connection.execute(
                "UPDATE documents SET usage_count = usage_count + 1, updated_at = ?, topic = ?, language = ? WHERE id = ?",
                (now, topic.strip(), language.strip(), int(existing["id"])),
            )
            connection.commit()
            row = connection.execute("SELECT * FROM documents WHERE id = ?", (int(existing["id"]),)).fetchone()
            return dict(row)
        cursor = connection.execute(
            """
            INSERT INTO documents (user_id, filename, file_hash, file_type, topic, language, text_content, usage_count, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (int(user_id), filename.strip(), file_hash, file_type.strip(), topic.strip(), language.strip(), text_content, 1, now, now),
        )
        connection.commit()
        row = connection.execute("SELECT * FROM documents WHERE id = ?", (int(cursor.lastrowid),)).fetchone()
        return dict(row)


def get_document(document_id: int | str, config: AppConfig | None = None) -> dict[str, Any] | None:
    initialize_database_schema(config)
    with _connect(config) as connection:
        row = connection.execute("SELECT * FROM documents WHERE id = ?", (int(document_id),)).fetchone()
    return dict(row) if row else None


def get_cached_summary(
    user_id: int | str,
    document_id: int | str,
    method: str,
    mode: str,
    target_language: str,
    config: AppConfig | None = None,
) -> dict[str, Any] | None:
    initialize_database_schema(config)
    summary_hash = hashlib.sha256(f"{user_id}|{document_id}|{method}|{mode}|{target_language}".encode("utf-8")).hexdigest()
    with _connect(config) as connection:
        row = connection.execute("SELECT * FROM summaries WHERE summary_hash = ?", (summary_hash,)).fetchone()
    return dict(row) if row else None


def store_summary(
    user_id: int | str,
    document_id: int | str,
    method: str,
    mode: str,
    target_language: str,
    summary_text: str,
    key_insights: list[dict[str, Any]],
    hierarchy: dict[str, Any],
    config: AppConfig | None = None,
) -> dict[str, Any]:
    initialize_database_schema(config)
    summary_hash = hashlib.sha256(f"{user_id}|{document_id}|{method}|{mode}|{target_language}".encode("utf-8")).hexdigest()
    with _connect(config) as connection:
        existing = connection.execute("SELECT id FROM summaries WHERE summary_hash = ?", (summary_hash,)).fetchone()
        if existing is not None:
            connection.execute(
                "UPDATE summaries SET summary_text = ?, key_insights = ?, hierarchy_json = ? WHERE id = ?",
                (summary_text, json.dumps(key_insights, ensure_ascii=False), json.dumps(hierarchy, ensure_ascii=False), int(existing["id"])),
            )
            connection.execute(
                "UPDATE quiz_results SET summary_cache = ? WHERE user_id = ? AND document_id = ?",
                (summary_text, int(user_id), int(document_id)),
            )
            connection.commit()
            row = connection.execute("SELECT * FROM summaries WHERE id = ?", (int(existing["id"]),)).fetchone()
            return dict(row)
        cursor = connection.execute(
            """
            INSERT INTO summaries (user_id, document_id, summary_hash, method, mode, target_language, summary_text, key_insights, hierarchy_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(user_id),
                int(document_id),
                summary_hash,
                method,
                mode,
                target_language,
                summary_text,
                json.dumps(key_insights, ensure_ascii=False),
                json.dumps(hierarchy, ensure_ascii=False),
                _now(),
            ),
        )
        connection.commit()
        row = connection.execute("SELECT * FROM summaries WHERE id = ?", (int(cursor.lastrowid),)).fetchone()
        return dict(row)


def store_quiz_questions(
    document_id: int | None,
    topic: str,
    questions: list[dict[str, Any]],
    config: AppConfig | None = None,
) -> list[int]:
    initialize_database_schema(config)
    inserted_ids: list[int] = []
    with _connect(config) as connection:
        for question in questions:
            cursor = connection.execute(
                """
                INSERT INTO quiz_questions (
                    document_id, topic, question_type, difficulty_level, skill_level,
                    question_text, options_json, answer_text, explanation, quality_score, struggle_count, created_at, last_used_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    None if document_id is None else int(document_id),
                    topic.strip(),
                    question.get("type", "multiple_choice"),
                    question.get("difficulty", "medium"),
                    question.get("skill_level", "intermediate"),
                    question.get("question", "").strip(),
                    json.dumps(question.get("options", []), ensure_ascii=False),
                    str(question.get("answer", "")),
                    question.get("explanation", ""),
                    float(question.get("quality_score", 0.75)),
                    int(question.get("struggle_count", 0)),
                    _now(),
                    _now(),
                ),
            )
            inserted_ids.append(int(cursor.lastrowid))
        connection.commit()
    return inserted_ids


def get_cached_questions(
    document_id: int | None,
    topic: str,
    difficulty_level: str,
    limit: int,
    config: AppConfig | None = None,
) -> list[dict[str, Any]]:
    initialize_database_schema(config)
    with _connect(config) as connection:
        if document_id is None:
            rows = connection.execute(
                "SELECT * FROM quiz_questions WHERE topic = ? AND difficulty_level = ? ORDER BY quality_score DESC, created_at DESC LIMIT ?",
                (topic.strip(), difficulty_level, int(limit)),
            ).fetchall()
        else:
            rows = connection.execute(
                "SELECT * FROM quiz_questions WHERE document_id = ? AND difficulty_level = ? ORDER BY quality_score DESC, created_at DESC LIMIT ?",
                (int(document_id), difficulty_level, int(limit)),
            ).fetchall()
    questions: list[dict[str, Any]] = []
    for row in rows:
        payload = dict(row)
        options = json.loads(payload.get("options_json") or "[]")
        if not isinstance(options, list) or len(options) < 2:
            continue
        payload["options"] = options
        payload["question"] = payload.pop("question_text")
        payload["answer"] = payload.pop("answer_text")
        payload["type"] = payload.pop("question_type")
        questions.append(payload)
    return questions


def update_question_quality(question_id: int | str, struggled: bool, config: AppConfig | None = None) -> None:
    initialize_database_schema(config)
    with _connect(config) as connection:
        if struggled:
            connection.execute(
                "UPDATE quiz_questions SET struggle_count = struggle_count + 1, quality_score = MAX(0, quality_score - 0.05), last_used_at = ? WHERE id = ?",
                (_now(), int(question_id)),
            )
        else:
            connection.execute(
                "UPDATE quiz_questions SET quality_score = MIN(1, quality_score + 0.02), last_used_at = ? WHERE id = ?",
                (_now(), int(question_id)),
            )
        connection.commit()


def log_study_session(
    user_id: int | str,
    subject: str,
    topic: str,
    time_spent: int,
    config: AppConfig | None = None,
    document_id: int | None = None,
    engagement_score: float = 0,
    completion_percentage: float = 0,
) -> int:
    initialize_database_schema(config)
    with _connect(config) as connection:
        cursor = connection.execute(
            """
            INSERT INTO study_sessions (user_id, document_id, subject, topic, time_spent, engagement_score, completion_percentage, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (int(user_id), document_id, subject.strip(), topic.strip(), int(time_spent), float(engagement_score), float(completion_percentage), _now()),
        )
        connection.commit()
        return int(cursor.lastrowid)


def save_quiz_result(
    user_id: int | str,
    subject: str,
    topic: str,
    score: float,
    total_questions: int,
    config: AppConfig | None = None,
    document_id: int | None = None,
    difficulty_level: str | None = None,
    question_types: list[str] | None = None,
    question_set_json: list[dict[str, Any]] | None = None,
) -> int:
    initialize_database_schema(config)
    score_percent = round((float(score) / max(int(total_questions), 1)) * 100, 2)
    with _connect(config) as connection:
        cursor = connection.execute(
            """
            INSERT INTO quiz_results (
                user_id, document_id, subject, topic, score, total_questions, score_percent,
                difficulty_level, question_types, question_set_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(user_id),
                document_id,
                subject.strip(),
                topic.strip(),
                float(score),
                int(total_questions),
                score_percent,
                difficulty_level,
                json.dumps(question_types or []),
                json.dumps(question_set_json or [], ensure_ascii=False),
                _now(),
            ),
        )
        connection.commit()
        return int(cursor.lastrowid)


def get_user_performance_summary(user_id: int | str, config: AppConfig | None = None) -> dict[str, Any]:
    initialize_database_schema(config)
    with _connect(config) as connection:
        row = connection.execute(
            "SELECT AVG(score_percent) AS avg_score, COUNT(*) AS attempts FROM quiz_results WHERE user_id = ?",
            (int(user_id),),
        ).fetchone()
    avg_score = float(row["avg_score"] or 0)
    if row["attempts"] == 0:
        level = "medium"
    elif avg_score >= 85:
        level = "hard"
    elif avg_score >= 60:
        level = "medium"
    else:
        level = "easy"
    return {"avg_score": avg_score, "attempts": int(row["attempts"] or 0), "recommended_difficulty": level}


def create_chat_session(user_id: int | str, title: str, topic: str, config: AppConfig | None = None, document_id: int | None = None) -> int:
    initialize_database_schema(config)
    now = _now()
    with _connect(config) as connection:
        cursor = connection.execute(
            "INSERT INTO chat_sessions (user_id, document_id, title, topic, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            (int(user_id), document_id, title.strip(), topic.strip(), now, now),
        )
        connection.commit()
        return int(cursor.lastrowid)


def touch_chat_session(session_id: int | str, config: AppConfig | None = None) -> None:
    initialize_database_schema(config)
    with _connect(config) as connection:
        connection.execute("UPDATE chat_sessions SET updated_at = ? WHERE id = ?", (_now(), int(session_id)))
        connection.commit()


def list_chat_sessions(user_id: int | str, config: AppConfig | None = None) -> list[dict[str, Any]]:
    initialize_database_schema(config)
    with _connect(config) as connection:
        rows = connection.execute(
            "SELECT * FROM chat_sessions WHERE user_id = ? ORDER BY updated_at DESC",
            (int(user_id),),
        ).fetchall()
    return [dict(row) for row in rows]


def add_chat_message(
    session_id: int | str,
    user_id: int | str,
    role: str,
    message_text: str,
    config: AppConfig | None = None,
    confidence_score: float | None = None,
    retrieval_metadata: dict[str, Any] | None = None,
) -> int:
    initialize_database_schema(config)
    with _connect(config) as connection:
        cursor = connection.execute(
            "INSERT INTO chat_messages (session_id, user_id, role, message_text, confidence_score, retrieval_metadata, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                int(session_id),
                int(user_id),
                role.strip(),
                message_text,
                confidence_score,
                json.dumps(retrieval_metadata or {}, ensure_ascii=False),
                _now(),
            ),
        )
        connection.commit()
    touch_chat_session(session_id, config)
    return int(cursor.lastrowid)


def list_chat_messages(session_id: int | str, config: AppConfig | None = None, limit: int = 50) -> list[dict[str, Any]]:
    initialize_database_schema(config)
    with _connect(config) as connection:
        rows = connection.execute(
            "SELECT * FROM chat_messages WHERE session_id = ? ORDER BY id ASC LIMIT ?",
            (int(session_id), int(limit)),
        ).fetchall()
    messages = [dict(row) for row in rows]
    for message in messages:
        message["retrieval_metadata"] = json.loads(message.get("retrieval_metadata") or "{}")
    return messages


def rate_chat_message(message_id: int | str, rating: int, config: AppConfig | None = None) -> None:
    initialize_database_schema(config)
    with _connect(config) as connection:
        connection.execute("UPDATE chat_messages SET feedback_rating = ? WHERE id = ?", (int(rating), int(message_id)))
        connection.commit()


def log_event(
    user_id: int | str | None,
    event_type: str,
    event_data: dict[str, Any] | str,
    config: AppConfig | None = None,
    *,
    activity_type: str | None = None,
    resource_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    duration_seconds: int = 0,
    engagement_score: float = 0,
    topics: list[str] | None = None,
    skill_level: str | None = None,
    completion_percentage: float = 0,
    session_id: str | None = None,
) -> int:
    initialize_database_schema(config)
    payload = event_data if isinstance(event_data, str) else json.dumps(event_data, ensure_ascii=False)
    metadata_json = json.dumps(metadata or {}, ensure_ascii=False)
    topics_json = json.dumps(topics or [], ensure_ascii=False)
    with _connect(config) as connection:
        cursor = connection.execute(
            """
            INSERT INTO events (
                user_id, activity_type, resource_id, metadata_json, duration_seconds, engagement_score,
                topics_json, skill_level, completion_percentage, session_id, event_type, event_data, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                None if user_id in {None, '', 'guest'} else int(user_id),
                activity_type or event_type,
                resource_id,
                metadata_json,
                int(duration_seconds),
                float(engagement_score),
                topics_json,
                skill_level,
                float(completion_percentage),
                session_id,
                event_type,
                payload,
                _now(),
            ),
        )
        connection.commit()
        return int(cursor.lastrowid)


def log_activity_batch(activities: list[dict[str, Any]], config: AppConfig | None = None) -> None:
    initialize_database_schema(config)
    for activity in activities:
        log_event(config=config, **activity)


def get_users_df(config: AppConfig | None = None) -> pd.DataFrame:
    initialize_database_schema(config)
    with _connect(config) as connection:
        return pd.read_sql_query("SELECT id, full_name, email, created_at FROM users ORDER BY id DESC", connection)


def get_documents_df(config: AppConfig | None = None) -> pd.DataFrame:
    initialize_database_schema(config)
    with _connect(config) as connection:
        return pd.read_sql_query("SELECT id, user_id, filename, file_type, topic, language, usage_count, created_at, updated_at FROM documents ORDER BY id DESC", connection)


def get_study_df(config: AppConfig | None = None) -> pd.DataFrame:
    initialize_database_schema(config)
    with _connect(config) as connection:
        return pd.read_sql_query(
            "SELECT id, user_id, document_id, subject, topic, time_spent, engagement_score, completion_percentage, created_at FROM study_sessions ORDER BY id DESC",
            connection,
        )


def get_quiz_df(config: AppConfig | None = None) -> pd.DataFrame:
    initialize_database_schema(config)
    with _connect(config) as connection:
        return pd.read_sql_query(
            "SELECT id, user_id, document_id, subject, topic, score, total_questions, score_percent, difficulty_level, question_types, created_at FROM quiz_results ORDER BY id DESC",
            connection,
        )


def get_summary_df(config: AppConfig | None = None) -> pd.DataFrame:
    initialize_database_schema(config)
    with _connect(config) as connection:
        return pd.read_sql_query(
            "SELECT id, user_id, document_id, method, mode, target_language, created_at FROM summaries ORDER BY id DESC",
            connection,
        )


def get_question_bank_df(config: AppConfig | None = None) -> pd.DataFrame:
    initialize_database_schema(config)
    with _connect(config) as connection:
        return pd.read_sql_query(
            "SELECT id, document_id, topic, question_type, difficulty_level, skill_level, quality_score, struggle_count, created_at, last_used_at FROM quiz_questions ORDER BY id DESC",
            connection,
        )


def get_events_df(limit: int = 500, config: AppConfig | None = None) -> pd.DataFrame:
    initialize_database_schema(config)
    with _connect(config) as connection:
        return pd.read_sql_query(
            "SELECT id, user_id, activity_type, resource_id, metadata_json, duration_seconds, engagement_score, topics_json, skill_level, completion_percentage, session_id, event_type, event_data, created_at FROM events ORDER BY id DESC LIMIT ?",
            connection,
            params=(int(limit),),
        )


def export_table(table_name: str, config: AppConfig | None = None) -> pd.DataFrame:
    allowed = {"users", "documents", "summaries", "study_sessions", "quiz_results", "quiz_questions", "chat_sessions", "chat_messages", "events"}
    if table_name not in allowed:
        raise ValueError(f"Unsupported export table: {table_name}")
    initialize_database_schema(config)
    with _connect(config) as connection:
        return pd.read_sql_query(f"SELECT * FROM {table_name}", connection)


def database_status(config: AppConfig | None = None) -> dict[str, Any]:
    app_config = config or get_config()
    db_path = _db_path(app_config)
    try:
        initialize_database_schema(app_config)
        with _connect(app_config) as connection:
            connection.execute("SELECT 1")
        return {"connected": True, "database": str(db_path), "exists": db_path.exists(), "database_configured": app_config.database_configured}
    except Exception as exc:
        return {"connected": False, "database": str(db_path), "exists": db_path.exists(), "database_configured": app_config.database_configured, "error": str(exc)}


def persist_pipeline_report(report: dict[str, Any], config: AppConfig | None = None) -> dict[str, Any]:
    run_id = log_event(None, "pipeline_report", report, config=config, activity_type="pipeline", resource_id=report.get("report_name", "pipeline"))
    return {"run_id": run_id, "stored": True}
