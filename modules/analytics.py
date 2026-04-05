from __future__ import annotations

from collections import Counter
from io import BytesIO
import json
from typing import Any

import numpy as np
import pandas as pd

from modules.llama_model import generate_llm_response, llm_is_available
from modules.utils import clean_token


STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "he", "in", "is", "it", "its", "of", "on", "that", "the", "to", "was",
    "were", "will", "with", "this", "these", "those", "or", "if", "but",
    "about", "into", "than", "then", "them", "they", "their", "you", "your",
    "we", "our", "can", "could", "should", "would", "may", "might", "not",
}


def _analytics_backend():
    from analytics import analytics as analytics_backend

    return analytics_backend


def load_structured_data(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        raise ValueError("No file was uploaded.")
    if not getattr(uploaded_file, "name", ""):
        raise ValueError("Uploaded file is missing a filename.")

    suffix = uploaded_file.name.lower()
    file_bytes = uploaded_file.getvalue()
    if not file_bytes:
        raise ValueError("The uploaded file is empty.")

    if suffix.endswith(".csv"):
        df = pd.read_csv(BytesIO(file_bytes))
    elif suffix.endswith(".json"):
        df = pd.read_json(BytesIO(file_bytes))
    elif suffix.endswith(".xlsx"):
        df = pd.read_excel(BytesIO(file_bytes))
    else:
        raise ValueError("Unsupported file type. Please upload CSV, JSON, or XLSX.")

    if df.empty:
        raise ValueError("The uploaded dataset contains no rows.")
    if df.columns.empty:
        raise ValueError("The uploaded dataset contains no columns.")
    return df


def profile_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    if df is None or df.empty:
        raise ValueError("Cannot profile an empty dataset.")

    numeric_df = df.select_dtypes(include=[np.number])

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_columns": list(numeric_df.columns),
        "categorical_columns": list(df.select_dtypes(exclude=[np.number]).columns),
        "column_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return pd.DataFrame()

    summary = numeric_df.describe().T
    summary["missing"] = numeric_df.isna().sum()
    summary["median"] = numeric_df.median()
    return summary[["count", "mean", "std", "min", "25%", "50%", "75%", "max", "median", "missing"]]


def top_categories(df: pd.DataFrame, column: str, limit: int = 10) -> pd.DataFrame:
    result = df[column].fillna("Missing").astype(str).value_counts().head(limit).reset_index()
    result.columns = [column, "count"]
    return result


def aggregate_metrics(df: pd.DataFrame, group_column: str, metric_column: str, aggregation: str) -> pd.DataFrame:
    if group_column not in df.columns:
        raise ValueError(f"Group column not found: {group_column}")
    if aggregation not in {"sum", "mean", "max", "min", "count"}:
        raise ValueError(f"Unsupported aggregation: {aggregation}")
    if metric_column not in df.columns:
        raise ValueError(f"Metric column not found: {metric_column}")

    grouped = df.groupby(group_column, dropna=False)[metric_column].agg(aggregation).reset_index()
    return grouped.sort_values(metric_column, ascending=False)


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return pd.DataFrame()
    return numeric_df.corr()


def detect_anomalies(df: pd.DataFrame, column: str, z_threshold: float = 3.0) -> pd.DataFrame:
    if column not in df.columns:
        raise ValueError(f"Column not found: {column}")
    if z_threshold <= 0:
        raise ValueError("z_threshold must be greater than 0.")

    series = pd.to_numeric(df[column], errors="coerce")
    valid = series.dropna()
    if valid.empty or valid.std() == 0:
        return pd.DataFrame()

    z_scores = (valid - valid.mean()) / valid.std()
    anomaly_index = z_scores[abs(z_scores) >= z_threshold].index
    result = df.loc[anomaly_index].copy()
    result["anomaly_score"] = z_scores.loc[anomaly_index].round(2)
    return result.reindex(result["anomaly_score"].abs().sort_values(ascending=False).index)


def infer_time_series(df: pd.DataFrame) -> tuple[str | None, pd.DataFrame]:
    if df is None or df.empty:
        return None, pd.DataFrame()

    for column in df.columns:
        parsed = pd.to_datetime(df[column], errors="coerce")
        if parsed.notna().sum() >= max(3, len(df) // 3):
            temp_df = df.copy()
            temp_df[column] = parsed
            return column, temp_df
    return None, df


def build_time_series(df: pd.DataFrame, date_column: str, metric_column: str) -> pd.DataFrame:
    if date_column not in df.columns:
        raise ValueError(f"Date column not found: {date_column}")
    if metric_column not in df.columns:
        raise ValueError(f"Metric column not found: {metric_column}")

    temp_df = df.copy()
    temp_df[date_column] = pd.to_datetime(temp_df[date_column], errors="coerce")
    temp_df = temp_df.dropna(subset=[date_column])
    if temp_df.empty:
        return pd.DataFrame()

    series = temp_df.groupby(temp_df[date_column].dt.to_period("M"))[metric_column].sum().reset_index()
    series[date_column] = series[date_column].astype(str)
    return series


def text_word_frequencies(text: str, limit: int = 15) -> pd.DataFrame:
    tokens = [clean_token(token) for token in text.split()]
    filtered_tokens = [token for token in tokens if token and token not in STOP_WORDS and len(token) > 2]
    counts = Counter(filtered_tokens).most_common(limit)
    return pd.DataFrame(counts, columns=["term", "frequency"])


def text_length_metrics(text: str) -> dict[str, int | float]:
    paragraphs = [line.strip() for line in text.splitlines() if line.strip()]
    words = [token for token in text.split() if token.strip()]
    sentences = [segment.strip() for segment in text.replace("\n", " ").split(".") if segment.strip()]

    return {
        "characters": len(text),
        "words": len(words),
        "sentences": len(sentences),
        "paragraphs": len(paragraphs),
        "avg_words_per_sentence": round(len(words) / max(len(sentences), 1), 2),
    }


def generate_analytics_insight(profile: dict[str, Any], numeric_table: pd.DataFrame) -> str:
    if not llm_is_available():
        insights = [
            f"- Dataset size: {profile['rows']} rows x {profile['columns']} columns.",
            f"- Missing cells: {profile['missing_cells']}; duplicate rows: {profile['duplicate_rows']}.",
        ]
        if not numeric_table.empty:
            first_metric = numeric_table.index[0]
            metric_row = numeric_table.loc[first_metric]
            insights.append(
                f"- Numeric highlight: `{first_metric}` has mean {round(float(metric_row['mean']), 2)} and median {round(float(metric_row['median']), 2)}."
            )
        return "\n".join(insights)

    compact_table = numeric_table.head(5).round(2).to_dict(orient="index") if not numeric_table.empty else {}
    prompt = (
        "You are a big data analytics assistant. "
        "Based on the dataset profile and numeric summary below, provide:\n"
        "1. Two important insights\n"
        "2. One data quality concern\n"
        "3. One business question worth exploring next\n\n"
        f"Profile:\n{json.dumps(profile, indent=2)}\n\n"
        f"Numeric summary:\n{json.dumps(compact_table, indent=2)}\n"
    )
    return generate_llm_response(prompt, max_tokens=300, temperature=0.4)


def summarize_pipeline_report(report: dict[str, Any]) -> str:
    preview = report.get("topic_metrics_preview", [])
    status = report.get("status", "unknown")
    if not llm_is_available():
        return (
            f"Pipeline status: {status}.\n\n"
            f"Records processed: {report.get('records_processed', 0)}.\n\n"
            f"Top topic rows available: {len(preview)}.\n\n"
            "Next action: Review the activity and quiz trends in the analytics dashboard."
        )

    prompt = (
        "You are an AI data engineer. Summarize this big data pipeline report in a short business-friendly format. "
        "Mention processing status, top topic performance, and next action.\n\n"
        f"{json.dumps(report, indent=2)}"
    )
    return generate_llm_response(prompt, max_tokens=220, temperature=0.4)


def spark_dashboard_metrics(limit: int = 10) -> dict[str, pd.DataFrame]:
    metrics = _analytics_backend().dashboard_metrics(limit=limit)
    return {key: pd.DataFrame(value) for key, value in metrics.items()}


def spark_hardest_topics(limit: int = 10) -> pd.DataFrame:
    return pd.DataFrame(_analytics_backend().hardest_topics(limit=limit))


def spark_weak_areas(limit: int = 20) -> pd.DataFrame:
    return pd.DataFrame(_analytics_backend().weak_areas_per_user(limit=limit))


def spark_top_students(limit: int = 10) -> pd.DataFrame:
    return pd.DataFrame(_analytics_backend().top_performing_students(limit=limit))


def spark_trends() -> pd.DataFrame:
    return pd.DataFrame(_analytics_backend().trend_analysis())


def recent_activity_history(user_id: str) -> dict[str, pd.DataFrame]:
    history = _analytics_backend().recent_user_history(user_id)
    return {key: pd.DataFrame(value) for key, value in history.items()}


def build_learning_profile(
    user_id: int | str,
    study_df: pd.DataFrame,
    quiz_df: pd.DataFrame,
    events_df: pd.DataFrame,
) -> dict[str, Any]:
    user_id = int(user_id)
    profile = {
        "avg_score": 0.0,
        "strong_topics": [],
        "weak_topics": [],
        "most_active_topics": [],
        "recommendations": [],
    }

    user_quiz_df = quiz_df[quiz_df["user_id"] == user_id].copy() if not quiz_df.empty and "user_id" in quiz_df.columns else pd.DataFrame()
    user_study_df = study_df[study_df["user_id"] == user_id].copy() if not study_df.empty and "user_id" in study_df.columns else pd.DataFrame()
    user_events_df = events_df[events_df["user_id"] == user_id].copy() if not events_df.empty and "user_id" in events_df.columns else pd.DataFrame()

    topic_scores = pd.DataFrame()
    if not user_quiz_df.empty and {"topic", "score_percent"}.issubset(user_quiz_df.columns):
        topic_scores = (
            user_quiz_df.groupby("topic", as_index=False)
            .agg(avg_score=("score_percent", "mean"), attempts=("topic", "size"))
            .sort_values(["avg_score", "attempts"], ascending=[False, False])
        )
        profile["avg_score"] = round(float(topic_scores["avg_score"].mean()), 2)
        profile["strong_topics"] = topic_scores[topic_scores["avg_score"] >= 75]["topic"].head(5).tolist()
        profile["weak_topics"] = topic_scores[topic_scores["avg_score"] < 60].sort_values("avg_score")["topic"].head(5).tolist()

    activity_topics = pd.Series(dtype=object)
    if not user_study_df.empty and "topic" in user_study_df.columns:
        activity_topics = pd.concat([activity_topics, user_study_df["topic"].dropna().astype(str)], ignore_index=True)
    if not user_events_df.empty and "topics_json" in user_events_df.columns:
        expanded_topics: list[str] = []
        for raw in user_events_df["topics_json"].fillna("[]"):
            try:
                expanded_topics.extend(json.loads(raw) or [])
            except Exception:
                continue
        if expanded_topics:
            activity_topics = pd.concat([activity_topics, pd.Series(expanded_topics, dtype=object)], ignore_index=True)
    if not activity_topics.empty:
        profile["most_active_topics"] = activity_topics.value_counts().head(5).index.tolist()

    recommendations: list[str] = []
    for topic in profile["weak_topics"][:3]:
        recommendations.append(f"Revise {topic} and try another medium quiz.")
    if not recommendations and profile["strong_topics"]:
        recommendations.append(f"Move to a harder quiz on {profile['strong_topics'][0]}.")
    if not recommendations and profile["most_active_topics"]:
        recommendations.append(f"Continue studying {profile['most_active_topics'][0]} and generate a summary revision sheet.")
    profile["recommendations"] = recommendations
    return profile
