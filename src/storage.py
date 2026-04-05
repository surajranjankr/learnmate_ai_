from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

from learnmate_ai.config import AppConfig, get_config


def ensure_data_directories(config: AppConfig | None = None) -> AppConfig:
    app_config = config or get_config()
    for directory in (
        app_config.data_dir,
        app_config.raw_dir,
        app_config.bronze_dir,
        app_config.silver_dir,
        app_config.gold_dir,
        app_config.report_dir,
        app_config.logs_dir,
        app_config.streaming_input_dir,
        app_config.streaming_output_dir,
        app_config.checkpoint_dir,
        app_config.lakehouse_dir,
        app_config.raw_events_dir,
        app_config.curated_events_dir,
        app_config.model_features_dir,
        app_config.kafka_checkpoint_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    return app_config


def timestamped_name(original_name: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{original_name}"


def save_uploaded_file(uploaded_file, destination_dir: Path) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    file_path = destination_dir / timestamped_name(uploaded_file.name)
    file_path.write_bytes(uploaded_file.getvalue())
    return file_path


def resolve_storage_uri(uri_or_path: str | Path) -> str:
    value = str(uri_or_path)
    if "://" in value:
        return value
    return str(Path(value))


def event_partition_path(config: AppConfig, event_type: str, event_timestamp: str) -> Path:
    stamp = datetime.fromisoformat(event_timestamp.replace("Z", "+00:00"))
    return (
        config.raw_events_dir
        / f"event_type={event_type}"
        / f"year={stamp:%Y}"
        / f"month={stamp:%m}"
        / f"day={stamp:%d}"
    )


def append_event_to_lake(config: AppConfig, event_type: str, payload: dict[str, Any]) -> Path:
    timestamp = str(payload.get("timestamp") or datetime.now(UTC).isoformat(timespec="seconds"))
    partition_dir = event_partition_path(config, event_type, timestamp)
    partition_dir.mkdir(parents=True, exist_ok=True)
    file_path = partition_dir / f"{timestamped_name(event_type)}.json"
    file_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return file_path
