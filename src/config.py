from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv() -> None:
        return None


load_dotenv()


BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class AppConfig:
    """Central application configuration for storage, Spark, and the local app database."""

    base_dir: Path = BASE_DIR
    data_dir: Path = BASE_DIR / "data"
    raw_dir: Path = BASE_DIR / "data" / "raw"
    bronze_dir: Path = BASE_DIR / "data" / "bronze"
    silver_dir: Path = BASE_DIR / "data" / "silver"
    gold_dir: Path = BASE_DIR / "data" / "gold"
    report_dir: Path = BASE_DIR / "data" / "reports"
    logs_dir: Path = BASE_DIR / "data" / "logs"
    streaming_input_dir: Path = BASE_DIR / "data" / "stream_input"
    streaming_output_dir: Path = BASE_DIR / "data" / "stream_output"
    checkpoint_dir: Path = BASE_DIR / "data" / "checkpoints"
    lakehouse_dir: Path = BASE_DIR / "data" / "lakehouse"
    raw_events_dir: Path = BASE_DIR / "data" / "lakehouse" / "raw_events"
    curated_events_dir: Path = BASE_DIR / "data" / "lakehouse" / "curated_events"
    model_features_dir: Path = BASE_DIR / "data" / "lakehouse" / "model_features"
    sqlite_db_path: Path = Path(os.getenv("SQLITE_DB_PATH", str(BASE_DIR / "data" / "learnmate_ai.db")))

    storage_backend: str = os.getenv("STORAGE_BACKEND", "local")
    data_lake_uri: str = os.getenv("DATA_LAKE_URI", str(BASE_DIR / "data" / "lakehouse"))
    bronze_uri: str = os.getenv("BRONZE_URI", str(BASE_DIR / "data" / "bronze"))
    silver_uri: str = os.getenv("SILVER_URI", str(BASE_DIR / "data" / "silver"))
    gold_uri: str = os.getenv("GOLD_URI", str(BASE_DIR / "data" / "gold"))

    spark_app_name: str = os.getenv("SPARK_APP_NAME", "LearnMateBigDataAI")
    spark_master: str = os.getenv("SPARK_MASTER", "local[*]")
    spark_warehouse_dir: str = os.getenv("SPARK_WAREHOUSE_DIR", str(BASE_DIR / "data" / "spark-warehouse"))
    spark_driver_memory: str = os.getenv("SPARK_DRIVER_MEMORY", "2g")
    spark_executor_memory: str = os.getenv("SPARK_EXECUTOR_MEMORY", "2g")
    spark_shuffle_partitions: int = int(os.getenv("SPARK_SHUFFLE_PARTITIONS", "8"))
    spark_default_parallelism: int = int(os.getenv("SPARK_DEFAULT_PARALLELISM", "8"))
    spark_streaming_trigger: str = os.getenv("SPARK_STREAMING_TRIGGER", "5 seconds")
    spark_packages: str = os.getenv("SPARK_PACKAGES", "")
    spark_local_cluster_workers: int = int(os.getenv("SPARK_LOCAL_CLUSTER_WORKERS", "2"))
    spark_local_cluster_cores: int = int(os.getenv("SPARK_LOCAL_CLUSTER_CORES", "2"))
    spark_local_cluster_memory: str = os.getenv("SPARK_LOCAL_CLUSTER_MEMORY", "1024")
    spark_event_log_enabled: bool = os.getenv("SPARK_EVENT_LOG_ENABLED", "false").lower() == "true"

    kafka_enabled: bool = os.getenv("KAFKA_ENABLED", "false").lower() == "true"
    kafka_bootstrap_servers: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    kafka_topic_user_activity: str = os.getenv("KAFKA_TOPIC_USER_ACTIVITY", "learnmate.user.activity")
    kafka_topic_chat: str = os.getenv("KAFKA_TOPIC_CHAT", "learnmate.chat")
    kafka_topic_quiz: str = os.getenv("KAFKA_TOPIC_QUIZ", "learnmate.quiz")
    kafka_checkpoint_dir: Path = BASE_DIR / "data" / "checkpoints" / "kafka_streaming"

    model_path: str = os.getenv("MODEL_PATH", str(BASE_DIR / "models" / "mistral-7b.Q4_K_M.gguf"))

    @property
    def database_configured(self) -> bool:
        return bool(self.sqlite_db_path and str(self.sqlite_db_path).strip())


def get_config() -> AppConfig:
    return AppConfig()
