from __future__ import annotations

from typing import Any

from learnmate_ai.config import AppConfig, get_config
from learnmate_ai.storage import ensure_data_directories

try:
    from pyspark.sql import SparkSession
except Exception:
    SparkSession = None


def _resolved_master(config: AppConfig) -> str:
    master = config.spark_master.strip()
    if master.lower() == 'local-cluster':
        return f'local-cluster[{config.spark_local_cluster_workers},{config.spark_local_cluster_cores},{config.spark_local_cluster_memory}]'
    return master


def get_spark_session(config: AppConfig | None = None):
    """Create an optimized Spark session for batch and streaming workloads."""
    app_config = ensure_data_directories(config or get_config())
    if SparkSession is None:
        raise RuntimeError(
            'PySpark is not available. Install dependencies from requirements.txt before running the big data pipeline.'
        )

    builder = (
        SparkSession.builder.appName(app_config.spark_app_name)
        .master(_resolved_master(app_config))
        .config('spark.sql.warehouse.dir', app_config.spark_warehouse_dir)
        .config('spark.driver.memory', app_config.spark_driver_memory)
        .config('spark.executor.memory', app_config.spark_executor_memory)
        .config('spark.sql.shuffle.partitions', str(app_config.spark_shuffle_partitions))
        .config('spark.default.parallelism', str(app_config.spark_default_parallelism))
        .config('spark.sql.adaptive.enabled', 'true')
        .config('spark.sql.adaptive.coalescePartitions.enabled', 'true')
        .config('spark.sql.adaptive.skewJoin.enabled', 'true')
        .config('spark.serializer', 'org.apache.spark.serializer.KryoSerializer')
        .config('spark.sql.execution.arrow.pyspark.enabled', 'true')
        .config('spark.streaming.stopGracefullyOnShutdown', 'true')
        .config('spark.sql.streaming.stateStore.providerClass', 'org.apache.spark.sql.execution.streaming.state.HDFSBackedStateStoreProvider')
        .config('spark.sql.files.maxPartitionBytes', str(128 * 1024 * 1024))
        .config('spark.sql.parquet.mergeSchema', 'false')
        .config('spark.sql.sources.partitionOverwriteMode', 'dynamic')
    )

    if app_config.spark_event_log_enabled:
        builder = builder.config('spark.eventLog.enabled', 'true').config('spark.eventLog.dir', str(app_config.checkpoint_dir / 'spark-events'))

    if app_config.spark_packages:
        builder = builder.config('spark.jars.packages', app_config.spark_packages)

    return builder.getOrCreate()


def spark_runtime_status(config: AppConfig | None = None) -> dict[str, Any]:
    """Return Spark availability and active runtime settings."""
    app_config = config or get_config()
    return {
        'spark_master': _resolved_master(app_config),
        'spark_app_name': app_config.spark_app_name,
        'spark_warehouse_dir': app_config.spark_warehouse_dir,
        'spark_driver_memory': app_config.spark_driver_memory,
        'spark_executor_memory': app_config.spark_executor_memory,
        'spark_shuffle_partitions': app_config.spark_shuffle_partitions,
        'spark_default_parallelism': app_config.spark_default_parallelism,
        'spark_packages': app_config.spark_packages,
        'kafka_enabled': app_config.kafka_enabled,
        'data_lake_uri': app_config.data_lake_uri,
        'pyspark_available': SparkSession is not None,
    }
