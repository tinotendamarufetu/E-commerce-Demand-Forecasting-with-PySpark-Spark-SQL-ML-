import os
import sys
import yaml
from pyspark.sql import SparkSession


def _load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _try_stop_existing_session():
    """
    Mirror your notebook behavior:
    try: spark.stop() ; except: pass
    """
    try:
        # If a SparkSession already exists, stop it
        from pyspark.sql import SparkSession as _SS

        _SS.builder.getOrCreate().stop()
        print("Existing SparkSession stopped.")
    except Exception:
        # No existing session, ignore
        pass


def create_spark(config_path: str = "configs/spark.yaml") -> SparkSession:
    """
    Create a SparkSession using the same logic as your notebook, but
    driven by configs/spark.yaml.
    """

    cfg = _load_yaml(config_path)

    # --- Environment variables (same as your notebook) ---
    os.environ["PYSPARK_PYTHON"] = sys.executable

    hadoop_home = cfg.get("hadoop_home", r"C:\hadoop")
    os.environ["HADOOP_HOME"] = hadoop_home
    os.environ["PATH"] = hadoop_home + r"\bin;" + os.environ["PATH"]

    # --- Stop existing session if any ---
    _try_stop_existing_session()

    # --- Build SparkSession (same configs as your notebook) ---
    builder = (
        SparkSession.builder
        .master(cfg.get("master", "local[*]"))
        .appName(cfg.get("app_name", "DemandForecast"))
        .config("spark.driver.bindAddress", cfg["ui"]["bind_address"])
        .config("spark.ui.port", str(cfg["ui"]["port"]))
        .config("spark.driver.memory", cfg["memory"]["driver_memory"])
        .config(
            "spark.driver.maxResultSize",
            cfg["memory"]["driver_max_result_size"],
        )
        .config(
            "spark.executor.memoryOverhead",
            cfg["memory"]["executor_memory_overhead"],
        )
    )

    spark = builder.getOrCreate()

    # Set wholeStage codegen flag exactly like your notebook
    whole_stage = cfg["codegen"]["whole_stage"]
    spark.conf.set(
        "spark.sql.codegen.wholeStage",
        "true" if whole_stage else "false",
    )

    print(f"SparkSession created successfully on UI port {cfg['ui']['port']}")
    print(f"Spark Version: {spark.version}")

    return spark
