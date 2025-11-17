from pyspark.sql import SparkSession, DataFrame
import yaml
import os


def _load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_raw_online_retail(
    spark: SparkSession,
    config_path: str = "configs/data.yaml",
    project_root: str = None,
) -> DataFrame:
    """
    Load the raw Online Retail dataset using the same logic as the notebook:

        df_raw = spark.read.csv(file_path, header=True, inferSchema=True)

    The file path is read from configs/data.yaml under:
        raw.online_retail
        
    Args:
        spark: SparkSession instance
        config_path: Path to the data configuration YAML file
        project_root: Optional project root directory for resolving relative paths
    """
    cfg = _load_yaml(config_path)
    file_path = cfg["raw"]["online_retail"]
    
    # If project_root is provided and file_path is relative, resolve it
    if project_root and not os.path.isabs(file_path):
        file_path = os.path.join(project_root, file_path)
    
    # Normalize path separators for the current OS
    file_path = os.path.normpath(file_path)

    print(f"Loading raw Online Retail data from: {file_path}")
    
    # Check if file exists before attempting to load
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Data file not found at: {file_path}\n"
            f"Please ensure the file exists at this location."
        )

    df_raw = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(file_path)
    )

    print("\n--- Data Schema (Column Names and Types) ---")
    df_raw.printSchema()

    return df_raw
