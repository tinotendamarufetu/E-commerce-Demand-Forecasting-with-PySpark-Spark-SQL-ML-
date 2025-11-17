#!/usr/bin/env python

import os
import sys

# Add the src directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

from pyspark.sql import SparkSession, DataFrame
import yaml

from demand_forecast.spark_session import create_spark
# from demand_forecast.data_ingestion import load_raw_online_retail
from demand_forecast.cleaning import clean_transactions
from demand_forecast.features import (
    aggregate_to_daily_panel,
    add_calendar_features,
    add_lag_and_rolling_features,
    finalize_features,
)
from demand_forecast.plotting import (
    plot_top_products,
    plot_top_countries,
    plot_weekly_sales,
    plot_monthly_sales,
    # the extra EDA you added earlier:
    plot_global_daily_quantity,
    plot_top_product_trends,
    plot_top_country_trends,
    plot_demand_volatility,
    plot_global_rolling_avg,
)


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


def main():
    # Use absolute paths relative to project root
    spark_config_path = os.path.join(project_root, "configs", "spark.yaml")
    data_config_path = os.path.join(project_root, "configs", "data.yaml")
    
    spark = create_spark(spark_config_path)

    # 1) Load raw data
    df_raw = load_raw_online_retail(spark, data_config_path, project_root)

    # 2) Clean data
    df_cleaned = clean_transactions(df_raw)

    # 2a) SAVE cleaned data -> data/interim
    interim_dir = os.path.join(project_root, "data", "interim")
    os.makedirs(interim_dir, exist_ok=True)
    cleaned_path = os.path.join(interim_dir, "transactions_cleaned.parquet")
    print(f"Saving cleaned data to: {cleaned_path}")
    (
        df_cleaned
        .write
        .mode("overwrite")
        .parquet(cleaned_path)
    )

    # 3) Feature engineering pipeline
    df_agg = aggregate_to_daily_panel(df_cleaned)
    df_cal = add_calendar_features(df_agg)
    df_lag = add_lag_and_rolling_features(df_cal)
    df_final = finalize_features(df_lag)

    # 3a) SAVE final feature data -> data/processed
    processed_dir = os.path.join(project_root, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    final_path = os.path.join(processed_dir, "demand_features.parquet")
    print(f"Saving final feature dataset to: {final_path}")
    (
        df_final
        .write
        .mode("overwrite")
        .parquet(final_path)
    )

    # 4) Register SQL view for EDA
    df_final.createOrReplaceTempView("sales")
    print("Temporary view 'sales' created successfully.")

    # 5) Run EDA plots (these also save images under plots/)
    plot_top_products(spark)
    plot_top_countries(df_cleaned)
    plot_weekly_sales(spark)
    plot_monthly_sales(spark)

    # Extra EDA
    plot_global_daily_quantity(df_final)
    plot_top_product_trends(df_final)
    plot_top_country_trends(df_final)
    plot_demand_volatility(df_final)
    plot_global_rolling_avg(df_final)

    print("\nEDA + data saving complete.")
    spark.stop()


if __name__ == "__main__":
    main()
