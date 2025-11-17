#!/usr/bin/env python

import os
import sys
import yaml

# Add the src directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

from pyspark.ml import PipelineModel
from pyspark.ml.regression import RandomForestRegressionModel

from demand_forecast.spark_session import create_spark
from demand_forecast.cleaning import clean_transactions
from demand_forecast.features import (
    aggregate_to_daily_panel,
    add_calendar_features,
    add_lag_and_rolling_features,
)


def validate_model_path(path: str, model_name: str) -> bool:
    """Validate that model path exists and contains required metadata."""
    if not os.path.exists(path):
        print(f"ERROR: {model_name} path does not exist: {path}")
        return False
    
    # Check for metadata directory (required for PySpark models)
    metadata_path = os.path.join(path, "metadata")
    if not os.path.exists(metadata_path):
        print(f"ERROR: {model_name} metadata directory missing: {metadata_path}")
        return False
    
    # Check for metadata/part-00000 file
    metadata_file = os.path.join(metadata_path, "part-00000")
    if not os.path.exists(metadata_file):
        print(f"ERROR: {model_name} metadata file missing: {metadata_file}")
        return False
    
    print(f"✓ {model_name} path validation passed: {path}")
    return True


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    # Construct absolute paths to config files relative to project root
    spark_config_path = os.path.join(project_root, "configs", "spark.yaml")
    data_config_path = os.path.join(project_root, "configs", "data.yaml")
    model_config_path = os.path.join(project_root, "configs", "model.yaml")

    # 1) Spark session with increased memory to prevent worker crashes
    print("Creating Spark session...")
    try:
        spark = create_spark(spark_config_path)
        
        # Additional configuration to prevent worker crashes
        spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "false")
        spark.conf.set("spark.python.worker.reuse", "true")
        
    except Exception as e:
        print(f"ERROR: Failed to create Spark session: {e}")
        sys.exit(1)

    # 2) Load configs
    try:
        model_cfg = load_yaml(model_config_path)
        data_cfg = load_yaml(data_config_path)
    except Exception as e:
        print(f"ERROR: Failed to load configuration files: {e}")
        spark.stop()
        sys.exit(1)

    target_col = model_cfg["target_column"]        # "TotalQuantity"
    label_col = model_cfg["label_column"]          # "label"
    feature_col = model_cfg["training"]["feature_col"]   # "features"
    prediction_col = model_cfg["training"]["prediction_col"]  # "prediction"

    save_dir_root = model_cfg["training"]["save_dir"]     # "models"
    experiment_name = model_cfg["training"]["experiment_name"]  # "rf_v1"

    # Construct absolute paths for model directories
    experiment_root = os.path.join(project_root, save_dir_root, experiment_name)
    rf_path = os.path.join(experiment_root, "rf_model")
    prep_path = os.path.join(experiment_root, "preprocess_pipeline")

    # Resolve input/output paths relative to project root if needed
    input_path = data_cfg["scoring"]["input_path"]
    output_path = data_cfg["scoring"]["output_path"]
    
    if not os.path.isabs(input_path):
        input_path = os.path.join(project_root, input_path)
    if not os.path.isabs(output_path):
        output_path = os.path.join(project_root, output_path)

    # Validate model paths before loading
    print("\n=== Validating Model Paths ===")
    prep_valid = validate_model_path(prep_path, "Preprocessing Pipeline")
    rf_valid = validate_model_path(rf_path, "Random Forest Model")
    
    if not prep_valid or not rf_valid:
        print("\nERROR: Model validation failed. Please ensure models are trained and saved correctly.")
        print(f"Expected preprocessing pipeline at: {prep_path}")
        print(f"Expected RF model at: {rf_path}")
        print("\nRun train_model.py first to generate the models.")
        spark.stop()
        sys.exit(1)

    # Load models with error handling
    print(f"\n=== Loading Models ===")
    try:
        print(f"Loading preprocessing pipeline from: {prep_path}")
        preprocess_model = PipelineModel.load(prep_path)
        print("✓ Preprocessing pipeline loaded successfully")
    except Exception as e:
        print(f"ERROR: Failed to load preprocessing pipeline: {e}")
        print("This may indicate model corruption or version mismatch.")
        spark.stop()
        sys.exit(1)

    try:
        print(f"Loading RandomForest model from: {rf_path}")
        rf_model = RandomForestRegressionModel.load(rf_path)
        print("✓ Random Forest model loaded successfully")
    except Exception as e:
        print(f"ERROR: Failed to load Random Forest model: {e}")
        spark.stop()
        sys.exit(1)

    # 3) Load new raw data (same format as training raw file)
    print(f"\n=== Loading Scoring Data ===")
    
    if not os.path.exists(input_path):
        print(f"ERROR: Input data file not found: {input_path}")
        spark.stop()
        sys.exit(1)
    
    print(f"Loading new scoring data from: {input_path}")
    try:
        df_new_raw = (
            spark.read
            .option("header", True)
            .option("inferSchema", True)
            .csv(input_path)
        )
        
        print("Raw scoring data schema:")
        df_new_raw.printSchema()
        
        row_count = df_new_raw.count()
        print(f"Loaded {row_count} rows")
        
        if row_count == 0:
            print("WARNING: Input data is empty!")
            
    except Exception as e:
        print(f"ERROR: Failed to load scoring data: {e}")
        spark.stop()
        sys.exit(1)

    # 4) Apply SAME cleaning & feature engineering as training
    print("\n=== Feature Engineering ===")
    try:
        print("Cleaning new data...")
        df_cleaned = clean_transactions(df_new_raw)

        print("Aggregating to daily panel...")
        df_agg = aggregate_to_daily_panel(df_cleaned)

        print("Adding calendar features...")
        df_cal = add_calendar_features(df_agg)

        print("Adding lag/rolling/price features...")
        df_final_features = add_lag_and_rolling_features(df_cal)
    except Exception as e:
        print(f"ERROR: Feature engineering failed: {e}")
        spark.stop()
        sys.exit(1)

    # 5) Rename target to label if it exists (if this is backtest data)
    if target_col in df_final_features.columns:
        print(f"Renaming target column {target_col} -> {label_col}")
        df_ml = df_final_features.withColumnRenamed(target_col, label_col)
    else:
        print(f"Target column '{target_col}' not found in scoring data. Proceeding without renaming.")
        df_ml = df_final_features

    # 6) Preprocess features using saved preprocessing pipeline
    print("\n=== Generating Predictions ===")
    try:
        print("Applying preprocessing pipeline to scoring data...")
        df_features = preprocess_model.transform(df_ml)

        # 7) Generate predictions with RF model
        print("Generating predictions...")
        df_predictions = rf_model.transform(df_features)
    except Exception as e:
        print(f"ERROR: Prediction generation failed: {e}")
        spark.stop()
        sys.exit(1)

    # 8) Select useful columns for output
    select_cols = ["Date", "StockCode", "Country", prediction_col]
    if label_col in df_predictions.columns:
        select_cols.append(label_col)

    df_output = df_predictions.select(*select_cols)

    # 9) Save predictions
    print(f"\n=== Saving Results ===")
    print(f"Saving predictions to: {output_path}")
    try:
        (
            df_output
            .write
            .mode("overwrite")
            .parquet(output_path)
        )
        print("✓ Predictions saved successfully")
    except Exception as e:
        print(f"ERROR: Failed to save predictions: {e}")
        spark.stop()
        sys.exit(1)

    print("\n=== Sample Predictions ===")
    df_output.show(10)

    spark.stop()
    print("\n=== Process Complete ===")


if __name__ == "__main__":
    main()
