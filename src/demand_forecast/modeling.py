# src/demand_forecast/modeling.py

import os
import yaml
from pyspark.sql import DataFrame
import pyspark.sql.functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor


def _load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def train_random_forest(
    df_final_features: DataFrame,
    config_path: str = "configs/model.yaml",
):
    """
    Train a Random Forest model using your EXACT notebook logic, but
    driven by model.yaml.

    Steps:
    - Rename TotalQuantity -> label
    - Build preprocessing pipeline (indexers + encoder + assembler)
    - Time-based train/test split on Date
    - Fit pipeline on train only
    - Train RandomForestRegressor on processed train data
    - Save RF model + preprocessing pipeline to disk

    Returns:
        rf_model, preprocess_model, train_processed_df, test_processed_df, cfg
    """
    cfg = _load_yaml(config_path)

    target_col = cfg["target_column"]
    label_col = cfg["label_column"]
    date_col = cfg["date_column"]

    cat_index = cfg["pipeline"]["categorical_index"]
    cat_ohe = cfg["pipeline"]["categorical_ohe"]
    numerical_cols = cfg["pipeline"]["numerical"]

    rf_params = cfg["model"]["params"]
    split_date = cfg["training"]["split_date"]
    feature_col = cfg["training"]["feature_col"]
    prediction_col = cfg["training"]["prediction_col"]

    save_dir_root = cfg["training"]["save_dir"]
    experiment_name = cfg["training"]["experiment_name"]

    # 1) Rename target column to 'label'
    df_ml_data = df_final_features.withColumnRenamed(target_col, label_col)

    print(f"Using {len(numerical_cols)} non-leaking features.")
    print("Building feature engineering pipeline...")

    # 2) Build preprocessing pipeline (same as your notebook)

    # Stage 1: StringIndexers
    indexers = [
        StringIndexer(
            inputCol=col,
            outputCol=f"{col}_idx",
            handleInvalid="keep",
        )
        for col in (cat_index + cat_ohe)
    ]

    # Stage 2: OneHotEncoders (for Country only, as per your code)
    encoders = [
        OneHotEncoder(
            inputCol=f"{col}_idx",
            outputCol=f"{col}_ohe",
        )
        for col in cat_ohe
    ]

    # Stage 3: VectorAssembler
    ohe_output_cols = [f"{col}_ohe" for col in cat_ohe]
    idx_output_cols = [f"{col}_idx" for col in cat_index]
    #all_feature_cols = numerical_cols + idx_output_cols + ohe_output_cols
    all_feature_cols = numerical_cols + ohe_output_cols


    assembler = VectorAssembler(
        inputCols=all_feature_cols,
        outputCol=feature_col,
    )

    preprocess_pipeline = Pipeline(stages=indexers + encoders + [assembler])

    # 3) Time-based train/test split (exactly your logic)

    print("Applying time-based train/test split...")
    train_df = df_ml_data.filter(F.col(date_col) <= split_date)
    test_df = df_ml_data.filter(F.col(date_col) > split_date)

    train_df = train_df.cache()
    test_df = test_df.cache()

    print(f"Training data: {train_df.count()} rows (<= {split_date})")
    print(f"Test data: {test_df.count()} rows (> {split_date})")

    # 4) Fit preprocessing pipeline on TRAIN only

    print("\nFitting preprocessing pipeline on training data...")
    preprocess_model = preprocess_pipeline.fit(train_df)

    print("Transforming train and test data...")
    train_processed_df = preprocess_model.transform(train_df)
    test_processed_df = preprocess_model.transform(test_df)

    print("Caching processed training data...")
    train_processed_df = train_processed_df.cache()

    # 5) Train Random Forest (your exact params)

    print("\n--- Training Random Forest ---")
    rf = RandomForestRegressor(
        featuresCol=feature_col,
        labelCol=label_col,
        **rf_params,
    )

    rf_model = rf.fit(train_processed_df)
    print("Random Forest trained successfully.")

    # 6) Save model + preprocessing pipeline

    experiment_root = os.path.join(save_dir_root, experiment_name)
    rf_path = os.path.join(experiment_root, "rf_model")
    prep_path = os.path.join(experiment_root, "preprocess_pipeline")

    os.makedirs(experiment_root, exist_ok=True)

    print(f"Saving Random Forest model to: {rf_path}")
    rf_model.write().overwrite().save(rf_path)

    print(f"Saving preprocessing pipeline to: {prep_path}")
    preprocess_model.write().overwrite().save(prep_path)

    print("Model and preprocessing pipeline saved successfully.")

    return rf_model, preprocess_model, train_processed_df, test_processed_df, cfg
