# src/demand_forecast/evaluation.py

import os
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.ml.evaluation import RegressionEvaluator
import pyspark.sql.functions as F


def calculate_mape(
    predictions_df: DataFrame,
    label_col: str = "label",
    prediction_col: str = "prediction",
) -> float:
    """
    Mean Absolute Percentage Error (MAPE), using your notebook logic:
    - Filter out rows where label == 0
    - Compute |(label - prediction) / label|
    - Return mean * 100 (percentage)
    """
    mape_df = predictions_df.filter(F.col(label_col) != 0)

    mape_df = mape_df.withColumn(
        "APE",
        F.abs((F.col(label_col) - F.col(prediction_col)) / F.col(label_col)),
    )

    mape_result = mape_df.select(F.mean("APE")).first()
    if mape_result and mape_result[0] is not None:
        return float(mape_result[0]) * 100.0

    return float("inf")


def evaluate_random_forest(
    rf_model,
    test_processed_df: DataFrame,
    label_col: str = "label",
    prediction_col: str = "prediction",
    csv_path: str = "reports/rf_v1_metrics.csv",
):
    """
    Evaluate Random Forest on the test set:
    - Computes RMSE, MAE, R², MAPE
    - Prints a nice comparison table (just RF)
    - Saves metrics to a CSV file

    Returns:
        metrics (dict), rf_preds (DataFrame)
    """
    print("\n--- Model Evaluation on Test Data ---")

    # 1) Get predictions
    rf_preds = rf_model.transform(test_processed_df)

    # 2) Evaluators
    eval_rmse = RegressionEvaluator(
        labelCol=label_col,
        predictionCol=prediction_col,
        metricName="rmse",
    )
    eval_mae = RegressionEvaluator(
        labelCol=label_col,
        predictionCol=prediction_col,
        metricName="mae",
    )
    eval_r2 = RegressionEvaluator(
        labelCol=label_col,
        predictionCol=prediction_col,
        metricName="r2",
    )

    rf_rmse = eval_rmse.evaluate(rf_preds)
    rf_mae = eval_mae.evaluate(rf_preds)
    rf_r2 = eval_r2.evaluate(rf_preds)
    rf_mape = calculate_mape(rf_preds, label_col=label_col, prediction_col=prediction_col)

    # 3) Print table (your style)
    print("\n--- Updated Model Comparison (Random Forest Only) ---")
    print("| Model               | RMSE     | MAE      | MAPE (%) | R-squared |")
    print("|---------------------|----------|----------|----------|-----------|")
    print(
        f"| Random Forest       | {rf_rmse:<8.2f} | {rf_mae:<8.2f} | "
        f"{rf_mape:<8.2f} | {rf_r2:<9.2f} |"
    )

    metrics = {
        "model": "Random Forest",
        "rmse": rf_rmse,
        "mae": rf_mae,
        "r2": rf_r2,
        "mape": rf_mape,
    }

    # 4) Save metrics to CSV
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df_metrics = pd.DataFrame([metrics])
    df_metrics.to_csv(csv_path, index=False)
    print(f"\nMetrics saved to: {csv_path}")

    return metrics, rf_preds
