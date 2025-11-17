# !/usr/bin/env python
import os
import sys

# Add the src directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

from demand_forecast.spark_session import create_spark
from demand_forecast.data_ingestion import load_raw_online_retail
from demand_forecast.cleaning import clean_transactions
from demand_forecast.features import (
    aggregate_to_daily_panel,
    add_calendar_features,
    add_lag_and_rolling_features,
    finalize_features,
)
from demand_forecast.modeling import train_random_forest
from demand_forecast.evaluation import evaluate_random_forest
from demand_forecast.diagnostics import (
    plot_actual_vs_predicted,
    plot_feature_importances,
    compute_expected_units_for_week,
)


def main():
    # Construct absolute paths to config files relative to project root
    spark_config_path = os.path.join(project_root, "configs", "spark.yaml")
    data_config_path = os.path.join(project_root, "configs", "data.yaml")
    model_config_path = os.path.join(project_root, "configs", "model.yaml")

    # Construct absolute paths for output directories
    reports_dir = os.path.join(project_root, "reports")
    plots_dir = os.path.join(project_root, "scripts", "plots")

    # Ensure output directories exist
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # 1) Start Spark
    spark = create_spark(spark_config_path)

    # 2) Load + clean raw data
    df_raw = load_raw_online_retail(spark, data_config_path, project_root)
    df_cleaned = clean_transactions(df_raw)

    # 3) Feature engineering
    df_agg = aggregate_to_daily_panel(df_cleaned)
    df_cal = add_calendar_features(df_agg)
    df_lag = add_lag_and_rolling_features(df_cal)
    df_final_features = finalize_features(df_lag)

    # 4) Train Random Forest (config-driven, saves model to models/rf_v1/)
    rf_model, preprocess_model, train_processed_df, test_processed_df, cfg = train_random_forest(
        df_final_features,
        config_path=model_config_path,
    )

    # 5) Evaluate RF on test set + save metrics CSV
    label_col = cfg["label_column"]  # "label"
    prediction_col = cfg["training"]["prediction_col"]  # "prediction"

    metrics, rf_preds = evaluate_random_forest(
        rf_model,
        test_processed_df,
        label_col=label_col,
        prediction_col=prediction_col,
        csv_path=os.path.join(reports_dir, "rf_v1_metrics.csv"),
    )

    print("\nFinal Test Metrics (Random Forest):")
    for k, v in metrics.items():
        if k == "model":
            continue
        if k == "mape":
            print(f"{k.upper()}: {v:.2f}%")
        else:
            print(f"{k.upper()}: {v:.4f}")

    # 6) Diagnostics: Actual vs Predicted for one product/country
    plot_actual_vs_predicted(
        rf_preds,
        stockcode="85123A",
        country="United Kingdom",
        label_col=label_col,
        prediction_col=prediction_col,
        save_path=os.path.join(plots_dir, "actual_vs_predicted_85123A_UK.png"),
    )

    # 7) Diagnostics: Feature importances
    plot_feature_importances(
        rf_model,
        preprocess_model,
        top_n=15,
        save_path=os.path.join(plots_dir, "rf_feature_importances.png"),
    )

    # 8) Week 39 forecast (2011)
    compute_expected_units_for_week(
        rf_preds,
        year=2011,
        week_of_year=39,
        prediction_col=prediction_col,
    )

    # 9) Optional: clean cache
    train_processed_df.unpersist()
    test_processed_df.unpersist()

    spark.stop()


if __name__ == "__main__":
    main()