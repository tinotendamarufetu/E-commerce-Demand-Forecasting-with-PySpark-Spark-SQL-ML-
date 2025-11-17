# src/demand_forecast/diagnostics.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pyspark.sql.functions as F
from pyspark.sql import DataFrame


def _save_plot(save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"Saved plot → {save_path}")
    plt.show()


def plot_actual_vs_predicted(
    rf_preds: DataFrame,
    stockcode: str = "85123A",
    country: str = "United Kingdom",
    label_col: str = "label",
    prediction_col: str = "prediction",
    save_path: str = "plots/actual_vs_predicted_85123A_UK.png",
):
    """
    Plot Actual vs Predicted for a single (StockCode, Country) pair.
    Mirrors your notebook logic.
    """
    print(f"\nFetching predictions for {stockcode} in {country}...")

    sample_preds_df = (
        rf_preds
        .filter(
            (F.col("StockCode") == stockcode)
            & (F.col("Country") == country)
        )
        .select("Date", label_col, prediction_col)
        .orderBy("Date")
    )

    pd_sample_preds = sample_preds_df.toPandas()

    if pd_sample_preds.empty:
        print("No data found for this StockCode/Country combination.")
        return

    print("Plotting Actual vs. Predicted sales...")

    plt.figure(figsize=(14, 6))
    plt.plot(
        pd_sample_preds["Date"],
        pd_sample_preds[label_col],
        label="Actual Sales",
        marker="o",
        markersize=4,
    )
    plt.plot(
        pd_sample_preds["Date"],
        pd_sample_preds[prediction_col],
        label="Predicted Sales",
        linestyle="--",
        marker="x",
        markersize=4,
    )
    plt.title(f"Actual vs. Predicted Sales for {stockcode} in {country}")
    plt.xlabel("Date")
    plt.ylabel("Total Quantity Sold")
    plt.legend()
    plt.grid(True)

    _save_plot(save_path)


def plot_feature_importances(
    rf_model,
    preprocess_model,
    top_n: int = 15,
    save_path: str = "plots/rf_feature_importances.png",
):
    """
    Extract and plot top N feature importances from the trained Random Forest,
    using the input columns of the fitted VectorAssembler.
    """
    print("\nExtracting and plotting feature importances from Random Forest...")

    importances = rf_model.featureImportances

    # The last stage of your preprocess_pipeline is the VectorAssembler
    assembler_stage = preprocess_model.stages[-1]
    feature_names = assembler_stage.getInputCols()

    feature_imp_df = pd.DataFrame(
        list(zip(feature_names, importances.toArray())),
        columns=["Feature", "Importance"],
    )

    top_features_df = feature_imp_df.sort_values(
        by="Importance", ascending=False
    ).head(top_n)

    plt.figure(figsize=(10, 8))
    sns.barplot(
        data=top_features_df,
        x="Importance",
        y="Feature",
        orient="h",
    )
    plt.title(f"Top {top_n} Feature Importances from Random Forest")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")

    _save_plot(save_path)


def compute_expected_units_for_week(
    rf_preds: DataFrame,
    year: int = 2011,
    week_of_year: int = 39,
    prediction_col: str = "prediction",
) -> int:
    """
    Compute total expected units to be sold in a specific week and year,
    based on predictions (WeekOfYear & Year columns must exist).

    Mirrors your 'Week 39 of 2011' logic.
    """
    print(f"\nCalculating expected units for Year={year}, Week={week_of_year}...")

    df_w = rf_preds.filter(
        (F.col("WeekOfYear") == week_of_year)
        & (F.col("Year") == year)
    )

    total_sales_df = df_w.agg(F.sum(prediction_col).alias("total_sales"))
    total_sales_row = total_sales_df.first()

    if total_sales_row is None or total_sales_row["total_sales"] is None:
        print("No predictions found for that week/year.")
        return 0

    total_sales_value = total_sales_row["total_sales"]
    quantity_sold = int(total_sales_value)

    print(
        f"Total expected units to be sold in Week {week_of_year} of {year}: "
        f"{quantity_sold}"
    )

    return quantity_sold
