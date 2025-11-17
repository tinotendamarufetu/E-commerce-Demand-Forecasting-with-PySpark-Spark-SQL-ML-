from pyspark.sql import DataFrame, Window
import pyspark.sql.functions as F

def aggregate_to_daily_panel(df_cleaned: DataFrame) -> DataFrame:
    """
    Aggregate transactional data to a daily panel by (Date, StockCode, Country).
    """
    print("Starting aggregation from transactional data...")

    df_aggregated = (
        df_cleaned
        # Create Date column (date only)
        .withColumn("Date", F.to_date("InvoiceDate"))
        # Aggregate
        .groupBy("Date", "StockCode", "Country")
        .agg(
            F.sum("Quantity").alias("TotalQuantity"),
            F.avg("UnitPrice").alias("AvgPrice"),
            F.countDistinct("InvoiceNo").alias("NumTransactions"),
            F.avg("Quantity").alias("AvgQtyPerTransaction")
        )
    )

    print("Aggregation complete.")
    return df_aggregated


def add_calendar_features(df: DataFrame) -> DataFrame:
    """
    Add calendar/time features (DayOfWeek, Month, etc.).
    """
    df = (
        df
        .withColumn("DayOfWeek", F.dayofweek("Date"))
        .withColumn("DayOfMonth", F.dayofmonth("Date"))
        .withColumn("WeekOfYear", F.weekofyear("Date"))
        .withColumn("Month", F.month("Date"))
        .withColumn("Quarter", F.quarter("Date"))
        .withColumn("Year", F.year("Date"))
        .withColumn("IsWeekend", F.when(F.dayofweek(F.col("Date")).isin([1, 7]), 1).otherwise(0))
        .withColumn("IsMonthStart", F.when(F.dayofmonth(F.col("Date")) == 1, 1).otherwise(0))
        .withColumn("IsMonthEnd", F.when(F.dayofmonth(F.col("Date")) == F.dayofmonth(F.last_day(F.col("Date"))), 1).otherwise(0))
        # You can add MonthStart / MonthEnd logic here if you did it
    )
    print("Calendar features added.")
    return df


def add_lag_and_rolling_features(df_with_calendar: DataFrame) -> DataFrame:
    """
    Create Lag, Rolling & Price Features

    This is the function version of your exact notebook code:

    - Window partition: per (StockCode, Country) ordered by Date
    - Lag features:
        TotalQuantity_lag_1, TotalQuantity_lag_7, TotalQuantity_lag_30
        AvgPrice_lag_1, NumTransactions_lag_1, AvgQtyPerTransaction_lag_1
    - Rolling features:
        Qty_moving_avg_7, Qty_moving_avg_30, Qty_volatility_7
    """

    print("Defining window specifications...")
    # This is our "key" for all window functions.
    # We want to calculate features *per item* and *per country*, ordered by date.
    base_window_spec = Window.partitionBy("StockCode", "Country").orderBy("Date")

    # Window spec for 7-day rolling features (i.e., last 7 sales days)
    window_roll_7d = base_window_spec.rowsBetween(-6, Window.currentRow)

    # Window spec for 30-day rolling features (i.e., last 30 sales days)
    window_roll_30d = base_window_spec.rowsBetween(-29, Window.currentRow)

    # Features represent what the model actually knows
    # (i.e., "what was the average price yesterday?")
    print("Adding lag and rolling features...")
    df = (
        df_with_calendar
        .withColumn("TotalQuantity_lag_1", F.lag("TotalQuantity", 1).over(base_window_spec))
        .withColumn("TotalQuantity_lag_7", F.lag("TotalQuantity", 7).over(base_window_spec))
        .withColumn("TotalQuantity_lag_30", F.lag("TotalQuantity", 30).over(base_window_spec))
        .withColumn("Qty_moving_avg_7", F.avg("TotalQuantity").over(window_roll_7d))
        .withColumn("Qty_moving_avg_30", F.avg("TotalQuantity").over(window_roll_30d))
        .withColumn("Qty_volatility_7", F.stddev("TotalQuantity").over(window_roll_7d))
        .withColumn("AvgPrice_lag_1", F.lag("AvgPrice", 1).over(base_window_spec))
        .withColumn("NumTransactions_lag_1", F.lag("NumTransactions", 1).over(base_window_spec))
        .withColumn("AvgQtyPerTransaction_lag_1", F.lag("AvgQtyPerTransaction", 1).over(base_window_spec),
        )
    )

    print("Lag and rolling features added.")
    return df


def finalize_features(df_features: DataFrame) -> DataFrame:
    """
    Create Price Change Feature & Fill Nulls

    This is the direct function version of your notebook block:

        df_final_features = df_features \
            .withColumn("Price_change_pct_lag_1", ...) \
            .drop("AvgPrice", "NumTransactions", "AvgQtyPerTransaction") \
            .na.fill(0)

    - Adds Price_change_pct_lag_1
    - Drops intermediate raw columns
    - Fills nulls from lag/rolling with 0
    """

    print("Calculating final features and cleaning up...")

    base_window_spec = Window.partitionBy("StockCode", "Country").orderBy("Date")

    df = (
        df_features
        .withColumn(
            "Price_change_pct_lag_1",
            (
                F.col("AvgPrice_lag_1")
                - F.lag("AvgPrice_lag_1", 1).over(base_window_spec)
            )
            / F.lag("AvgPrice_lag_1", 1).over(base_window_spec),
        )
        .drop("AvgPrice", "NumTransactions", "AvgQtyPerTransaction")
        .na.fill(0)  # Fill all nulls created by lags/rolling windows
    )

    print("Feature Engineering complete!")
    return df