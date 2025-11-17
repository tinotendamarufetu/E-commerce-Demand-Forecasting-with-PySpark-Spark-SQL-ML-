# src/demand_forecast/plotting.py

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pyspark.sql.functions as F
from pyspark.sql import SparkSession, DataFrame


def _save_plot(save_path: str):
    """Helper to save the current matplotlib figure."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"Saved plot → {save_path}")
    plt.show()


# 1. Top Products
def plot_top_products(spark: SparkSession, save_path: str = "plots/top_products.png"):
    print("Generating plot for: Top 10 Products")

    df = spark.sql("""
        SELECT 
            StockCode,
            SUM(TotalQuantity) AS TotalSales
        FROM sales
        GROUP BY StockCode
        ORDER BY TotalSales DESC
        LIMIT 10
    """).toPandas()

    plt.figure(figsize=(10,5))
    sns.barplot(data=df, x="TotalSales", y="StockCode", orient="h")
    plt.title("Top 10 Products by Total Quantity Sold")
    plt.xlabel("Total Quantity Sold")
    plt.ylabel("StockCode")

    _save_plot(save_path)


# 2. Top Countries by Revenue
def plot_top_countries(df_cleaned: DataFrame, save_path: str = "plots/top_countries.png"):
    print("Generating plot for: Top 10 Countries by True Total Revenue")

    df = (
        df_cleaned
        .withColumn("Revenue", F.col("Quantity") * F.col("UnitPrice"))
        .groupBy("Country")
        .agg(F.sum("Revenue").alias("TotalRevenue"))
        .orderBy(F.col("TotalRevenue").desc())
        .limit(10)
        .toPandas()
    )

    plt.figure(figsize=(10,5))
    sns.barplot(data=df, x="TotalRevenue", y="Country", orient="h")
    plt.title("Top 10 Countries by True Total Revenue")
    plt.xlabel("Total Revenue")
    plt.ylabel("Country")

    _save_plot(save_path)


# 3. Weekly Seasonality
def plot_weekly_sales(spark: SparkSession, save_path: str = "plots/weekly_sales.png"):
    print("Generating plot for: Weekly Sales Seasonality")

    df = spark.sql("""
        SELECT 
            DayOfWeek,
            AVG(TotalQuantity) AS AverageSales
        FROM sales
        GROUP BY DayOfWeek
        ORDER BY DayOfWeek
    """).toPandas()

    plt.figure(figsize=(10,4))
    sns.barplot(data=df, x="DayOfWeek", y="AverageSales")
    plt.title("Average Sales by Day of Week")
    plt.xlabel("Day of Week (1=Sunday, 7=Saturday)")
    plt.ylabel("Average Sales Quantity")

    _save_plot(save_path)


# 4. Monthly Seasonality
def plot_monthly_sales(spark: SparkSession, save_path: str = "plots/monthly_sales.png"):
    print("Generating plot for: Monthly Sales Seasonality")

    df = spark.sql("""
        SELECT 
            Month,
            AVG(TotalQuantity) AS AverageSales
        FROM sales
        GROUP BY Month
        ORDER BY Month
    """).toPandas()

    plt.figure(figsize=(10,4))
    sns.barplot(data=df, x="Month", y="AverageSales")
    plt.title("Average Sales by Month")
    plt.xlabel("Month")
    plt.ylabel("Average Sales Quantity")

    _save_plot(save_path)

def plot_global_daily_quantity(df_final: DataFrame, save_path="plots/global_daily_quantity.png"):
    print("Generating: Global Daily Quantity Trend")

    pdf = (
        df_final.groupBy("Date")
        .sum("TotalQuantity")
        .orderBy("Date")
        .withColumnRenamed("sum(TotalQuantity)", "TotalQuantity")
        .toPandas()
    )

    plt.figure(figsize=(14,5))
    sns.lineplot(data=pdf, x="Date", y="TotalQuantity")
    plt.title("Global Total Quantity Sold Per Day")
    plt.xlabel("Date")
    plt.ylabel("Total Quantity Sold")

    _save_plot(save_path)

def plot_top_product_trends(df_final: DataFrame, top_n=5, save_path="plots/top_product_trends.png"):
    print(f"Generating: Top {top_n} Product Trends")

    # Top products by total quantity
    top_products = (
        df_final.groupBy("StockCode")
        .sum("TotalQuantity")
        .orderBy(F.col("sum(TotalQuantity)").desc())
        .limit(top_n)
        .toPandas()["StockCode"]
        .tolist()
    )

    pdf = (
        df_final.filter(F.col("StockCode").isin(top_products))
        .groupBy("Date", "StockCode")
        .sum("TotalQuantity")
        .orderBy("Date")
        .toPandas()
    )

    plt.figure(figsize=(14,5))
    sns.lineplot(data=pdf, x="Date", y="sum(TotalQuantity)", hue="StockCode")
    plt.title(f"Daily Sales Trends for Top {top_n} Products")
    plt.ylabel("Quantity Sold")
    plt.xlabel("Date")

    _save_plot(save_path)

def plot_top_country_trends(df_final: DataFrame, top_n=5, save_path="plots/top_country_trends.png"):
    print(f"Generating: Top {top_n} Country Trends")

    top_countries = (
        df_final.groupBy("Country")
        .sum("TotalQuantity")
        .orderBy(F.col("sum(TotalQuantity)").desc())
        .limit(top_n)
        .toPandas()["Country"]
        .tolist()
    )

    pdf = (
        df_final.filter(F.col("Country").isin(top_countries))
        .groupBy("Date", "Country")
        .sum("TotalQuantity")
        .orderBy("Date")
        .toPandas()
    )

    plt.figure(figsize=(14,5))
    sns.lineplot(data=pdf, x="Date", y="sum(TotalQuantity)", hue="Country")
    plt.title(f"Daily Sales Trends for Top {top_n} Countries")
    plt.ylabel("Quantity Sold")
    plt.xlabel("Date")

    _save_plot(save_path)


def plot_demand_volatility(df_final: DataFrame, save_path="plots/demand_volatility.png"):
    print("Generating: Demand Volatility Distribution")

    pdf = (
        df_final.groupBy("StockCode")
        .agg(F.stddev("TotalQuantity").alias("volatility"))
        .orderBy("volatility", ascending=False)
        .toPandas()
    )

    plt.figure(figsize=(12,4))
    sns.histplot(pdf["volatility"], bins=50, kde=True)
    plt.title("Distribution of Product Demand Volatility")
    plt.xlabel("Volatility (Std Dev)")
    plt.ylabel("Number of Products")

    _save_plot(save_path)


def plot_global_rolling_avg(df_final: DataFrame, save_path="plots/global_rolling_avg.png"):
    print("Generating: Global 7-Day Rolling Average Demand")

    pdf = (
        df_final.groupBy("Date")
        .sum("TotalQuantity")
        .orderBy("Date")
        .toPandas()
        .rename(columns={"sum(TotalQuantity)": "TotalQuantity"})
    )

    pdf["Rolling7"] = pdf["TotalQuantity"].rolling(window=7).mean()

    plt.figure(figsize=(14,5))
    sns.lineplot(data=pdf, x="Date", y="TotalQuantity", label="Daily Total")
    sns.lineplot(data=pdf, x="Date", y="Rolling7", label="7-Day Rolling Avg")
    plt.title("Global 7-Day Rolling Average of Total Demand")

    _save_plot(save_path)


