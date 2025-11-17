# src/demand_forecast/cleaning.py

from pyspark.sql import DataFrame
import pyspark.sql.functions as F


def clean_transactions(df_raw: DataFrame) -> DataFrame:
    """
    Apply all cleaning and preprocessing steps in a single chain.

    - Remove all transactions with zero or negative quantities
    - Remove items that were given away for free (UnitPrice <= 0)
    - Drop rows where the Description is null
    - Drop rows with a null CustomerID
    - Cast InvoiceNo to string
    - Filter out rows where InvoiceNo starts with 'C' (cancelled transactions)

    This is a direct function version of your original notebook code.
    """

    df_cleaned = (
        df_raw
        .filter(F.col("Quantity") > 0)
        .filter(F.col("UnitPrice") > 0)
        .dropna(subset=["Description", "CustomerID"])
        .withColumn("InvoiceNo", F.col("InvoiceNo").cast("string"))
        .filter(~F.col("InvoiceNo").startswith("C"))
    )

    return df_cleaned
