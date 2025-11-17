# E-commerce Demand Forecasting with PySpark (Spark SQL & ML)

This repo contains a **production-style PySpark project** that forecasts daily product demand for an e-commerce retailer using the classic **Online Retail** dataset.

The goal is to move beyond "just notebooks" into a **software-engineering style ML project** with:

- Clear folder structure
- Reusable modules under `src/`
- Config-driven experiments (`configs/*.yaml`)
- CLI scripts under `scripts/`
- Model artifacts in `models/`

---

## 🚀 Project Goal & Business Context

You are a data scientist for the **Sales & Operations Planning (S&OP)** team at a multinational e-commerce company.

- **Problem:** Uncertain demand leads to poor inventory planning.  
  - Overstock → cash tied in inventory, markdowns  
  - Understock → lost sales, angry customers  
- **Solution:** Build a demand forecasting model to predict **daily quantities sold** by product and country.

This project forecasts **daily `TotalQuantity`** for each `(StockCode, Country)` combination.

---

## 🧱 Project Structure

```bash
ecommerce-demand-forecast-pyspark/
├── README.md
├── requirements.txt           # Python dependencies
├── .gitignore
├── configs/
│   ├── spark.yaml             # Spark session config (master, memory, etc.)
│   ├── data.yaml              # Data paths and column names
│   └── model.yaml             # Pipeline, model, and training config
├── data/
│   ├── raw/                   # Raw source data (e.g., Online Retail.csv)
│   ├── interim/               # Optional intermediate tables
│   └── processed/             # Final ML-ready tables
├── notebooks/
│   └── spark-demand-forecast.ipynb   # Exploration & scratch work
├── src/
│   └── demand_forecast/
│       ├── __init__.py
│       ├── spark_session.py          # SparkSession factory (config-driven)
│       ├── data_ingestion.py         # Load raw data
│       ├── cleaning.py               # Data cleaning rules
│       ├── features.py               # Aggregation & feature engineering
│       ├── modeling.py               # Pipeline + model training
│       ├── evaluation.py             # Metrics (RMSE, MAE, R2, SMAPE)
│       ├── plotting.py               # EDA & diagnostic plots
│       └── io_utils.py               # Model saving helpers
├── scripts/
│   ├── train_model.py                # Full training + evaluation + save
│   └── run_eda.py                    # Quick EDA plots
├── models/
│   └── rf_baseline/                  # Saved Spark PipelineModel
└── tests/
    └── ...                           # (Optional) unit tests
