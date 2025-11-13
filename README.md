# 🛒 E-commerce Demand Forecasting with PySpark (Spark SQL & ML)
### *End-to-End Scalable Forecasting Pipeline using Spark SQL & MLlib*

This project builds a **production-style demand forecasting system** for an e-commerce retailer using **Apache Spark (PySpark)**. It covers ingestion, cleaning, feature engineering, model building, debugging target leakage, and solving the “bully feature” problem to create an **honest and realistic forecasting model**.

---

## 🚀 Project Goal & Business Context

As a data scientist supporting the **Sales & Operations Planning (S&OP)** team at a multinational e-commerce company, your job is to improve the accuracy of demand planning during the critical end-of-year period.

### **Business Problem**
- ❗ Uncertain demand leads to poor inventory decisions  
- 📦 Overstocking → lost cash, storage, markdowns  
- 😡 Understocking → lost sales, angry customers  

### **Solution**
Build a **daily product-level demand forecast** using PySpark machine learning and engineered features— enabling smarter decisions around:
- Inventory  
- Logistics  
- Promotions  
- Cash flow planning  

---

## 🛠️ Tech Stack

| Layer | Tools |
|------|-------|
| **Core Engine** | Apache Spark (PySpark) |
| **Data Prep** | PySpark DataFrame API, Window Functions |
| **Analytics** | Spark SQL |
| **Machine Learning** | PySpark MLlib (Pipeline, VectorAssembler, RandomForestRegressor, etc.) |
| **Visualization** | Matplotlib, Seaborn |
| **Environment** | PyCharm, Jupyter Notebook |

---

## 📁 Dataset Overview

Using the **Online Retail Dataset**, which contains real e-commerce transactions for a UK retailer.

| Column | Description |
|--------|-------------|
| InvoiceNo | Unique transaction ID |
| StockCode | Unique product identifier |
| Description | Product name |
| Quantity | Units sold |
| UnitPrice | Price per unit |
| CustomerID | Unique customer ID |
| Country | Customer location |
| InvoiceDate | Transaction timestamp |

---

# 📈 Project Workflow

## **Milestone 1: Data Ingestion & Spark Setup**
- Configured SparkSession (`spark.driver.memory`, `maxResultSize`)
- Loaded dataset using `spark.read.csv()`
- Inspected schema with `.printSchema()`

---

## **Milestone 2: Data Cleaning & Preprocessing**
- Converted `InvoiceDate` → timestamp  
- Removed negative/zero quantity (cancellations/returns)  
- Dropped rows missing `CustomerID` or `Description`

---

## **Milestone 3: Feature Engineering (Critical Step)**

Raw transactional logs → **Daily time-series panel**

### ✔ Aggregated Features  
Grouped by **Date + StockCode + Country**

Created:
- `TotalQuantity` (target variable)
- Calendar features:  
  `DayOfWeek`, `Month`, `IsWeekend`, `IsMonthEnd`
- Lag features:  
  `TotalQuantity_lag_1`, `TotalQuantity_lag_7`, …
- Rolling features using **Window Functions**  
  `Qty_moving_avg_7`, `Qty_moving_avg_30`, volatility metrics

### ⚠️ Solving Target Leakage  
Originally features used same-day stats → model cheated.

**Fix:**  
Only lagged & historical features allowed.

---

## **Milestone 4: Exploratory Data Analysis**
Analyzed patterns using **Spark SQL**:

- 🔝 Top products & top revenue countries  
  <img width="881" height="473" alt="image" src="https://github.com/user-attachments/assets/fc5cd141-576f-486e-92d3-4560e5a59d6f" />

  <img width="937" height="473" alt="image" src="https://github.com/user-attachments/assets/1bbdec47-434d-4c07-ab8b-97e3473081ef" />

- 📆 Monthly & weekly seasonality  
  <img width="845" height="396" alt="image" src="https://github.com/user-attachments/assets/46c45881-e694-42f2-93e5-4dc11db25d43" />

  <img width="845" height="396" alt="image" src="https://github.com/user-attachments/assets/541f04ac-9ada-4af8-8e23-dcd67af26df1" />

- 🔗 Feature correlations (e.g., strong lag relationships)

---

## **Milestone 5: Model Building & Hyperparameter Tuning**

### Model Pipeline
- `StringIndexer` for `StockCode` and `Country`
- `VectorAssembler` for features
- Time-aware train/test split (pre-2011-09-25 for training)

### Models Tested
- **DecisionTreeRegressor** (baseline)
- **RandomForestRegressor**

### ⚠️ Solving the “Bully Feature Problem”
`Qty_moving_avg_7` was dominating predictions.

**Fix:**  
Removed dominant moving averages → forced the model to actually *learn* from multiple features.

---

## **Milestone 6: Final Model & Results**

### ✔ Best Model  
**RandomForestRegressor**  
- 50 trees  
- Max depth = 8  
- No bully features  

### 📊 Final Performance
- **R²:** 0.47  
- **MAE:** ~7.9  
- **MAPE:** ~170% (ignored because of low-volume days)  

### 📉 Prediction Tracking  
The final model successfully captures sales spikes and volatility.  
<img width="1164" height="550" alt="image" src="https://github.com/user-attachments/assets/a095ac5d-6018-4652-b3fd-96a7727629f3" />


### ⭐ Feature Importance Highlights
1. Recent Trend (Qty_moving_avg_7)  
2. Product ID (`StockCode_idx`)  
3. Long-Term Trend (Qty_moving_avg_30)  
4. Volatility (Qty_volatility_7)  
5. Yesterday’s Sales (TotalQuantity_lag_1)  

<img width="1027" height="704" alt="image" src="https://github.com/user-attachments/assets/cc2a3206-778f-4b70-b107-fa17b3ef36c3" />


---

# 🖥️ How to Run the Project

### **1. Clone the Repository**
```bash
git clone https://github.com/Your-Username/Your-Repo-Name.git
cd Your-Repo-Name
```

### **2. Install Dependencies**
```bash
pip install pyspark pandas matplotlib seaborn jupyterlab
```

### **3. Windows Spark Setup**
Requirements:
- Java JDK 17
- Hadoop `winutils.exe` in `C:\hadoop\bin`

Environment variables:
```
JAVA_HOME = C:\Program Files\Java\jdk-17
HADOOP_HOME = C:\hadoop
```

Add to PATH:
```
%JAVA_HOME%\bin
%HADOOP_HOME%\bin
```

---

### **4. Run the Notebook**
```bash
jupyter-lab
```
Open **project_notebook.ipynb** and run all cells.

---

## ⭐ Future Improvements
- Add Prophet/LSTM comparison  
- Build a Spark Streaming version  
- Deploy model with MLflow + Databricks  
- Create a dashboard (Tableau/PowerBI) for demand insights  

---
