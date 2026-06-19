# Customer Behavioral Analytics & Feature Engineering Pipeline

This repository contains an end-to-end data preprocessing and feature engineering pipeline that transforms raw transaction-level e-commerce data into structured, standardized, and dimensionally optimized customer-centric profiles. 

It is designed to handle mass continuous feature extraction—generating over 20 distinct behavioral signals per customer—making the output perfectly tailored for machine learning models, customer segmentation (e.g., K-Means), and lifetime value (CLV) predictions.

## 🚀 Features

*   **Automated Data Ingestion:** Integrated download and extraction of the Online Retail Dataset using `kagglehub`.
*   **Robust Data Cleaning:** Automated filtering of missing identifier tokens (e.g., `CustomerID`), invalid entries, and returns (negative quantities and unit prices).
*   **Mass Feature Engineering (23 Continuous Metrics):**
    *   *Core RFM Metrics:* Recency, Frequency, Monetary value, Total Quantity, and Average Unit Price.
    *   *Temporal Analysis:* Hour of day (Morning, Afternoon, Evening), Day of week (Weekday vs. Weekend), and Quarterly spending behaviors (Q1–Q4).
    *   *Categorical Spending Footprints:* Quantitative tracking of keyword-specific purchases (e.g., 'HEART', 'VINTAGE', 'CHRISTMAS', 'BAG').
*   **Statistical Normalization:** Pipeline-wide integration of `sklearn.preprocessing.StandardScaler` to output zero-mean ($~0$) and unit-variance ($~1$) features.
*   **Dimensionality Reduction:** Built-in Principal Component Analysis (PCA) and scree plotting to evaluate information retention and identify optimal variance thresholds.

## 📊 Pipeline Architecture

1. **Ingest:** Download transaction logs via `kagglehub`.
2. **Clean:** Drop anomalies, calculate isolated pricing matrix (`TotalAmount`), and tokenize date/time markers.
3. **Aggregate & Engineer:** Group records by unique `CustomerID` to establish structural behavioral footprints across 23 continuous feature dimensions.
4. **Scale:** Standardize metrics to eliminate scaling bias across algorithms.
5. **Analyze:** Run PCA to visualize cumulative explained variance thresholds.

## 🛠️ Prerequisites & Installation

To run this notebook, ensure you have Python 3 installed along with the following required libraries:

```bash
pip install kagglehub pandas numpy scikit-learn matplotlib
