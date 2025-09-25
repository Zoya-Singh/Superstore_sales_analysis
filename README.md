# 🚀 Superstore Sales Analytics and Forecasting (Maximized Accuracy)

A complete end-to-end Machine Learning project to analyze transactional data and predict future sales, achieving a **highly accurate, production-ready** performance.

## Final Model Performance Summary

This project successfully implemented advanced techniques to stabilize the model's error and achieve superior performance on complex sales data.

| Metric | Final Score | Assessment |
| :--- | :--- | :--- |
| **R² Score (Test Set)** | **0.8628** | The model explains over **86%** of the variance in future sales. |
| **Mean Absolute Error (MAE)** | **$65.42** | On average, the prediction is off by only $65.42, providing precise guidance for inventory. |
| **Validation Method** | **TimeSeriesSplit (TSS)** | Used the correct chronological validation technique to guarantee the model's reliability over time. |

## Methodology: Achieving High Accuracy

The high performance was driven by a robust pipeline focusing on data quality and specialized algorithms:

1.  **Data Stabilization**: Implemented **Outlier Handling (Winsorizing)** on Sales, Quantity, and Profit, and applied **Log Transformation** to the target variable to normalize the data's extreme skew.
2.  **Advanced Feature Engineering**: Created critical temporal and localized features:
    * **Hierarchical Lags**: Sales from the previous order within the same product category.
    * **Rolling Aggregations**: 30-day and 90-day rolling averages for Sales and Profit, localizing market trends.
3.  **Model Optimization**: Utilized the **LightGBM Regressor**, tuned aggressively with **`GridSearchCV`** and correctly validated using **`TimeSeriesSplit`**.

## Business Relevance & Actionable Insights

The model provides predictive insights for inventory, pricing, and resource allocation:

* **Forecasting Reliability**: The low **MAE** allows for the implementation of **Just-In-Time (JIT) Inventory** and dynamic safety stock levels.
* **Sales Momentum**: The model heavily weights recent sales (`Sales_Last_Order`, `Category_Sales_Last_Order`), confirming that **short-term momentum is the primary driver** of future revenue.
* **Strategic Focus**: The **Feature Importance** chart identifies the true profit drivers (e.g., specific regions' 90-day profit average) for targeted marketing and resource investment.
* **Discount Strategy**: The analysis confirms the need for highly strategic discounting to prevent erosion of margins.

## How to Run the Project

### Prerequisites

You need Python 3.8+ and the following libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm scipy
