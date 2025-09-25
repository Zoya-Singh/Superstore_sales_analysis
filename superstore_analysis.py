# Section 1: Data Loading, Preprocessing, and Outlier Handling

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.impute import SimpleImputer
from scipy.stats.mstats import winsorize
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import matplotlib 
matplotlib.use('TkAgg') # FIX: Explicitly set the backend for consistent display in terminals

print("--- Starting Data Preprocessing and Outlier Handling ---")

# Define the expected file name based on your input
file_name = 'superstore_sales.csv'

# Load the dataset
try:
    df = pd.read_csv(file_name, encoding='latin1')
except FileNotFoundError:
    print(f"Error: The file '{file_name}' was not found.")
    print(f"Please ensure the file is named '{file_name}' and is in the same directory as this script.")
    exit()

## --- INSTRUCTIONS FOR NEW FILES: UNCOMMENT AND EDIT THIS BLOCK ---
# If you load a new store's CSV, you must map its column names to the names
# expected by the rest of the pipeline (e.g., map 'Transaction_Value' to 'Sales').

# column_map = {
#     'New_Store_Sales_Column': 'Sales',
#     'New_Store_Order_Date_Column': 'Order Date',
#     'New_Store_Ship_Date_Column': 'Ship Date',
#     'New_Store_Product_Category': 'Category',
#     'New_Store_Discount_Rate': 'Discount',
#     # ... map all relevant columns here ...
# }
# df.rename(columns=column_map, inplace=True)
# ------------------------------------------------------------------

# Drop irrelevant columns
df = df.drop(columns=['Row ID', 'Order ID', 'Customer ID', 'Customer Name', 'Product ID', 'Ship Mode', 'Postal Code'])

# Fix the date parsing issue: The dates are in MM/DD/YYYY format.
df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Ship Date'] = pd.to_datetime(df['Ship Date'])

# --- Outlier Handling (Winsorizing) ---
skewed_cols = ['Sales', 'Quantity', 'Profit']
for col in skewed_cols:
    df[col] = winsorize(df[col], limits=[0.01, 0.01])

# --- Impute Missing Values (for production readiness) ---
numerical_cols = df.select_dtypes(include=np.number).columns
categorical_cols_to_encode = ['Country', 'City', 'State', 'Segment', 'Region', 'Category', 'Sub-Category']

num_imputer = SimpleImputer(strategy='mean')
df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])

cat_imputer = SimpleImputer(strategy='most_frequent')
df[categorical_cols_to_encode] = cat_imputer.fit_transform(df[categorical_cols_to_encode])

# Encode categorical features
for col in categorical_cols_to_encode:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

print("\nData preprocessing complete.")


# Section 2: Exploratory Data Analysis (EDA)

print("\n--- Starting Exploratory Data Analysis (EDA) ---")
sns.set(style="whitegrid")
plt.style.use('seaborn-v0_8-darkgrid')

# Sales Trend Over Time
df['Year-Month'] = df['Order Date'].dt.to_period('M')
monthly_sales = df.groupby('Year-Month')['Sales'].sum().reset_index()
monthly_sales['Year-Month'] = monthly_sales['Year-Month'].astype(str)

plt.figure(figsize=(15, 6))
sns.lineplot(data=monthly_sales, x='Year-Month', y='Sales', marker='o', color='b')
plt.title('Monthly Sales Trend', fontsize=16)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Total Sales', fontsize=12)
plt.xticks(rotation=90, fontsize=8)
plt.tight_layout()
plt.show()

# Profit and Sales by Category
category_data = df.groupby('Category')[['Sales', 'Profit']].sum().reset_index()
category_data.set_index('Category').plot(kind='bar', figsize=(10, 6), color=['skyblue', 'salmon'])
plt.title('Total Sales and Profit by Product Category', fontsize=16)
plt.xlabel('Category', fontsize=12)
plt.ylabel('Amount', fontsize=12)
plt.xticks(rotation=0)
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()


# Section 3: Feature Engineering and Data Preparation (Advanced Global Signals)

print("\n--- Starting Feature Engineering (Final Set) ---")

# Sort the data by date for correct time-series calculations
df.sort_values(by='Order Date', inplace=True)

# Create new meaningful features
df['Shipping_Duration'] = (df['Ship Date'] - df['Order Date']).dt.days
df['Order_Year'] = df['Order Date'].dt.year
df['Order_Month'] = df['Order Date'].dt.month
df['Order_Quarter'] = df['Order Date'].dt.quarter
df['Order_Day_of_Week'] = df['Order Date'].dt.dayofweek

# --- Global Aggregation Features ---
avg_sub_cat_sales = df.groupby('Sub-Category')['Sales'].mean()
df['Avg_Sales_SubCategory_Total'] = df['Sub-Category'].map(avg_sub_cat_sales)

segment_order_count = df.groupby('Segment')['Sales'].count()
df['Segment_Order_Count'] = df['Segment'].map(segment_order_count)

# --- Time-Series Lag and Rolling Features ---
df['Sales_Last_Order'] = df['Sales'].shift(1)
df['Category_Sales_Last_Order'] = df.groupby('Category')['Sales'].shift(1)

ROLLING_WINDOW_30 = 30
ROLLING_WINDOW_90 = 90
df['Sales_30_Day_Avg'] = df['Sales'].rolling(window=ROLLING_WINDOW_30).mean().shift(1)
df['Category_Sales_90D_Avg'] = df.groupby('Category')['Sales'].transform(
    lambda x: x.rolling(window=ROLLING_WINDOW_90).mean().shift(1)
)
df['Region_Profit_90D_Avg'] = df.groupby('Region')['Profit'].transform(
    lambda x: x.rolling(window=ROLLING_WINDOW_90).mean().shift(1)
)

# Final cleanup of NaN values created by shift/rolling
df.fillna(0, inplace=True) 

# Drop original date columns
df = df.drop(columns=['Order Date', 'Ship Date'])

# Define features (X) and target (y) for sales forecasting
features = ['Category', 'Sub-Category', 'Quantity', 'Discount', 'Profit',
            'Shipping_Duration', 'Order_Year', 'Order_Month', 'Order_Quarter',
            'Order_Day_of_Week', 'Sales_Last_Order',
            'Category_Sales_Last_Order', 'Sales_30_Day_Avg',
            'Avg_Sales_SubCategory_Total', 'Segment_Order_Count',
            'Category_Sales_90D_Avg', 'Region_Profit_90D_Avg']
target = 'Sales'
X = df[features]
y = df[target]

# --- FINAL ACCURACY BOOST: Target Variable Transformation ---
y_transformed = np.log1p(y)

# --- Scaling (Ensuring robustness by converting to float64) ---
numerical_features = X.select_dtypes(include=np.number).columns
X_to_scale = X.loc[:, numerical_features].astype('float64') 
X.loc[:, numerical_features] = StandardScaler().fit_transform(X_to_scale)

print("Feature engineering complete. Data is ready for modeling.")


# Section 4: Modeling and Evaluation (with LightGBM Hyperparameter Tuning)

print("\n--- Starting Model Training and Hyperparameter Tuning (LightGBM with TimeSeriesSplit) ---")

# Split the data, using the transformed target
X_train, X_test, y_train, y_test = train_test_split(X, y_transformed, test_size=0.2, random_state=42, shuffle=False)

# Initialize TimeSeriesSplit for proper cross-validation
tscv = TimeSeriesSplit(n_splits=5) 

# Hyperparameter Tuning
param_grid = {
    'n_estimators': [300, 500],
    'learning_rate': [0.05],
    'max_depth': [7, 10],
    'num_leaves': [31, 63],
    'subsample': [0.7],
    'reg_alpha': [0.1],
}

lgbm_reg = lgb.LGBMRegressor(objective='regression', random_state=42, n_jobs=-1, verbose=-1)

grid_search = GridSearchCV(estimator=lgbm_reg, param_grid=param_grid, cv=tscv, scoring='r2', n_jobs=-1, verbose=1)

grid_search.fit(X_train, y_train)

print(f"\nBest Hyperparameters found: {grid_search.best_params_}")
print(f"Best R² Score found during tuning (TimeSeriesSplit): {grid_search.best_score_:.4f}")

final_lgbm_reg = grid_search.best_estimator_
y_pred_lgbm_log = final_lgbm_reg.predict(X_test)

# --- FINAL STEP: Inverse Transform Predictions ---
y_test_original = np.expm1(y_test)
y_pred_lgbm_original = np.expm1(y_pred_lgbm_log)

r2_lgbm = r2_score(y_test_original, y_pred_lgbm_original)
mae_lgbm = mean_absolute_error(y_test_original, y_pred_lgbm_original)
rmse_lgbm = np.sqrt(mean_squared_error(y_test_original, y_pred_lgbm_original))

print("\n--- Final Tuned LightGBM Regressor Performance (Log Transformed) ---")
print(f"R² Score (Test Set): {r2_lgbm:.4f}")
print(f"Mean Absolute Error (MAE): {mae_lgbm:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse_lgbm:.4f}")

print("\nModel training and evaluation complete.")


# Section 5: Visualization of Results and Insights-

print("\n--- Generating Final Visualizations and Insights ---")

# Plot Actual vs. Predicted values for the final, tuned LGBM model
plt.figure(figsize=(12, 6))
plt.scatter(y_test_original, y_pred_lgbm_original, alpha=0.5)
plt.plot([y_test_original.min(), y_test_original.max()], [y_test_original.min(), y_test_original.max()], 'r--', lw=2)
plt.title('Tuned LightGBM: Actual vs. Predicted Sales (Inverse Transformed)', fontsize=16)
plt.xlabel('Actual Sales', fontsize=12)
plt.ylabel('Predicted Sales', fontsize=12)
plt.show()

# Feature Importances from the final, tuned LGBM model
feature_importances = pd.Series(final_lgbm_reg.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(12, 8))
sns.barplot(x=feature_importances, y=feature_importances.index, palette='crest')
plt.title('Feature Importances (Tuned LightGBM)', fontsize=16)
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.tight_layout()
plt.show()


