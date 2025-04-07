import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Load preprocessed data
X_train = pd.read_pickle("X_train.pkl")
X_test = pd.read_pickle("X_test.pkl")
y_train = pd.read_pickle("y_train.pkl")
y_test = pd.read_pickle("y_test.pkl")

# Combine for feature engineering
X_full = pd.concat([X_train, X_test], axis=0)
y_full = pd.concat([y_train, y_test], axis=0)

# Create new features
X_full['rm_squared'] = X_full['rm'] ** 2
X_full['crim_lstat'] = X_full['crim'] * X_full['lstat']
X_full['dis_rad'] = X_full['dis'] * X_full['rad']

# Split back
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42
)

# Baseline
lr_baseline = LinearRegression()
lr_baseline.fit(X_train, y_train)
y_pred_baseline = lr_baseline.predict(X_test)
mse_baseline = mean_squared_error(y_test, y_pred_baseline)
r2_baseline = r2_score(y_test, y_pred_baseline)

# Model 1: Original + rm_squared
features_1 = ['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat', 'rm_squared']
lr_model_1 = LinearRegression()
lr_model_1.fit(X_train_new[features_1], y_train_new)
y_pred_1 = lr_model_1.predict(X_test_new[features_1])
mse_1 = mean_squared_error(y_test_new, y_pred_1)
r2_1 = r2_score(y_test_new, y_pred_1)

# Model 2: Original + rm_squared + crim_lstat
features_2 = features_1 + ['crim_lstat']
lr_model_2 = LinearRegression()
lr_model_2.fit(X_train_new[features_2], y_train_new)
y_pred_2 = lr_model_2.predict(X_test_new[features_2])
mse_2 = mean_squared_error(y_test_new, y_pred_2)
r2_2 = r2_score(y_test_new, y_pred_2)

# Model 3: All new features
features_3 = features_2 + ['dis_rad']
lr_model_3 = LinearRegression()
lr_model_3.fit(X_train_new[features_3], y_train_new)
y_pred_3 = lr_model_3.predict(X_test_new[features_3])
mse_3 = mean_squared_error(y_test_new, y_pred_3)
r2_3 = r2_score(y_test_new, y_pred_3)

# Print results
print("\nModel Performance Comparison:")
print(f"Baseline (Original Features): MSE = {mse_baseline:.2f}, R-squared = {r2_baseline:.2f}")
print(f"Model 1 (Original + rm_squared): MSE = {mse_1:.2f}, R-squared = {r2_1:.2f}")
print(f"Model 2 (Original + rm_squared + crim_lstat): MSE = {mse_2:.2f}, R-squared = {r2_2:.2f}")
print(f"Model 3 (All New Features): MSE = {mse_3:.2f}, R-squared = {r2_3:.2f}")