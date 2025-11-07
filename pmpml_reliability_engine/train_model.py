# train_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

print("Starting model training...")

# --- 1. Load Data ---
try:
    df = pd.read_csv('master_data.csv')
except FileNotFoundError:
    print("Error: master_data.csv not found.\nPlease run create_mock_data.py first to generate the data.")
    raise SystemExit(1)

# Parse and sort by time
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by=['Bus_ID', 'Date']).reset_index(drop=True)
print(f"Data loaded. {len(df)} rows.")

# --- 2. Feature Engineering ---
# One-hot encode categorical variables
cat_cols = ['Depot_Name', 'Bus_Type', 'Owner_Type']
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# Lag features by bus
df['breakdown_history_lag_1m'] = df.groupby('Bus_ID')['Total_Breakdowns'].shift(1)
df['kms_history_lag_1m'] = df.groupby('Bus_ID')['KMs_Run_Monthly'].shift(1)
# Rolling average of past 3 months breakdowns (excluding current month)
rolling_series = df.groupby('Bus_ID')['Total_Breakdowns'].shift(1).rolling(3, min_periods=1).mean()
df['breakdown_history_avg_3m'] = rolling_series.reset_index(level=0, drop=True)

# --- 3. Target: Will the bus break down next month? ---
df['target_breakdown_next_month'] = df.groupby('Bus_ID')['Total_Breakdowns'].shift(-1)
df['target_will_fail_next_month'] = (df['target_breakdown_next_month'] > 0).astype(int)

# --- 4. Prepare modeling frame ---
# Define feature columns programmatically to keep app consistent
base_features = [
    'Bus_Age_Months', 'KMs_Run_Monthly',
    'breakdown_history_lag_1m', 'kms_history_lag_1m', 'breakdown_history_avg_3m'
]
# Include dummy columns
dummy_features = [c for c in df.columns if c.startswith('Depot_Name_') or c.startswith('Bus_Type_') or c.startswith('Owner_Type_')]
FEATURES = base_features + dummy_features
TARGET = 'target_will_fail_next_month'

# Drop rows with missing values due to shifting
df_model = df.dropna(subset=FEATURES + ['Date', TARGET]).copy()
X = df_model[FEATURES]
y = df_model[TARGET]

# --- 5. Time-based split (last 6 months for testing) ---
split_date = df_model['Date'].max() - pd.DateOffset(months=6)
train_mask = df_model['Date'] < split_date
test_mask = ~train_mask
X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[test_mask], y[test_mask]
print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples.")

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)
print("Model training complete.")

# --- 6. Evaluate ---
y_pred = model.predict(X_test)
# Also get probabilities for analysis
y_proba = model.predict_proba(X_test)[:, 1]
print("\n--- Model Evaluation (Test Set) ---")
print(classification_report(y_test, y_pred, digits=3))

# Persist evaluation payload for the app's Training Insights page
try:
    eval_df = df_model.loc[test_mask, ['Bus_ID', 'Date']].copy()
    eval_df['y_true'] = y_test.values
    eval_df['y_proba'] = y_proba
    eval_df.to_csv('test_eval.csv', index=False)
except Exception:
    pass

# --- 7. Save artifacts ---
joblib.dump(model, 'bus_failure_model.pkl')
joblib.dump(FEATURES, 'model_features.pkl')
print("Model and features saved to 'bus_failure_model.pkl' and 'model_features.pkl'")
print("Saved test evaluation data to 'test_eval.csv'")
