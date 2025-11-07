# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from typing import List

st.set_page_config(page_title="PMPML Reliability Engine", layout="wide")
st.title('🚌 PMPML Reliability Engine')
st.markdown('Predictive Maintenance Dashboard to Prioritize At-Risk Vehicles')
st.markdown('---')

# --- 1. Load Model and Features ---
try:
    model = joblib.load('bus_failure_model.pkl')
    FEATURES: List[str] = joblib.load('model_features.pkl')
except FileNotFoundError:
    st.error("Model files not found. Please run `python train_model.py` first.")
    st.stop()

# --- 2. Load data for current predictions ---
# Prefer a provided mock_fleet_status.csv; otherwise derive latest per-bus from master_data.csv
@st.cache_data
def load_fleet_or_latest():
    try:
        df = pd.read_csv('mock_fleet_status.csv')
        df['Date'] = pd.to_datetime(df[['Year','Month']].assign(DAY=1))
        source = 'mock_fleet_status.csv'
        return df, source
    except FileNotFoundError:
        # Fallback to master_data.csv and compute latest row per bus with engineered lags
        m = pd.read_csv('master_data.csv')
        m['Date'] = pd.to_datetime(m['Date'])
        m = m.sort_values(['Bus_ID','Date'])
        
        # Compute the same engineered features used in training
        m_fe = m.copy()
        
        # --- FIX 1: Removed dummification from this function. ---
        # Let the score_fleet function handle dummification consistently.
        # m_fe = pd.get_dummies(m_fe, columns=['Depot_Name','Bus_Type','Owner_Type'], drop_first=True) # <--- REMOVED
        
        m_fe['breakdown_history_lag_1m'] = m.groupby('Bus_ID')['Total_Breakdowns'].shift(1)
        m_fe['kms_history_lag_1m'] = m.groupby('Bus_ID')['KMs_Run_Monthly'].shift(1)
        roll = m.groupby('Bus_ID')['Total_Breakdowns'].shift(1).rolling(3, min_periods=1).mean()
        m_fe['breakdown_history_avg_3m'] = roll.reset_index(level=0, drop=True)
        
        # Take the latest row per bus from original (retain human-readable cols)
        latest_idx = m.groupby('Bus_ID')['Date'].idxmax()
        latest = m.loc[latest_idx].copy().reset_index(drop=True)
        
        # Merge engineered features from m_fe on Bus_ID and Date
        # --- FIX 1 (Continued): Simplified merge to only add lag features ---
        merge_features = ['Bus_ID', 'Date', 'breakdown_history_lag_1m', 'kms_history_lag_1m', 'breakdown_history_avg_3m']
        latest = latest.merge(m_fe[merge_features], on=['Bus_ID', 'Date'], how='left')
        
        source = 'master_data.csv (latest)'
        return latest, source

fleet_df, source_used = load_fleet_or_latest()

# --- 3. Prediction helper ---
@st.cache_data
def score_fleet(fleet_df: pd.DataFrame, features: List[str]):
    # Ensure one-hot encoding columns exist similar to training
    df_pred = fleet_df.copy()
    # If the file wasn't already dummified (mock file case), dummify it
    cat_cols = [c for c in ['Depot_Name','Bus_Type','Owner_Type'] if c in df_pred.columns]
    if cat_cols:
        df_pred = pd.get_dummies(df_pred, columns=cat_cols, drop_first=True)
    # Ensure all model features exist
    for f in features:
        if f not in df_pred.columns:
            df_pred[f] = 0
            
    # --- FIX 2: Create a .copy() to avoid SettingWithCopyWarning ---
    X = df_pred[features].copy()
    
    # --- FIX 3: Robust fillna(0) for ALL features ---
    # This prevents errors if columns other than lags (e.g., KMs_Run_Monthly) have NaNs
    # and fixes the SettingWithCopyWarning.
    X = X.fillna(0)
    
    # The old, brittle loop is replaced:
    # for col in ['breakdown_history_lag_1m','kms_history_lag_1m','breakdown_history_avg_3m']:
    #     if col in X.columns:
    #         X[col] = X[col].fillna(0) # <--- This caused the warning
            
    probs = model.predict_proba(X)[:, 1]
    out = fleet_df.copy()
    out['Failure_Risk_Score'] = probs
    return out.sort_values('Failure_Risk_Score', ascending=False)

risk_df = score_fleet(fleet_df, FEATURES)

# --- 4. Sidebar ---
st.sidebar.header('Model Insights')
try:
    importances = pd.Series(model.feature_importances_, index=FEATURES)
    st.sidebar.bar_chart(importances.sort_values(ascending=False).head(10))
except Exception:
    st.sidebar.write("Feature importances not available for this model type.")

# Threshold slider for predicted positives
threshold = st.sidebar.slider('Decision threshold (positive if risk ≥ threshold)', min_value=0.05, max_value=0.95, value=0.5, step=0.05)
st.sidebar.caption("Adjust to trade precision vs recall. Training tab shows metrics across thresholds.")

st.sidebar.caption("Note: e.g., 'Owner_Type_Private' being high indicates private buses are a key predictor, aligning with reports.")

# Filter by Depot
depot_col = 'Depot_Name' if 'Depot_Name' in risk_df.columns else None
if depot_col:
    depots = ['All'] + sorted([d for d in risk_df[depot_col].dropna().unique().tolist()])
    selected = st.sidebar.selectbox('Filter by Depot', depots)
    display_df = risk_df if selected == 'All' else risk_df[risk_df[depot_col] == selected]
else:
    display_df = risk_df

# Label positives under current threshold
risk_df['Predicted_Positive'] = (risk_df['Failure_Risk_Score'] >= threshold).astype(int)
display_df['Predicted_Positive'] = (display_df['Failure_Risk_Score'] >= threshold).astype(int)

# --- Tabs ---
main_tab, training_tab = st.tabs(["Priority List", "Training Insights"]) 

with main_tab:
    # --- Main content: priority table, breakdown charts, export ---
    st.markdown(f"Data source: {source_used}")

    sub = 'High-Risk Vehicle Priority List'
    if depot_col and selected != 'All':
        sub += f" — {selected}"
    st.subheader(sub)

    cols_to_show = [c for c in ['Bus_ID','Depot_Name','Owner_Type','Bus_Type','Bus_Age_Months','KMs_Run_Monthly','breakdown_history_lag_1m','breakdown_history_avg_3m','Failure_Risk_Score','Predicted_Positive'] if c in display_df.columns]

    tbl_df = display_df[cols_to_show].copy()
    if 'Failure_Risk_Score' in tbl_df.columns:
        tbl_df['Failure_Risk_Score'] = (tbl_df['Failure_Risk_Score'] * 100).round(2)

    st.dataframe(tbl_df, use_container_width=True)

    st.caption("Buses are sorted from highest to lowest predicted failure risk next month. Prioritize inspections accordingly.")

    # Export Top N
    st.subheader("Export Top-N by Risk")
    top_n = st.number_input('Top N', min_value=5, max_value=1000, value=50, step=5)
    top_df = display_df.sort_values('Failure_Risk_Score', ascending=False).head(int(top_n))
    csv = top_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", data=csv, file_name="top_risk_buses.csv", mime="text/csv")

    # Breakdown charts
    st.subheader("Breakdown Charts")
    chart_cols = st.columns(3)

    with chart_cols[0]:
        if depot_col and 'Failure_Risk_Score' in risk_df.columns:
            depot_risk = risk_df.groupby('Depot_Name', dropna=True)['Failure_Risk_Score'].mean().sort_values(ascending=False)
            st.bar_chart(depot_risk, height=280)
            st.caption("Avg risk by depot")

    with chart_cols[1]:
        if 'Owner_Type' in risk_df.columns:
            owner_risk = risk_df.groupby('Owner_Type', dropna=True)['Failure_Risk_Score'].mean().sort_values(ascending=False)
            st.bar_chart(owner_risk, height=280)
            st.caption("Avg risk by owner")

    with chart_cols[2]:
        if depot_col:
            depot_pos = risk_df.groupby('Depot_Name', dropna=True)['Predicted_Positive'].sum().sort_values(ascending=False)
            st.bar_chart(depot_pos, height=280)
            st.caption(f"Count predicted positive (≥ {threshold:.2f}) by depot")

with training_tab:
    st.subheader("Training Insights (held-out last 6 months)")
    try:
        eval_df = pd.read_csv('test_eval.csv')
        from sklearn.metrics import precision_recall_curve, confusion_matrix, precision_score, recall_score, f1_score
        if {'y_true','y_proba'}.issubset(eval_df.columns):
            # Interactive threshold
            thr = st.slider('Evaluation threshold', min_value=0.05, max_value=0.95, value=float(threshold), step=0.05)
            y_true = eval_df['y_true'].astype(int).values
            y_proba = eval_df['y_proba'].astype(float).values
            y_hat = (y_proba >= thr).astype(int)

            # Confusion matrix and metrics
            tn, fp, fn, tp = confusion_matrix(y_true, y_hat).ravel()
            prec = precision_score(y_true, y_hat, zero_division=0)
            rec = recall_score(y_true, y_hat, zero_division=0)
            f1 = f1_score(y_true, y_hat, zero_division=0)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("TP", int(tp))
            m2.metric("FP", int(fp))
            m3.metric("FN", int(fn))
            m4.metric("TN", int(tn))

            st.write(f"Precision: {prec:.3f}  |  Recall: {rec:.3f}  |  F1: {f1:.3f}")

            # PR curve
            precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
            pr_df = pd.DataFrame({'recall': recall, 'precision': precision})
            st.line_chart(pr_df.set_index('recall'))
            st.caption("Precision–Recall curve (higher is better).")
        else:
            st.info("test_eval.csv found but missing required columns 'y_true' and 'y_proba'. Re-run training.")
    except FileNotFoundError:
        st.info("Training evaluation data not found. Re-run `python train_model.py` to regenerate 'test_eval.csv'.")