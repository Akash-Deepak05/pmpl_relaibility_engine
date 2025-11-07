# PMPML Reliability Engine (Streamlit MVP)

An end-to-end MVP to predict monthly bus breakdown risk and surface a priority list for inspections.

Stack
- Python, Pandas, NumPy
- Scikit-learn (RandomForest)
- Streamlit (dashboard)
- Joblib (model persistence)

Quick start
1) Create project data
```
python create_mock_data.py
```
2) Train the model
```
python train_model.py
```
3) Run the dashboard
```
streamlit run app.py
```

Notes
- The app prefers `mock_fleet_status.csv` if present. If missing, it will compute the latest per-bus records from `master_data.csv` and score them.
- Feature engineering in the app mirrors the training script for consistency.
- You can edit `create_mock_data.py` to adjust fleet size, depots, or failure dynamics.
