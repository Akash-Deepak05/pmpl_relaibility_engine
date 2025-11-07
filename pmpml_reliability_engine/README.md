# PMPML Reliability Engine

An end-to-end predictive maintenance dashboard to predict monthly bus breakdown risk and prioritize inspections for the Pune Mahanagar Parivahan Mahamandal Limited (PMPML) fleet.

## Features
- 🔮 Predicts breakdown risk for each bus in the fleet
- 📊 Interactive dashboard with risk prioritization
- 📈 Training insights with precision-recall curves
- 🎯 Adjustable decision thresholds
- 📥 Export high-risk vehicles for action

## Tech Stack
- **Python** - Data processing and ML
- **Pandas & NumPy** - Data manipulation
- **Scikit-learn** - RandomForest classifier
- **Streamlit** - Interactive dashboard
- **Joblib** - Model persistence

## Quick Start

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Akash-Deepak05/pmpl_relaibility_engine.git
   cd pmpl_relaibility_engine/pmpml_reliability_engine
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\activate

   # macOS/Linux
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Usage

#### Option 1: Use Pre-trained Model (Fastest)
The repository includes pre-trained model files, so you can run the dashboard immediately:
```bash
streamlit run app.py
```

#### Option 2: Train from Scratch
To generate fresh data and retrain the model:

1. **Generate mock data**
   ```bash
   python create_mock_data.py
   ```
   This creates `master_data.csv` with 3 years of monthly data for 500 buses.

2. **Train the model**
   ```bash
   python train_model.py
   ```
   This generates:
   - `bus_failure_model.pkl` - Trained RandomForest model
   - `model_features.pkl` - Feature list for consistency
   - `test_eval.csv` - Test set evaluation data

3. **Run the dashboard**
   ```bash
   streamlit run app.py
   ```

## Project Structure
```
pmpml_reliability_engine/
├── app.py                    # Main Streamlit dashboard
├── train_model.py           # Model training script
├── create_mock_data.py      # Mock data generator
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── bus_failure_model.pkl   # Pre-trained model
├── model_features.pkl      # Feature definitions
├── master_data.csv         # Training data (generated)
└── test_eval.csv          # Evaluation metrics (generated)
```

## How It Works

### Data Features
- **Bus characteristics**: Age, type (Diesel/CNG/e-Bus), depot, ownership (PMPML/Private)
- **Operational metrics**: Monthly kilometers run
- **Historical patterns**: Lag features (1-month history, 3-month rolling average)

### Model
- **Algorithm**: Random Forest Classifier with 200 trees
- **Target**: Binary prediction of breakdown in the next month
- **Training**: Time-based split (last 6 months held out for testing)
- **Class balancing**: Handles imbalanced failure rates

### Dashboard Features
1. **Priority List Tab**
   - Sorted list of vehicles by risk score
   - Filter by depot
   - Adjustable risk threshold
   - Export top-N vehicles to CSV
   - Breakdown charts by depot/owner type

2. **Training Insights Tab**
   - Confusion matrix (TP/FP/FN/TN)
   - Precision, Recall, F1 scores
   - Interactive precision-recall curve
   - Threshold adjustment

## Customization

Edit `create_mock_data.py` to customize:
- Fleet size (default: 500 buses)
- Depots and locations
- Bus types and characteristics
- Breakdown probability factors
- Time range (default: 3 years)

## Notes
- The app prioritizes `mock_fleet_status.csv` if present; otherwise, it uses the latest records from `master_data.csv`
- Feature engineering in the app mirrors the training script for consistency
- Private-owned buses and older vehicles show higher failure rates (as per domain knowledge)

## Repository
[https://github.com/Akash-Deepak05/pmpl_relaibility_engine](https://github.com/Akash-Deepak05/pmpl_relaibility_engine)
