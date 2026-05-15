# 🏎️ Formula Racing Telemetry Intelligence System

An AI-powered motorsport analytics platform for analyzing Formula 1 telemetry, predicting lap performance, estimating tire degradation, and generating pit strategy recommendations.

---

# 📊 Features

- Formula 1 telemetry collection using FastF1
- Multi-session race data pipeline
- Automated preprocessing and feature engineering
- Next-lap time prediction using machine learning
- Tire degradation and race trend analysis
- Pit strategy recommendation system
- Interactive F1-style Streamlit dashboard
- Data visualization with Plotly

---

# 🧠 Machine Learning

The project currently includes:

- Feature engineering pipeline
- Leakage-free next lap prediction
- Baseline regression models:
  - Linear Regression
  - Random Forest Regressor
  - HistGradientBoosting Regressor

Current best model:

| Metric | Value |
|---|---|
| MAE | ~1.83s |
| RMSE | ~4.26s |
| R² | ~0.21 |

---

# 🖥️ Dashboard

The Streamlit dashboard includes:

- Interactive driver filtering
- Compound selection
- Lap range filtering
- Lap time trend analysis
- Compound performance comparison
- Prediction visualization
- Pit strategy insights
- Race-control inspired UI

---

# 🛠️ Tech Stack

## Languages
- Python

## Data & ML
- Pandas
- NumPy
- Scikit-learn
- FastF1

## Visualization
- Plotly
- Matplotlib
- Streamlit

## Tools
- Git
- GitHub
- VS Code

---

# 📂 Project Structure

```text
Formula-Racing-Telemetry-Intelligence-System/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── notebooks/
│   ├── 01_day1_fastf1_test.py
│   ├── 02_day2_collect_data.py
│   └── 03_day3_eda.py
│
├── reports/
│   └── figures/
│
├── scripts/
│   ├── day4_feature_engineering.py
│   ├── day5_train_model.py
│   └── day6_evaluate_and_strategy.py
│
├── src/
│   ├── config.py
│   ├── data_loader.py
│   └── preprocess.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# 🚀 How To Run

## 1. Clone the repository

```bash
git clone https://github.com/ShaeeshB/Formula-Racing-Telemetry-Intelligence-System.git
cd Formula-Racing-Telemetry-Intelligence-System
```

---

## 2. Create virtual environment

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Run the dashboard

```bash
streamlit run app/streamlit_app.py
```

---

# 📈 Current Capabilities

The system can currently:

- Load Formula 1 sessions using FastF1
- Cache telemetry data locally
- Export combined race CSVs
- Export per-driver telemetry CSVs
- Process and clean raw race data
- Generate engineered ML features
- Train lap prediction models
- Evaluate prediction accuracy
- Generate pit strategy recommendations
- Visualize telemetry in an interactive dashboard

---

# 🔮 Planned Features

- Real-time telemetry streaming
- Driver-to-driver comparison mode
- Tire wear prediction
- Fuel strategy estimation
- Sector-by-sector pace analysis
- Live leaderboard simulation
- Deep learning models
- Race simulation engine

---

# 📸 Screenshots

## Race Control Dashboard

_Add dashboard screenshots here_

---

# 📌 Status

✅ Week 1 Complete  
✅ Telemetry Pipeline Complete  
✅ Feature Engineering Complete  
✅ Baseline ML Models Complete  
✅ Dashboard MVP Complete

---

# 👨‍💻 Author

Shaeesh Bhowmik

---

# 📄 License

This project is for educational and research purposes.