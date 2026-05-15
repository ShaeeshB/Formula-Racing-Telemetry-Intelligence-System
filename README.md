# Formula Racing Telemetry Intelligence System

An AI-powered motorsport analytics platform for Formula 1 telemetry analysis, lap-time prediction, tire degradation analysis, and race strategy intelligence.

---

# Project Overview

This project uses Formula 1 telemetry data collected through FastF1 and applies machine learning techniques to analyze race performance and predict future lap times.

The long-term goal is to simulate a lightweight race-engineering analytics system capable of:
- telemetry analysis
- lap-time prediction
- tire degradation estimation
- driver comparison
- pit strategy recommendations
- interactive dashboards

---

# Current Features

## Telemetry Data Pipeline
- FastF1 session loading
- Multi-session race collection
- Combined lap exports
- Per-driver CSV exports
- Raw + processed dataset structure

## Data Processing
- Cleaning and preprocessing pipeline
- Lap-time normalization
- Feature engineering
- Next-lap prediction target generation

## Exploratory Data Analysis
- Lap-time trends
- Tire compound analysis
- Driver pace comparison
- Race progression visualization

## Machine Learning
- Leakage-free next-lap prediction pipeline
- Linear Regression baseline
- Random Forest Regressor
- HistGradientBoostingRegressor
- Model evaluation and persistence

---

# Tech Stack

## Languages
- Python

## Data & ML
- Pandas
- NumPy
- Scikit-learn
- FastF1

## Visualization
- Matplotlib
- Plotly
- Streamlit

## Tools
- Git
- GitHub
- VS Code

---

# Current Best Model

| Model | RMSE | R² |
|---|---|---|
| RandomForestRegressor | ~4.26s | ~0.21 |

The current model predicts the next lap time using historical race context and telemetry-derived features.

---

# Project Structure

```text
motorsport-telemetry-ai/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_day1_fastf1_test.py
│   ├── 02_day2_collect_data.py
│   └── 03_day3_eda.py
│
├── scripts/
│   ├── check_environment.py
│   ├── process_raw_data.py
│   ├── day4_feature_engineering.py
│   └── day5_train_model.py
│
├── src/
│   ├── config.py
│   ├── data_loader.py
│   └── preprocess.py
│
├── models/
│
├── reports/
│   └── figures/
│
├── requirements.txt
├── .gitignore
└── README.md