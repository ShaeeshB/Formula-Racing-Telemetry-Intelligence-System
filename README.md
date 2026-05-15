# Formula Racing Telemetry Intelligence System

An AI-powered motorsport analytics platform for Formula 1 telemetry analysis, next-lap prediction, tire degradation analysis, and race strategy intelligence.

---

# Project Overview

This project uses Formula 1 telemetry and lap data collected through FastF1 and applies machine learning techniques to analyze race performance and predict future lap times.

The goal is to simulate a lightweight race-engineering analytics system capable of:
- telemetry analysis
- next-lap prediction
- tire degradation estimation
- driver comparison
- pit strategy recommendations
- interactive dashboard visualization

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
- Leakage-free next-lap target generation

## Exploratory Data Analysis
- Lap-time trend visualization
- Tire compound analysis
- Driver pace comparison
- Race progression analysis
- Residual analysis

## Machine Learning
- Leakage-free next-lap prediction pipeline
- Linear Regression baseline
- Random Forest Regressor
- HistGradientBoostingRegressor
- Model evaluation and persistence
- Feature importance analysis

## Strategy Intelligence
- Pit-window recommendation logic
- Pace degradation detection
- Simple race-strategy rule engine

---

# Current Best Model

| Model | MAE | RMSE | R² |
|---|---|---|---|
| RandomForestRegressor | ~1.83s | ~4.26s | ~0.21 |

The current model predicts the next lap time using historical race context and telemetry-derived features.

---

# Evaluation Outputs

The project currently generates:
- Actual vs predicted lap-time plots
- Residual distribution analysis
- Permutation feature importance analysis
- Pit recommendation reports

Saved outputs:
```text
reports/
├── figures/
│   ├── day6_actual_vs_predicted.png
│   ├── day6_residual_distribution.png
│   └── day6_feature_importance.png
│
└── day6_pit_recommendations.csv

