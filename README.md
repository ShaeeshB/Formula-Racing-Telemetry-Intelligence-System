# Formula Racing Telemetry Intelligence System

<<<<<<< HEAD
Motorsport telemetry intelligence system using FastF1, machine learning, and data analytics for race performance analysis.

An AI-powered motorsport analytics project that analyzes Formula 1 telemetry, predicts lap performance, estimates tire degradation, and suggests pit strategy.
=======
An AI-powered motorsport analytics platform for analyzing Formula 1 telemetry, predicting lap performance, estimating tire degradation, and recommending pit strategy.
>>>>>>> 44e2564 (Add multi-session telemetry collection and preprocessing pipeline)

---

## Project Goals

- Analyze Formula 1 telemetry data
- Predict lap times using machine learning
- Estimate tire degradation trends
- Recommend pit windows and strategy
- Compare driver performance
- Build an interactive telemetry dashboard

---

## Tech Stack

### Languages
- Python

### Data & Analytics
- Pandas
- NumPy
- Scikit-learn
- FastF1

### Visualization
- Plotly
- Matplotlib
- Streamlit

<<<<<<< HEAD
## Week 1 Progress
- Repo created
- Environment setup
- FastF1 testing in progress
- Project configuration and telemetry loader added
=======
### Tools
- Git
- GitHub
- VS Code

---

## Current Progress

### Week 1
- Repository setup complete
- Virtual environment configured
- FastF1 telemetry integration working
- Initial race session loaded successfully
- CSV export pipeline working

### Week 1 — Data Pipeline Progress
- Multiple Formula 1 race sessions collected
- Combined race CSV export working
- Per-driver CSV export working
- Initial preprocessing pipeline created
- Raw and processed data folders structured

---

## Planned Features

- Lap time prediction
- Tire degradation analysis
- Driver comparison
- Pit strategy recommendation
- Telemetry visualization dashboard
- Interactive web application

---
## Current Capabilities

The system can currently:

- Load Formula 1 sessions using FastF1
- Cache telemetry data locally
- Export combined session CSVs
- Export per-driver lap CSVs
- Process and clean raw lap data
- Organize datasets into raw and processed pipelines

---
## Folder Structure

```text
motorsport-telemetry-ai/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│
├── src/
│   ├── config.py
│   ├── data_loader.py
│   └── preprocess.py
│
├── scripts/
│   └── process_raw_data.py
│
├── app/
├── models/
├── reports/
│   └── figures/
│
├── requirements.txt
├── .gitignore
└── README.md
```
>>>>>>> 44e2564 (Add multi-session telemetry collection and preprocessing pipeline)
