import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

st.set_page_config(
    page_title="Race Control Dashboard",
    page_icon="🏁",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .stApp {
        background:
            radial-gradient(circle at top, rgba(120, 0, 0, 0.28) 0%, rgba(0, 0, 0, 0.96) 40%),
            linear-gradient(180deg, #090909 0%, #050505 100%);
        color: #f4f4f4;
    }

    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 1600px;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(16,16,16,0.98) 0%, rgba(8,8,8,0.98) 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    .hero {
        background: linear-gradient(135deg, rgba(72, 0, 0, 0.50) 0%, rgba(18, 8, 8, 0.92) 50%, rgba(12, 12, 12, 0.98) 100%);
        border: 1px solid rgba(255, 60, 60, 0.35);
        border-radius: 28px;
        padding: 1.45rem 1.7rem 1.55rem 1.7rem;
        box-shadow:
            0 0 0 1px rgba(255,255,255,0.02) inset,
            0 0 38px rgba(255, 0, 0, 0.10);
        margin-bottom: 1.35rem;
    }

    .hero h1 {
        margin: 0;
        font-size: 2.05rem;
        font-weight: 900;
        letter-spacing: 0.02em;
    }

    .hero p {
        margin: 0.8rem 0 0 0;
        font-size: 1.0rem;
        color: rgba(255,255,255,0.76);
    }

    .kpi {
        background: linear-gradient(180deg, rgba(26,26,26,0.98) 0%, rgba(18,18,18,0.98) 100%);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 22px;
        padding: 1rem 1rem 0.95rem 1rem;
        min-height: 128px;
        box-shadow: 0 0 22px rgba(0,0,0,0.18);
    }

    .kpi-label {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.16em;
        color: rgba(255,255,255,0.72);
        margin-bottom: 0.55rem;
    }

    .kpi-value {
        font-size: 2.0rem;
        font-weight: 900;
        line-height: 1;
        margin-bottom: 0.55rem;
        color: #f7f7f7;
    }

    .kpi-sub {
        color: rgba(255,255,255,0.64);
        font-size: 0.9rem;
    }

    .section-title {
        margin: 0.6rem 0 0.75rem 0;
        color: #ff4b4b;
        font-size: 0.86rem;
        font-weight: 800;
        letter-spacing: 0.18em;
        text-transform: uppercase;
    }

    

    .stButton button {
        background: linear-gradient(90deg, #d81e1e 0%, #ff4b4b 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.7rem 1rem;
        font-weight: 800;
    }

    .stButton button:hover {
        filter: brightness(1.05);
        border: none;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="slider"] {
        background-color: rgba(255,255,255,0.04);
        border-radius: 12px;
    }

    [data-testid="stDataFrame"] {
        border-radius: 14px;
        overflow: hidden;
    }

    .small-muted {
        color: rgba(255,255,255,0.62);
        font-size: 0.88rem;
    }

    .status-pill {
        display: inline-block;
        padding: 0.18rem 0.5rem;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        background: rgba(255,75,75,0.12);
        color: #ff5858;
        border: 1px solid rgba(255,75,75,0.28);
    }
</style>
""",
    unsafe_allow_html=True,
)

@st.cache_data
def load_feature_data():
    path = PROCESSED_DIR / "day4_feature_engineered_data.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)

@st.cache_resource
def load_model():
    path = MODELS_DIR / "day5_best_next_lap_time_model.joblib"
    if not path.exists():
        return None
    return joblib.load(path)

@st.cache_data
def load_metrics():
    path = MODELS_DIR / "day6_evaluation_metrics.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_recommendations():
    path = REPORTS_DIR / "day6_pit_recommendations.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)

def num(s):
    return pd.to_numeric(s, errors="coerce")

def kpi(label, value, sub):
    st.markdown(
        f"""
<div class="kpi">
    <div class="kpi-label">{label}</div>
    <div class="kpi-value">{value}</div>
    <div class="kpi-sub">{sub}</div>
</div>
""",
        unsafe_allow_html=True,
    )

df = load_feature_data()
model = load_model()
metrics = load_metrics()
recs = load_recommendations()

if df is None:
    st.error("Feature data not found. Run Day 4 first.")
    st.stop()

df = df.copy()
df["Driver"] = df["Driver"].astype(str)
if "Compound" in df.columns:
    df["Compound"] = df["Compound"].astype(str)

for col in [
    "LapNumber",
    "Stint",
    "Stint_filled",
    "Target_NextLapTimeSeconds",
    "PreviousLapTimeSeconds",
    "PreviousLapTimeRollingMean3",
]:
    if col in df.columns:
        df[col] = num(df[col])

st.sidebar.markdown("## Race Control")
st.sidebar.markdown('<div class="small-muted">Telemetry Intelligence</div>', unsafe_allow_html=True)

drivers = sorted(df["Driver"].dropna().unique().tolist())
selected_driver = st.sidebar.selectbox("Driver", drivers)

if "Compound" in df.columns:
    compounds = ["All"] + sorted(df["Compound"].dropna().unique().tolist())
    selected_compound = st.sidebar.selectbox("Compound", compounds)
else:
    selected_compound = "All"

if "LapNumber" in df.columns and df["LapNumber"].notna().any():
    lap_min = int(df["LapNumber"].min())
    lap_max = int(df["LapNumber"].max())
    lap_range = st.sidebar.slider("Lap Range", lap_min, lap_max, (lap_min, lap_max))
else:
    lap_range = (0, 0)

show_only_pit_signals = st.sidebar.toggle("Show only pit signals", value=False)

st.markdown(
    """
<div class="hero">
    <h1>Race Control Dashboard</h1>
    <p>Formula 1 telemetry intelligence, next-lap prediction, and pit strategy signals.</p>
</div>
""",
    unsafe_allow_html=True,
)

driver_df = df[df["Driver"] == selected_driver].copy()

if selected_compound != "All" and "Compound" in driver_df.columns:
    driver_df = driver_df[driver_df["Compound"] == selected_compound].copy()

if "LapNumber" in driver_df.columns:
    driver_df = driver_df[
        driver_df["LapNumber"].between(lap_range[0], lap_range[1], inclusive="both")
    ].copy()

driver_df = driver_df.sort_values("LapNumber") if "LapNumber" in driver_df.columns else driver_df

if driver_df.empty:
    st.warning("No data matches the selected filters.")
    st.stop()

rows_count = len(driver_df)
avg_next = driver_df["Target_NextLapTimeSeconds"].mean() if "Target_NextLapTimeSeconds" in driver_df.columns else np.nan
best_next = driver_df["Target_NextLapTimeSeconds"].min() if "Target_NextLapTimeSeconds" in driver_df.columns else np.nan
avg_stint = driver_df["Stint_filled"].mean() if "Stint_filled" in driver_df.columns else np.nan

c1, c2, c3, c4 = st.columns(4)
with c1:
    kpi("Driver", selected_driver, f"{rows_count} laps in view")
with c2:
    kpi("Avg Next Lap", f"{avg_next:.2f}s" if pd.notna(avg_next) else "N/A", "From filtered sample")
with c3:
    kpi("Best Next Lap", f"{best_next:.2f}s" if pd.notna(best_next) else "N/A", "Quickest observed")
with c4:
    kpi("Avg Stint", f"{avg_stint:.1f}" if pd.notna(avg_stint) else "N/A", "Stint context")


tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Telemetry Trends", "Prediction Lab", "Pit Wall"])

with tab1:
    st.markdown('<div class="section-title">Session Snapshot</div>', unsafe_allow_html=True)
    a, b = st.columns([1.25, 1], gap="large")

    with a:
        preview_cols = [
            c for c in [
                "LapNumber",
                "Compound",
                "Stint",
                "PreviousLapTimeSeconds",
                "PreviousLapTimeRollingMean3",
                "Target_NextLapTimeSeconds",
            ] if c in driver_df.columns
        ]
        st.dataframe(driver_df[preview_cols].head(20), use_container_width=True, height=360)

    with b:
        if metrics:
            st.metric("MAE", f"{metrics.get('MAE', 0):.2f}s")
            st.metric("RMSE", f"{metrics.get('RMSE', 0):.2f}s")
            st.metric("R²", f"{metrics.get('R2', 0):.2f}")
        else:
            st.metric("MAE", "N/A")
            st.metric("RMSE", "N/A")
            st.metric("R²", "N/A")

        st.markdown('<div class="section-title">Current Status</div>', unsafe_allow_html=True)
        st.markdown('<div class="small-muted">Feature Data <span class="status-pill">Loaded</span></div>', unsafe_allow_html=True)
        st.markdown('<div class="small-muted" style="margin-top:0.5rem;">Model <span class="status-pill">Loaded</span></div>', unsafe_allow_html=True)
        st.markdown('<div class="small-muted" style="margin-top:0.5rem;">Metrics <span class="status-pill">Loaded</span></div>', unsafe_allow_html=True)
        st.markdown('<div class="small-muted" style="margin-top:0.5rem;">Pit Recs <span class="status-pill">Loaded</span></div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="section-title">Next Lap Time Trend</div>', unsafe_allow_html=True)
    if "LapNumber" in driver_df.columns and "Target_NextLapTimeSeconds" in driver_df.columns:
        trend_df = driver_df[["LapNumber", "Target_NextLapTimeSeconds"]].dropna()
        if not trend_df.empty:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=trend_df["LapNumber"],
                    y=trend_df["Target_NextLapTimeSeconds"],
                    mode="lines+markers",
                    line=dict(color="#ff4b4b", width=2.5),
                    marker=dict(size=6, color="#ff4b4b"),
                )
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,0.03)",
                font=dict(color="white"),
                xaxis_title="Lap",
                yaxis_title="Time (s)",
                margin=dict(l=10, r=10, t=20, b=10),
                height=420,
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Next Lap Time by Compound</div>', unsafe_allow_html=True)
    if "Compound" in driver_df.columns and "Target_NextLapTimeSeconds" in driver_df.columns:
        comp_df = driver_df[["Compound", "Target_NextLapTimeSeconds"]].dropna()
        if not comp_df.empty:
            fig2 = px.box(
                comp_df,
                x="Compound",
                y="Target_NextLapTimeSeconds",
                color="Compound",
                color_discrete_sequence=["#ff4b4b", "#ffb000", "#cfcfcf"],
            )
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,0.03)",
                font=dict(color="white"),
                margin=dict(l=10, r=10, t=20, b=10),
                height=420,
            )
            st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.markdown('<div class="section-title">Prediction Lab</div>', unsafe_allow_html=True)
    if model is None:
        st.warning("Model not found. Run Day 5 first.")
    else:
        feature_cols = [c for c in driver_df.columns if c not in ["Target_NextLapTimeSeconds", "LapTimeSeconds", "LapTime"]]
        predict_df = driver_df[feature_cols].copy()

        if st.button("Run Prediction"):
            preds = model.predict(predict_df)
            pred_df = driver_df.copy()
            pred_df["PredictedNextLapTime"] = preds

            show_cols = [
                c for c in [
                    "LapNumber",
                    "Compound",
                    "PreviousLapTimeSeconds",
                    "Target_NextLapTimeSeconds",
                    "PredictedNextLapTime",
                ] if c in pred_df.columns
            ]

            st.dataframe(pred_df[show_cols].head(25), use_container_width=True, height=340)

            if "Target_NextLapTimeSeconds" in pred_df.columns:
                compare_df = pred_df.dropna(subset=["Target_NextLapTimeSeconds", "PredictedNextLapTime"])
                if not compare_df.empty:
                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x=compare_df["Target_NextLapTimeSeconds"],
                            y=compare_df["PredictedNextLapTime"],
                            mode="markers",
                            marker=dict(size=8, color="#ff4b4b"),
                        )
                    )
                    min_val = float(min(compare_df["Target_NextLapTimeSeconds"].min(), compare_df["PredictedNextLapTime"].min()))
                    max_val = float(max(compare_df["Target_NextLapTimeSeconds"].max(), compare_df["PredictedNextLapTime"].max()))
                    fig.add_trace(
                        go.Scatter(
                            x=[min_val, max_val],
                            y=[min_val, max_val],
                            mode="lines",
                            line=dict(color="white", dash="dash"),
                        )
                    )
                    fig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(255,255,255,0.03)",
                        font=dict(color="white"),
                        xaxis_title="Actual",
                        yaxis_title="Predicted",
                        margin=dict(l=10, r=10, t=20, b=10),
                        height=400,
                    )
                    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.markdown('<div class="section-title">Pit Wall</div>', unsafe_allow_html=True)
    if recs is None:
        st.warning("Pit recommendation file not found. Run Day 6 first.")
    else:
        recs2 = recs.copy()
        recs2["Driver"] = recs2["Driver"].astype(str)
        current_recs = recs2[recs2["Driver"] == selected_driver].copy()

        if show_only_pit_signals:
            current_recs = current_recs[current_recs["Recommendation"].isin(["Pit soon", "Consider pit window"])]

        if current_recs.empty:
            st.info("No strategy signals for current selection.")
        else:
            st.dataframe(current_recs.head(25), use_container_width=True, height=340)

            counts = current_recs["Recommendation"].value_counts().reset_index()
            counts.columns = ["Recommendation", "Count"]

            fig3 = px.bar(
                counts,
                x="Recommendation",
                y="Count",
                color="Recommendation",
                color_discrete_sequence=["#ff4b4b", "#ffb000", "#bfbfbf"],
            )
            fig3.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,0.03)",
                font=dict(color="white"),
                margin=dict(l=10, r=10, t=20, b=10),
                height=380,
            )
            st.plotly_chart(fig3, use_container_width=True)

st.markdown('<div class="section-title">Overall Dataset Slice</div>', unsafe_allow_html=True)
full_cols = [
    c for c in [
        "Driver",
        "Compound",
        "TrackStatus",
        "LapNumber",
        "Stint",
        "Target_NextLapTimeSeconds",
    ] if c in df.columns
]
st.dataframe(df[full_cols].head(20), use_container_width=True, height=240)