"""
Day 4 — Feature Engineering

Goal:
Create an ML-ready dataset that predicts the NEXT lap time using only
information that would be known before that next lap starts.

Why this version is better:
- avoids target leakage
- uses only pre-lap features
- creates a next-lap target
- saves a clean feature file for model training
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd().resolve()
if not (ROOT / "src").exists():
    ROOT = Path.cwd().resolve().parent

sys.path.append(str(ROOT / "src"))

from config import PROCESSED_DIR  # noqa: E402


def load_latest_cleaned_file() -> pd.DataFrame:
    files = sorted(PROCESSED_DIR.glob("*_cleaned.csv"))
    if not files:
        raise FileNotFoundError("No cleaned CSV files found in data/processed.")
    latest_file = files[-1]
    print(f"Loading cleaned file: {latest_file.name}")
    return pd.read_csv(latest_file)


def prepare_base_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    if "LapTime" in df.columns:
        df["LapTime"] = pd.to_timedelta(df["LapTime"], errors="coerce")
        df["LapTimeSeconds"] = df["LapTime"].dt.total_seconds()

    if "LapNumber" in df.columns:
        df["LapNumber"] = pd.to_numeric(df["LapNumber"], errors="coerce")

    if "Stint" in df.columns:
        df["Stint"] = pd.to_numeric(df["Stint"], errors="coerce")

    for col in ["Driver", "Compound", "Team", "TrackStatus"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Sort so lag-based operations make sense
    sort_cols = [c for c in ["Driver", "LapNumber"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols)

    # Target: next lap time for the same driver
    if "Driver" in df.columns and "LapTimeSeconds" in df.columns:
        df["Target_NextLapTimeSeconds"] = (
            df.groupby("Driver")["LapTimeSeconds"].shift(-1)
        )

    # Basic race progression feature
    if "LapNumber" in df.columns:
        max_lap = df["LapNumber"].max()
        if pd.notna(max_lap) and max_lap != 0:
            df["RaceProgress"] = df["LapNumber"] / max_lap
        else:
            df["RaceProgress"] = np.nan

        df["LapNumber_filled"] = df["LapNumber"].fillna(df["LapNumber"].median())
        df["LapNumberSquared"] = df["LapNumber_filled"] ** 2

    # Stint features
    if "Stint" in df.columns:
        df["Stint_filled"] = df["Stint"].fillna(df["Stint"].median())
    else:
        df["Stint_filled"] = np.nan

    # Simple compound flags
    if "Compound" in df.columns:
        compound_upper = df["Compound"].str.upper()
        df["IsSoft"] = (compound_upper == "SOFT").astype(int)
        df["IsMedium"] = (compound_upper == "MEDIUM").astype(int)
        df["IsHard"] = (compound_upper == "HARD").astype(int)
    else:
        df["IsSoft"] = 0
        df["IsMedium"] = 0
        df["IsHard"] = 0

    # Previous lap time is a valid feature because it is known before the next lap starts
    if "Driver" in df.columns and "LapTimeSeconds" in df.columns:
        df["PreviousLapTimeSeconds"] = df.groupby("Driver")["LapTimeSeconds"].shift(1)

        # Change from previous lap to current lap, based only on past information
        df["PreviousLapTimeDelta"] = df["PreviousLapTimeSeconds"] - df["PreviousLapTimeSeconds"].groupby(df["Driver"]).shift(1)

        # Rolling average over the last 3 known lap times
        df["PreviousLapTimeRollingMean3"] = (
            df.groupby("Driver")["LapTimeSeconds"]
            .transform(lambda s: s.shift(1).rolling(window=3, min_periods=1).mean())
        )

        # Rolling std over the last 3 known lap times
        df["PreviousLapTimeRollingStd3"] = (
            df.groupby("Driver")["LapTimeSeconds"]
            .transform(lambda s: s.shift(1).rolling(window=3, min_periods=2).std())
        )

    return df


def select_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "Driver",
        "Team",
        "Compound",
        "TrackStatus",
        "LapNumber",
        "LapNumber_filled",
        "LapNumberSquared",
        "RaceProgress",
        "Stint",
        "Stint_filled",
        "PreviousLapTimeSeconds",
        "PreviousLapTimeDelta",
        "PreviousLapTimeRollingMean3",
        "PreviousLapTimeRollingStd3",
        "IsSoft",
        "IsMedium",
        "IsHard",
        "Target_NextLapTimeSeconds",
    ]

    available_cols = [c for c in keep_cols if c in df.columns]
    return df[available_cols].copy()


def save_features(df: pd.DataFrame) -> Path:
    output_path = PROCESSED_DIR / "day4_feature_engineered_data.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved feature table to: {output_path}")
    return output_path


def main():
    print("## Day 4 — Feature Engineering\n")

    df = load_latest_cleaned_file()
    print(f"Loaded shape: {df.shape}")
    print("\nSample data:")
    print(df.head())

    df = prepare_base_columns(df)
    df = engineer_features(df)

    feature_df = select_feature_columns(df)

    # Remove rows where the NEXT lap time is unknown
    feature_df = feature_df[feature_df["Target_NextLapTimeSeconds"].notna()].copy()

    print("\nFinal feature table shape:")
    print(feature_df.shape)
    print(feature_df.head())

    save_features(feature_df)

    print("\n## Day 4 complete")
    print("You now have a leakage-free feature dataset ready for model training.")


if __name__ == "__main__":
    main()