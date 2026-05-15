import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "src"))

from config import PROCESSED_DIR


def load_latest_cleaned_file() -> pd.DataFrame:
    files = sorted(PROCESSED_DIR.glob("*_cleaned.csv"))
    if not files:
        raise FileNotFoundError("No cleaned CSV files found in data/processed.")
    latest_file = files[-1]
    print(f"Loading: {latest_file.name}")
    return pd.read_csv(latest_file)


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "LapTime" in df.columns:
        df["LapTime"] = pd.to_timedelta(df["LapTime"], errors="coerce")
        df["LapTimeSeconds"] = df["LapTime"].dt.total_seconds()

    if "LapNumber" in df.columns:
        df["LapNumber"] = pd.to_numeric(df["LapNumber"], errors="coerce")

    if "Driver" in df.columns:
        df["Driver"] = df["Driver"].astype(str)

    if "Compound" in df.columns:
        df["Compound"] = df["Compound"].astype(str)

    return df


def plot_lap_time_by_driver(df: pd.DataFrame) -> None:
    if "Driver" not in df.columns or "LapTimeSeconds" not in df.columns:
        print("Skipping lap time by driver plot: required columns missing.")
        return

    plt.figure(figsize=(12, 6))
    for driver in df["Driver"].dropna().unique()[:10]:
        driver_df = df[df["Driver"] == driver].sort_values("LapNumber")
        plt.plot(driver_df["LapNumber"], driver_df["LapTimeSeconds"], label=driver, alpha=0.8)

    plt.title("Lap Time Trend by Driver")
    plt.xlabel("Lap Number")
    plt.ylabel("Lap Time (seconds)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_lap_time_distribution(df: pd.DataFrame) -> None:
    if "LapTimeSeconds" not in df.columns:
        print("Skipping lap time distribution plot: LapTimeSeconds missing.")
        return

    plt.figure(figsize=(10, 6))
    plt.hist(df["LapTimeSeconds"].dropna(), bins=30)
    plt.title("Lap Time Distribution")
    plt.xlabel("Lap Time (seconds)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


def plot_compound_counts(df: pd.DataFrame) -> None:
    if "Compound" not in df.columns:
        print("Skipping compound plot: Compound missing.")
        return

    compound_counts = df["Compound"].value_counts(dropna=False)

    plt.figure(figsize=(10, 5))
    compound_counts.plot(kind="bar")
    plt.title("Tire Compound Usage")
    plt.xlabel("Compound")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def plot_lap_time_by_lap_number(df: pd.DataFrame) -> None:
    if "LapNumber" not in df.columns or "LapTimeSeconds" not in df.columns:
        print("Skipping lap number plot: required columns missing.")
        return

    avg_laps = df.groupby("LapNumber")["LapTimeSeconds"].mean().reset_index()

    plt.figure(figsize=(12, 6))
    plt.plot(avg_laps["LapNumber"], avg_laps["LapTimeSeconds"], marker="o")
    plt.title("Average Lap Time by Lap Number")
    plt.xlabel("Lap Number")
    plt.ylabel("Average Lap Time (seconds)")
    plt.tight_layout()
    plt.show()


def main():
    df = load_latest_cleaned_file()
    df = prepare_data(df)

    print("\nColumns available:")
    print(df.columns.tolist())

    print("\nHead of cleaned data:")
    print(df.head())

    plot_lap_time_by_driver(df)
    plot_lap_time_distribution(df)
    plot_compound_counts(df)
    plot_lap_time_by_lap_number(df)


if __name__ == "__main__":
    main()