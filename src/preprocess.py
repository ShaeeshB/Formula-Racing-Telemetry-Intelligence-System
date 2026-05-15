from pathlib import Path
import pandas as pd


def load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def basic_clean_laps(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df.columns = [c.strip() for c in df.columns]

    if "LapTime" in df.columns:
        df = df[df["LapTime"].notna()]

    if "Driver" in df.columns:
        df["Driver"] = df["Driver"].astype(str).str.strip()

    if "Compound" in df.columns:
        df["Compound"] = df["Compound"].astype(str).str.strip()

    return df


def save_clean_csv(df: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)