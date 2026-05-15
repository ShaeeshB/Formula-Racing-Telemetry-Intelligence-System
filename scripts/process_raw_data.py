from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "src"))

from config import RAW_DIR, PROCESSED_DIR
from preprocess import load_csv, basic_clean_laps, save_clean_csv


def main():
    combined_files = sorted(RAW_DIR.glob("*_all_drivers_laps.csv"))

    if not combined_files:
        print("No combined raw CSV files found in data/raw.")
        return

    for raw_file in combined_files:
        print(f"Processing {raw_file.name} ...")
        df = load_csv(raw_file)
        cleaned = basic_clean_laps(df)

        output_name = raw_file.name.replace("_all_drivers_laps.csv", "_cleaned.csv")
        output_path = PROCESSED_DIR / output_name

        save_clean_csv(cleaned, output_path)
        print(f"Saved cleaned file to {output_path}")


if __name__ == "__main__":
    main()