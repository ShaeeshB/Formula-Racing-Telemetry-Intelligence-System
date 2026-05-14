import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "src"))

from config import CACHE_DIR, RAW_DIR
from data_loader import enable_cache, load_session, get_driver_laps

def main():
    enable_cache(CACHE_DIR)

    year = 2024
    gp_name = "British Grand Prix"
    session_code = "R"

    print(f"Loading {year} {gp_name} {session_code} ...")
    session = load_session(year, gp_name, session_code)

    print("\nSession loaded successfully.")
    print("Available drivers:", session.drivers[:10])

    driver_code = session.drivers[0]
    laps = get_driver_laps(session, driver_code)

    print(f"\nLaps for {driver_code}:")
    columns_to_show = ["Driver", "LapNumber", "LapTime", "Compound"]
    available_columns = [c for c in columns_to_show if c in laps.columns]
    print(laps[available_columns].head())

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    output_file = RAW_DIR / f"{year}_{gp_name.replace(' ', '_')}_{session_code}_{driver_code}_laps.csv"
    laps.to_csv(output_file, index=False)
    print(f"\nSaved sample lap data to: {output_file}")

if __name__ == "__main__":
    main()