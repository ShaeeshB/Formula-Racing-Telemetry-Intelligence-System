import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "src"))

from config import CACHE_DIR, RAW_DIR
from data_loader import enable_cache, load_session, save_laps_all_and_per_driver


def main():
    enable_cache(CACHE_DIR)

    year = 2024
    gp_name = "British Grand Prix"
    session_code = "R"

    print(f"Loading {year} {gp_name} {session_code} ...")
    session = load_session(year, gp_name, session_code)

    print("\nSession loaded successfully.")
    print("Available drivers:", session.drivers[:10])

    base_name = f"{year}_{gp_name.replace(' ', '_')}_{session_code}"
    save_laps_all_and_per_driver(session, RAW_DIR, base_name)

if __name__ == "__main__":
    main()