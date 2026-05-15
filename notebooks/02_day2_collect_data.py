import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "src"))

from config import CACHE_DIR, RAW_DIR
from data_loader import enable_cache, load_session, save_laps_all_and_per_driver


def collect_session(year: int, gp_name: str, session_code: str) -> None:
    print(f"\nLoading {year} {gp_name} {session_code} ...")
    session = load_session(year, gp_name, session_code)

    print("Session loaded successfully.")
    print("Drivers:", session.drivers)

    base_name = f"{year}_{gp_name.replace(' ', '_')}_{session_code}"
    save_laps_all_and_per_driver(session, RAW_DIR, base_name)


def main():
    enable_cache(CACHE_DIR)

    sessions = [
        (2024, "British Grand Prix", "R"),
        (2024, "Italian Grand Prix", "R"),
        (2024, "Hungarian Grand Prix", "R"),
    ]

    for year, gp_name, session_code in sessions:
        try:
            collect_session(year, gp_name, session_code)
        except Exception as e:
            print(f"Failed to load {year} {gp_name} {session_code}: {e}")


if __name__ == "__main__":
    main()