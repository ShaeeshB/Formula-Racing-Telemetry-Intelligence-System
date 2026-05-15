from pathlib import Path
import fastf1
import pandas as pd


def enable_cache(cache_dir: str | Path) -> None:
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_path))


def load_session(year: int, gp_name: str, session_code: str):
    session = fastf1.get_session(year, gp_name, session_code)
    session.load()
    return session


def get_all_laps(session) -> pd.DataFrame:
    return session.laps.copy()


def get_driver_laps(session, driver_code: str) -> pd.DataFrame:
    return session.laps.pick_drivers(driver_code).copy()


def get_fastest_lap(session):
    return session.laps.pick_fastest()


def save_laps_all_and_per_driver(session, output_dir: str | Path, base_name: str) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_laps = get_all_laps(session)
    combined_file = output_dir / f"{base_name}_all_drivers_laps.csv"
    all_laps.to_csv(combined_file, index=False)

    for driver_code in session.drivers:
        driver_laps = get_driver_laps(session, driver_code)
        driver_file = output_dir / f"{base_name}_{driver_code}_laps.csv"
        driver_laps.to_csv(driver_file, index=False)

    print(f"Saved combined file: {combined_file}")
    print(f"Saved per-driver files in: {output_dir}")