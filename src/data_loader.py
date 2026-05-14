from pathlib import Path
import fastf1
import pandas as pd


def enable_cache(cache_dir: str | Path) -> None:
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    fastf1.Cache.enable_cache(str(cache_path))


def load_session(year: int, gp_name: str, session_code: str):
    """
    Load a Formula 1 session.

    Examples:
    - R = Race
    - Q = Qualifying
    - FP1 = Free Practice 1
    """

    session = fastf1.get_session(year, gp_name, session_code)
    session.load()

    return session


def get_driver_laps(session, driver_code: str) -> pd.DataFrame:
    """
    Return laps for a specific driver.
    """

    return session.laps.pick_drivers(driver_code).copy()


def get_fastest_lap(session):
    """
    Return fastest lap in the session.
    """

    return session.laps.pick_fastest()