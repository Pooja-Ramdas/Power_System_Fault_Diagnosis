"""
Downloads CEN's published hourly telemetry spreadsheets once, caches them,
and slices a time window around each fault event for the numeric modality.

IMPORTANT: CEN's xlsx files change column layout across releases/years, and
CEN re-dates the upload folder periodically (so a URL that works today may
404 in a few months -- see config.py for how to refresh it). This module
auto-detects a date/time column heuristically; the first time you run it,
inspect the printed column list for each file and, if detection picks the
wrong column, pass `datetime_col` explicitly.
"""
import os
import requests
import pandas as pd
from datetime import timedelta

USER_AGENT = "Mozilla/5.0 (academic research script; multimodal fault diagnosis project)"


def _session():
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def download_once(url, dest_path, session=None, timeout=180):
    if os.path.exists(dest_path):
        return dest_path
    session = session or _session()
    r = session.get(url, timeout=timeout)
    r.raise_for_status()
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(dest_path, "wb") as f:
        f.write(r.content)
    return dest_path


def _guess_datetime_col(df):
    for col in df.columns:
        if df[col].dtype == "datetime64[ns]":
            return col
    for col in df.columns:
        name = str(col).lower()
        if "fecha" in name or "hora" in name or "date" in name or "time" in name:
            return col
    return None


def load_xlsx(xlsx_path):
    df = pd.read_excel(xlsx_path)
    return df


def extract_window(df, event_dt, window_hours=48, datetime_col=None):
    """Returns the rows of df within +/- window_hours/2 of event_dt.
    Raises ValueError with the available columns if no datetime column can
    be found/guessed, so you can pass datetime_col explicitly next time."""
    dt_col = datetime_col or _guess_datetime_col(df)
    if dt_col is None:
        raise ValueError(
            f"Could not detect a date/time column. Columns present: {list(df.columns)}. "
            f"Pass datetime_col=... explicitly once you've identified it."
        )
    df = df.copy()
    df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
    start = event_dt - timedelta(hours=window_hours / 2)
    end = event_dt + timedelta(hours=window_hours / 2)
    window = df[(df[dt_col] >= start) & (df[dt_col] <= end)]
    return window
