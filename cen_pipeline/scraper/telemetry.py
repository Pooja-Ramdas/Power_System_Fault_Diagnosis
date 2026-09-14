"""
Downloads CEN's published hourly telemetry spreadsheets once, caches them,
and slices a time window around each fault event for the numeric modality.

All of the per-plant hourly generation files use a WIDE format:
  - One row per (plant, date)
  - Columns HORA 1 / HORA 2 / ... / HORA 24  (or 1 / 2 / ... / 24 for 2024)
  - A FECHA column (Timestamp for 2016-2023 and 2025; "DD-mes" string for 2024)

extract_window() melts these into a proper long-format dataframe
(one row per plant per hour) with a real 'timestamp' column, then
slices the +/- window around the event datetime.

The fallback 'hourly_generation.xlsx' is already long-format per-year-sheet.
"""

import os
import re
import requests
import pandas as pd
from datetime import datetime, timedelta

USER_AGENT = "Mozilla/5.0 (academic research script; multimodal fault diagnosis project)"

# Spanish month abbreviations used in the 2024 file's "DD-mes" date strings
_ES_MONTHS = {
    "ene": 1, "feb": 2, "mar": 3, "abr": 4, "may": 5, "jun": 6,
    "jul": 7, "ago": 8, "sep": 9, "oct": 10, "nov": 11, "dic": 12,
}


def _session():
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def download_once(url, dest_path, session=None, timeout=180):
    """Download `url` to `dest_path` if not already cached. Returns dest_path."""
    if os.path.exists(dest_path):
        return dest_path
    session = session or _session()
    r = session.get(url, timeout=timeout)
    r.raise_for_status()
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(dest_path, "wb") as f:
        f.write(r.content)
    return dest_path


# ---------------------------------------------------------------------------
# Sheet-metadata lookup: maps each file key to (sheet_name, date_col, hour_col_prefix)
# ---------------------------------------------------------------------------
# Confirmed by inspecting the real downloaded files with debug_telemetry_sheets.py

_FILE_META = {
    # generation_by_plant_2016_2019.xlsx
    "2016_2019": {
        "sheet": "2016 al 2019",
        "date_col": "FECHA",          # Timestamp (auto-parsed by pandas)
        "hour_prefix": "HORA ",       # columns "HORA 1" ... "HORA 24"
        "year_from_date": True,       # year is embedded in FECHA
    },
    # generation_by_plant_2020_2023.xlsx -- has TWO data sheets
    "2020_2022": {
        "sheet": "2020 al 2022",
        "date_col": "FECHA",
        "hour_prefix": "HORA ",
        "year_from_date": True,
    },
    "2023": {
        "sheet": "2023",
        "date_col": "FECHA",
        "hour_prefix": "HORA ",
        "year_from_date": True,
    },
    # generation_by_plant_2024_2024.xlsx -- yearless Spanish "DD-mes" dates
    "2024": {
        "sheet": "Hoja2",
        "date_col": "_ops_por_hora_new.fecha",  # e.g. "01-ene"
        "hour_prefix": "",            # columns are bare "1" .. "25"
        "year_from_date": False,      # must be supplied externally (it's always 2024)
        "fixed_year": 2024,
    },
    # generation_by_plant_2025_2025.xlsx
    "2025": {
        "sheet": "Principal",
        "date_col": "Fecha",          # Timestamp
        "hour_prefix": "Hora ",       # "Hora 1" ... "Hora 24"
        "year_from_date": True,
    },
}


def load_wide_sheet(xlsx_path, meta_key, needed_dates=None):
    """Load one of the per-plant wide-format sheets into a melted long-format df.

    Returns a DataFrame with columns:
        timestamp (datetime), mwh, plus plant metadata columns.

    Args:
        xlsx_path: Path to the xlsx file.
        meta_key: Key into _FILE_META describing the sheet structure.
        needed_dates: Optional set/list of datetime.date objects. When supplied,
            only rows whose FECHA matches one of these dates (±1 day buffer for
            the window boundary) are loaded before the melt. This dramatically
            reduces memory for the large 2024/2025 files (450K+ rows). Leave
            None to load all rows (needed for 2016-2019/2020-2023 which are
            loaded once and reused across many cases).
    """
    meta = _FILE_META[meta_key]
    df = pd.read_excel(xlsx_path, sheet_name=meta["sheet"])

    date_col = meta["date_col"]
    hour_prefix = meta["hour_prefix"]
    fixed_year = meta.get("fixed_year")

    # Identify the hour columns
    if hour_prefix:
        # "HORA 1" .. "HORA 24" or "Hora 1" .. "Hora 24"
        hour_cols = [c for c in df.columns if str(c).startswith(hour_prefix)
                     and str(c)[len(hour_prefix):].strip().isdigit()]
        # exclude HORA 25 (it's a duplicate of HORA 1 in some years, or empty)
        hour_cols = [c for c in hour_cols
                     if int(str(c)[len(hour_prefix):].strip()) <= 24]
    else:
        # 2024 file: bare numeric string columns "1" .. "25"
        hour_cols = [c for c in df.columns
                     if str(c).strip().isdigit() and 1 <= int(str(c).strip()) <= 24]

    # Detect the plant-name column (first non-date, non-hour column)
    meta_cols = [c for c in df.columns if c != date_col and c not in hour_cols]

    # Parse the date column
    if not meta.get("year_from_date"):
        # 2024: convert "DD-mes" -> date using the fixed year
        def _parse_dd_mes(s, yr=fixed_year):
            try:
                s = str(s).strip().lower()
                day_str, mon_str = s.split("-")
                day = int(day_str)
                month = _ES_MONTHS.get(mon_str)
                if month is None:
                    return pd.NaT
                return datetime(yr, month, day)
            except Exception:
                return pd.NaT
        df["_fecha_parsed"] = df[date_col].apply(_parse_dd_mes)
    else:
        df["_fecha_parsed"] = pd.to_datetime(df[date_col], errors="coerce")

    # Drop rows with no parseable date
    df = df.dropna(subset=["_fecha_parsed"])

    # Optional: filter to only the needed date rows before the expensive melt.
    # needed_dates should be a set of datetime.date objects; we expand by ±1 day
    # to safely cover window boundaries (build_dataset passes event dates ±1d).
    if needed_dates is not None and len(needed_dates) > 0:
        needed_dates_set = set(needed_dates)
        df = df[df["_fecha_parsed"].apply(lambda ts: ts.date() in needed_dates_set)]
        if len(df) == 0:
            # No rows matched -- return empty df with right shape (avoids error downstream)
            return pd.DataFrame(columns=["timestamp", "mwh"] + meta_cols)

    # Melt: (plant_cols + _fecha_parsed) × hour_cols -> long format
    id_vars = ["_fecha_parsed"] + meta_cols
    df_melted = df[id_vars + hour_cols].melt(
        id_vars=id_vars,
        value_vars=hour_cols,
        var_name="_hora_col",
        value_name="mwh",
    )

    # Extract the hour number (1..24) from the column name
    if hour_prefix:
        df_melted["hour"] = df_melted["_hora_col"].apply(
            lambda c: int(str(c)[len(hour_prefix):].strip())
        )
    else:
        df_melted["hour"] = df_melted["_hora_col"].apply(
            lambda c: int(str(c).strip())
        )

    # Build a proper datetime: date + (hour-1) hours (hour=1 means 00:00-01:00)
    df_melted["timestamp"] = df_melted.apply(
        lambda r: r["_fecha_parsed"] + timedelta(hours=r["hour"] - 1), axis=1
    )

    df_melted = df_melted.drop(columns=["_hora_col", "_fecha_parsed"])
    df_melted = df_melted.dropna(subset=["mwh"])
    df_melted = df_melted.sort_values("timestamp").reset_index(drop=True)
    return translate_columns(df_melted)


_COLUMN_TRANSLATION_MAP = {
    # 2016-2023 columns
    "NOMBRE CENTRAL": "plant_name",
    "LLAVE NOMBRE": "plant_key",
    "TIPO": "generation_type",
    "SUBTIPO": "generation_subtype",
    "REGIÓN": "region",
    "ernc/convencional": "energy_type",
    "Factor ERNC": "renewable_factor",
    # 2024 columns
    "central_name": "plant_name",
    "llave_nombre": "plant_key",
    "tipo_fuente": "generation_type",
    "subtipo_fuente": "generation_subtype",
    "region": "region",
    "tipo_energia": "energy_type",
    "ernc_factor": "renewable_factor",
    # 2025 columns
    "Central": "plant_name",
    "Llave": "plant_key",
    "Coordinado": "coordinator",
    "Grupo reporte": "reporting_group",
    "Tipo": "generation_type",
    "Subtipo": "generation_subtype",
    "Región": "region",
    # Aggregates / totals / general
    "TOTAL": "total",
    "Total": "total",
    "gen_mwh": "mwh",
}


def translate_columns(df):
    """Translates Spanish column names in telemetry DataFrame to English."""
    if df is None or len(df) == 0:
        return df

    new_cols = {}
    for col in df.columns:
        c_str = str(col).strip()
        if c_str in _COLUMN_TRANSLATION_MAP:
            new_cols[col] = _COLUMN_TRANSLATION_MAP[c_str]
        elif c_str.lower().startswith("regi"):
            new_cols[col] = "region"
        elif c_str.lower() in ("fecha", "timestamp"):
            new_cols[col] = "timestamp"
        elif c_str.lower() in ("hora", "hour"):
            new_cols[col] = "hour"
        elif c_str.lower() in ("gen_mwh", "mwh"):
            new_cols[col] = "mwh"
        else:
            new_cols[col] = c_str

    return df.rename(columns=new_cols)


def load_hourly_generation_sheet(xlsx_path, year):
    """Load one year-sheet from hourly_generation.xlsx (the aggregate fallback).

    That file has one sheet per year named '2000', '2001', ..., '2025'.
    Row 0 is a header row: the real columns are
      col 1 = 'fecha', col 2 = 'dia', col 3 = 'hora', col 4 = 'SEN' (MWh/h)
    as confirmed by debug_telemetry_sheets.py.
    """
    sheet_name = str(year)
    try:
        raw = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=None)
    except Exception:
        return None

    # Row 0 should be the header; find 'fecha' column index
    header_row = raw.iloc[0].astype(str).str.lower().tolist()
    try:
        fecha_idx = next(i for i, v in enumerate(header_row) if "fecha" in v)
        hora_idx  = next(i for i, v in enumerate(header_row) if "hora" in v)
        gen_idx   = next(i for i, v in enumerate(header_row)
                         if "generaci" in v or "sen" in v.upper())
    except StopIteration:
        return None

    df = raw.iloc[1:].reset_index(drop=True)
    df = df[[fecha_idx, hora_idx, gen_idx]].copy()
    df.columns = ["fecha", "hora", "gen_mwh"]
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df["hora"]  = pd.to_numeric(df["hora"],  errors="coerce")
    df = df.dropna(subset=["fecha", "hora"])
    # Build timestamp: fecha + (hora - 1) hours
    df["timestamp"] = df.apply(
        lambda r: r["fecha"] + timedelta(hours=int(r["hora"]) - 1), axis=1
    )
    df = df.drop(columns=["fecha", "hora"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return translate_columns(df)


def extract_window(df, event_dt, window_hours=48):
    """Slice rows within +/- window_hours/2 of event_dt.

    Expects df to have a 'timestamp' column (as returned by load_wide_sheet
    or load_hourly_generation_sheet). Returns an empty DataFrame if no rows
    fall in the window or if df is None/empty.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame()
    start = event_dt - timedelta(hours=window_hours / 2)
    end   = event_dt + timedelta(hours=window_hours / 2)
    mask = (df["timestamp"] >= start) & (df["timestamp"] <= end)
    df_window = df[mask].reset_index(drop=True)
    return translate_columns(df_window)

