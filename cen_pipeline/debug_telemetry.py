"""
Standalone, read-only diagnostic -- doesn't touch cen_dataset/ or any other
file. Downloads/loads each configured telemetry xlsx and prints its actual
shape and structure, so we can see whether extract_window() is failing due
to a download problem, a wide/pivot layout, or a date-range mismatch.

Run:  python debug_telemetry.py
"""
import os
import config
from scraper import eaf_scraper, telemetry

session = eaf_scraper.new_session()  # same domain as EAF reports, needs the Cloudflare bypass

# combine both the year-matched per-plant files and the aggregate fallbacks
# into one (name, url) list to inspect
all_files = {
    f"generation_by_plant_{y0}_{y1}": url
    for (y0, y1), url in config.TELEMETRY_XLSX_BY_YEAR_RANGE.items()
}
all_files.update(config.TELEMETRY_XLSX_URLS)

for name, url in all_files.items():
    print(f"\n{'='*70}\n{name}\n{url}\n{'='*70}")
    dest = os.path.join(config.RAW_CACHE_DIR, f"{name}.xlsx")
    try:
        telemetry.download_once(url, dest, session=session)
    except Exception as e:
        print(f"  DOWNLOAD FAILED: {e}")
        continue

    size_kb = os.path.getsize(dest) / 1024
    print(f"  downloaded OK, {size_kb:.0f} KB on disk")

    try:
        df = telemetry.load_xlsx(dest)
    except Exception as e:
        print(f"  LOAD FAILED (corrupt file? wrong format?): {e}")
        continue

    print(f"  shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"  column names (first 15): {list(df.columns)[:15]}")
    print(f"  dtypes (first 8 cols):\n{df.dtypes.head(8)}")
    print(f"  first 3 rows (first 6 cols):\n{df.iloc[:3, :6]}")

    dt_col = telemetry._guess_datetime_col(df)
    print(f"  auto-detected datetime column: {dt_col}")
    if dt_col:
        import pandas as pd
        parsed = pd.to_datetime(df[dt_col], errors="coerce")
        print(f"  parsed date range: {parsed.min()} to {parsed.max()}")
        print(f"  rows that failed to parse as a date: {parsed.isna().sum()} / {len(parsed)}")
    else:
        print("  NO datetime column found -- this file is likely in a wide/pivot "
              "format (e.g. one column per hour, or per date) rather than one "
              "row per timestamp. Paste this script's output back and we'll "
              "write a format-specific extractor for this file.")
