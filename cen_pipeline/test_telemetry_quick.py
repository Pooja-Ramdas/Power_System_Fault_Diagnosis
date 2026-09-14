"""
Quick smoke test for rewritten telemetry.py -- uses needed_dates filter
to avoid loading millions of unneeded rows (fast even for the big files).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scraper import telemetry
from datetime import datetime, date, timedelta
import config

# Test events: one per year relevant to our dataset
TESTS = [
    ("2024", "generation_by_plant_2024_2024.xlsx", "2024",  datetime(2024, 6, 15, 13, 0)),
    ("2025", "generation_by_plant_2025_2025.xlsx", "2025",  datetime(2025, 3, 20, 10, 0)),
    ("2023", "generation_by_plant_2020_2023.xlsx", "2023",  datetime(2023, 8, 10, 9, 0)),
    ("2020", "generation_by_plant_2020_2023.xlsx", "2020_2022", datetime(2021, 5, 3, 14, 0)),
    ("2016-2019", "generation_by_plant_2016_2019.xlsx", "2016_2019", datetime(2019, 2, 7, 8, 0)),
]

for label, fname, meta_key, ev_dt in TESTS:
    path = os.path.join(config.RAW_CACHE_DIR, fname)
    if not os.path.exists(path):
        print(f"SKIP {label}: {fname} not in cache")
        continue

    # Build needed_dates: event date ±1 day
    needed = set()
    for delta in (-1, 0, 1):
        needed.add((ev_dt + timedelta(days=delta)).date())

    print(f"\nTesting {label} (event={ev_dt.date()}, {len(needed)} unique dates) ...")
    try:
        df = telemetry.load_wide_sheet(path, meta_key, needed_dates=needed)
    except Exception as exc:
        print(f"  ERROR loading: {exc}")
        continue

    print(f"  shape: {df.shape}")
    print(f"  columns: {list(df.columns)[:8]}")
    if "timestamp" in df.columns:
        print(f"  timestamp range: {df['timestamp'].min()} -> {df['timestamp'].max()}")
    w = telemetry.extract_window(df, ev_dt, window_hours=48)
    print(f"  window ({ev_dt}, ±24h): {len(w)} rows")
    if len(w) == 0:
        print("  WARNING: empty window -- check date parsing!")
    else:
        print(w[["timestamp", "mwh"]].head(3).to_string())

# Also test the hourly_generation fallback
print("\n--- Testing hourly_generation fallback (2024) ---")
hg_path = os.path.join(config.RAW_CACHE_DIR, "hourly_generation.xlsx")
if os.path.exists(hg_path):
    agg = telemetry.load_hourly_generation_sheet(hg_path, 2024)
    if agg is not None:
        print(f"  shape: {agg.shape}, cols: {list(agg.columns)}")
        ev = datetime(2024, 6, 15, 13, 0)
        w = telemetry.extract_window(agg, ev, window_hours=48)
        print(f"  window: {len(w)} rows")
        if len(w):
            print(w.head(3).to_string())
    else:
        print("  FAILED to load 2024 sheet")
else:
    print("  hourly_generation.xlsx not in cache")

print("\nDone.")
