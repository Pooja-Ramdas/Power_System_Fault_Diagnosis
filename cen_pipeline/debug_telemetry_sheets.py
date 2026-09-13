"""
Read-only, LOCAL-ONLY diagnostic -- inspects the xlsx files already
downloaded into _raw_cache/ by the previous debug_telemetry.py run. Does
NOT re-download anything, so this is fast even though some of these files
are 100-270MB.

For each cached file: lists every sheet name + a cheap row/column count
(via openpyxl in read-only/streaming mode, so this doesn't load the whole
sheet into memory), then previews the first few rows of whichever sheet
looks like the real data (most rows), plus raw (unparsed) sample values of
likely date/hour columns so we can see the actual format CEN used.

Run:  python debug_telemetry_sheets.py
"""
import glob
import os
import openpyxl
import pandas as pd
import config

for path in sorted(glob.glob(os.path.join(config.RAW_CACHE_DIR, "*.xlsx"))):
    print(f"\n{'='*70}\n{os.path.basename(path)}\n{'='*70}")
    try:
        wb = openpyxl.load_workbook(path, read_only=True)
    except Exception as e:
        print(f"  could not open with openpyxl: {e}")
        continue

    sheet_info = []
    for ws in wb.worksheets:
        sheet_info.append((ws.title, ws.max_row, ws.max_column))
        print(f"  sheet '{ws.title}': {ws.max_row} rows x {ws.max_column} cols")
    wb.close()

    if not sheet_info:
        continue
    # the real data sheet is almost certainly the one with the most rows
    best_sheet = max(sheet_info, key=lambda t: t[1])[0]
    print(f"  -> inspecting largest sheet: '{best_sheet}'")

    try:
        head = pd.read_excel(path, sheet_name=best_sheet, nrows=8)
    except Exception as e:
        print(f"  could not read head of '{best_sheet}': {e}")
        continue

    print(f"  columns ({len(head.columns)} total): {list(head.columns)[:20]}")
    print(head.head(5).to_string()[:1200])

    # look for anything date-like or hour-like and print RAW sample values
    for col in head.columns:
        cname = str(col).lower()
        if "fecha" in cname or "date" in cname:
            raw = pd.read_excel(path, sheet_name=best_sheet, usecols=[col], nrows=10)
            print(f"  raw sample values of date-like column '{col}': "
                  f"{raw[col].tolist()}")
