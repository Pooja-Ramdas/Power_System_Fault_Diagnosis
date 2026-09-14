"""Verify the 5 built cases have correct data in all modalities."""
import pandas as pd, json, os, glob

dataset_dir = "cen_dataset"
cases = sorted(glob.glob(os.path.join(dataset_dir, "CASE-*")))
print(f"Found {len(cases)} cases\n")

all_ok = True
for case_dir in cases:
    case_id = os.path.basename(case_dir)
    files = os.listdir(case_dir)
    
    # Check all expected files exist
    expected = {"image.png", "telemetry.csv", "report_es.txt", "report_en.txt", "meta.json"}
    missing = expected - set(files)
    if missing:
        print(f"[{case_id}] MISSING FILES: {missing}")
        all_ok = False
        continue
    
    # Check telemetry.csv
    tdf = pd.read_csv(os.path.join(case_dir, "telemetry.csv"))
    tel_ok = len(tdf) > 0 and "timestamp" in tdf.columns
    
    # Check reports
    es_size = os.path.getsize(os.path.join(case_dir, "report_es.txt"))
    en_size = os.path.getsize(os.path.join(case_dir, "report_en.txt"))
    
    # Check image
    img_size = os.path.getsize(os.path.join(case_dir, "image.png"))
    
    # Load meta
    meta = json.load(open(os.path.join(case_dir, "meta.json"), encoding="utf-8"))
    
    status = "OK" if tel_ok else "EMPTY_TELEMETRY"
    if not tel_ok:
        all_ok = False
    
    print(f"[{case_id}] {status}")
    print(f"  telemetry: {len(tdf)} rows, cols={list(tdf.columns)[:5]}")
    if len(tdf) > 0:
        print(f"  timestamp range: {tdf['timestamp'].min()} -> {tdf['timestamp'].max()}")
    print(f"  image.png: {img_size/1024:.0f}KB")
    print(f"  report_es: {es_size/1024:.0f}KB, report_en: {en_size/1024:.0f}KB")
    print(f"  diagram_source: {meta.get('diagram_source')}")
    print(f"  fault_label: {meta.get('fault_label')}")
    print(f"  event: {meta.get('event_datetime')}")
    print()

if all_ok:
    print("All cases verified OK!")
else:
    print("WARNING: Some cases have issues (see above)")
