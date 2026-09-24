"""
Audit disk usage across _raw_cache and cen_dataset/
"""
import os
import glob
import json
from collections import defaultdict

cache_dir = "_raw_cache"
dataset_dir = "cen_dataset"

def get_size(path):
    if os.path.isfile(path):
        return os.path.getsize(path)
    total = 0
    for root, dirs, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            if not os.path.islink(fp):
                total += os.path.getsize(fp)
    return total

def main():
    print("======================================================================")
    print("STEP 1 — DISK USAGE AUDIT REPORT")
    print("======================================================================\n")

    # 1. raw cache breakdown
    print("1. Raw Cache (_raw_cache/):")
    cache_total = 0
    if os.path.exists(cache_dir):
        for f in sorted(os.listdir(cache_dir)):
            fp = os.path.join(cache_dir, f)
            sz = get_size(fp)
            cache_total += sz
            print(f"   {f:45s}: {sz / (1024*1024):7.2f} MB")
    print(f"   --> TOTAL _raw_cache: {cache_total / (1024*1024):7.2f} MB\n")

    # 2. cen_dataset total
    ds_total = get_size(dataset_dir) if os.path.exists(dataset_dir) else 0
    print(f"2. Total cen_dataset/ size: {ds_total / (1024*1024):7.2f} MB\n")

    # 3. Breakdown per diagram_source
    source_stats = defaultdict(lambda: {
        "count": 0,
        "size": 0,
        "file_type_bytes": defaultdict(int),
        "image_png_sizes": []
    })
    pdf_and_png_count = 0
    pdf_and_png_bytes = 0

    cases = sorted(glob.glob(os.path.join(dataset_dir, "CASE-*")))
    for c in cases:
        meta_path = os.path.join(c, "meta.json")
        if not os.path.exists(meta_path):
            continue
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            continue
        
        ds = meta.get("diagram_source", "unknown")
        c_size = get_size(c)
        source_stats[ds]["count"] += 1
        source_stats[ds]["size"] += c_size
        
        files = os.listdir(c)
        has_pdf = False
        has_png = False
        for fn in files:
            fp = os.path.join(c, fn)
            sz = get_size(fp)
            # categorize file
            if fn.endswith(".pdf"):
                source_stats[ds]["file_type_bytes"]["raw_pdf"] += sz
                has_pdf = True
                if fn.startswith("official_diagram"):
                    pdf_and_png_bytes += sz
            elif fn.endswith(".png"):
                source_stats[ds]["file_type_bytes"]["png_image"] += sz
                if fn == "image.png":
                    source_stats[ds]["image_png_sizes"].append(sz)
                    has_png = True
            elif fn.endswith(".rar") or fn.endswith(".dwg") or fn.endswith(".zip") or fn.endswith(".7z"):
                source_stats[ds]["file_type_bytes"]["raw_archive"] += sz
            elif fn.endswith(".csv"):
                source_stats[ds]["file_type_bytes"]["telemetry_csv"] += sz
            elif fn.endswith(".txt"):
                source_stats[ds]["file_type_bytes"]["report_txt"] += sz
            elif fn.endswith(".json"):
                source_stats[ds]["file_type_bytes"]["meta_json"] += sz

        if has_pdf and has_png:
            pdf_and_png_count += 1

    print("3. Breakdown by diagram_source in cen_dataset/:")
    hdr = f"   {'diagram_source':38s} | {'Cases':5s} | {'Total MB':9s} | {'Avg KB/case':12s}"
    print(hdr)
    print("   " + "-" * len(hdr))
    for ds, st in source_stats.items():
        cnt = st["count"]
        tot_mb = st["size"] / (1024*1024)
        avg_kb = (st["size"] / 1024) / cnt if cnt > 0 else 0
        print(f"   {ds:38s} | {cnt:5d} | {tot_mb:9.2f} | {avg_kb:12.2f}")
        for ft, b in st["file_type_bytes"].items():
            ft_mb = b / (1024*1024)
            ft_avg = (b / 1024) / cnt if cnt > 0 else 0
            print(f"       -> {ft:20s}: {ft_mb:6.2f} MB total ({ft_avg:7.2f} KB/case avg)")

    print("\n4. Redundant Raw PDF Check (official_diagram.pdf AND image.png present):")
    print(f"   Cases with BOTH official_diagram.pdf & image.png: {pdf_and_png_count}")
    print(f"   Total redundant space taken by official_diagram.pdf: {pdf_and_png_bytes / (1024*1024):.2f} MB\n")

    print("5. image.png Average Size by diagram_source:")
    for ds, st in source_stats.items():
        pngs = st["image_png_sizes"]
        if pngs:
            avg_png_kb = (sum(pngs) / len(pngs)) / 1024
            max_png_kb = max(pngs) / 1024
            min_png_kb = min(pngs) / 1024
            print(f"   {ds:38s}: {len(pngs)} images, avg {avg_png_kb:.2f} KB (min {min_png_kb:.2f} KB, max {max_png_kb:.2f} KB)")

    print("\n======================================================================")

if __name__ == "__main__":
    main()
