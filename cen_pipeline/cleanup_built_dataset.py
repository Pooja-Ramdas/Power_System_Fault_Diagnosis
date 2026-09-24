"""
Retroactive disk cleanup script for cen_dataset/ and _raw_cache/
"""
import os
import glob
import json
import shutil

dataset_dir = "cen_dataset"
cache_dir = "_raw_cache"

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
    print("STEP 3 — RETROACTIVE DISK CLEANUP")
    print("======================================================================\n")

    initial_dataset_size = get_size(dataset_dir) if os.path.exists(dataset_dir) else 0
    initial_cache_size = get_size(cache_dir) if os.path.exists(cache_dir) else 0

    pdf_deleted_cnt = 0
    pdf_bytes_saved = 0

    archive_deleted_cnt = 0
    archive_bytes_saved = 0

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

        ds = meta.get("diagram_source")

        # 1. Redundant PDF cleanup
        pdf_path = os.path.join(c, "official_diagram.pdf")
        png_path = os.path.join(c, "image.png")
        if os.path.exists(pdf_path) and os.path.exists(png_path) and os.path.getsize(png_path) > 0:
            sz = os.path.getsize(pdf_path)
            os.remove(pdf_path)
            pdf_deleted_cnt += 1
            pdf_bytes_saved += sz

        # 2. Archive cleanup (pending conversion)
        archive_files = [f for f in os.listdir(c) if f.startswith("official_diagram.") and not f.endswith(".pdf") and not f.endswith(".png")]
        if archive_files:
            for af in archive_files:
                af_path = os.path.join(c, af)
                sz = os.path.getsize(af_path)
                ext = os.path.splitext(af)[1]

                # Update meta.json with pending conversion metadata if not present
                if "pending_conversion_metadata" not in meta or meta["pending_conversion_metadata"] is None:
                    meta["pending_conversion_metadata"] = {
                        "filename": af,
                        "extension": ext,
                        "note": "Re-fetch on demand via InfoTecnica API"
                    }
                    meta["pending_conversion_path"] = None
                    with open(meta_path, "w", encoding="utf-8") as f:
                        json.dump(meta, f, ensure_ascii=False, indent=2)

                os.remove(af_path)
                archive_deleted_cnt += 1
                archive_bytes_saved += sz

    print(f"1. Redundant raw PDFs deleted: {pdf_deleted_cnt} files ({pdf_bytes_saved / (1024*1024):.2f} MB recovered)")
    print(f"2. Raw pending-conversion archives deleted: {archive_deleted_cnt} files ({archive_bytes_saved / (1024*1024):.2f} MB recovered)")

    # 3. Clean up _raw_cache/*.xlsx files
    cache_xlsx_deleted_cnt = 0
    cache_xlsx_bytes_saved = 0
    if os.path.exists(cache_dir):
        for f in os.listdir(cache_dir):
            if f.endswith(".xlsx"):
                fp = os.path.join(cache_dir, f)
                sz = os.path.getsize(fp)
                os.remove(fp)
                cache_xlsx_deleted_cnt += 1
                cache_xlsx_bytes_saved += sz

    print(f"3. Raw cache telemetry xlsx files deleted: {cache_xlsx_deleted_cnt} files ({cache_xlsx_bytes_saved / (1024*1024):.2f} MB recovered)")

    final_dataset_size = get_size(dataset_dir) if os.path.exists(dataset_dir) else 0
    final_cache_size = get_size(cache_dir) if os.path.exists(cache_dir) else 0

    total_recovered_mb = (pdf_bytes_saved + archive_bytes_saved + cache_xlsx_bytes_saved) / (1024*1024)
    print("\n----------------------------------------------------------------------")
    print(f"TOTAL RECOVERED SPACE: {total_recovered_mb:.2f} MB")
    print(f"cen_dataset/ size: {initial_dataset_size / (1024*1024):.2f} MB  ==>  {final_dataset_size / (1024*1024):.2f} MB")
    print(f"_raw_cache/ size:  {initial_cache_size / (1024*1024):.2f} MB  ==>  {final_cache_size / (1024*1024):.2f} MB")
    print("======================================================================\n")

if __name__ == "__main__":
    main()
