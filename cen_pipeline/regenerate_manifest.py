"""
Regenerate a complete manifest.csv from all existing meta.json files.
Run this after build_dataset.py completes (or anytime) to get a manifest
that includes ALL built cases, not just those from the most recent run.
"""
import os
import sys
import json
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def regenerate_manifest():
    output_dir = config.OUTPUT_DIR
    manifest_path = os.path.join(output_dir, "manifest.csv")

    rows = []
    case_dirs = sorted(
        d for d in os.listdir(output_dir)
        if d.startswith("CASE-") and os.path.isdir(os.path.join(output_dir, d))
    )

    for case_id in case_dirs:
        meta_path = os.path.join(output_dir, case_id, "meta.json")
        if not os.path.exists(meta_path):
            continue
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        rows.append({
            "case_id": meta.get("case_id", case_id),
            "eaf_id": meta.get("eaf_id"),
            "event_datetime": meta.get("event_datetime"),
            "description": meta.get("description"),
            "fault_label": meta.get("fault_label"),
            "telemetry_available": meta.get("telemetry_available"),
            "diagram_source": meta.get("diagram_source"),
            "image_is_real_infotecnica": meta.get("image_is_real_infotecnica"),
            "image_path": f"{case_id}/image.png",
            "telemetry_path": f"{case_id}/telemetry.csv",
            "report_en_path": f"{case_id}/report_en.txt",
            "report_es_path": f"{case_id}/report_es.txt",
        })

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(manifest_path, index=False)
        print(f"Manifest regenerated: {len(rows)} cases -> {manifest_path}")
    else:
        print("No completed cases found.")

    return len(rows)


if __name__ == "__main__":
    regenerate_manifest()
