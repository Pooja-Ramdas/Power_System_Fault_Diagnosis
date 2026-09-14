"""
Diagnostic script to analyze pattern matching in topology.py across all
built dataset cases and report descriptions.
"""
import glob
import json
import os
import sys

# Ensure cen_pipeline directory is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scraper.topology import _parse_elements_from_description

def check_built_cases():
    case_files = sorted(glob.glob("cen_dataset/CASE-*/meta.json"))
    if not case_files:
        print("No built cases found in cen_dataset/")
        return []

    entries = []
    for cf in case_files:
        try:
            with open(cf, "r", encoding="utf-8") as f:
                entries.append(json.load(f))
        except Exception:
            pass
    return entries

def main():
    entries = check_built_cases()
    if not entries:
        print("No cases to check.")
        return

    line_count = 0
    substation_count = 0
    no_match_count = 0

    no_matches = []

    for entry in entries:
        desc = entry.get("description", "")
        case_id = entry.get("case_id", "UNKNOWN")
        substations, line_pair = _parse_elements_from_description(desc)

        if line_pair:
            line_count += 1
        elif substations:
            substation_count += 1
        else:
            no_match_count += 1
            no_matches.append((case_id, desc))

    total = len(entries)
    print("=" * 70)
    print(f"DIAGRAM MATCH BREAKDOWN ({total} built cases total)")
    print("=" * 70)
    print(f"  LINE matches:            {line_count:4d} ({line_count/total*100:5.1f}%)")
    print(f"  SUBSTATION(S) ONLY:      {substation_count:4d} ({substation_count/total*100:5.1f}%)")
    print(f"  NO MATCH (bare circle):  {no_match_count:4d} ({no_match_count/total*100:5.1f}%)")
    print("=" * 70)

    if no_matches:
        print("\n--- NO MATCH CASES (SAMPLE DESCRIPTIONS) ---")
        for cid, desc in no_matches:
            print(f"[{cid}] {desc}")

if __name__ == "__main__":
    main()
