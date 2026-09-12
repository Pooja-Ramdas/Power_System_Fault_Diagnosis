"""
Builds the unified CEN multimodal fault-diagnosis dataset:

  cen_dataset/
    CASE-0001/
      image.png          <- diagram derived from the case's named grid elements
      telemetry.csv       <- numeric window around the fault timestamp
      report_es.txt       <- original Spanish EAF report text (trimmed)
      report_en.txt       <- English translation
      meta.json           <- eaf_id, date/time, description, fault label, pdf_url
    CASE-0002/
    ...
    manifest.csv          <- one row per case, indexes all of the above

Run:  python build_dataset.py

Safe to interrupt and re-run -- already-built case folders are skipped.
"""
import os
import sys
import json
import random
import traceback
from datetime import datetime

import pandas as pd
from tqdm import tqdm

import config
from scraper import eaf_scraper, telemetry, topology, translate, infotecnica_api


def dir_size_mb(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            if os.path.exists(fp):
                total += os.path.getsize(fp)
    return total / (1024 * 1024)


def infer_fault_label(description):
    """Very lightweight rule-based label from the report title, so you have
    *something* to train/evaluate against immediately. Refine this against
    your rubric's actual fault taxonomy (short circuit / open circuit /
    breaker / transformer / overload / overheating) -- these EAF titles
    usually contain enough signal to do that with a slightly richer rule set
    or a small manually-labeled validation subset."""
    d = description.lower()
    if "sobrecarga" in d or "sobrecalent" in d:
        return "overload_overheating"
    if "transformador" in d:
        return "transformer_fault"
    if "interruptor" in d:
        return "breaker_fault"
    if "línea" in d or "linea" in d:
        return "line_fault"
    if "barra" in d:
        return "busbar_fault"
    return "other"


def main():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.RAW_CACHE_DIR, exist_ok=True)
    random.seed(config.RANDOM_SEED)

    # ---- 1. Scrape the EAF listing for every configured year -------------
    session = eaf_scraper.new_session()
    all_entries = []
    for year in config.EAF_YEARS:
        print(f"[list] scraping EAF index for {year} ...")
        entries = eaf_scraper.list_eaf_entries(
            year, config.EAF_LISTING_URL_TEMPLATE,
            max_pages=config.MAX_PAGES_PER_YEAR,
            delay=config.REQUEST_DELAY_SEC,
            session=session,
        )
        print(f"[list] {year}: found {len(entries)} reports")
        all_entries.extend(entries)

    if not all_entries:
        print("No EAF entries found -- CEN likely changed their page layout. "
              "Open EAF_LISTING_URL_TEMPLATE in a browser, compare against "
              "scraper/eaf_scraper.py's TITLE_RE / DOM-walk, and adjust.")
        sys.exit(1)

    if config.TARGET_N_CASES and len(all_entries) > config.TARGET_N_CASES:
        all_entries = random.sample(all_entries, config.TARGET_N_CASES)
    print(f"[list] processing {len(all_entries)} cases total")

    # ---- 2. Download + cache telemetry spreadsheets once ------------------
    telemetry_dfs = {}
    for name, url in config.TELEMETRY_XLSX_URLS.items():
        dest = os.path.join(config.RAW_CACHE_DIR, f"{name}.xlsx")
        try:
            telemetry.download_once(url, dest, session=session)
            df = telemetry.load_xlsx(dest)
            telemetry_dfs[name] = df
            print(f"[telemetry] loaded {name}: {df.shape[0]} rows, "
                  f"columns={list(df.columns)[:6]}...")
        except Exception as e:
            print(f"[telemetry] WARNING could not load '{name}' ({url}): {e}")

    # Preferred order to try when slicing a window for a given case
    telemetry_priority = ["hourly_generation", "generation_by_plant_2025",
                           "generation_by_technology"]

    # cache of installation lists per tipo, shared across all cases so we
    # only hit list_installations() once per type, not once per case
    installations_cache = {}

    # ---- 3. Build each case -------------------------------------------
    manifest_rows = []
    failures = []
    for i, entry in enumerate(tqdm(all_entries, desc="building cases")):
        case_id = f"CASE-{i+1:04d}"
        case_dir = os.path.join(config.OUTPUT_DIR, case_id)
        if os.path.exists(os.path.join(case_dir, "meta.json")):
            continue  # already built, resumable run

        try:
            os.makedirs(case_dir, exist_ok=True)
            event_dt = datetime.strptime(
                f"{entry['date']} {entry['time']}", "%d-%m-%Y %H:%M"
            )

            # -- text modality --
            pdf_path = os.path.join(config.RAW_CACHE_DIR, f"{entry['eaf_id']}.pdf")
            eaf_scraper.download_pdf(entry["pdf_url"], pdf_path, session=session)
            report_es = eaf_scraper.extract_report_text(
                pdf_path, max_pages=config.PDF_TEXT_MAX_PAGES
            )
            report_en = translate.translate_es_to_en(report_es)
            with open(os.path.join(case_dir, "report_es.txt"), "w", encoding="utf-8") as f:
                f.write(report_es)
            with open(os.path.join(case_dir, "report_en.txt"), "w", encoding="utf-8") as f:
                f.write(report_en)
            if not config.KEEP_RAW_PDFS and os.path.exists(pdf_path):
                os.remove(pdf_path)

            # -- image modality --
            image_path = os.path.join(case_dir, "image.png")
            image_is_real = False
            if config.USE_REAL_INFOTECNICA_IMAGES:
                substations, line_pair = topology._parse_elements_from_description(
                    entry["description"]
                )
                candidates = list(substations) + (list(line_pair) if line_pair else [])
                for name in candidates:
                    for tipo in ("subestaciones", "lineas"):
                        try:
                            ok = infotecnica_api.fetch_real_diagram(
                                tipo, name, image_path, installations_cache, session=session
                            )
                        except Exception:
                            ok = False
                        if ok:
                            image_is_real = True
                            break
                    if image_is_real:
                        break
            if not image_is_real:
                topology.build_case_diagram(
                    entry["description"], image_path, dpi=config.DIAGRAM_DPI
                )

            # -- numeric modality --
            telemetry_saved = False
            for name in telemetry_priority:
                if name not in telemetry_dfs:
                    continue
                try:
                    window = telemetry.extract_window(
                        telemetry_dfs[name], event_dt,
                        window_hours=config.TELEMETRY_WINDOW_HOURS,
                    )
                    if len(window) > 0:
                        window.to_csv(os.path.join(case_dir, "telemetry.csv"), index=False)
                        telemetry_saved = True
                        break
                except ValueError:
                    continue
            if not telemetry_saved:
                # still record the case; flag missing telemetry rather than
                # silently dropping the whole case, so you can see coverage
                pd.DataFrame().to_csv(os.path.join(case_dir, "telemetry.csv"), index=False)

            # -- metadata --
            meta = {
                **entry,
                "case_id": case_id,
                "event_datetime": event_dt.isoformat(),
                "fault_label": infer_fault_label(entry["description"]),
                "telemetry_available": telemetry_saved,
                "image_is_real_infotecnica": image_is_real,
            }
            with open(os.path.join(case_dir, "meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            manifest_rows.append({
                "case_id": case_id, "eaf_id": entry["eaf_id"],
                "event_datetime": event_dt.isoformat(),
                "description": entry["description"],
                "fault_label": meta["fault_label"],
                "telemetry_available": telemetry_saved,
                "image_is_real_infotecnica": image_is_real,
                "image_path": f"{case_id}/image.png",
                "telemetry_path": f"{case_id}/telemetry.csv",
                "report_en_path": f"{case_id}/report_en.txt",
                "report_es_path": f"{case_id}/report_es.txt",
            })

        except Exception as e:
            failures.append((entry.get("eaf_id", "?"), str(e)))
            traceback.print_exc()
            continue

        if i % 25 == 0:
            size_now = dir_size_mb(config.OUTPUT_DIR)
            if size_now > config.DATASET_SIZE_HARD_CAP_MB:
                print(f"[budget] hit {size_now:.0f} MB cap, stopping early "
                      f"at case {i+1}/{len(all_entries)}")
                break

    # ---- 4. Write manifest + summary --------------------------------------
    manifest_path = os.path.join(config.OUTPUT_DIR, "manifest.csv")
    if manifest_rows:
        pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    print(f"\nDone. {len(manifest_rows)} cases built, {len(failures)} failed.")
    if manifest_rows:
        n_real = sum(1 for r in manifest_rows if r["image_is_real_infotecnica"])
        print(f"Real InfoTecnica diagrams: {n_real}/{len(manifest_rows)} "
              f"({n_real/len(manifest_rows)*100:.0f}%) -- the rest used the "
              f"generated schematic fallback. If this is 0%, the API field "
              f"names in scraper/infotecnica_api.py likely need adjusting -- "
              f"re-run probe_infotecnica.py.")
    print(f"Dataset size: {dir_size_mb(config.OUTPUT_DIR):.1f} MB "
          f"(cap was {config.DATASET_SIZE_HARD_CAP_MB} MB)")
    if failures:
        print("First few failures (see traceback above for each):")
        for eaf_id, err in failures[:10]:
            print(f"  {eaf_id}: {err}")


if __name__ == "__main__":
    main()
