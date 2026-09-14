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
from scraper import eaf_scraper, telemetry, topology, translate, infotecnica_api, render_pdf


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
    session = eaf_scraper.new_session()          # www.coordinador.cl (Cloudflare-cleared)
    api_session = infotecnica_api._session()      # api-infotecnica.coordinador.cl (plain --
                                                    # this API never needed the Cloudflare bypass,
                                                    # and the `session` above's Chrome-impersonation
                                                    # headers actually break it, so keep separate)
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
    # Per-plant wide-format sheets, melted to long once at startup.
    # Keyed by event year -> melted long-format DataFrame with 'timestamp' col.
    telemetry_by_year = {}   # year (int) -> melted df  (or None if unavailable)

    # Map: event year -> (xlsx_path, meta_key, multi_sheet_keys)
    # multi_sheet_keys: for the 2020-2023 file which has 2 data sheets
    _year_to_source = {
        2016: ("generation_by_plant_2016_2019.xlsx", ["2016_2019"]),
        2017: ("generation_by_plant_2016_2019.xlsx", ["2016_2019"]),
        2018: ("generation_by_plant_2016_2019.xlsx", ["2016_2019"]),
        2019: ("generation_by_plant_2016_2019.xlsx", ["2016_2019"]),
        2020: ("generation_by_plant_2020_2023.xlsx", ["2020_2022"]),
        2021: ("generation_by_plant_2020_2023.xlsx", ["2020_2022"]),
        2022: ("generation_by_plant_2020_2023.xlsx", ["2020_2022"]),
        2023: ("generation_by_plant_2020_2023.xlsx", ["2023"]),
        2024: ("generation_by_plant_2024_2024.xlsx", ["2024"]),
        2025: ("generation_by_plant_2025_2025.xlsx", ["2025"]),
    }

    # Download the xlsx files (cached after first run)
    for (y0, y1), url in config.TELEMETRY_XLSX_BY_YEAR_RANGE.items():
        name = f"generation_by_plant_{y0}_{y1}"
        dest = os.path.join(config.RAW_CACHE_DIR, f"{name}.xlsx")
        try:
            telemetry.download_once(url, dest, session=session)
        except Exception as e:
            print(f"[telemetry] WARNING: could not download '{name}': {e}")

    # Download the aggregate hourly_generation.xlsx (for fallback)
    hourly_gen_path = os.path.join(config.RAW_CACHE_DIR, "hourly_generation.xlsx")
    for name, url in config.TELEMETRY_XLSX_URLS.items():
        if name != "hourly_generation":
            continue
        try:
            telemetry.download_once(url, hourly_gen_path, session=session)
        except Exception as e:
            print(f"[telemetry] WARNING: could not download hourly_generation: {e}")

    # Melt each required sheet once upfront, filtered to just the dates we need.
    # Pre-compute: for each meta_key, the set of date objects needed across all
    # cases that map to that sheet. We expand ±1 day to safely cover window edges.
    _loaded_meta_keys = {}   # meta_key -> melted df, shared across years
    event_years = set(e.get("year") or datetime.strptime(
        f"{e['date']} {e['time']}", "%d-%m-%Y %H:%M").year for e in all_entries)

    # Build: meta_key -> set of date objects (from event_dt ±1 day for window buffer)
    from datetime import date as _date, timedelta as _td
    _meta_key_needed_dates = {}
    for entry in all_entries:
        try:
            ev_dt = datetime.strptime(f"{entry['date']} {entry['time']}", "%d-%m-%Y %H:%M")
            yr = ev_dt.year
        except Exception:
            continue
        if yr not in _year_to_source:
            continue
        _, mk_list = _year_to_source[yr]
        # ±1 day around event to cover the 48h window boundaries
        for delta in (-1, 0, 1):
            d = (ev_dt + _td(days=delta)).date()
            for mk in mk_list:
                _meta_key_needed_dates.setdefault(mk, set()).add(d)

    for yr in sorted(event_years):
        if yr in telemetry_by_year:
            continue
        if yr not in _year_to_source:
            telemetry_by_year[yr] = None
            continue
        fname, meta_keys = _year_to_source[yr]
        xlsx_path = os.path.join(config.RAW_CACHE_DIR, fname)
        if not os.path.exists(xlsx_path):
            print(f"[telemetry] WARNING: {fname} not found in cache, skipping year {yr}")
            telemetry_by_year[yr] = None
            continue
        # load / reuse each meta_key sheet (with date filter for speed)
        frames = []
        for mk in meta_keys:
            if mk not in _loaded_meta_keys:
                needed = _meta_key_needed_dates.get(mk)
                n_dates = len(needed) if needed else 0
                print(f"[telemetry] loading sheet '{mk}' from {fname} "
                      f"(filtering to {n_dates} unique dates) ...")
                try:
                    _loaded_meta_keys[mk] = telemetry.load_wide_sheet(
                        xlsx_path, mk, needed_dates=needed
                    )
                    print(f"[telemetry] '{mk}' -> {len(_loaded_meta_keys[mk])} rows")
                except Exception as e:
                    print(f"[telemetry] WARNING: could not load sheet '{mk}': {e}")
                    _loaded_meta_keys[mk] = None
            if _loaded_meta_keys[mk] is not None:
                frames.append(_loaded_meta_keys[mk])
        telemetry_by_year[yr] = pd.concat(frames, ignore_index=True) if frames else None

    # Also load the aggregate hourly-generation per-year fallback sheets lazily
    _hourly_gen_by_year = {}   # year -> df (loaded on demand)

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
            diagram_source = "synthetic_fallback"
            real_hit = None
            if config.USE_REAL_INFOTECNICA_IMAGES:
                substations, line_pair = topology._parse_elements_from_description(
                    entry["description"]
                )
                candidates = list(substations) + (list(line_pair) if line_pair else [])
                raw_dest_no_ext = os.path.join(case_dir, "official_diagram")
                for name in candidates:
                    for tipo in ("subestaciones", "lineas"):
                        try:
                            result = infotecnica_api.fetch_real_diagram(
                                tipo, name, raw_dest_no_ext, installations_cache,
                                session=api_session,
                            )
                        except Exception:
                            result = {"found": False}
                        if result.get("found"):
                            real_hit = result
                            break
                    if real_hit:
                        break

            if real_hit and real_hit["is_pdf"]:
                try:
                    render_pdf.render_pdf_page(real_hit["path"], image_path)
                    diagram_source = "official_pdf"
                except Exception as e:
                    print(f"[image] PDF render failed for {case_id} ({e}); "
                          f"using synthetic fallback")
            elif real_hit:
                # real official document acquired (e.g. .rar/.dwg) but not yet
                # renderable to a pixel image -- keep the raw file (it's on
                # disk at real_hit["path"], NOT deleted) and mark this case so
                # a later DWG/archive->PNG conversion pass can pick it up
                # without re-scraping. image.png still gets the synthetic
                # fallback for now so every case has a directly usable image.
                diagram_source = "official_archive_pending_conversion"

            if diagram_source != "official_pdf":
                topology.build_case_diagram(
                    entry["description"], image_path, dpi=config.DIAGRAM_DPI
                )
            image_is_real = diagram_source == "official_pdf"

            # -- numeric modality --
            telemetry_saved = False
            ev_year = event_dt.year
            # Try per-plant wide-format df first (year-matched)
            per_plant_df = telemetry_by_year.get(ev_year)
            if per_plant_df is not None and len(per_plant_df) > 0:
                window = telemetry.extract_window(
                    per_plant_df, event_dt,
                    window_hours=config.TELEMETRY_WINDOW_HOURS,
                )
                if len(window) > 0:
                    window.to_csv(os.path.join(case_dir, "telemetry.csv"), index=False)
                    telemetry_saved = True
            # Fallback: hourly_generation.xlsx (system-wide aggregate, long-format)
            if not telemetry_saved and os.path.exists(hourly_gen_path):
                if ev_year not in _hourly_gen_by_year:
                    _hourly_gen_by_year[ev_year] = telemetry.load_hourly_generation_sheet(
                        hourly_gen_path, ev_year
                    )
                agg_df = _hourly_gen_by_year.get(ev_year)
                if agg_df is not None and len(agg_df) > 0:
                    window = telemetry.extract_window(
                        agg_df, event_dt,
                        window_hours=config.TELEMETRY_WINDOW_HOURS,
                    )
                    if len(window) > 0:
                        window.to_csv(os.path.join(case_dir, "telemetry.csv"), index=False)
                        telemetry_saved = True
            if not telemetry_saved:
                # Still record the case; flag missing telemetry rather than
                # silently dropping the whole case, so you can see coverage.
                pd.DataFrame().to_csv(os.path.join(case_dir, "telemetry.csv"), index=False)

            # -- metadata --
            meta = {
                **entry,
                "case_id": case_id,
                "event_datetime": event_dt.isoformat(),
                "fault_label": infer_fault_label(entry["description"]),
                "telemetry_available": telemetry_saved,
                "diagram_source": diagram_source,
                "image_is_real_infotecnica": image_is_real,  # kept for backwards-compat
                "pending_conversion_path": (
                    os.path.relpath(real_hit["path"], config.OUTPUT_DIR)
                    if diagram_source == "official_archive_pending_conversion" else None
                ),
            }
            with open(os.path.join(case_dir, "meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            manifest_rows.append({
                "case_id": case_id, "eaf_id": entry["eaf_id"],
                "event_datetime": event_dt.isoformat(),
                "description": entry["description"],
                "fault_label": meta["fault_label"],
                "telemetry_available": telemetry_saved,
                "diagram_source": diagram_source,
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
        from collections import Counter
        counts = Counter(r["diagram_source"] for r in manifest_rows)
        n = len(manifest_rows)
        print("Image modality source breakdown:")
        for source in ("official_pdf", "official_archive_pending_conversion", "synthetic_fallback"):
            c = counts.get(source, 0)
            print(f"  {source:35s} {c:5d}  ({c/n*100:.1f}%)")
        n_pending = counts.get("official_archive_pending_conversion", 0)
        if n_pending:
            print(f"\n{n_pending} cases have a real official document saved on disk "
                  f"(CASE-XXXX/official_diagram.<ext>) but still need the "
                  f"archive/DWG -> PNG conversion step to become image.png. "
                  f"Nothing needs to be re-scraped for those when that's built.")
    print(f"Dataset size: {dir_size_mb(config.OUTPUT_DIR):.1f} MB "
          f"(cap was {config.DATASET_SIZE_HARD_CAP_MB} MB)")
    if failures:
        print("First few failures (see traceback above for each):")
        for eaf_id, err in failures[:10]:
            print(f"  {eaf_id}: {err}")


if __name__ == "__main__":
    main()
