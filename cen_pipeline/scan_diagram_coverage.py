"""
Read-only diagnostic -- does NOT modify build_dataset.py, topology.py, or
infotecnica_api.py, and does not write anything into cen_dataset/.

Scrapes your actually-configured EAF cases (config.EAF_YEARS), extracts the
substation/line name(s) each one names, resolves each unique name against
the InfoTecnica API, and reports how many would get a PDF diagram, an
archive-only (rar/zip/dwg) diagram, or no diagram at all -- so you know real
coverage numbers before investing more time in DWG conversion tooling.

Run:  python scan_diagram_coverage.py
"""
from collections import Counter

import config
from scraper import eaf_scraper, topology, infotecnica_api


def classify_documents(documents):
    """Returns ('pdf' | 'archive' | 'other' | 'none', matched_doc_or_None)."""
    matches = []
    for doc in documents:
        label = " ".join([
            str(doc.get("nombre", "")),
            str(doc.get("filename", "")),
        ]).lower()
        if any(k in label for k in infotecnica_api.DIAGRAM_KEYWORDS):
            matches.append(doc)
    if not matches:
        return "none", None
    # prefer a PDF match if any exists among the candidates
    for doc in matches:
        ext = str(doc.get("extension", "")).lower()
        if ext == ".pdf" or str(doc.get("filename", "")).lower().endswith(".pdf"):
            return "pdf", doc
    ext = str(matches[0].get("extension", "")).lower()
    if ext in (".rar", ".zip", ".7z", ".dwg"):
        return "archive", matches[0]
    return "other", matches[0]


def main():
    session = eaf_scraper.new_session()          # for www.coordinador.cl (Cloudflare-cleared)
    api_session = infotecnica_api._session()      # for api-infotecnica.coordinador.cl (plain --
                                                    # that API never needed the Cloudflare bypass,
                                                    # and the Chrome-impersonation headers on the
                                                    # `session` above actually break it, so keep
                                                    # these two fully separate)

    print(f"Scraping EAF cases for years {config.EAF_YEARS} ...")
    all_entries = []
    for year in config.EAF_YEARS:
        entries = eaf_scraper.list_eaf_entries(
            year, config.EAF_LISTING_URL_TEMPLATE,
            max_pages=config.MAX_PAGES_PER_YEAR,
            delay=config.REQUEST_DELAY_SEC, session=session,
        )
        print(f"  {year}: {len(entries)} cases")
        all_entries.extend(entries)
    print(f"Total cases: {len(all_entries)}")

    # extract every candidate installation name per case
    case_candidates = []
    all_names = set()
    for entry in all_entries:
        substations, line_pair = topology._parse_elements_from_description(entry["description"])
        names = list(substations) + (list(line_pair) if line_pair else [])
        case_candidates.append(names)
        all_names.update(names)
    print(f"Unique installation names referenced across all cases: {len(all_names)}")

    # resolve each unique name ONCE (not once per case) against both tipos
    print("\nLoading installation lists (subestaciones, lineas) ...")
    installations_cache = {}
    for tipo in ("subestaciones", "lineas"):
        try:
            installations_cache[tipo] = infotecnica_api.list_installations(tipo, session=api_session)
            print(f"  {tipo}: {len(installations_cache[tipo])} installations")
        except Exception as e:
            print(f"  {tipo}: FAILED ({e})")
            installations_cache[tipo] = []

    print("\nResolving each unique name -> installation -> documents -> diagram type ...")
    name_result = {}  # name -> ('pdf'|'archive'|'other'|'none'|'no_installation_match', doc)
    for i, name in enumerate(sorted(all_names)):
        best = None
        for tipo in ("subestaciones", "lineas"):
            inst = infotecnica_api.find_installation_by_name(installations_cache[tipo], name)
            if inst is None:
                continue
            try:
                docs = infotecnica_api.list_documents(tipo, inst[infotecnica_api.ID_FIELD], session=api_session)
            except Exception:
                continue
            kind, doc = classify_documents(docs)
            if kind == "pdf":
                best = (kind, doc)
                break  # can't do better than a PDF match
            if best is None or (best[0] == "none"):
                best = (kind, doc)
        name_result[name] = best if best else ("no_installation_match", None)
        if (i + 1) % 25 == 0:
            print(f"  ...{i+1}/{len(all_names)} names resolved")

    # ---- per-name summary --------------------------------------------------
    name_kind_counts = Counter(v[0] for v in name_result.values())
    print("\n=== Per unique installation name ===")
    for kind, count in name_kind_counts.most_common():
        print(f"  {kind:22s} {count}")

    # ---- per-case summary (what fetch_real_diagram's OR-across-candidates
    #      logic would actually achieve for each fault case) ---------------
    case_best = []
    for names in case_candidates:
        kinds = [name_result[n][0] for n in names if n in name_result]
        if "pdf" in kinds:
            case_best.append("pdf")
        elif "archive" in kinds:
            case_best.append("archive")
        elif "other" in kinds:
            case_best.append("other")
        else:
            case_best.append("none")

    case_kind_counts = Counter(case_best)
    n = len(case_best) or 1
    print(f"\n=== Per fault case ({len(case_best)} total) ===")
    for kind in ("pdf", "archive", "other", "none"):
        c = case_kind_counts.get(kind, 0)
        print(f"  {kind:10s} {c:5d}  ({c/n*100:.1f}%)")

    print("\nInterpretation:")
    print("  'pdf'      -> gettable as a real image today with a simple PDF->PNG step")
    print("  'archive'  -> real diagram exists but needs the DWG conversion pipeline")
    print("  'other'    -> a diagram-labeled doc exists in an unexpected format, inspect by hand")
    print("  'none'     -> would use the synthetic topology.py fallback")


if __name__ == "__main__":
    main()