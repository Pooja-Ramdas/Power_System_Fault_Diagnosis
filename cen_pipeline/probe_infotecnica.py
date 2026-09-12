"""
Run this FIRST, by itself, before touching the rest of the pipeline:

    python probe_infotecnica.py

It hits the InfoTecnica public API for one installation type, prints the raw
JSON shape it gets back (keys + one sample record), then tries a few common
REST conventions for that installation's document list (where the real
"diagrama unilineal" / single-line diagram file should be). Whichever works
tells us exactly what to hard-code into scraper/infotecnica_api.py.

If everything 404s or returns HTML instead of JSON, that tells us the API
needs auth after all (their developer portal at portal.api.coordinador.cl
does require OpenID/API-key registration for some services) -- paste the
output back and we'll adjust the approach (e.g. register for a key, or fall
back to the per-case schematic + a manual formal-request diagram for a
curated subset).
"""
import json
import requests

BASE = "https://api-infotecnica.coordinador.cl/v1"
HEADERS = {"User-Agent": "Mozilla/5.0 (academic research script)"}


def try_get(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        print(f"\nGET {url}\n  status={r.status_code}  content-type={r.headers.get('content-type')}")
        if "json" in r.headers.get("content-type", ""):
            data = r.json()
            print("  JSON OK. Top-level type:", type(data).__name__)
            if isinstance(data, dict):
                print("  top-level keys:", list(data.keys())[:20])
                # try to find the list of records inside common wrapper keys
                for key in ("results", "data", "items", "instalaciones"):
                    if key in data and isinstance(data[key], list) and data[key]:
                        print(f"  found list under '{key}', first record:")
                        print(" ", json.dumps(data[key][0], indent=2, ensure_ascii=False)[:800])
                        return data[key][0]
            elif isinstance(data, list) and data:
                print("  first record:")
                print(" ", json.dumps(data[0], indent=2, ensure_ascii=False)[:800])
                return data[0]
            return data
        else:
            print("  (not JSON) first 300 chars:", r.text[:300])
    except Exception as e:
        print(f"  ERROR: {e}")
    return None


if __name__ == "__main__":
    print("=== Step 1: list substations ===")
    record = try_get(f"{BASE}/subestaciones/")
    if record is None:
        record = try_get(f"{BASE}/subestaciones")

    inst_id = None
    if isinstance(record, dict):
        for key in ("id", "id_instalacion", "idInstalacion", "codigo", "mnemotecnico"):
            if key in record:
                inst_id = record[key]
                print(f"\nUsing id field '{key}' = {inst_id}")
                break

    if inst_id is not None:
        print("\n=== Step 2: try common document-list URL shapes ===")
        for url in [
            f"{BASE}/subestaciones/{inst_id}/documentos/",
            f"{BASE}/subestaciones/{inst_id}/documentos",
            f"{BASE}/documentos/?instalacion={inst_id}",
            f"{BASE}/subestaciones/documentos/?id={inst_id}",
        ]:
            try_get(url)
    else:
        print("\nCouldn't find an id field automatically -- look at the sample "
              "record printed above and re-run Step 2 by hand with the right key.")
