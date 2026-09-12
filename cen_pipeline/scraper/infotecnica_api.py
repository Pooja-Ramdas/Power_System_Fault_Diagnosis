"""
Client for InfoTecnica's public API -- this is what gets you REAL CEN images
(official "diagrama unilineal" documents per substation/line) instead of the
generated schematic in topology.py.

Field names below confirmed via probe_infotecnica.py against the live API.
"""
import os
import re
import requests
from urllib.parse import quote

BASE_URL = "https://api-infotecnica.coordinador.cl/v1"
USER_AGENT = "Mozilla/5.0 (academic research script; multimodal fault diagnosis project)"

# --- confirmed from probe_infotecnica.py -----------------------------------
LIST_RESULTS_KEY = None            # /v1/subestaciones/ returns a bare list, no wrapper key
NAME_FIELD = "nombre"              # e.g. "S/E CENTRAL ALFALFAL"
ID_FIELD = "id"                    # e.g. 199
DOCUMENTS_URL_TEMPLATE = BASE_URL + "/{tipo}/{id}/documentos/"   # confirmed: returns a bare list
DOCUMENT_TYPE_FIELD = "nombre"     # document records have no separate type field; "nombre" IS the label, e.g. "Diagrama unilinial"
# document records have no "url" field -- only id + filename. Download URL is
# built in download_document() below from the confirmed
# infotecnica.coordinador.cl/documents/{id}/{filename} pattern (unverified
# against this exact endpoint -- test on one case before a full run).
DOCUMENT_URL_FIELD = None
# -----------------------------------------------------------------------------

DIAGRAM_KEYWORDS = ("unilineal", "unilinial", "diagrama", "plano")


def _session():
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def list_installations(tipo="subestaciones", session=None):
    session = session or _session()
    r = session.get(f"{BASE_URL}/{tipo}/", timeout=30)
    r.raise_for_status()
    data = r.json()
    return data[LIST_RESULTS_KEY] if LIST_RESULTS_KEY else data


def find_installation_by_name(installations, target_name):
    """Fuzzy-ish match: normalize and look for the target substring inside
    each installation's name (and vice versa). Good enough for short CEN
    names like 'Quelentaro' or 'Puente Alto'; tighten with rapidfuzz if you
    get false matches on a larger installation list."""
    def norm(s):
        return re.sub(r"[^a-z0-9]", "", s.lower())

    t = norm(target_name)
    if not t:
        return None
    best = None
    for inst in installations:
        name = str(inst.get(NAME_FIELD, ""))
        n = norm(name)
        if not n:
            continue
        if t in n or n in t:
            best = inst
            break
    return best


def list_documents(tipo, installation_id, session=None):
    session = session or _session()
    url = DOCUMENTS_URL_TEMPLATE.format(tipo=tipo, id=installation_id)
    r = session.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()
    return data[LIST_RESULTS_KEY] if (LIST_RESULTS_KEY and isinstance(data, dict)) else data


def find_diagram_document(documents):
    for doc in documents:
        label = str(doc.get(DOCUMENT_TYPE_FIELD, "")) + " " + str(doc.get("nombre", ""))
        if any(k in label.lower() for k in DIAGRAM_KEYWORDS):
            return doc
    return None


def download_document(doc, dest_path, session=None):
    session = session or _session()
    if DOCUMENT_URL_FIELD and doc.get(DOCUMENT_URL_FIELD):
        url = doc[DOCUMENT_URL_FIELD]
    else:
        # No direct URL field in the API response -- build it from the
        # confirmed infotecnica.coordinador.cl/documents/{id}/{filename}
        # pattern. Unverified against this specific endpoint; if this 404s,
        # run one manual test and adjust this line.
        url = f"https://infotecnica.coordinador.cl/documents/{doc['id']}/{quote(doc['filename'])}"
    r = session.get(url, timeout=60)
    r.raise_for_status()
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(dest_path, "wb") as f:
        f.write(r.content)
    return dest_path


def fetch_real_diagram(tipo, installation_name, dest_path, installations_cache, session=None):
    """High-level helper: returns True + writes dest_path if a real diagram
    document was found and downloaded for the named installation, else False."""
    if tipo not in installations_cache:
        installations_cache[tipo] = list_installations(tipo, session=session)
    inst = find_installation_by_name(installations_cache[tipo], installation_name)
    if inst is None:
        return False
    try:
        docs = list_documents(tipo, inst[ID_FIELD], session=session)
    except Exception:
        return False
    diagram = find_diagram_document(docs)
    if diagram is None:
        return False
    try:
        download_document(diagram, dest_path, session=session)
        return True
    except Exception:
        return False