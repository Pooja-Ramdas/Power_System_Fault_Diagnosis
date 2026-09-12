"""
Client for InfoTecnica's public API -- this is what gets you REAL CEN images
(official "diagrama unilineal" documents per substation/line) instead of the
generated schematic in topology.py.

*** RUN probe_infotecnica.py FIRST. *** The three constants right below are
my best inference from CEN's own documentation and a third-party integration
that already uses this API, but I could not execute a live request against
it from my side to confirm exact field names -- update these three constants
from what the probe prints, then this module should work as-is.
"""
import os
import re
import requests

BASE_URL = "https://api-infotecnica.coordinador.cl/v1"
USER_AGENT = "Mozilla/5.0 (academic research script; multimodal fault diagnosis project)"

# --- update these three after running probe_infotecnica.py -----------------
LIST_RESULTS_KEY = "results"       # wrapper key holding the array, or None if the endpoint returns a bare list
NAME_FIELD = "nombre"              # field holding the installation's display name
ID_FIELD = "id"                    # field holding the installation's id
DOCUMENTS_URL_TEMPLATE = BASE_URL + "/{tipo}/{id}/documentos/"   # confirm/adjust from probe Step 2
DOCUMENT_TYPE_FIELD = "tipo_documento"   # field on a document record that names its type
DOCUMENT_URL_FIELD = "url"               # field on a document record holding the downloadable file URL
# -----------------------------------------------------------------------------

DIAGRAM_KEYWORDS = ("unilineal", "diagrama", "plano")


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
    url = doc[DOCUMENT_URL_FIELD]
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
