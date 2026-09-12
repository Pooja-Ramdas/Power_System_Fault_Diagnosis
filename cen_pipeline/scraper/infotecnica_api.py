"""
Client for InfoTecnica's public API -- this is what gets you REAL CEN images
(official "diagrama unilineal" documents per substation/line) instead of the
generated schematic in topology.py.

Field names + download mechanism below confirmed by hand against the live
API (see project notes): list endpoints return bare lists, document records
have no "url" field, and the actual file bytes come from the NESTED
document-detail endpoint -- which returns them with a misleading
"Content-Type: text/html" header, so do not branch on content-type here.
"""
import os
import re
import requests

BASE_URL = "https://api-infotecnica.coordinador.cl/v1"
USER_AGENT = "Mozilla/5.0 (academic research script; multimodal fault diagnosis project)"

# --- confirmed against the live API -----------------------------------
LIST_RESULTS_KEY = None            # /v1/subestaciones/ returns a bare list, no wrapper key
NAME_FIELD = "nombre"              # e.g. "S/E CENTRAL ALFALFAL"
ID_FIELD = "id"                    # e.g. 199 -- this is the INSTALLATION id, distinct from a
                                    # document's own "id_instalacion" field, which is NOT the same thing
DOCUMENTS_URL_TEMPLATE = BASE_URL + "/{tipo}/{id}/documentos/"   # confirmed: returns a bare list of document metadata
# document records have no separate "type" field or "url" field -- the label
# lives in "nombre" (sometimes misspelled "unilinial") and the actual file
# lives at the nested detail endpoint built in download_document() below,
# not at any field on the metadata record itself.
DOCUMENT_TYPE_FIELD = "nombre"
# -----------------------------------------------------------------------------

DIAGRAM_KEYWORDS = ("unilineal", "unilinial", "diagrama", "plano", "single line", "single-line")


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
    """Returns the first document whose nombre or filename mentions a
    single-line-diagram keyword. NOTE: some installations have several
    candidates (e.g. an older .rar and a newer, better .pdf) -- this just
    takes the first match for now. Ranking by extension/date is a later
    step, deliberately deferred until the basic download is verified."""
    for doc in documents:
        label = " ".join([
            str(doc.get(DOCUMENT_TYPE_FIELD, "")),
            str(doc.get("nombre", "")),
            str(doc.get("filename", "")),
        ])
        if any(k in label.lower() for k in DIAGRAM_KEYWORDS):
            return doc
    return None


def download_document(tipo, installation_id, doc, dest_path, session=None):
    """Downloads the actual file bytes for `doc` (as returned by
    find_diagram_document/list_documents) belonging to `installation_id`
    (the installation's own "id", e.g. 199 -- NOT the document's
    "id_instalacion" field, which is a different, unrelated id).

    The real file lives at the nested document-detail endpoint:
        {BASE_URL}/{tipo}/{installation_id}/documentos/{document_id}/
    This endpoint streams raw binary content but mislabels it as
    "Content-Type: text/html" -- do NOT call .json() or branch on
    content-type here, just write response.content to disk as-is.
    """
    session = session or _session()
    url = f"{BASE_URL}/{tipo}/{installation_id}/documentos/{doc['id']}/"
    r = session.get(url, timeout=120, allow_redirects=True)
    r.raise_for_status()
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(dest_path, "wb") as f:
        f.write(r.content)
    return dest_path


def fetch_real_diagram(tipo, installation_name, dest_path, installations_cache, session=None):
    """High-level helper: returns True + writes dest_path if a real diagram
    document was found and downloaded for the named installation, else False.

    NOTE: dest_path currently gets whatever bytes come back (which may be a
    .rar/.zip/.pdf/.dwg, not necessarily a directly-usable image) -- that is
    intentional for now per the staged plan: verify the raw download works
    end to end first, decide on archive-extraction/conversion afterward,
    and only then adjust what build_dataset.py does with the result.
    """
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
        download_document(tipo, inst[ID_FIELD], diagram, dest_path, session=session)
        return True
    except Exception:
        return False