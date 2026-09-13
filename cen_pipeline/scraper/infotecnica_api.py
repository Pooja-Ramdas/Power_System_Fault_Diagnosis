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


def _doc_extension(doc):
    ext = str(doc.get("extension", "")).lower().strip()
    if ext and not ext.startswith("."):
        ext = "." + ext
    if not ext:
        fn = str(doc.get("filename", ""))
        if "." in fn:
            ext = "." + fn.rsplit(".", 1)[-1].lower()
    return ext


def _doc_rank(doc):
    """Lower is better. PDFs (and already-raster images, if any ever show
    up) win outright since they need no further conversion; everything
    else (rar/zip/7z/dwg) is usable but needs the archive/DWG conversion
    step, so it's ranked behind a directly-usable PDF."""
    ext = _doc_extension(doc)
    if ext == ".pdf":
        return 0
    if ext in (".png", ".jpg", ".jpeg"):
        return 0
    if ext in (".rar", ".zip", ".7z", ".dwg"):
        return 2
    return 1


def find_diagram_document(documents):
    """Returns the BEST matching diagram document: prefers a directly
    usable PDF/image over an archive (.rar/.zip/.dwg) when an installation
    has both, since a PDF needs no further conversion step."""
    matches = []
    for doc in documents:
        label = " ".join([
            str(doc.get(DOCUMENT_TYPE_FIELD, "")),
            str(doc.get("nombre", "")),
            str(doc.get("filename", "")),
        ])
        if any(k in label.lower() for k in DIAGRAM_KEYWORDS):
            matches.append(doc)
    if not matches:
        return None
    matches.sort(key=_doc_rank)
    return matches[0]


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


def fetch_real_diagram(tipo, installation_name, dest_path_no_ext, installations_cache, session=None):
    """High-level helper. Returns a dict:
        {"found": False}
      or
        {"found": True, "path": <actual saved path, with real extension>,
         "extension": ".pdf" | ".rar" | ".dwg" | ..., "is_pdf": bool,
         "installation_name": <matched name>, "document": <doc metadata>}

    `dest_path_no_ext` should be a path WITHOUT an extension (e.g.
    ".../CASE-0001/diagram") -- the real extension is appended based on
    what actually came back, since it may be a PDF, an archive, or (rarely)
    something else, and saving arbitrary binary content under a ".png" name
    would silently corrupt the image modality for that case.
    """
    if tipo not in installations_cache:
        installations_cache[tipo] = list_installations(tipo, session=session)
    inst = find_installation_by_name(installations_cache[tipo], installation_name)
    if inst is None:
        return {"found": False}
    try:
        docs = list_documents(tipo, inst[ID_FIELD], session=session)
    except Exception:
        return {"found": False}
    diagram = find_diagram_document(docs)
    if diagram is None:
        return {"found": False}
    ext = _doc_extension(diagram) or ".bin"
    dest_path = dest_path_no_ext + ext
    try:
        download_document(tipo, inst[ID_FIELD], diagram, dest_path, session=session)
    except Exception:
        return {"found": False}
    return {
        "found": True,
        "path": dest_path,
        "extension": ext,
        "is_pdf": ext == ".pdf",
        "installation_name": inst.get(NAME_FIELD),
        "document": diagram,
    }
