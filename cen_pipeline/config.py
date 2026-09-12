"""
Central configuration for the CEN multimodal dataset pipeline.
Tune everything here rather than editing the scraper modules.
"""
import os

# ---------------------------------------------------------------------------
# Where the final dataset lands. Point this at your project directory.
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cen_dataset")
RAW_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_raw_cache")
# _raw_cache holds temporary downloads (PDFs, xlsx). Safe to delete after a run.
# It is NOT part of your 1.5-2GB budget calculation below -- see build_dataset.py.

# ---------------------------------------------------------------------------
# EAF (text modality) scraping scope
# ---------------------------------------------------------------------------
EAF_YEARS = [2023, 2024, 2025]          # add/remove years as needed (2013-2026 exist)
EAF_LISTING_URL_TEMPLATE = (
    "https://www.coordinador.cl/operacion/documentos/estudios-operacionales/"
    "estudios-de-analisis-de-falla/{year}-estudios-de-analisis-de-falla"
)
MAX_PAGES_PER_YEAR = 40                 # 2025 alone has ~29 pages of ~20 reports each
REQUEST_DELAY_SEC = 1.0                 # be polite to CEN's servers
PDF_TEXT_MAX_PAGES = 6                  # stop before the raw SCADA/event-log annexes

# ---------------------------------------------------------------------------
# Overall case budget
# ---------------------------------------------------------------------------
# You do NOT need to shrink this to save disk space -- a cleaned, per-case
# record (small PNG + small telemetry CSV + short text) is on the order of
# 60-150 KB, so even 2,000 cases lands well under 300 MB. Cap this by DATA
# QUALITY / BALANCE across fault types, not by an arbitrary byte ceiling.
# Set to None to process every EAF report found in EAF_YEARS.
TARGET_N_CASES = 1200
DATASET_SIZE_HARD_CAP_MB = 2000         # safety net; the run stops if this is hit

# ---------------------------------------------------------------------------
# Telemetry (numeric modality)
# ---------------------------------------------------------------------------
TELEMETRY_WINDOW_HOURS = 48             # +/- 24h around each fault timestamp
TELEMETRY_XLSX_URLS = {
    "hourly_generation": "https://www.coordinador.cl/wp-content/uploads/2026/05/CENhist_gen_de_energia_por_hora_26.xlsx",
    "generation_by_plant_2025": "https://www.coordinador.cl/wp-content/uploads/2026/05/CENhist_gen_de_energia_por_central_2025.xlsx",
    "generation_by_technology": "https://www.coordinador.cl/wp-content/uploads/2026/05/CENhist_gen_de_energia_por_tecnologia.xlsx",
    "installed_capacity": "https://www.coordinador.cl/wp-content/uploads/2026/05/CENhist_cap_inst_por_tecnologia.xlsx",
    "max_demand": "https://www.coordinador.cl/wp-content/uploads/2026/05/CENhist_ddas_maximas_anual_y_punta.xlsx",
}
# NOTE: CEN periodically renames/re-dates these files (the "2026/05" folder
# will change). If a URL 404s, go to the "Telemetry" links in the dataset
# proposal PDF or https://www.coordinador.cl and grab the current link,
# then update this dict. The script will tell you exactly which one failed.

# ---------------------------------------------------------------------------
# Translation (Spanish EAF text -> English)
# ---------------------------------------------------------------------------
TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-es-en"   # local, offline, free, ~300MB
                                                     # cached once in ~/.cache/huggingface
                                                     # (not counted in the dataset budget)

# ---------------------------------------------------------------------------
# Diagram (image modality) generation
# ---------------------------------------------------------------------------
# The REAL official single-line diagram PDF requires a manual, free request
# to CEN (see README.md). It cannot be bulk-scraped. This pipeline instead
# renders a small schematic per case from the real named grid elements
# (substations / lines) mentioned in that case's own EAF report text -- see
# scraper/topology.py and the README for how to upgrade this to pull real
# coordinates from the InfoTecnica API.
DIAGRAM_DPI = 110

# Delete the raw downloaded EAF PDFs after extracting their text (recommended --
# this is what keeps the final dataset small; the PDFs themselves can be
# re-downloaded any time from pdf_url in manifest.csv if you ever need them).
KEEP_RAW_PDFS = False

# Reproducible sampling when TARGET_N_CASES < number of scraped reports
RANDOM_SEED = 13

# ---------------------------------------------------------------------------
# Image modality: try the REAL InfoTecnica "diagrama unilineal" document for
# the substation/line named in each case first; only fall back to the
# generated schematic (scraper/topology.py) when no real document is found
# for that installation. See scraper/infotecnica_api.py -- run
# probe_infotecnica.py once before your first real build to confirm the API
# field names hard-coded there.
USE_REAL_INFOTECNICA_IMAGES = True


