# CEN Multimodal Fault Diagnosis — Dataset Builder

Scrapes CEN's public EAF fault reports (text), matches them to CEN's hourly
telemetry (numeric), and generates a per-case diagram (image) from the real
grid elements named in each report — producing one folder per fault event,
all keyed to the same case ID, ready to feed your early/late/hybrid fusion
and unimodal baselines.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 0 — get the REAL images working first (do this before the full build)

```bash
python probe_infotecnica.py
```

This hits InfoTecnica's public API once and prints its actual JSON shape.
`scraper/infotecnica_api.py` has my best inference of the field names
(`LIST_RESULTS_KEY`, `NAME_FIELD`, `ID_FIELD`, `DOCUMENTS_URL_TEMPLATE`,
`DOCUMENT_TYPE_FIELD`, `DOCUMENT_URL_FIELD`) at the top of the file — after
running the probe, update those few constants to match what you actually
saw printed. This is the one part of the pipeline I could not execute
myself to confirm (their API docs page blocks automated fetching), so
budget 10–15 minutes here before the full run.

If the probe returns HTML or a 401/403 instead of JSON, that means this
particular endpoint needs registration after all — go to
`portal.api.coordinador.cl`, register for a free API key, and add an
`Authorization` header in `scraper/infotecnica_api.py`'s `_session()`. Tell
me what the probe printed and I'll adjust the client for you.

### Step 1 — build the dataset

```bash
python build_dataset.py
```

First run will download the MarianMT translation model (~300MB, one-time,
cached in `~/.cache/huggingface`, does not count against your dataset size
budget) and the CEN telemetry spreadsheets (cached in `_raw_cache/`).

The run is resumable: it skips any `CASE-XXXX/` folder that already has a
`meta.json`, so if it's interrupted or a case fails, just re-run.

## What you get

```
cen_dataset/
  CASE-0001/
    image.png          # diagram of the substations/line named in this case
    telemetry.csv       # numeric window around the fault timestamp
    report_es.txt        # original Spanish report text (trimmed to narrative)
    report_en.txt        # English translation
    meta.json            # eaf_id, timestamp, description, rule-based fault_label
  CASE-0002/
  ...
  manifest.csv           # one row per case indexing everything above
```

## Important caveats — read before you rely on this for grading

1. **The image modality now tries the real official CEN document first.**
   `scraper/infotecnica_api.py` queries InfoTecnica's public API
   (`api-infotecnica.coordinador.cl/v1/...`) for the substation/line named in
   each case and downloads its actual "diagrama unilineal" document when one
   exists. This is confirmed to be a real, per-installation document type in
   CEN's own platform documentation — but not every installation has one
   uploaded, and I could not execute a live call against this API from my
   side to lock in the exact JSON field names (see Step 0 above). When no
   real document is found for an installation, the pipeline automatically
   falls back to the generated schematic in `scraper/topology.py` so no case
   is left without *an* image — `meta.json`'s `image_is_real_infotecnica`
   field tells you, per case, which one you got, and `build_dataset.py`
   prints a real-vs-fallback percentage at the end of the run.

   The consolidated, official single-line diagram for the whole grid is a
   separate thing and still requires the manual free request via CEN's form
   — useful for a report figure, not something to automate per case.

2. **Telemetry alignment is best-effort.** CEN's spreadsheets change column
   layout across releases, and the URLs in `config.py` include a dated
   upload folder (`2026/05/...`) that CEN will eventually rotate — if a
   download 404s, grab the current link from the dataset proposal PDF or
   coordinador.cl and update `config.TELEMETRY_XLSX_URLS`. The window
   extractor logs the columns it saw if it can't find a date/time column, so
   you can pass `datetime_col=` explicitly per file.

3. **`fault_label` is a quick rule-based guess** from keywords in the report
   title (línea/transformador/interruptor/barra/sobrecarga), so you have
   something to train against immediately. Cross-check it against your
   rubric's actual fault taxonomy and, ideally, hand-verify a validation
   subset — EAF report titles are terse and this heuristic will misclassify
   some cases.

4. **Respect the source.** CEN publishes this under a public-transparency
   mandate and it's explicitly free/open, but the script still rate-limits
   requests (`REQUEST_DELAY_SEC`) and sends a descriptive User-Agent. Don't
   drop the delay to zero or fan out parallel requests.

## On the storage budget you mentioned

A cleaned case (small PNG + small telemetry CSV + a few KB of text) runs
roughly 60–150 KB. Even 1,500–2,000 cases — more than you'll get from EAF
reports in a 2–3 year window — lands well under 300 MB, because raw PDFs and
xlsx files are deleted/not retained after extraction (`KEEP_RAW_PDFS=False`
in `config.py`). So you don't need to shrink the dataset to hit 1.5–2GB;
that cap in `config.py` is just a safety net. For a paired multimodal
dataset this size, more matched, correctly-labeled cases generally helps
more than an artificially small one — cap by class balance across fault
types, not by disk space.

## Early vs. late vs. hybrid fusion — same preprocessing?

Yes, keep one shared, cleaned `cen_dataset/` as the single source of truth
for all four models (early, late, hybrid, unimodal). What differs is what
each *architecture* does downstream of that shared data, not the cleaning
itself:

- **Unimodal baselines** — same per-modality preprocessing, each model just
  ignores the other two modalities' folders.
- **Early fusion** — the three modality encoders' feature vectors are
  concatenated (or projected to a shared dim) *before* the main task layers,
  so all three need to be encoded per-sample in a way that can be
  concatenated (e.g. fixed-length embeddings from an image CNN, a
  telemetry TCN, and a text transformer, all pooled to vectors).
- **Late fusion** — each modality trains/predicts (mostly) independently
  and only the final logits/probabilities are combined, so per-branch
  preprocessing *can* differ slightly (e.g. different image augmentation
  per branch) without breaking anything, since there's no shared
  intermediate feature space to align.
- **Hybrid fusion** — typically cross-modal attention at an intermediate
  layer, which — like early fusion — needs aligned intermediate
  representations, so treat its preprocessing requirements like early
  fusion's.

Practical takeaway: build the cleaning/alignment pipeline once (this
script), and let fusion-specific transforms (augmentation, normalization
stats, tokenization) live inside each model's own `Dataset`/`DataLoader`,
not in the shared scraping step.
