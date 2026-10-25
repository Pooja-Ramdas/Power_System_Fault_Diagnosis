<USER_REQUEST>
# Multimodal Fault Console (build, deploy, push)

You are building a live-demo web dashboard on top of an EXISTING, WORKING university project ("Multimodal Reasoning for Power-System Fault Diagnosis", Group 13, real data from Chile's CEN grid: InfoTécnica single-line diagrams + hourly telemetry spreadsheets + EAF fault reports; Early, Late and Hybrid fusion models already trained and evaluated; final results are in `final_results`, older ones archived as `.first-level results`).

Go through this entire project and understand the architecture well.
Your job, end to end: (1) export the three trained fusion models, (2) build a small LIVE inference + explainability API around them, plus a precomputed fallback cache so the demo can never fail, (3) build a fast, beautiful retro-futuristic dashboard that calls the API, (4) deploy both (static frontend + API) to public URLs, (5) push everything to git. Do all of it without modifying, moving, renaming or deleting any existing file.

**Key requirement from the user: the diagnosis AND the explainability must be computed LIVE by the models when a fault is injected, not just displayed from stored results.** The demo events themselves are fixed held-out test cases (their diagram, telemetry window and report are the model inputs); the models run on them for real at click time. The cache exists only as an automatic safety net and must be labelled as such.

Work autonomously. Do not stop to ask questions unless a HARD GATE below fails. When you face a choice, pick the stated default and log it in `demo/DECISIONS.md` (one line: decision, reason).

---

## 0. NON-NEGOTIABLE RULES (read twice)

1. **Additive only.** Every file you create lives under a new top-level folder `demo/`. The only permitted edit outside `demo/` is NOTHING. No root `.gitignore` edits (use `demo/.gitignore`), no root README edits, no dependency file edits.
2. **Never import-execute training code.** If importing an existing module would run training or write files (missing `if __name__ == "__main__"`), do NOT edit it. Copy the needed class/function definitions into `demo/export/vendor/` with a header comment stating the source path and source git SHA.
3. **Never overwrite existing outputs** (`final_results`, `.first-level results`, `cen_dataset`, checkpoints, logs). Write only to `demo/`.
4. **Never upgrade or downgrade packages in the existing environment.** Create `demo/.venv-export` with `python -m venv --system-site-packages`, install only what is missing, pinned. Run `pip freeze` before and after in the existing env and confirm it is unchanged.
5. **Git safety.** Before any change:
   - `git status --porcelain`. If the tree is dirty, do NOT stage, stash, reset or commit those files; record them in `demo/_recon/dirty_files.txt` and never touch them.
   - Create a tag `pre-demo-backup` at HEAD (`git tag pre-demo-backup`), push it.
   - Record `git ls-files | xargs sha256sum > demo/_recon/tracked_files.sha256` (this file itself is the only thing committed from `_recon` besides notes).
   - Work on branch `feat/demo-dashboard`. Never force-push. Never rewrite history. Never `git add -A` or `git add .` at the repo root; stage explicit paths under `demo/` only.
6. **Final proof of safety (required):** at the end, `git diff --name-status pre-demo-backup..HEAD` must list ONLY `A` (added) entries under `demo/`. Re-verify the sha256 manifest of all pre-existing tracked files. If anything differs, revert your change and report it.
7. **Honesty of the demo.** Events are replays of held-out test cases (fixed inputs). Diagnosis and explanations are computed live by the API whenever it is reachable; every result carries `source: "live" | "cached"` and the UI shows a truthful LIVE or CACHED badge. Never label cached results as live. Every number shown must come from a real file/computation. No invented metrics, no artificial padding of latency, no cherry-picking by model correctness (see 2.2).
8. **No third-party runtime dependencies:** no map tiles, no CDN fonts, no analytics. The frontend is a static site; its only network calls go to the demo's own API (URL from `VITE_API_URL`, with the restricted override described in 3.1).

---

## 1. PHASE 0 — RECON (read-only, produce `demo/_recon/RECON.md`)

Inspect the repo and write a short report answering:

1. Repo layout; where the trained model code, training entry points, dataloaders, evaluation and `final_results` live.
2. For each of Early, Late, Hybrid fusion: architecture classes, encoders (diagram CNN/ViT, telemetry TCN, report text transformer — confirm what was actually used), fusion mechanism, input preprocessing (image transform, telemetry columns/window length/scaling, text language used by the model: Spanish or English, tokenizer, max length), class label order, and whether the model can run with a modality masked/missing.
3. Saved artifacts: search for `*.pt, *.pth, *.ckpt, *.safetensors, *.pkl, *.joblib, *.onnx`, scalers, tokenizers, label encoders. List path, size, which model, and timestamp.
4. Train/val/test split: is the test-set case list saved (IDs or indices)? If not, locate the seeded split code so it can be reproduced exactly.
5. Reported test metrics per model (accuracy, macro-F1, per-class F1) from `final_results` — these are the ground truth numbers the export must reproduce.
6. Dataset facts: number of cases, class distribution in the test split, how many test cases have `image_is_real_infotecnica == true`, telemetry columns and units (the telemetry is hourly generation data — do NOT label it as voltage/current anywhere in the UI), average report length, any `.rar`-derived images.
7. Repo size, any files >50 MB, Git LFS usage, remote URL, default branch, existing CI.

**HARD GATE A:** if no checkpoints exist for a model, retrain ONLY via thin wrappers around the existing training entry points (same config, same seeds, outputs redirected to `demo/artifacts/`). Then reproduce the test metrics; they must match `final_results` within ±0.5 percentage points accuracy/macro-F1. If not, STOP and report exactly what differs. Never present numbers that do not match the project's reported results.

**HARD GATE B:** if the exact test split cannot be reproduced, STOP and report. Never use training/validation cases in the demo.

---

## 2. PHASE 1 — EXPORT, LIVE API & FALLBACK CACHE (Python, under `demo/export/` and `demo/api/`)

### 2.1 Model registry (`demo/artifacts/models/`)

Name the three models (use these names everywhere; they are project-themed):

| id | Display name | Fusion | Tooltip text (verify against real code, edit if inaccurate) |
|---|---|---|---|
| `conflux` | **CONFLUX** | EARLY FUSION | Features from diagram, telemetry and report are merged into one joint representation at the input, before any decision is made. |
| `quorum` | **QUORUM** | LATE FUSION | Each modality has its own specialist model that reaches an independent verdict; a learned vote combines them. |
| `nexus` | **NEXUS** | HYBRID FUSION | Encoders exchange evidence through cross-modal attention mid-network, then a decision-level stage combines the result. |

For each model create `demo/artifacts/models/<id>/` containing: `model.pt` (state_dict), `config.json` (architecture hyperparameters, class order, preprocessing params, scaler values, tokenizer name/path, image transform), `metrics.json` (copied from `final_results`), `README.md` (how it was trained, source commit SHA, seed, date). Also `demo/artifacts/manifest.json` (sha256 of every artifact, torch/transformers versions, source git SHA, seed, UTC date) and `demo/export/load_models.py` exposing `load(model_id)` returning an object with `predict(case, mask=("D","T","R")) -> probs`, where `mask` selects which modalities are active (inactive ones use the same neutral baseline the model was designed for: zeros / learned null / dropped from the vote — document the choice per model).

Large files: if any artifact >50 MB, track with Git LFS (`demo/.gitattributes`), otherwise commit normally. The deployed site never loads checkpoints, so if LFS is unavailable add them to `demo/.gitignore` and document where they live. Do not block the demo on this.

### 2.2 Case curation (`demo/export/select_cases.py`)

- Source: TEST split only.
- Keep a case only if: `image_is_real_infotecnica == true`, the image decodes, telemetry has rows around t0, and the report text is non-empty.
- Stratify by the four classes (SHORT CIRCUIT, OPEN CIRCUIT, EQUIPMENT FAULT, OVERHEATING/OVERLOAD — use the exact label strings in the dataset, display friendly names). Target 12 per class, 40–48 total; if a class has fewer eligible test cases, use all of them.
- Selection must be seeded random (seed 13) with diversity across grid elements. **Do NOT filter or rank by whether any model got the case right.**
- Write `demo/export/selection_report.json`: counts per class, and each model's accuracy on the curated set versus on the full test set.

### 2.3 Core diagnose + explain functions (`demo/export/explain_core.py`)

Implement ONE shared module used by both the live API (2.6) and the cache builder (`demo/export/build_demo_data.py`), so live and cached results are produced by identical code. Every method must be LIVE-SAFE (fast; budgets in 2.6). For a given case and model compute:

1. **Prediction:** class probabilities, predicted class, confidence, correctness vs ground truth.
2. **Modality contribution (common, comparable metric for all three models):** exact Shapley values over the 3 modalities (D = diagram, T = telemetry, R = report) for the predicted-class probability. That is 8 forward passes per model per case (all subsets of masked modalities). Normalise to shares: `share_m = |phi_m| / sum|phi|`; keep the sign (positive = pushed toward the predicted class, negative = pushed against). Assert Shapley efficiency: `sum(phi) == f(all) - f(none)` within 1e-4, else fail the build.
3. **Ablation table (stretch feature, cheap since you already ran all 8 subsets):** prediction + confidence for each modality subset: D, T, R, DT, DR, TR, DTR.
4. **Within-modality evidence (LIVE-SAFE gradient methods only: gradient×input or Integrated Gradients with a small step count via Captum, Grad-CAM for the diagram; use occlusion only if you measure that it fits the latency budget):**
   - Telemetry: importance per time step (row) of the window.
   - Report: importance per token/word (aggregate sub-word tokens to words); store the top 8 words plus per-word scores for the displayed report excerpt.
   - Diagram: a coarse importance heatmap (Grad-CAM, or 6x6 patch occlusion) rendered as a transparent WebP overlay `img/<case>.<model>.cam.webp` (small, ≤ 40 KB).
5. **Model-specific "how" data:**
   - QUORUM (late): each unimodal specialist's own predicted class, confidence, and its learned vote weight.
   - NEXUS (hybrid): cross-modal attention weights and/or gate values per modality (whatever the real architecture exposes).
   - CONFLUX (early): the joint-representation attribution only (Shapley + within-modality evidence).
6. **Narrative (`narrative: string[]`, 3–4 short lines), generated by deterministic templates from the numbers above — never by an LLM and never invented.** Example shape: "REPORT carried 61% of the decision (words: 'tripped', 'breaker'). TELEMETRY carried 27%: generation fell 84 MW within 1 h of t0. DIAGRAM carried 12%: topology context only." Every figure in a sentence must be traceable to the case data. Include the model-specific line (QUORUM: the three specialist verdicts and weights; NEXUS: attention/gate summary).

### 2.4 Web assets and fallback cache (`demo/web/public/`)

- Images: for each case produce (a) `img/<case>.hud.webp` — a dark "HUD" version made offline with Pillow (detect light background by mean luminance; invert; tint line-art cyan on near-black; keep symbols legible), and (b) `img/<case>.raw.webp` original colours. Max width 2000 px, WebP quality ~80, each ≤ 250 KB. Record width/height.
- Optional per-case `highlight: {x,y,w,h} | null` (normalised 0-1 box around the affected element). If you cannot derive it reliably, set `null` (UI then shows a pulsing frame + element name). ALSO generate a tiny dev page `demo/export/annotate.html` (pure HTML/JS, no deps) that lets a human click-drag boxes on each case image and download `highlights.json`, which the build merges if present.
- Data files: `data/index.json` (models, classes, case list with id/class/element, full-test-set metrics per model from `final_results`) and `data/cases/<case>.json` per case. Schema (validate with a zod schema in `demo/web/src/schema.ts` AND a Python check; build fails on mismatch):

```
case: {
  id, event_ts, element_name, true_class,
  image: { hud, raw, width, height, highlight|null, is_real_infotecnica: true },
  telemetry: { columns[], units[], t0_index, rows[][] (<= 48 rows, <= 6 columns) },
  report: { en, es, shown_language, word_scores[] },
  predictions: { <model_id>: { pred, probs{class:p}, confidence, correct } },
  explanations: { <model_id>: { method, contrib: {D,T,R: {share, signed}},
      dominant, telemetry_importance[], report_top_words[], cam: "img/..." | null,
      model_specific: {...}, narrative[] } },
  ablations: { <model_id>: { "D":{pred,conf}, ... "DTR":{pred,conf} } }
}
```
- Per-case JSON ≤ 40 KB, image payload per case ≤ 450 KB total.

### 2.5 Verification gates for Phase 1
- Exported models reproduce `final_results` test accuracy/macro-F1 within ±0.5 pp on the FULL test set (print a comparison table into `demo/_recon/metrics_check.md`).
- Shapley efficiency assertion passes for every case/model.
- Every asset referenced by any JSON exists; total `public/` size reported.

### 2.6 Live API (`demo/api/`)

- FastAPI + uvicorn, one worker (models live in memory), CPU only, `torch.inference_mode()`, `torch.set_num_threads` = available cores.
- Startup: load all three models, and for every curated case preload and cache the PREPROCESSED model inputs (image tensor, tokenised report, scaled telemetry window) in RAM. Requests never read the raw dataset. Accept only whitelisted `case_id`s from the curated set (no uploads, no arbitrary paths).
- Endpoints:
  - `GET /health` → models loaded, artifact sha256s, source git SHA, uptime.
  - `GET /warmup` → runs one dummy forward per model and returns timings (this also wakes a sleeping host).
  - `POST /diagnose` body `{case_id, model_id, mask?: ["D","T","R"]}` → prediction, probabilities, confidence, correctness flag, Shapley modality contribution (8 batched masked passes), contribution narrative, `source:"live"`, `latency_ms`. Budget on the target host, warm: p50 ≤ 700 ms, p95 ≤ 1.5 s per model.
  - `POST /evidence` same body → within-modality evidence (telemetry importance per row, report top words + per-word scores, diagram CAM overlay as a small base64 WebP ≤ 40 KB), model-specific data (QUORUM specialist votes, NEXUS attention/gates), evidence narrative lines. Budget p95 ≤ 2.5 s.
  - Response field names match the cached case JSON (`predictions[model]`, `explanations[model]`) so the frontend has a single code path.
- CORS restricted to the Vercel production domain, Vercel preview pattern and localhost; no cookies; light rate limit (~60 req/min/IP).
- Frontend flow: at injection fire the three `/diagnose` calls in parallel; as each returns, show its bubble; fetch `/evidence` for the selected model immediately and prefetch it for the others in the background.
- **HARD GATE C (latency):** measure on the real deployment target. If `/diagnose` p95 > 2 s, degrade in this order: (a) fewer IG steps / lighter CAM layer, (b) `/evidence` on demand only (when a model is clicked), (c) as a last resort serve the heavy within-modality evidence from cache with a CACHED badge. Prediction + Shapley contribution must stay live. Record final numbers in `demo/_recon/latency.md`.
- **Cache parity:** `build_demo_data.py` generates the cached JSON by calling the same `explain_core` functions; a test asserts live vs cached probabilities and contribution shares agree within 1e-4 for at least 10 random case/model pairs.
- The cache is the safety net: the frontend falls back per request (timeout 3.5 s or error) and shows a CACHED badge.

---

## 3. PHASE 2 — FRONTEND (`demo/web/`)

You may start this in parallel with Phase 1 using a fixture generated from the schema; swap in real data at integration.

### 3.1 Stack and budgets
- Vite + React 18 + TypeScript. Plain CSS with CSS variables (no Tailwind, no UI kit, no chart library, no map library). Only runtime deps: `react`, `react-dom`, `three` (used ONLY inside the lazy-loaded fx chunk, see 3.9), `@fontsource/jetbrains-mono`, `@fontsource/vt323` (+ `zod` dev/build-time if you want). Pan/zoom for the diagram via pointer events + CSS transform (or a ≤ 5 KB lib).
- Fonts: self-hosted through `@fontsource` packages (bundled woff2, no CDN, no Google Fonts). Exact families, weights, subsets and preload rules are in 3.8. Wait for `document.fonts.ready` (max 800 ms) before the first reveal so there is no font flash during the demo.
- Budgets: initial JS (excluding the lazy `fx` chunk) ≤ 150 KB gzip; the lazy `fx` chunk (three.js + effects) ≤ 180 KB gzip, loaded on idle after first paint and never blocking first paint or the INJECT button; CSS ≤ 20 KB gzip, first meaningful paint < 1.5 s on throttled Fast 3G-ish, Lighthouse performance ≥ 90 mobile / ≥ 95 desktop. Animate only `transform` and `opacity`. No `backdrop-filter`, no full-screen `filter: blur`, glows via small `box-shadow`/`text-shadow` (max two layers).
- Case inputs (diagram, telemetry, report) and the fallback cache are static files. After first paint, `requestIdleCallback` prefetches one case per fault class (data + image); other cases load on demand. Add a tiny hand-rolled service worker (cache-first for `/data`, `/img`, `/fonts`, app shell; cache name includes the build hash; NEVER cache API calls) so the UI shell and case files survive venue Wi-Fi dying after the first load.
- API client (`src/api.ts`): base URL from `VITE_API_URL`, overridable at demo time via `?api=<https-url>` (persist in localStorage; allow only hostnames ending in `.hf.space`, `.onrender.com`, `.trycloudflare.com`, or `localhost`; show an `API OVERRIDE` chip when active) so the deployed frontend can be pointed at a laptop tunnel without a rebuild. On page load call `/warmup` (3 s timeout) and re-ping every 4 minutes while the tab is open to keep the host awake. Per-request timeout 3.5 s; on timeout/error fall back to the cached JSON for that case/model and mark the result `cached`.

### 3.2 Layout (desktop 1920×1080 primary; must also work at 1366×768; graceful stacked layout on mobile)

Reference mock: dark console, header bar on top, big diagram area filling the left/centre, a fixed-width control column on the right, a small data dock along the bottom, a legend overlaid at left. Use the same arrangement, with these zones:

1. **Header bar:** left: wordmark `TRIAD-13` (VT323, phosphor glow) + sub-label `multimodal fault console // SEN · Chile`. Far right corner: the revolving neon **Three.js logo** (3.9, 72×72 px, click opens ABOUT). To its left: status chips `SYSTEM NOMINAL` (green) → `FAULT ACTIVE` (fault colour) after injection, `MODELS 3/3 ONLINE · LIVE` (from `/health`; becomes `API OFFLINE · CACHED MODE` in amber if the API is unreachable), event timestamp (from the case), small `ABOUT` button (see 3.7).
2. **Centre — diagram stage (largest area):** the real InfoTécnica single-line diagram (HUD version by default) with pan/zoom, a `HUD | RAW` toggle, a `FIT` button, corner brackets, faint scanline texture. Overlays: highlight frame/reticle on the affected element + label with the element name (appears at injection), optional CAM heatmap overlay (toggled from the explainability panel), diagnosis bubbles (3.5).
3. **Left overlay — legend (collapsible):** explains (a) what the dashboard draws (affected element reticle, evidence heatmap, nominal vs fault status) and (b) the single-line symbols that ACTUALLY appear in the diagrams. Open several of the real images first and build the symbol legend from what you see (busbar, line, breaker, transformer, generator, etc.); do not invent symbology.
4. **Bottom dock (below/inside the stage):** two panes. **Telemetry table** (compact, sticky header, `t0` row marked, real column names + units from the case) with a tiny inline-SVG sparkline; **Operator report** pane (English translation by default, ES toggle; show the language the model actually consumed as a small tag). Before injection: telemetry shows pre-fault rows only and the report pane shows `NO ACTIVE INCIDENT`. After injection: post-fault rows stream in and the report "decrypts" in (≤ 400 ms text-scramble). Evidence from the selected model highlights rows / words here.
5. **Right column (fixed ~380–420 px), top to bottom:**
   - `FAULT SIMULATOR` panel: four fault-type chips (SHORT CIRCUIT, OPEN CIRCUIT, EQUIPMENT FAULT, OVERHEATING/OVERLOAD) each with its own colour and key hint 1–4, a `SHUFFLE FEEDER` link, and the big `INJECT FAULT` button.
   - `DIAGNOSTIC MODELS`: three model cards (CONFLUX, QUORUM, NEXUS) — each its own box with name, a one-line status, and a result chip after analysis. **On hover/focus a tooltip/flip reveals the fusion type** (`EARLY FUSION` etc. + tooltip text from 2.1). Keyboard accessible.
   - `EXPLAINABILITY` panel: shows the explanation of the currently selected model (3.6).

### 3.3 Interaction flow (this resolves "simulate vs test set")
- Idle state: the app arms a feeder (a held-out test case) matching the selected fault type and shows its diagram NOMINAL (green status, pre-fault telemetry only). Selecting a different fault type or pressing `SHUFFLE FEEDER` swaps to another preloaded case of that class (crossfade ≤ 150 ms; never repeat the same case twice in a row).
- `INJECT FAULT` (or Space/Enter) reveals the fault on that diagram. The three models are called live at this moment; the case's own telemetry rows and report (the model inputs) are already in memory.
- Keys: `1–4` select fault type, `Space` inject, `Q/W/E` select CONFLUX/QUORUM/NEXUS, `R` reset to nominal, `H` toggle HUD/RAW. Query param `?fast=1` disables all theatrics.

### 3.4 Animation timeline after injection (target ≤ 2.2 s end to end including the live calls)
- 0–250 ms: short glitch/flash (2-frame chromatic offset), status chip → FAULT ACTIVE, reticle pulses on the affected element, Three.js shockwave ring expands from the reticle (3.9), logo shifts to the fault-class colour.
- 100–500 ms: telemetry rows stream in; 200–600 ms: report decrypts.
- 300 ms: model cards switch to `ANALYSING` with a staggered pulse and the Three.js scan-sweep across the diagram (3.9); this state lasts exactly as long as the real request (minimum 400 ms for animation continuity, no further artificial delay).
- As each model's `/diagnose` returns, its bubble appears and types its text in ≤ 300 ms (bubbles may therefore arrive at different times; that is fine). Cards show a small `LIVE` / `CACHED` tag.
- ~1.5 s: `GROUND TRUTH: <class>` stamp appears under the bubbles and each model card gets a ✓ or ✗. If the models disagree, show a small `MODELS DISAGREE` flag (this is a good story, not a bug).
- `prefers-reduced-motion` and `?fast=1`: skip to the final state immediately.

### 3.5 Diagnosis bubbles (must look native to the aesthetic)
- A futuristic callout, not a rounded cartoon bubble: chamfered corners (clip-path), 1 px neon outline (draw the border with the outer-wrapper-as-border-colour + inset inner clip-path trick), faint translucent dark fill, a small angular tail, soft glow in the model's colour, a thin header row `CONFLUX // EARLY` and body `SHORT CIRCUIT · 87%`.
- Three bubbles, one per model, stacked along the right edge of the diagram stage; each bubble is vertically aligned to its model card in the right column with its tail pointing right at that card (use refs + `getBoundingClientRect`, recompute on resize). Below ~1100 px width, stack bubbles at the bottom of the stage without tails.
- Bubble colour = model colour; the predicted-class chip inside uses the fault-class colour. Clicking a bubble selects that model.
- `aria-live="polite"` announcing results.

### 3.6 Explainability panel (for the selected model; defaults to the first model after injection)
1. Header `EXPLAIN // <MODEL>`, the fusion-type tag, a `LIVE`/`CACHED` badge and the `latency_ms` of the call in small text.
2. **Contribution bar** (from `/diagnose`): one horizontal segmented bar (D / T / R) in modality colours with % labels and `DOMINANT: <MODALITY>`; negative contributors drawn hatched with a minus. Animate widths with `transform: scaleX`.
3. **HOW (narrative):** the template lines from `/diagnose` (contribution) followed by those from `/evidence` (words, rows, regions) as they arrive; show a thin scanning skeleton for the evidence part until it returns.
4. **Model-specific block:** QUORUM → three mini-rows (specialist verdict + confidence + vote weight); NEXUS → attention/gate mini-bars; CONFLUX → a one-line note that modalities were merged before decision, so attribution is by Shapley + within-modality evidence.
5. **Evidence links:** clicking a modality segment cross-highlights the evidence: D → CAM heatmap on the diagram, T → highlighted telemetry rows + sparkline markers, R → highlighted words in the report.
6. **Live ablation toggles (build this; the API supports it):** three toggles `D T R` that call `/diagnose` with the chosen `mask` for the selected model and show the real, live prediction change when a modality is switched off. Fallback: the cached ablation table (CACHED badge).

### 3.7 About / honesty overlay
`ABOUT` opens a modal listing: what the demo is (replay of held-out test events; the saved checkpoints at build `<git sha>` are run live by the API, with a labelled cached fallback), the three models with their real full-test-set metrics from `final_results`, the dataset sources (InfoTécnica / CEN hourly generation / EAF reports), and that telemetry is hourly generation data. Also a one-line footer on the main screen: `MODE: HELD-OUT TEST EVENT REPLAY · LIVE INFERENCE` (switches to `· CACHED INFERENCE` when in fallback).

### 3.8 Visual design system — "analogpunk": a retro-futuristic control-room terminal, clean, with soft neon

**Concept.** A salvaged analog control-room console from a future that never happened: phosphor-glow screens, etched panel labels, physical-feeling switches and LED indicators, hairline rules, lots of dark negative space. The analogpunk character comes from tactile details; the "clean" comes from restraint: an 8 px grid, generous spacing, at most 2–3 accent hues visible at once, effects as accents rather than wallpaper. Readability always beats decoration.

**Typography (all self-hosted, SIL Open Font License).**
- **UI, body, data: JetBrains Mono** — weights 400, 500, 700 via `@fontsource/jetbrains-mono`; import only `latin-400.css`, `latin-500.css`, `latin-700.css` (this subset covers Spanish accents á é í ó ú ñ ¿ ¡ and ° used in the ES report toggle; verify rendering). Base 12–13 px, line-height 1.5. In tables/telemetry: `font-variant-numeric: tabular-nums; font-variant-ligatures: none;`.
- **Display / CRT readouts: VT323** via `@fontsource/vt323/latin-400.css` — for the `TRIAD-13` wordmark, big confidence percentages, latency readouts, status text and the boot screen, at 28–56 px with a phosphor glow. Never for paragraphs.
- **Etched labels:** JetBrains Mono 500, 10–11 px, UPPERCASE, `letter-spacing: .14em`.
- Fallback stacks: `"JetBrains Mono", ui-monospace, "SF Mono", Menlo, Consolas, monospace` and `"VT323", "JetBrains Mono", monospace`. `font-display: swap`; preload JetBrains 400 and VT323 400 (use the hashed asset URLs Vite emits).
- Do not rely on font glyphs for icons (✓ ✗ ▶ may be missing from the Latin subset): draw check/cross/arrow icons as small inline SVG.

**Palette** (define as CSS variables; keep usage consistent). Surfaces: `--bg-0:#05080d`, panels `#0a1018` at ~86% opacity, hairlines `--line:#16313f` / `--line-hi:#24566b`, text `--ink:#cfe9ee`, dim text `--ink-dim:#6f8f99` (never pure white). Models: CONFLUX cyan `#38e8ff`, QUORUM mint `#5dffb0`, NEXUS periwinkle `#8fa8ff`. Fault classes: SHORT CIRCUIT `#ff4d6d`, OPEN CIRCUIT `#ffb347`, EQUIPMENT FAULT `#c77dff`, OVERHEATING/OVERLOAD `#ff7b3a`. Modalities: DIAGRAM `#6ea8ff`, TELEMETRY `#ffd84d`, REPORT `#ff8bd0`. Nominal `#5dff9a`. Amber `#ffb347` doubles as the CACHED/API-offline colour.

**Soft neon glow recipe** (the glow is a whisper around a crisp bright core, never a blob):
```css
.neon      { --c: var(--cyan);
             border: 1px solid color-mix(in srgb, var(--c) 55%, transparent);
             box-shadow: 0 0 8px color-mix(in srgb, var(--c) 28%, transparent),
                         inset 0 0 12px color-mix(in srgb, var(--c) 8%, transparent); }
.neon-text { color: var(--c);
             text-shadow: 0 0 8px color-mix(in srgb, var(--c) 45%, transparent); }
.panel     { clip-path: polygon(10px 0,100% 0,100% calc(100% - 10px),calc(100% - 10px) 100%,0 100%,0 10px); }
.crt::after{ content:""; position:fixed; inset:0; pointer-events:none; z-index:50;
             background: repeating-linear-gradient(to bottom, rgba(255,255,255,.025) 0 1px, transparent 1px 3px);
             mix-blend-mode: overlay; }
.vignette  { background: radial-gradient(ellipse at center, transparent 55%, rgba(0,0,0,.45) 100%); }
```
Rules: max two shadow layers per element; glow alpha ≤ 0.35 on small elements and ≤ 0.12 on large areas; the core colour stays bright and readable; scanlines static (≤ 4% opacity); no animated backgrounds behind text.

**Analogpunk component details** (apply with restraint, keep everything aligned and tidy):
- **Panels:** chamfered corners, 1 px neon hairline at 40–55% alpha, tiny corner screws or `+` registration marks, an etched title tab such as `FAULT SIMULATOR // 01`. Faint hairline grid (24 px, ~4% alpha) and edge tick-rulers on the diagram stage; tiny stencil text like `UNIT TRIAD-13 · REV C`.
- **INJECT FAULT:** an arming-switch style button with a low-contrast diagonal hazard-stripe edge that becomes active (fault colour) only when armed; pressed state moves 1 px down; hover raises the glow.
- **Fault-type chips:** rocker-switch / tape-label look with a small LED dot that lights up when selected.
- **Model cards:** rack-module look — LED (dim idle, blinking while ANALYSING, solid when done), serial label `UNIT 01 · CONFLUX`, and a 12-segment VU-style confidence bar.
- **Contribution bar and confidence meters:** segmented LED/VU bars (e.g. 40 discrete segments) in modality colours, not smooth gradients.
- **D/T/R ablation toggles:** drawn as physical toggle switches with an LED each.
- **Numbers:** confidence %, latency and counters in VT323 LED-style readouts.
- **Ground-truth stamp:** rubber-stamp look (outlined uppercase text, −2° rotation, no glow).
- **Hazard stripes** appear only in the FAULT ACTIVE state (header chip and inject button edge).
- **Telemetry table:** etched uppercase header, rows separated by ≤ 3% alpha hairlines, the `t0` row marked by a left neon bar, evidence rows highlighted by a soft tint.
- **Motion:** transitions ≤ 200 ms, crisp easing; a 2-frame chromatic-aberration offset (red/cyan text-shadow, 1–2 px, ≤ 120 ms) only at injection and when a bubble appears; status LED/chip flicker at most once every 8–10 s; `prefers-reduced-motion` disables all of it.
- **Don'ts:** no pure-white text, no rainbow gradients, no heavy blur, no more than two glow layers, no curved-screen distortion of the UI, no effects that reduce legibility. Body text contrast ≥ 4.5:1; visible 2 px neon focus rings.
- After building, take screenshots and critique them against this section: if it looks busy rather than clean, remove decoration until it is clean.

### 3.9 Three.js layer (logo, boot scene, stage effects)

**Principles.** Progressive enhancement: the whole app must work with Three.js absent. All 3D code lives in `src/fx/`, loaded with a dynamic `import()` after first paint (`requestIdleCallback`). Import only named symbols from `three` (no `import * as THREE`) so it tree-shakes; NO EffectComposer/UnrealBloom/postprocessing, NO network textures or models (procedural geometry only). Budget for the whole fx chunk ≤ 180 KB gzip (if exceeded, drop fat lines and use plain `LineSegments`). At most **2 WebGL contexts**: (a) the logo, (b) one shared "stage" renderer used first for the boot scene and then for stage effects. Cap DPR at 2 (1.5 for the stage). Pause rendering when the tab is hidden; the stage renderer runs its loop only while an effect is active. Glow without bloom: draw each neon object twice — a thin full-brightness core plus a slightly larger (×1.04–1.08) additive-blended halo at 15–25% opacity — and add `filter: drop-shadow(0 0 6px <colour>)` on the small logo canvas only.

**Static fallbacks** (used when WebGL is unavailable, `?fx=0`, `?fast=1`, `prefers-reduced-motion`, viewport < 900 px, `navigator.hardwareConcurrency <= 2`, or the chunk fails to load): an inline SVG hexagon logo with a slow CSS rotation (or static under reduced motion), CSS/HTML-only boot screen, no ambient or shockwave effects. Auto-degrade at runtime: if frame time exceeds 24 ms for 30 consecutive frames, disable ambient first, then the scan sweep, and keep the logo. Effects must never delay or block the live API calls or the bubbles.

**Elements**
1. **Logo (P0), top-right corner of the header.** A 72×72 px transparent canvas: a cyan neon **hexagonal prism** wireframe (`EdgesGeometry` of a 6-segment cylinder) with a smaller **cube** wireframe inside, counter-rotating on a different axis, plus three tiny dots orbiting in the diagram, telemetry and report colours. Slow continuous revolve (~0.4 rad/s about Y, slight X wobble), halo as above. States: nominal = cyan, slow; any model `ANALYSING` = spin ×3 with halo pulse; FAULT ACTIVE = colour lerps to the injected class colour over 600 ms, then returns to cyan after results; API offline / CACHED = amber and slow; hover = spin up; click = open ABOUT. Paint the SVG fallback first at the same fixed 72×72 box and crossfade to the canvas when ready (no layout shift). Canvas is `aria-hidden`; the wrapper has `role="img"` and `aria-label="TRIAD-13 logo"`.
2. **Boot / loading scene (P0).** Full-viewport overlay shown on first load only if real loading takes longer than 300 ms (never add artificial delay; hard cap 4 s, then proceed, falling back to cached mode if the API is not ready). It starts as plain HTML/CSS (VT323 title + CSS progress ring) so first paint stays fast; the Three.js scene mounts into it as soon as the fx chunk arrives: a perspective wireframe grid floor and ~600 particles converging into a hexagonal prism whose edges draw in. A typed checklist reflects REAL tasks — `FONTS`, `CASE INDEX`, `FIRST FEEDER (DIAGRAM DECODE)`, `MODEL API WARMUP` — each flipping `[ .. ]` → `[ OK ]` (or `[ -- ] CACHED MODE` if the warmup times out after 3 s). The progress ring = completed tasks / total tasks. Fade out in ≤ 250 ms when done; click or `Esc` skips. If the fx chunk isn't ready by the time loading finishes, the overlay simply closes without 3D.
3. **Stage effects (P1).** One transparent canvas over the diagram stage (`pointer-events: none`; above the diagram image, below bubbles and all UI text):
   - **Injection shockwave:** thin expanding ring(s) plus a small particle burst (≤ 150 points) from the reticle position (stage centre if there is no highlight), in the fault-class colour, ≤ 700 ms.
   - **Analysis scan sweep:** while any model is `ANALYSING`, a thin luminous scan line/plane sweeps left→right (~700 ms loop, cyan, 20–30% alpha, trailing fade); stops when the last model returns.
   - **Ambient (lowest priority):** ≤ 120 sparse drifting dust particles and a faint perspective grid at the stage edges, ≤ 0.2 alpha, rendered at half resolution, capped at 30 fps, paused during other effects and when the tab is hidden. Never placed behind text.
4. **Feeder-swap wipe (P2, only after everything else passes):** a brief wireframe grid wipe (~200 ms) when SHUFFLE FEEDER or a fault-type change swaps the case.

The About modal credits three.js (MIT), JetBrains Mono (OFL) and VT323 (OFL).

---

## 4. PHASE 3 — TESTS (all must pass before deploy)

- Python: metrics reproduction (2.5), Shapley efficiency, schema validation, asset existence, size budgets.
- API: live-vs-cached parity (2.6), whitelist enforcement (unknown `case_id` → 404), CORS check, and a latency benchmark script (`demo/api/bench.py`, 50 sequential + 10 concurrent calls per model) run against the local container AND the deployed API; write p50/p95 to `demo/_recon/latency.md`.
- Playwright resilience test: block the API and assert the UI falls back to cached results within 3.5 s, shows the CACHED badge and still completes the full flow.
- `npm run build` with zero TypeScript errors/warnings; bundle report confirms the budgets in 3.1.
- Playwright (headless Chromium) end-to-end: for each of the 4 fault types → arm, inject, assert three bubbles visible within 2.5 s (or 300 ms with `?fast=1`), click each model, assert the explanation panel updates and the contribution shares sum to 100%, hover each model card and assert the fusion label appears, press `R` and assert return to nominal. Capture screenshots at 1920×1080, 1366×768 and 390×844 into `demo/_recon/screens/`.
- Lighthouse run against `vite preview` and (later) the live URL; record scores (measured with fx enabled on desktop).
- fx tests: report the fx chunk size and assert it is ≤ 180 KB gzip and that the initial bundle excludes `three`; assert at most 2 `<canvas>` elements / WebGL contexts exist; with `?fx=0` and with WebGL disabled assert the SVG logo shows and the full inject → bubbles → explain flow still passes; assert click-to-first-bubble time with fx on is within 60 ms of fx off (excluding network); confirm no WebGL console errors or context-lost warnings; verify the boot checklist reflects real task completion (block the API and confirm it ends in `CACHED MODE` within the 4 s cap); verify fonts load from same-origin and that á é í ó ú ñ render in the report pane.
- Manual-style checks scripted where possible: no console errors, no network requests to third-party hosts (assert all requests are same-origin), works offline after first load (Playwright offline mode).
- Visually inspect your own screenshots and fix anything cut off, overlapping or ugly before moving on (bubble tails must line up with the right cards at both desktop sizes).

---

## 5. PHASE 4 — DEPLOY (two parts: static frontend + live API)

**Hosting reality check (log any deviation in DECISIONS.md):**
- Frontend → **Vercel** (static, CDN, no cold start).
- API → needs PyTorch plus the text and image encoders, so it needs real RAM and CPU. **Render's FREE web service (512 MB RAM, 0.1 CPU, sleeps after 15 minutes, ~1 minute to wake) is NOT suitable.** Use the first option that works:
  1. **Hugging Face Docker Space, CPU Basic** (2 vCPU, 16 GB RAM, free tier sleeps after ~48 h idle; port 7860). HF documentation currently indicates that creating Docker/Gradio Spaces may require a paid plan on some accounts; check the account and report.
  2. **Render web service, Standard (2 GB)** for the demo window: this costs money, so do NOT purchase anything; prepare `demo/render.yaml` and document it.
  3. **Laptop + free tunnel:** `demo/api/run_local.sh` starts uvicorn and a Cloudflare quick tunnel (`cloudflared tunnel --url http://localhost:8000`); free, no cold start; the URL changes per run, which is why the frontend supports `?api=`.
- Never make purchases or sign-ups. If a login is needed and the CLI isn't authenticated, finish everything else, build and test the Docker image locally, and write exact manual steps into `demo/DEPLOY.md`.

**Steps**
1. `demo/api/Dockerfile` (python:3.11-slim, CPU-only torch wheels pinned to the versions the models were trained with, runtime deps only, copies only the needed artifacts and preprocessed curated inputs, non-root user, listens on `$PORT` default 7860), `.dockerignore`, pinned `requirements.txt`. Report image size and cold-start time to `/warmup`-ready.
2. Local test: `docker run` the image; run the parity tests and `bench.py`.
3. Deploy the API per the ladder above; verify `/health` from an unauthenticated context; run `bench.py` against the deployed URL (note the network RTT from the dev machine).
4. Frontend: `demo/web/vercel.json` with long-lived immutable cache headers for `/assets`, `/fonts`, `/img`, short cache for `/data/index.json` and the service worker. Set `VITE_API_URL` to the deployed API. If `vercel whoami` works: `vercel link` (project `triad-13-fault-console`, root `demo/web`, framework Vite, output `dist`), `vercel deploy --prod`, connect the Git repo. If not authenticated: document the click-path in `DEPLOY.md` (Add New Project → import repo → Root Directory `demo/web` → Framework Vite → env var `VITE_API_URL` → Deploy). Backup: `demo/render.yaml` can also define a Render **Static Site** (build `cd demo/web && npm ci && npm run build`, publish `demo/web/dist`).
5. Verify from a clean, unauthenticated context: HTTP 200 on the **production** Vercel URL (no Vercel login wall), CORS works from that origin, the Playwright suite passes against the live pair, Lighthouse recorded.
6. `demo/RUNBOOK.md` for demo day: both URLs; warm-up (open the frontend 10 minutes early; the page pings `/warmup`; click through all four fault types once); what the badges mean; how to switch to the laptop-tunnel backup (`run_local.sh`, then open the frontend with `?api=<tunnel-url>`); cached-mode fallback; offline preview (`cd demo/web && npm ci && npm run build && npm run preview`).

---

## 6. PHASE 5 — GIT

1. Commit in logical, conventional-commit chunks, staging explicit paths under `demo/` only: `chore(demo): recon + safety manifest`, `feat(demo): model export + registry`, `feat(demo): case curation + explanations`, `feat(demo): web dashboard`, `test(demo): e2e + data checks`, `docs(demo): runbook + deploy`.
2. Check that no file >50 MB is staged (use LFS or ignore per 2.1) and that no secrets/tokens are present.
3. `git push -u origin feat/demo-dashboard` (and the `pre-demo-backup` tag).
4. Run the safety proof from rule 6. Only if it passes and all Phase 3 checks pass: merge into the default branch with `git merge --no-ff feat/demo-dashboard` and push. If the default branch is protected or the push is rejected, open a PR instead (via `gh` if available) and report. Never force anything.

---

## 7. FINAL REPORT (print at the end, concisely)

- Frontend URL and API URL (or the exact remaining manual step), which API hosting option was used, and the backup status.
- Branch, commit SHA, tag `pre-demo-backup`, merge status.
- Safety proof output (only additions under `demo/`; pre-existing files' checksums unchanged).
- Metrics reproduction table (reported vs exported, per model).
- Curated set size per class; curated-set vs full-test accuracy per model.
- Performance numbers: bundle sizes (initial and fx chunk), Lighthouse, API p50/p95 per endpoint (local and deployed), click-to-bubbles time, live-vs-cached parity result.
- Anything you could not do, any assumption made (`demo/DECISIONS.md`), and a short list of open questions for the user (e.g. optional manual highlight boxes via `annotate.html`).

**Definition of done:** a public link where a person can pick a fault type, press INJECT FAULT, and within ~2 seconds see the real diagram, telemetry and report for a held-out event, three named models (CONFLUX / QUORUM / NEXUS) each running LIVE and showing a diagnosis bubble, click any model to see, computed live, which modality drove its decision and how (and toggle modalities off to watch the prediction change), all in a clean analogpunk retro-futuristic UI (JetBrains Mono + VT323, soft neon glows) with a revolving Three.js neon logo in the top-right corner, a Three.js boot sequence and stage effects (shockwave, scan sweep) that degrade gracefully to static fallbacks, with a truthfully labelled cached fallback if the API is down, and with zero changes to any pre-existing file in the repo.
</USER_REQUEST>
<ADDITIONAL_METADATA>
The current local time is: 2026-10-25T02:52:36+05:30.
</ADDITIONAL_METADATA>
<USER_SETTINGS_CHANGE>
The user changed setting `Model Selection` from None to Gemini 3.8 Flash (Medium). No need to comment on this change if the user doesn't ask about it. If reporting what model you are, please use a human readable name instead of the exact string.
</USER_SETTINGS_CHANGE>