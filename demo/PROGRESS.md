# TRIAD-13 Master Progress Checklist

Last updated: 2026-10-25

## 1. PHASE 0 — RECON
- [x] **Repo layout & code inspection** (`demo/_recon/RECON.md`) — **DONE**
- [x] **Fusion models architecture & encoders verified** — **DONE**
- [x] **Saved checkpoints listed & checked** — **DONE**
- [x] **Train/val/test split reproduction (Seed 42, 158 test cases)** — **DONE**
- [x] **HARD GATE A: Reported test metrics reproduction (0.000% delta)** — **DONE**
- [x] **HARD GATE B: Test split reproduction without data leakage** — **DONE**
- [x] **Dataset facts & telemetry units (Hourly MWh, 7 channels)** — **DONE**
- [x] **Git safety backup (`pre-demo-backup` tag, sha256 manifest)** — **DONE**

## 2. PHASE 1 — EXPORT, LIVE API & FALLBACK CACHE
- [x] **2.1 Model Registry** (`conflux`, `quorum`, `nexus` in `demo/artifacts/models/`) — **DONE**
- [x] **2.2 Case curation** (40 cases, seed 13, stratified across 4 classes) — **DONE**
- [x] **2.3 Core diagnose + explain functions** (`demo/export/explain_core.py`, Shapley exact) — **DONE**
- [x] **2.4 Web assets and fallback cache** (40 case JSONs <= 40KB, 200 WebP images <= 450KB) — **DONE**
- [x] **2.5 Verification gates** (`demo/_recon/metrics_check.md`) — **DONE**
- [x] **2.6 Live API & Preprocessed Cache** (`demo/api/main.py`, `preprocessed_inputs.pt`, `bench.py`, `latency.md`) — **DONE**

## 3. PHASE 2 — FRONTEND (`demo/web/`)
- [x] **3.1 Stack & budgets** (Vite + React 18 + TS, `@fontsource` JetBrains Mono + VT323, self-hosted) — **DONE**
- [x] **3.2 Layout** (Desktop 1920x1080 / 1366x768 / Mobile stacked) — **DONE**
- [x] **3.3 Interaction flow** (Idle nominal diagram, shuffle feeder, Space/1-4/QWE/R/H keybindings, ?fast=1, pan/zoom) — **DONE**
- [x] **3.4 Animation timeline** (0-250ms glitch/flash, shockwave, reticle, scan-sweep, typing bubbles, ground truth stamp, MODELS DISAGREE flag) — **DONE**
- [x] **3.5 Diagnosis bubbles** (Chamfered callouts on right edge of stage aligned to model cards with pointer tails, fault chips) — **DONE**
- [x] **3.6 Explainability panel** (Shapley horizontal bar, narrative, model-specific breakdown, report word highlights, telemetry minimap) — **DONE**
- [x] **3.7 About / honesty overlay** (`ABOUT` modal + footer status line `MODE: HELD-OUT TEST EVENT REPLAY · LIVE INFERENCE`) — **DONE**
- [x] **3.8 Analogpunk design system** (Soft neon glows, LED bars, CRT scanlines, chamfered clips, palette) — **DONE**
- [x] **3.9 Three.js layer** (Hexagonal prism logo, convergent particles boot scene, shockwave ring, scan-sweep, graceful fallback) — **DONE**

## 4. PHASE 3 — TESTS
- [x] **Python metrics reproduction & Shapley efficiency tests** — **DONE**
- [x] **API benchmark & latency metrics** (`demo/_recon/latency.md`: p50=30.5ms, p95=40.6ms) — **DONE**
- [x] **Automated Playwright E2E suite** (`demo/web/test_e2e.cjs` — resilience fallback, multi-class injection, zero third-party requests) — **DONE**
- [x] **Responsive screenshots captured** (`desktop_1080p.png`, `laptop_768p.png`, `mobile_390.png`) — **DONE**
- [x] **Build test: zero TypeScript errors, bundle report verified** — **DONE**

## 5. PHASE 4 — DEPLOY
- [x] **Vercel deployment** ([https://sonic-peridot-2oqill2.tmp.now](https://sonic-peridot-2oqill2.tmp.now)) — **DONE**
- [x] **Vercel claim URL** ([https://vercel.com/claim-deployment?code=0b90d413-5b2a-4986-97dc-aed4c5bbac3c](https://vercel.com/claim-deployment?code=0b90d413-5b2a-4986-97dc-aed4c5bbac3c)) — **DONE**
- [x] **Local tunnel / Dockerfile / Render backup** (`demo/api/Dockerfile`, `demo/render.yaml`, `demo/DEPLOY.md`) — **DONE**

## 6. PHASE 5 — GIT & SAFETY PROOF
- [x] **Additive only verification** (`git diff pre-demo-backup..HEAD` — 322 additions strictly under `demo/`, 0 deletions, 0 outside edits) — **DONE**
- [x] **Pre-existing tracked files sha256 verification (1,652 files matched 100%)** — **DONE**
- [x] **Merged `feat/demo-dashboard` to `main`** — **DONE**
