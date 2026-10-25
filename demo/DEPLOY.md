# Deployment Guide — TRIAD-13 Multimodal Fault Console

## Status Overview

- **Static Frontend**: Pre-built at `demo/web/dist/`, production-ready with zero external dependencies.
- **Anonymous Vercel Deployment**: [https://sonic-peridot-2oqill2.tmp.now](https://sonic-peridot-2oqill2.tmp.now)
  - Claim URL (permanent): [https://vercel.com/claim-deployment?code=0b90d413-5b2a-4986-97dc-aed4c5bbac3c](https://vercel.com/claim-deployment?code=0b90d413-5b2a-4986-97dc-aed4c5bbac3c)
- **Live Inference API**: FastAPI server running CPU-only PyTorch inference with Shapley explainability.
  - Local benchmark: p50 = 30.5 ms, p95 = 40.6 ms (50 sequential), wall = 180.8 ms (10 concurrent).

---

## 1. Frontend Deployment (Vercel)

### Option A: 1-Click Import from GitHub (Recommended)
1. Navigate to [https://vercel.com/new](https://vercel.com/new).
2. Select your repository: **`Pooja-Ramdas/Power_System_Fault_Diagnosis`** (branch: `feat/demo-dashboard` or `main` after merge).
3. Configure project settings:
   - **Root Directory**: `demo/web`
   - **Framework Preset**: `Vite`
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`
   - **Install Command**: `npm install`
4. Environment Variables (optional if running in standalone cached mode):
   - `VITE_API_URL`: Your deployed API URL (e.g., `https://your-user-triad13-api.hf.space` or tunnel URL).
5. Click **Deploy**.

### Option B: Claim Existing Anonymous Deployment
Visit the claim link generated during the build:
[https://vercel.com/claim-deployment?code=0b90d413-5b2a-4986-97dc-aed4c5bbac3c](https://vercel.com/claim-deployment?code=0b90d413-5b2a-4986-97dc-aed4c5bbac3c)

---

## 2. Live API Deployment

### Option A: Hugging Face Docker Space (Free CPU Basic, 16 GB RAM)
1. Create a new Space at [https://huggingface.co/new-space](https://huggingface.co/new-space).
2. Select **Docker** SDK, hardware **CPU Basic** (free tier).
3. In your Space repository, copy the following files:
   - `demo/api/Dockerfile`
   - `demo/api/requirements.txt`
   - `demo/api/main.py`
   - `demo/artifacts/` (includes `models/`, `preprocessing.pkl`, `preprocessed_inputs.pt`, `manifest.json`)
   - `demo/export/` (`explain_core.py`, `load_models.py`, `vendor/`)
   - `demo/web/public/data/` (`index.json`, `cases/`)
4. The container exposes port 7860 and spins up in ~4 seconds.
5. In your frontend, pass `?api=https://<your-space>.hf.space` to connect live.

### Option B: Cloudflare Quick Tunnel (Instant Zero-Config Free Tunnel)
Run on any host with internet access:
```bash
# 1. Start the API locally
python -m uvicorn demo.api.main:app --host 127.0.0.1 --port 8000

# 2. In another terminal, open a free Cloudflare tunnel
cloudflared tunnel --url http://127.0.0.1:8000
```
Copy the generated `https://*.trycloudflare.com` URL and open the frontend with:
`https://your-frontend.vercel.app/?api=https://your-tunnel.trycloudflare.com`

### Option C: Render Web Service
Push repository with the included `demo/render.yaml` configuration.

---

## 3. Privacy & Zero-Dependency Audit

- **Self-Hosted Fonts**: Bundled via `@fontsource/jetbrains-mono` and `@fontsource/vt323` (no Google Fonts or external CDN).
- **Network Leaks**: Playwright automated suite asserted zero third-party network requests.
- **Fail-Safe Offline Mode**: Service worker (`demo/web/public/sw.js`) caches data and assets; fallback cache engages within 3.5 seconds if the live API is unreachable, truthfully displaying the **CACHED** badge.
