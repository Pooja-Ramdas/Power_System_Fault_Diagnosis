# demo/DEPLOY.md
# Deployment Guide — Multimodal Fault Console

## Architecture Overview

```
┌─────────────────────────────────────────────┐
│  STATIC FRONTEND (Vercel)                   │
│  demo/web/  →  https://*.vercel.app         │
│  • React 18 + Vite + TypeScript             │
│  • Self-hosted fonts (@fontsource)          │
│  • No CDN, no external resources            │
│  • Pre-built fallback cache in /data/       │
└──────────────────┬──────────────────────────┘
                   │ POST /diagnose, /evidence
                   ▼ (3.5s timeout → fallback)
┌─────────────────────────────────────────────┐
│  LIVE INFERENCE API (HF Docker Space / PC)  │
│  demo/api/  →  FastAPI + uvicorn            │
│  • CONFLUX / QUORUM / NEXUS models          │
│  • Shapley explainability                   │
│  • Returns source: "live"                   │
└─────────────────────────────────────────────┘
```

---

## Phase 1: Deploy Static Frontend to Vercel

### Prerequisites
- Vercel account: https://vercel.com
- Vercel CLI: `npm install -g vercel`

### Steps

```powershell
cd demo/web
npm run build        # Produces demo/web/dist/
vercel --prod        # Follow prompts; set root to demo/web
```

### Environment Variables on Vercel
In Vercel dashboard → Project Settings → Environment Variables:
```
VITE_API_URL = https://your-api-space.hf.space
```

If no API is deployed yet, leave blank — the frontend runs in full cache-only mode
(all results labeled CACHED with an orange triangle badge).

### Cache-Only Mode
The site works 100% without the API. Every pre-built result in `/data/cases/` is
displayed immediately with a clear **CACHED** badge. Live results replace them when
the API is reachable.

---

## Phase 2: Deploy Live Inference API

### Option A: Hugging Face Docker Space (Recommended)

1. Create a Space at https://huggingface.co/spaces
   - SDK: Docker
   - Hardware: CPU Basic (2 vCPU, 16 GB RAM — sufficient for inference-only)
   - Port: 7860

2. The Space needs these files (copy from repo):
   ```
   demo/api/main.py
   demo/api/requirements.txt
   demo/api/Dockerfile
   demo/artifacts/    (models + preprocessing)
   demo/export/       (explain_core + load_models + vendor)
   demo/web/public/data/  (cases + index.json)
   ```

3. Push to HF:
   ```bash
   git clone https://huggingface.co/spaces/YOUR_USER/triad13-fault-api
   # Copy relevant files, push
   git push
   ```

4. Update Vercel env: `VITE_API_URL = https://YOUR_USER-triad13-fault-api.hf.space`

### Option B: Local Machine + cloudflared Tunnel

```bash
# 1. Start the API locally
cd d:/Downloads/Power_System_Fault_Diagnosis
python -m uvicorn demo.api.main:app --host 127.0.0.1 --port 8000

# 2. Create a public tunnel (no account needed for 30 day URLs)
cloudflared tunnel --url http://localhost:8000

# 3. Copy the tunnel URL (e.g. https://xxxx.trycloudflare.com)
# 4. Open the Vercel app with ?api=https://xxxx.trycloudflare.com
```

### Option C: render.com (see demo/render.yaml)

---

## Phase 3: Update CORS for Production

In `demo/api/main.py`, the CORS regex already allows any `.vercel.app` origin:
```python
allow_origin_regex=r"(https://.*\.vercel\.app|http://localhost:\d+|...)"
```

If you use a custom domain, add it to this regex.

---

## Verification

After deployment, verify:
1. Open the Vercel URL → Boot animation plays, 4 fault class buttons visible
2. Select SHORT CIRCUIT → case list populates
3. Pick any case → single-line diagram loads
4. Press INJECT FAULT → 3 model bubbles show results
   - If API is live: badges show **LIVE** (green dot)
   - If API is offline: badges show **CACHED** (orange triangle)
5. Ablation table tab shows 7 modality combination rows
6. Telemetry minimap shows 48-hour hourly generation data (MWh)
7. Report excerpt shows highlighted words

---

## Safety Proof

At any time, verify no pre-existing files were modified:
```bash
git diff --name-status pre-demo-backup..HEAD
# Must show ONLY "A" (added) entries, all under demo/
```

Regenerate sha256 manifest to verify:
```bash
git ls-files | grep -v "^demo/" | xargs sha256sum > /tmp/current.sha256
diff demo/_recon/tracked_files.sha256 /tmp/current.sha256
# Must be empty (no differences)
```
