"""
Multimodal Fault Console — Live Inference API
FastAPI service wrapping CONFLUX / QUORUM / NEXUS models.

All inferences are computed live; the only cache role is the pre-built
fallback in demo/web/public/data/. Every response carries source:"live".
"""

import sys, os, time, json, hashlib, asyncio, logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from datetime import datetime, timezone

# Make demo/ importable
DEMO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DEMO_DIR))

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("fault-api")

# ── Paths ─────────────────────────────────────────────────────────────────────
ARTIFACTS_DIR = DEMO_DIR / "artifacts"
MODELS_DIR    = ARTIFACTS_DIR / "models"
EXPORT_DIR    = DEMO_DIR / "export"
CASES_DIR     = DEMO_DIR / "web" / "public" / "data" / "cases"
INDEX_FILE    = DEMO_DIR / "web" / "public" / "data" / "index.json"

sys.path.insert(0, str(EXPORT_DIR))

# ── Lazy-loaded globals ───────────────────────────────────────────────────────
_models: Dict[str, object] = {}
_cases:  Dict[str, dict]   = {}
_startup_time = time.time()

# ── Rate limiter (simple sliding-window, no external deps) ────────────────────
_rate_buckets: Dict[str, list] = defaultdict(list)
RATE_LIMIT = 120       # requests per minute per IP
RATE_WINDOW = 60.0     # seconds

def _check_rate(ip: str) -> bool:
    if ip in ("127.0.0.1", "localhost", "testclient"):
        return True
    now = time.time()
    bucket = _rate_buckets[ip]
    # Prune old
    _rate_buckets[ip] = [t for t in bucket if now - t < RATE_WINDOW]
    if len(_rate_buckets[ip]) >= RATE_LIMIT:
        return False
    _rate_buckets[ip].append(now)
    return True

# ── Startup ───────────────────────────────────────────────────────────────────
def _load_all_models():
    from export.load_models import load as load_model
    for mid in ("conflux", "quorum", "nexus"):
        logger.info(f"Loading model: {mid}")
        t0 = time.perf_counter()
        _models[mid] = load_model(mid)
        logger.info(f"  {mid} loaded in {time.perf_counter()-t0:.2f}s")

def _load_all_cases():
    logger.info("Loading curated cases into RAM…")
    index = json.loads(INDEX_FILE.read_text())
    for entry in index["cases"]:
        cid = entry["id"]
        case_file = CASES_DIR / f"{cid}.json"
        if case_file.exists():
            _cases[cid] = json.loads(case_file.read_text())
    from export.explain_core import _get_prep, _get_preprocessed_inputs
    _get_prep()
    _get_preprocessed_inputs()
    logger.info(f"  {len(_cases)} cases loaded")

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Multimodal Fault Console API",
    description="Live inference for CONFLUX / QUORUM / NEXUS fusion models",
    version="1.0.0",
)

# CORS — allow Vercel deployments + localhost dev
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"(https://.*\.vercel\.app|http://localhost:\d+|http://127\.0\.0\.1:\d+)",
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
    max_age=600,
)

def _ensure_loaded():
    if not _models:
        _load_all_models()
    if not _cases:
        _load_all_cases()

@app.on_event("startup")
def startup_event():
    _ensure_loaded()
    logger.info("API ready ✓")

# ── Middleware: rate limiting ─────────────────────────────────────────────────
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    ip = request.client.host if request.client else "unknown"
    if request.method not in ("GET", "POST"):
        return await call_next(request)
    if not _check_rate(ip):
        return JSONResponse({"detail": "Rate limit exceeded (60 req/min)"}, status_code=429)
    return await call_next(request)

# ── Schemas ───────────────────────────────────────────────────────────────────
class DiagnoseRequest(BaseModel):
    case_id:  str
    model_id: str
    mask:     Optional[List[str]] = None   # e.g. ["D","T","R"]

class EvidenceRequest(BaseModel):
    case_id:  str
    model_id: str
    mask:     Optional[List[str]] = None

def _validate_request(case_id: str, model_id: str):
    _ensure_loaded()
    if case_id not in _cases:
        raise HTTPException(404, f"case_id '{case_id}' not in curated set")
    if model_id not in _models:
        raise HTTPException(404, f"model_id '{model_id}' must be conflux|quorum|nexus")

def _build_case_input(case_data: dict) -> dict:
    """Reconstruct the case_input dict from the cached JSON + raw data files."""
    from export.explain_core import build_case_input_from_json
    return build_case_input_from_json(case_data)

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check — lists loaded models and artifact hashes."""
    model_status = {mid: mid in _models for mid in ("conflux", "quorum", "nexus")}
    sha_map = {}
    for mid in ("conflux", "quorum", "nexus"):
        pt = MODELS_DIR / mid / "model.pt"
        if pt.exists():
            sha_map[mid] = hashlib.sha256(pt.read_bytes()).hexdigest()[:16]
    return {
        "status": "ok" if all(model_status.values()) else "loading",
        "models_loaded": model_status,
        "artifact_sha256s": sha_map,
        "cases_loaded": len(_cases),
        "uptime_s": round(time.time() - _startup_time, 1),
        "source_git_sha": _git_sha(),
        "utc": datetime.now(timezone.utc).isoformat(),
    }

@app.get("/warmup")
async def warmup():
    """Run a quick forward pass on each model (uses first curated case)."""
    if not _cases:
        raise HTTPException(503, "Cases not yet loaded")
    results = {}
    first_case = next(iter(_cases.values()))
    for mid in ("conflux", "quorum", "nexus"):
        if mid not in _models:
            results[mid] = {"error": "not loaded"}
            continue
        t0 = time.perf_counter()
        try:
            ci = _build_case_input(first_case)
            _models[mid].predict(ci, mask=("D","T","R"))
            results[mid] = {"latency_ms": round((time.perf_counter()-t0)*1000, 1), "ok": True}
        except Exception as e:
            results[mid] = {"error": str(e), "ok": False}
    return {"warmup": results}

@app.post("/diagnose")
def diagnose(req: DiagnoseRequest):
    """
    Live model inference: class probabilities + Shapley modality contributions.
    Returns source:"live".
    """
    _validate_request(req.case_id, req.model_id)
    mask = tuple(req.mask) if req.mask else ("D","T","R")
    case_data = _cases[req.case_id]

    t0 = time.perf_counter()
    try:
        from export.explain_core import run_full_diagnosis
        result = run_full_diagnosis(case_data, req.model_id, mask, _models[req.model_id])
    except Exception as e:
        logger.error(f"diagnose error {req.case_id}/{req.model_id}: {e}", exc_info=True)
        raise HTTPException(500, f"Inference error: {e}")

    latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    result["latency_ms"] = latency_ms
    result["source"] = "live"
    result["model_id"] = req.model_id
    result["case_id"] = req.case_id
    result["mask"] = list(mask)
    return result

@app.post("/evidence")
def evidence(req: EvidenceRequest):
    """
    Live evidence computation: telemetry importance, report word scores, CAM base64, narrative.
    Returns source:"live".
    """
    _validate_request(req.case_id, req.model_id)
    mask = tuple(req.mask) if req.mask else ("D","T","R")
    case_data = _cases[req.case_id]

    t0 = time.perf_counter()
    try:
        from export.explain_core import compute_full_evidence
        result = compute_full_evidence(case_data, req.model_id, mask, _models[req.model_id])
    except Exception as e:
        logger.error(f"evidence error {req.case_id}/{req.model_id}: {e}", exc_info=True)
        raise HTTPException(500, f"Evidence error: {e}")

    latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    result["latency_ms"] = latency_ms
    result["source"] = "live"
    result["model_id"] = req.model_id
    result["case_id"] = req.case_id
    result["mask"] = list(mask)
    return result

@app.get("/cases")
async def list_cases():
    """Return the index of all curated cases (same as /data/index.json)."""
    return {"cases": list(_cases.keys()), "count": len(_cases)}

# ── Helpers ───────────────────────────────────────────────────────────────────
def _git_sha() -> str:
    try:
        import subprocess
        r = subprocess.run(["git","rev-parse","--short","HEAD"],
                           capture_output=True, text=True, cwd=str(DEMO_DIR.parent))
        return r.stdout.strip()
    except Exception:
        return "unknown"
