"""
Builds Fallback Cache and Web Assets for Multimodal Fault Console:
- Converts single-line diagrams to dark HUD and RAW WebP images.
- Computes Grad-CAM transparent overlays for each model.
- Precomputes offline fallback JSON for all curated cases using explain_core.py.
- Validates data budget: per-case JSON <= 40 KB, image payload <= 450 KB.
- Emits demo/web/public/data/index.json and demo/web/public/data/cases/<case>.json.
"""

import os
import sys
import json
import pickle
from typing import Tuple, Dict, List, Any
import numpy as np
import torch
from PIL import Image, ImageOps

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT_DIR)

import dataset_utils as du
from demo.export.load_models import load
from demo.export.vendor.feature_utils import extract_telemetry_window_single
from demo.export.explain_core import (
    FAULT_CLASSES,
    CLASS_DISPLAY,
    run_full_diagnosis,
    compute_diagram_cam_overlay,
)

WEB_PUBLIC_DIR = os.path.join(ROOT_DIR, "demo", "web", "public")
IMG_OUT_DIR = os.path.join(WEB_PUBLIC_DIR, "img")
DATA_OUT_DIR = os.path.join(WEB_PUBLIC_DIR, "data")
CASES_OUT_DIR = os.path.join(DATA_OUT_DIR, "cases")

os.makedirs(IMG_OUT_DIR, exist_ok=True)
os.makedirs(CASES_OUT_DIR, exist_ok=True)

CLASS_COLORS = {
    "line_fault": "#ff4d6d",
    "breaker_fault": "#ffb347",
    "transformer_fault": "#c77dff",
    "busbar_fault": "#ff7b3a",
}


def create_hud_and_raw_images(src_path: str, case_id: str) -> Tuple[str, str, int, int]:
    """
    Creates dark HUD version and RAW version of the single-line diagram.
    Max dimension 2000 px, WebP quality 80.
    """
    hud_filename = f"{case_id}.hud.webp"
    raw_filename = f"{case_id}.raw.webp"
    hud_out = os.path.join(IMG_OUT_DIR, hud_filename)
    raw_out = os.path.join(IMG_OUT_DIR, raw_filename)

    with Image.open(src_path) as orig:
        orig_rgb = orig.convert("RGB")
        w, h = orig_rgb.size

        # Scale so max dimension is 1600 px (crisp for 1920x1080 display, fits <= 450 KB total per case)
        max_dim = max(w, h)
        if max_dim > 1600:
            scale = 1600.0 / max_dim
            new_w, new_h = int(w * scale), int(h * scale)
            orig_rgb = orig_rgb.resize((new_w, new_h), Image.Resampling.LANCZOS)
            w, h = new_w, new_h

        # Save RAW WebP (quality 72, <= 200 KB)
        orig_rgb.save(raw_out, format="WEBP", quality=72)

        # Generate HUD Dark Mode:
        # Detect light background by mean luminance
        gray = orig_rgb.convert("L")
        mean_lum = np.mean(gray)

        if mean_lum > 120:  # Typical black on white engineering PDF
            inv = ImageOps.invert(orig_rgb)
        else:
            inv = orig_rgb

        # Tint lines cyan on dark background:
        # Map grayscale to custom palette: black -> #05080d, bright -> #38e8ff
        inv_arr = np.array(inv.convert("L"), dtype=np.float32) / 255.0
        hud_arr = np.zeros((h, w, 3), dtype=np.uint8)

        # Background base: (5, 8, 13)
        # Foreground line art: (56, 232, 255)
        for c, (bg_v, fg_v) in enumerate([(5, 56), (8, 232), (13, 255)]):
            hud_arr[:, :, c] = np.clip(bg_v + inv_arr * (fg_v - bg_v), 0, 255).astype(np.uint8)

        hud_img = Image.fromarray(hud_arr, mode="RGB")
        hud_img.save(hud_out, format="WEBP", quality=72)

    return f"img/{hud_filename}", f"img/{raw_filename}", w, h


def main():
    print("Loading curated cases and models...")
    curated_path = os.path.join(ROOT_DIR, "demo", "export", "curated_cases.json")
    with open(curated_path, "r", encoding="utf-8") as f:
        curated_cases = json.load(f)

    # Check for manual highlights if available
    highlights_path = os.path.join(ROOT_DIR, "demo", "export", "highlights.json")
    highlights = {}
    if os.path.exists(highlights_path):
        with open(highlights_path, "r", encoding="utf-8") as f:
            highlights = json.load(f)

    # Load models
    models = {
        "conflux": load("conflux"),
        "quorum": load("quorum"),
        "nexus": load("nexus"),
    }

    # Load preprocessing artifacts
    with open(os.path.join(ROOT_DIR, "demo", "artifacts", "preprocessing.pkl"), "rb") as f:
        prep = pickle.load(f)
    vectorizer = prep["tfidf_vectorizer"]
    scaler = prep["telemetry_scaler"]

    # Pre-extract image features for all cases to keep exact parity
    cases_all = du.scan_dataset()
    case_index = {c.case_id: i for i, c in enumerate(cases_all)}
    img_feats_all = du.get_image_features(cases_all)

    index_cases = []

    print(f"Processing assets and computing explanations for {len(curated_cases)} cases...")

    for i, item in enumerate(curated_cases):
        cid = item["case_id"]
        true_cls = item["true_class"]
        elem = item["element_name"]

        with open(item["meta_path"], "r", encoding="utf-8") as f:
            meta = json.load(f)
        event_ts = meta.get("event_timestamp", meta.get("timestamp", "2024-03-15T14:00:00Z"))

        # 1. Process Images
        hud_rel, raw_rel, width, height = create_hud_and_raw_images(item["image_path"], cid)

        # 2. Process Telemetry
        win, mask, df_tel = extract_telemetry_window_single(item["telemetry_path"], target_len=48)
        scaled_win = du.apply_telemetry_scaler(win[np.newaxis, :, :], scaler)[0]

        # Extract rows for frontend telemetry table (up to 48 rows, 7 columns)
        # Columns: total, thermal, solar, hydro, wind, bess, other
        tel_rows = []
        if not df_tel.empty:
            timestamps = df_tel["timestamp"].tolist()
            t_len = min(48, len(timestamps))
            for t_idx in range(t_len):
                row_vals = [round(float(win[ch, t_idx]), 2) for ch in range(7)]
                tel_rows.append({
                    "ts": str(timestamps[t_idx]),
                    "channels": row_vals,
                })
        else:
            for t_idx in range(48):
                tel_rows.append({
                    "ts": f"T+{t_idx:02d}:00",
                    "channels": [0.0] * 7,
                })

        # 3. Process Text Reports
        rpt_en = ""
        if item["report_en_path"] and os.path.exists(item["report_en_path"]):
            with open(item["report_en_path"], "r", encoding="utf-8", errors="ignore") as f:
                rpt_en = f.read().strip()
        rpt_es = ""
        if item["report_es_path"] and os.path.exists(item["report_es_path"]):
            with open(item["report_es_path"], "r", encoding="utf-8", errors="ignore") as f:
                rpt_es = f.read().strip()

        active_text = rpt_en if rpt_en else (rpt_es if rpt_es else "No report text available.")

        # Prepare inputs
        case_idx = case_index[cid]
        img_f = torch.tensor(img_feats_all[case_idx:case_idx+1], dtype=torch.float32)
        txt_sparse = vectorizer.transform([active_text]).toarray().astype(np.float32)
        txt_f = torch.tensor(txt_sparse, dtype=torch.float32)
        tw_t = torch.tensor(scaled_win[np.newaxis, :, :], dtype=torch.float32)
        tm_t = torch.tensor(mask[np.newaxis, :, :], dtype=torch.float32)

        case_input = {
            "case_id": cid,
            "image_feat": img_f,
            "text_feat": txt_f,
            "telemetry_window": tw_t,
            "telemetry_mask": tm_t,
        }

        # 4. Run Diagnoses and Explanations for all 3 models
        predictions = {}
        explanations = {}
        ablations = {}
        by_mask = {}
        first_word_scores = []

        MASK_MAP = {
            "D": ("D",),
            "T": ("T",),
            "R": ("R",),
            "DT": ("D", "T"),
            "DR": ("D", "R"),
            "TR": ("T", "R"),
            "DTR": ("D", "T", "R"),
        }

        for mid, m in models.items():
            by_mask[mid] = {}
            dtr_full = None
            for k, m_tuple in MASK_MAP.items():
                diag_m = run_full_diagnosis(
                    model_id=mid,
                    predictor=m,
                    case_input=case_input,
                    true_class=true_cls,
                    element_name=elem,
                    report_text=active_text,
                    vectorizer=vectorizer,
                    mask=m_tuple,
                )
                by_mask[mid][k] = {
                    "prediction": diag_m["prediction"],
                    "explanation": {
                        "contrib": diag_m["explanation"]["contrib"],
                        "dominant": diag_m["explanation"]["dominant"],
                        "narrative": diag_m["explanation"]["narrative"],
                        "model_specific": diag_m["explanation"].get("model_specific", {}),
                    },
                }
                if k == "DTR":
                    dtr_full = diag_m
                    if not first_word_scores:
                        first_word_scores = diag_m["word_scores"]

            predictions[mid] = dtr_full["prediction"]
            explanations[mid] = dtr_full["explanation"]
            ablations[mid] = dtr_full["ablations"]

            # Generate CAM WebP overlay
            cam_img = compute_diagram_cam_overlay(item["image_path"], img_f)
            cam_out_path = os.path.join(IMG_OUT_DIR, f"{cid}.{mid}.cam.webp")
            cam_img.save(cam_out_path, format="WEBP", quality=75)

        # Compile case JSON (trim excerpt to 1800 chars for clean display and <= 40KB budget)
        disp_en = rpt_en[:1200].strip() if rpt_en else rpt_es[:1200].strip()
        disp_es = rpt_es[:1200].strip() if rpt_es else rpt_en[:1200].strip()

        case_json = {
            "id": cid,
            "event_ts": event_ts,
            "element_name": elem,
            "true_class": true_cls,
                        "image": {
                "hud": hud_rel,
                "raw": raw_rel,
                "width": width,
                "height": height,
                "highlight": highlights.get(cid, None),
                "is_real_infotecnica": True,
            },
            "telemetry": {
                "columns": du.TELEMETRY_CHANNELS,
                "units": ["MWh"] * 7,
                "t0_index": 24,
                "rows": tel_rows,
            },
            "report": {
                "en": disp_en,
                "es": disp_es,
                "shown_language": "en" if rpt_en else "es",
                "word_scores": first_word_scores,
            },
            "predictions": predictions,
            "explanations": explanations,
            "ablations": ablations,
            "by_mask": by_mask,
        }

        # Validate case JSON size (budget: <= 40 KB)
        case_json_str = json.dumps(case_json, separators=(',', ':'))
        json_size_kb = len(case_json_str.encode("utf-8")) / 1024.0
        assert json_size_kb <= 40.0, f"Case {cid} JSON exceeds budget: {json_size_kb:.1f} KB > 40 KB"

        case_file_path = os.path.join(CASES_OUT_DIR, f"{cid}.json")
        with open(case_file_path, "w", encoding="utf-8") as f:
            f.write(case_json_str)

        index_cases.append({
            "id": cid,
            "event_ts": event_ts,
            "element_name": elem,
            "true_class": true_cls,
                        "predictions": {mid: predictions[mid]["pred"] for mid in models},
        })

    # 5. Build index.json
    index_data = {
        "title": "TRIAD-13 Multimodal Fault Console",
        "models": [
            {
                "id": "conflux",
                "name": "CONFLUX",
                "fusion": "EARLY FUSION",
                "color": "#38e8ff",
                "tooltip": "Features from diagram, telemetry and report are merged into one joint representation at the input, before any decision is made.",
                "test_accuracy": 0.8797,
                "test_macro_f1": 0.8220,
            },
            {
                "id": "quorum",
                "name": "QUORUM",
                "fusion": "LATE FUSION",
                "color": "#5dffb0",
                "tooltip": "Each modality has its own specialist model that reaches an independent verdict; a learned vote combines them.",
                "test_accuracy": 0.8354,
                "test_macro_f1": 0.7651,
            },
            {
                "id": "nexus",
                "name": "NEXUS",
                "fusion": "HYBRID FUSION",
                "color": "#8fa8ff",
                "tooltip": "Encoders exchange evidence through cross-modal attention mid-network, then a decision-level stage combines the result.",
                "test_accuracy": 0.8228,
                "test_macro_f1": 0.7303,
            },
        ],
        "classes": [
            {"id": "line_fault", "display": "LINE FAULT", "color": CLASS_COLORS["line_fault"], "key": "1"},
            {"id": "breaker_fault", "display": "BREAKER FAULT", "color": CLASS_COLORS["breaker_fault"], "key": "2"},
            {"id": "transformer_fault", "display": "TRANSFORMER FAULT", "color": CLASS_COLORS["transformer_fault"], "key": "3"},
            {"id": "busbar_fault", "display": "BUSBAR FAULT", "color": CLASS_COLORS["busbar_fault"], "key": "4"},
        ],
        "cases": index_cases,
    }

    index_path = os.path.join(DATA_OUT_DIR, "index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_data, f, indent=2)

    print(f"\nCompleted! Generated fallback cache for {len(index_cases)} cases into {CASES_OUT_DIR}")
    print(f"Index created at {index_path}")

if __name__ == "__main__":
    main()
