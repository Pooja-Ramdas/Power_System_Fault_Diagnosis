"""
Curate test-set cases for the live demo dashboard:
- TEST split only (seed 42).
- image_is_real_infotecnica == True (or official_pdf), decodable, non-empty telemetry and report.
- Seeded random (seed 13) with grid element diversity.
- Never filtered or ranked by model correctness.
- Generates demo/export/selection_report.json.
"""

import os
import sys
import json
import random
from collections import Counter
from PIL import Image
import numpy as np
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT_DIR)

import dataset_utils as du
from demo.export.load_models import load

# Fault class mapping
CLASS_MAP = {
    "line_fault": {"display": "LINE FAULT", "sub": "Transmission Line Fault"},
    "breaker_fault": {"display": "BREAKER FAULT", "sub": "Breaker / Switchgear Fault"},
    "transformer_fault": {"display": "TRANSFORMER FAULT", "sub": "Power Transformer Fault"},
    "busbar_fault": {"display": "BUSBAR FAULT", "sub": "Busbar Section Overload"},
}

def main():
    cases = du.scan_dataset()
    train_cases, val_cases, test_cases = du.stratified_case_split(cases, seed=42)

    eligible = {c: [] for c in du.FAULT_CLASSES}

    for c in test_cases:
        with open(c.meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        is_real = meta.get("image_is_real_infotecnica", False) or meta.get("diagram_source") == "official_pdf"
        if not is_real:
            continue

        try:
            with Image.open(c.image_path) as img:
                img.verify()
        except Exception:
            continue

        if not os.path.exists(c.telemetry_path) or os.path.getsize(c.telemetry_path) < 100:
            continue

        txt_en = open(c.report_en_path, encoding="utf-8", errors="ignore").read().strip() if c.report_en_path and os.path.exists(c.report_en_path) else ""
        txt_es = open(c.report_es_path, encoding="utf-8", errors="ignore").read().strip() if c.report_es_path and os.path.exists(c.report_es_path) else ""
        if not (txt_en or txt_es):
            continue

        element = meta.get("affected_element", meta.get("element_name", meta.get("substation", "Grid Asset")))
        eligible[c.label].append({
            "case": c,
            "meta": meta,
            "element": element,
        })

    rng = random.Random(13)

    selected = []
    # For classes with <= 12, select all. For line_fault, select 17 to reach 40 total cases
    targets = {
        "line_fault": 17,
        "breaker_fault": 12,
        "transformer_fault": 12,
        "busbar_fault": 12,
    }

    for cls in du.FAULT_CLASSES:
        pool = eligible[cls]
        target = targets[cls]
        if len(pool) <= target:
            chosen = list(pool)
        else:
            # Sort by element name to group and shuffle deterministically for diversity
            pool_sorted = sorted(pool, key=lambda x: (x["element"], x["case"].case_id))
            rng.shuffle(pool_sorted)
            # Pick distinct elements first
            seen_elements = set()
            chosen = []
            remaining = []
            for item in pool_sorted:
                el = item["element"]
                if el not in seen_elements and len(chosen) < target:
                    seen_elements.add(el)
                    chosen.append(item)
                else:
                    remaining.append(item)
            while len(chosen) < target and remaining:
                chosen.append(remaining.pop(0))

        selected.extend(chosen)

    # Sort selected cases by case_id
    selected = sorted(selected, key=lambda x: x["case"].case_id)

    curated_ids = [x["case"].case_id for x in selected]
    print(f"Curated {len(selected)} total cases across classes:")
    counts = Counter(x["case"].label for x in selected)
    for cls, count in counts.items():
        print(f"  {cls} ({CLASS_MAP[cls]['display']}): {count}")

    # Compute models accuracy on full test set vs curated set
    print("\nEvaluating model performance on full test set vs curated set...")
    models = {
        "conflux": load("conflux"),
        "quorum": load("quorum"),
        "nexus": load("nexus"),
    }

    # Load test set features
    case_index = {c.case_id: i for i, c in enumerate(cases)}
    img_feats_all = du.get_image_features(cases)
    tel_windows_all, tel_masks_all, n_channels = du.load_telemetry_windows(cases)
    class_to_idx = {c: i for i, c in enumerate(du.FAULT_CLASSES)}

    tr_idx = [case_index[c.case_id] for c in train_cases]
    scaler = du.fit_telemetry_scaler(tel_windows_all[tr_idx], tel_masks_all[tr_idx])

    import pickle
    with open(os.path.join(ROOT_DIR, "demo", "artifacts", "preprocessing.pkl"), "rb") as f:
        prep = pickle.load(f)
    vectorizer = prep["tfidf_vectorizer"]

    def prepare_case_input(c_list):
        idx = [case_index[c.case_id] for c in c_list]
        img_f = torch.tensor(img_feats_all[idx], dtype=torch.float32)

        texts = du.load_texts(c_list)
        txt_f = torch.tensor(vectorizer.transform(texts).toarray().astype(np.float32), dtype=torch.float32)

        tw = tel_windows_all[idx]
        tm = tel_masks_all[idx]
        scaled_tw = du.apply_telemetry_scaler(tw, scaler)
        tw_t = torch.tensor(scaled_tw, dtype=torch.float32)
        tm_t = torch.tensor(tm, dtype=torch.float32)

        y = [class_to_idx[c.label] for c in c_list]
        return {"image_feat": img_f, "text_feat": txt_f, "telemetry_window": tw_t, "telemetry_mask": tm_t}, y

    full_inputs, full_y = prepare_case_input(test_cases)
    curated_cases = [x["case"] for x in selected]
    curated_inputs, curated_y = prepare_case_input(curated_cases)

    comparison = {}
    for mid, m in models.items():
        full_preds = m.predict(full_inputs).argmax(axis=1)
        cur_preds = m.predict(curated_inputs).argmax(axis=1)

        full_acc = float(np.mean(full_preds == full_y))
        cur_acc = float(np.mean(cur_preds == curated_y))
        comparison[mid] = {
            "full_test_accuracy": full_acc,
            "curated_set_accuracy": cur_acc,
            "curated_correct_count": int(np.sum(cur_preds == curated_y)),
            "curated_total": len(curated_y),
        }
        print(f"  {mid.upper()}: Full Test Acc = {full_acc*100:.2f}%, Curated Acc = {cur_acc*100:.2f}%")

    report = {
        "curated_total": len(selected),
        "target_range": "40-48",
        "seed": 13,
        "criteria": {
            "source": "held-out test split (seed 42)",
            "image_is_real_infotecnica": True,
            "image_decodes": True,
            "telemetry_present": True,
            "report_text_non_empty": True,
            "no_filter_by_model_correctness": True,
        },
        "counts_per_class": {cls: counts[cls] for cls in du.FAULT_CLASSES},
        "display_names": {cls: CLASS_MAP[cls]["display"] for cls in du.FAULT_CLASSES},
        "model_performance_comparison": comparison,
        "selected_case_ids": curated_ids,
    }

    report_path = os.path.join(ROOT_DIR, "demo", "export", "selection_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Also save selected case objects for subsequent export steps
    curated_meta = []
    for item in selected:
        c = item["case"]
        meta = item["meta"]
        curated_meta.append({
            "case_id": c.case_id,
            "true_class": c.label,
            "display_class": CLASS_MAP[c.label]["display"],
            "element_name": item["element"],
            "case_dir": c.case_dir,
            "image_path": c.image_path,
            "telemetry_path": c.telemetry_path,
            "report_en_path": c.report_en_path,
            "report_es_path": c.report_es_path,
            "meta_path": c.meta_path,
        })
    with open(os.path.join(ROOT_DIR, "demo", "export", "curated_cases.json"), "w", encoding="utf-8") as f:
        json.dump(curated_meta, f, indent=2)

    print(f"\nSelection report written to {report_path}")

if __name__ == "__main__":
    main()
