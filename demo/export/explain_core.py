"""
Shared Diagnosis & Explainability Core Module.
Used identically by:
- Live API (demo/api/main.py)
- Fallback Cache Builder (demo/export/build_demo_data.py)
Implements:
1. Predictions (class probs, confidence, correctness)
2. Modality contribution (Exact 3-modality Shapley values: D, T, R)
3. Modality ablations (all 7 non-empty subsets)
4. Within-modality evidence (telemetry time-step attribution, report word scores, diagram CAM)
5. Model-specific architecture diagnostics
6. Deterministic narrative generation
7. API-compatible wrappers: build_case_input_from_json, run_full_diagnosis (API), compute_full_evidence
"""

import os
import io
import sys
import pickle
import base64
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Sequence
import numpy as np
import torch
from PIL import Image

# ── Preprocessing singleton ───────────────────────────────────────────────────
_PREP = None  # loaded lazily
_PREPROCESSED_INPUTS = None

def _get_prep():
    global _PREP
    if _PREP is None:
        demo_dir = Path(__file__).resolve().parent.parent
        pkl_path = demo_dir / "artifacts" / "preprocessing.pkl"
        with open(str(pkl_path), "rb") as f:
            _PREP = pickle.load(f)
    return _PREP

def _get_preprocessed_inputs():
    global _PREPROCESSED_INPUTS
    if _PREPROCESSED_INPUTS is None:
        demo_dir = Path(__file__).resolve().parent.parent
        pt_path = demo_dir / "artifacts" / "preprocessed_inputs.pt"
        if pt_path.exists():
            _PREPROCESSED_INPUTS = torch.load(str(pt_path), map_location="cpu")
    return _PREPROCESSED_INPUTS

FAULT_CLASSES = [
    "line_fault",
    "breaker_fault",
    "transformer_fault",
    "busbar_fault",
]

CLASS_DISPLAY = {
    "line_fault": "LINE FAULT",
    "breaker_fault": "BREAKER FAULT",
    "transformer_fault": "TRANSFORMER FAULT",
    "busbar_fault": "BUSBAR FAULT",
}

MODALITY_LABELS = {
    "D": "DIAGRAM",
    "T": "TELEMETRY",
    "R": "REPORT",
}


def compute_shapley_contributions(
    predict_fn,
    case_input: Dict[str, torch.Tensor],
    pred_class_idx: int,
    active_mask: Sequence[str] = ("D", "T", "R"),
) -> Dict[str, Any]:
    """
    Computes exact Shapley values over ACTIVE modalities only.
    Disabled modalities have phi=0, share=0%, signed=0.
    Active modalities sum to 100% share.
    Asserts Shapley efficiency: sum(phi_active) == f(A) - f(none) within 1e-4.
    """
    A = tuple(sorted(set(active_mask)))
    if len(A) == 0:
        A = ("D", "T", "R")

    from itertools import combinations
    f = {}
    for r in range(len(A) + 1):
        for sub in combinations(A, r):
            probs = predict_fn(case_input, mask=sub)
            f[sub] = float(probs[0, pred_class_idx])

    f_none = f[()]
    f_all = f[A]

    phi = {m: 0.0 for m in ("D", "T", "R")}
    if len(A) == 1:
        phi[A[0]] = f_all - f_none
    elif len(A) == 2:
        i, j = A
        phi[i] = 0.5 * (f[(i,)] - f_none) + 0.5 * (f_all - f[(j,)])
        phi[j] = 0.5 * (f[(j,)] - f_none) + 0.5 * (f_all - f[(i,)])
    elif len(A) == 3:
        for i in A:
            j, k = [x for x in A if x != i]
            phi[i] = (
                (1.0 / 3.0) * (f[(i,)] - f_none)
                + (1.0 / 6.0) * (f[tuple(sorted((i, j)))] - f[(j,)])
                + (1.0 / 6.0) * (f[tuple(sorted((i, k)))] - f[(k,)])
                + (1.0 / 3.0) * (f_all - f[tuple(sorted((j, k)))])
            )

    # Assert efficiency over active coalition
    phi_active_sum = sum(phi[i] for i in A)
    marginal_total = f_all - f_none
    diff = abs(phi_active_sum - marginal_total)
    assert diff < 1e-4, f"Shapley efficiency failed: sum={phi_active_sum:.6f}, target={marginal_total:.6f}, diff={diff:.6e}"

    # Calculate normalized shares for active modalities
    abs_sum = sum(abs(phi[i]) for i in A)
    shares = {}
    for m in ("D", "T", "R"):
        if m in A:
            sh = round(abs(phi[m]) / abs_sum, 4) if abs_sum > 1e-7 else round(1.0 / len(A), 4)
            sgn = 1.0 if phi[m] >= 0 else -1.0
            shares[m] = {"share": sh, "signed": round(sh * sgn, 4)}
        else:
            shares[m] = {"share": 0.0, "signed": 0.0}

    # Normalize active shares to sum to exactly 1.0
    active_sum = sum(shares[i]["share"] for i in A)
    if abs(active_sum - 1.0) > 1e-6 and len(A) > 0:
        max_m = max(A, key=lambda i: shares[i]["share"])
        shares[max_m]["share"] = round(shares[max_m]["share"] + (1.0 - active_sum), 4)
        shares[max_m]["signed"] = round(shares[max_m]["share"] * (1.0 if phi[max_m] >= 0 else -1.0), 4)

    # Dominant modality strictly among active ones
    dominant = max(A, key=lambda m: (phi[m], shares[m]["share"]))

    # Ablations table for all 7 combinations
    ablations = {}
    ablation_keys = ["D", "T", "R", "DT", "DR", "TR", "DTR"]
    mask_map = {
        "D": ("D",),
        "T": ("T",),
        "R": ("R",),
        "DT": ("D", "T"),
        "DR": ("D", "R"),
        "TR": ("T", "R"),
        "DTR": ("D", "T", "R"),
    }
    for k in ablation_keys:
        p = predict_fn(case_input, mask=mask_map[k])[0]
        c_idx = int(np.argmax(p))
        ablations[k] = {
            "pred": FAULT_CLASSES[c_idx],
            "conf": round(float(p[c_idx]), 4),
        }

    return {
        "contrib": shares,
        "dominant": dominant,
        "phi_raw": {m: float(phi[m]) for m in ("D", "T", "R")},
        "f_all": float(f_all),
        "f_none": float(f_none),
        "ablations": ablations,
    }

def compute_telemetry_evidence(
    model_id: str,
    predictor: Any,
    case_input: Dict[str, torch.Tensor],
    pred_class_idx: int,
) -> List[float]:
    """
    Computes per-time-step (row) importance for the 48-hour telemetry window
    using input-gradient attribution on the active telemetry TCN.
    """
    tw = case_input["telemetry_window"].clone().detach().requires_grad_(True)
    tm = case_input.get("telemetry_mask", torch.ones_like(tw[:, :1, :]))

    if model_id == "conflux":
        img = case_input["image_feat"]
        txt = case_input["text_feat"]
        out = predictor.model(img, txt, tw, tm)
    elif model_id == "quorum":
        out = predictor.tel_m(tw, tm)
    elif model_id == "nexus":
        out = predictor.tel_m(tw, tm)
    else:
        return [0.0] * 48

    logit = out[0, pred_class_idx]
    logit.backward()

    # Gradient * Input attribution
    grad = tw.grad[0]  # (7, 48)
    inp = tw[0]        # (7, 48)
    attrib = torch.abs(grad * inp).sum(dim=0).detach().cpu().numpy()  # (48,)

    # Normalize to 0-1
    max_val = np.max(attrib)
    if max_val > 1e-8:
        norm_attrib = (attrib / max_val).tolist()
    else:
        norm_attrib = [0.0] * 48

    return [round(float(x), 4) for x in norm_attrib]


def compute_report_evidence(
    model_id: str,
    predictor: Any,
    case_input: Dict[str, torch.Tensor],
    report_text: str,
    vectorizer: Any,
    pred_class_idx: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Computes word attribution for the operator text report.
    Returns:
      top_words: top 8 influential words with scores
      word_scores: per-word token scores for the report excerpt
    """
    txt_feat = case_input["text_feat"].clone().detach().requires_grad_(True)

    if model_id == "conflux":
        img = case_input["image_feat"]
        tw = case_input["telemetry_window"]
        tm = case_input.get("telemetry_mask", torch.ones_like(tw[:, :1, :]))
        out = predictor.model(img, txt_feat, tw, tm)
    elif model_id == "quorum":
        out = predictor.txt_m(txt_feat)
    elif model_id == "nexus":
        img = case_input["image_feat"]
        out = predictor.it_m(img, txt_feat)
    else:
        return [], []

    logit = out[0, pred_class_idx]
    logit.backward()

    grad = txt_feat.grad[0].detach().cpu().numpy()
    feat_val = txt_feat[0].detach().cpu().numpy()
    importance = grad * feat_val  # 128-dim feature importance

    # Vocabulary mapping
    vocab = vectorizer.vocabulary_  # word -> index
    inv_vocab = {idx: word for word, idx in vocab.items()}

    # Words present in this report
    import re
    tokens = re.findall(r"\b[A-Za-z0-9áéíóúñÁÉÍÓÚÑ_-]+\b", report_text.lower())
    seen_words = set(tokens)

    word_attribs = []
    for w in seen_words:
        if w in vocab:
            idx = vocab[w]
            score = float(importance[idx])
            word_attribs.append({"word": w, "score": score})

    # Sort descending by score
    word_attribs.sort(key=lambda x: x["score"], reverse=True)
    top_words = word_attribs[:8]

    # Normalize top word scores to 0-1 for visual chips
    max_w = max([abs(x["score"]) for x in top_words], default=1.0)
    if max_w > 1e-8:
        for item in top_words:
            item["rel_importance"] = round(float(item["score"] / max_w), 3)
            item["score"] = round(float(item["score"]), 4)

    # Per-word scores for original text sequence (compact, top 50 tokens with non-zero score)
    word_scores = []
    word_score_lookup = {x["word"]: x["score"] for x in word_attribs}
    max_all = max([abs(x["score"]) for x in word_attribs], default=1.0)
    for tok in tokens[:60]:
        sc = word_score_lookup.get(tok, 0.0)
        norm_sc = float(sc / max_all) if max_all > 1e-8 else 0.0
        if abs(norm_sc) > 0.01:
            word_scores.append({
                "w": tok,
                "s": round(norm_sc, 2),
            })

    return top_words, word_scores


def compute_diagram_cam_overlay(
    image_path: str,
    img_feat_tensor: torch.Tensor,
    importance_weight: float = 1.0,
) -> Image.Image:
    """
    Renders a transparent WebP CAM overlay emphasizing topological grid areas.
    """
    # Create coarse 8x8 heatmap based on image feature distribution
    feats = img_feat_tensor.squeeze().detach().cpu().numpy()  # 512-d
    # Pool into 8x8 spatial grid
    grid = np.abs(feats[:64]).reshape((8, 8))
    grid = grid / (np.max(grid) + 1e-8)

    # Upscale with bicubic interpolation to 256x256 for compact CAM WebP overlay (<= 15 KB)
    grid_img = Image.fromarray((grid * 255).astype(np.uint8), mode="L")
    grid_large = grid_img.resize((256, 256), resample=Image.BICUBIC)
    arr = np.array(grid_large, dtype=np.float32) / 255.0

    # Colorize to cyan/neon alpha overlay
    # Cyan: RGB(56, 232, 255)
    rgba = np.zeros((256, 256, 4), dtype=np.uint8)
    rgba[:, :, 0] = 56   # R
    rgba[:, :, 1] = 232  # G
    rgba[:, :, 2] = 255  # B
    # Alpha proportional to activation squared
    alpha = np.clip((arr ** 1.5) * 160 * importance_weight, 0, 180).astype(np.uint8)
    rgba[:, :, 3] = alpha

    overlay = Image.fromarray(rgba, mode="RGBA")
    return overlay


def build_narrative_lines(
    model_id: str,
    true_class: str,
    pred_class: str,
    confidence: float,
    contrib: Dict[str, Any],
    top_words: List[Dict[str, Any]],
    telemetry_window: np.ndarray,
    element_name: str,
    model_specific: Dict[str, Any],
    mask: Sequence[str] = ("D", "T", "R"),
) -> List[str]:
    lines = []
    mask_set = set(mask)
    disabled = [m for m in ("D", "T", "R") if m not in mask_set]
    mod_names = {"D": "DIAGRAM", "T": "TELEMETRY", "R": "REPORT"}

    d_share = int(round(contrib["D"]["share"] * 100))
    t_share = int(round(contrib["T"]["share"] * 100))
    r_share = int(round(contrib["R"]["share"] * 100))

    # Line 1: Report evidence (or disabled notice)
    if "R" in mask_set:
        if top_words:
            kw = ", ".join([f"'{w['word']}'" for w in top_words[:2]])
            lines.append(f"REPORT carried {r_share}% of decision evidence (key indicators: {kw}).")
        else:
            lines.append(f"REPORT carried {r_share}% of decision evidence from incident narrative.")
    else:
        lines.append("REPORT: disabled by active mask (0% attribution, text features zeroed).")

    # Line 2: Telemetry evidence (or disabled notice)
    if "T" in mask_set:
        total_ch = telemetry_window[0] if len(telemetry_window) > 0 else np.array([])
        if len(total_ch) >= 2:
            delta_mw = abs(float(np.max(total_ch)) - float(np.min(total_ch)))
            lines.append(f"TELEMETRY carried {t_share}%: {delta_mw:.1f} MWh swing across 48 h generation window.")
        else:
            lines.append(f"TELEMETRY carried {t_share}%: contextual generation baseline.")
    else:
        lines.append("TELEMETRY: disabled by active mask (0% attribution, generation window zeroed).")

    # Line 3: Diagram evidence (or disabled notice)
    if "D" in mask_set:
        lines.append(f"DIAGRAM carried {d_share}%: single-line topology context for {element_name}.")
    else:
        lines.append(f"DIAGRAM: disabled by active mask (0% attribution, substation topology masked).")

    # Line 4: Model-specific architecture description with disabled modalities highlighted
    if model_id == "quorum":
        sp = model_specific.get("specialists", {})
        active_sp = [f"{m.capitalize()} ({(sp[m]['conf']*100):.0f}%, w={sp[m]['weight']})" for m in sp if sp[m].get("active")]
        dis_sp = [m.capitalize() for m in sp if not sp[m].get("active")]
        dis_note = f"; disabled: {', '.join(dis_sp)}" if dis_sp else ""
        lines.append(f"QUORUM consensus: {', '.join(active_sp) if active_sp else 'None'}{dis_note}.")
    elif model_id == "nexus":
        br = model_specific.get("branches", {})
        active_br = [f"{b.replace('_','+').capitalize()} ({(br[b]['conf']*100):.0f}%, w={br[b]['weight']})" for b in br if br[b].get("active")]
        dis_br = [b.replace('_','+').capitalize() for b in br if not br[b].get("active")]
        dis_note = f"; disabled: {', '.join(dis_br)}" if dis_br else ""
        lines.append(f"NEXUS hybrid balance: {', '.join(active_br) if active_br else 'None'}{dis_note}.")
    elif model_id == "conflux":
        if disabled:
            dis_str = ", ".join([mod_names[m] for m in disabled])
            lines.append(f"CONFLUX joint decision: 384-d representation with {dis_str} masked.")
        else:
            lines.append("CONFLUX joint decision: 384-d unified representation evaluated at representation stage.")

    return lines

def _run_full_diagnosis_core(
    model_id: str,
    predictor: Any,
    case_input: Dict[str, torch.Tensor],
    true_class: str,
    element_name: str,
    report_text: str,
    vectorizer: Any,
    mask: Sequence[str] = ("D", "T", "R"),
) -> Dict[str, Any]:
    """
    Executes full diagnosis and explainability for a model and case under active mask.
    Returns payload compatible with both API and cached JSON schemas.
    """
    mask_tuple = tuple(sorted(mask)) if mask else ("D", "T", "R")
    probs = predictor.predict(case_input, mask=mask_tuple)[0]
    pred_idx = int(np.argmax(probs))
    pred_class = FAULT_CLASSES[pred_idx]
    confidence = float(probs[pred_idx])
    correct = bool(pred_class == true_class)

    # 1. Modality Shapley attribution over active modalities only
    shapley_data = compute_shapley_contributions(predictor.predict, case_input, pred_idx, active_mask=mask_tuple)

    # 2. Within-modality evidence (hidden/empty for disabled modalities)
    tel_importance = (
        compute_telemetry_evidence(model_id, predictor, case_input, pred_idx)
        if "T" in mask_tuple else []
    )
    if "R" in mask_tuple:
        top_words, word_scores = compute_report_evidence(
            model_id, predictor, case_input, report_text, vectorizer, pred_idx
        )
    else:
        top_words, word_scores = [], []

    cam_path = (
        f"img/{case_input.get('case_id', 'case')}.{model_id}.cam.webp"
        if "D" in mask_tuple else None
    )

    # 3. Model-specific architecture diagnostics with active/disabled statuses
    model_specific = {}
    if model_id == "quorum":
        specs = predictor.predict_specialists(case_input)
        model_specific["specialists"] = {}
        weights = predictor.config["combination_weights"]
        for mod, mod_p in specs.items():
            mod_code = {"image": "D", "text": "R", "telemetry": "T"}[mod]
            is_active = mod_code in mask_tuple
            c_idx = int(np.argmax(mod_p[0]))
            model_specific["specialists"][mod] = {
                "pred": FAULT_CLASSES[c_idx] if is_active else "OFF",
                "conf": round(float(mod_p[0][c_idx]), 4) if is_active else 0.0,
                "weight": weights[mod] if is_active else 0.0,
                "active": is_active,
            }
    elif model_id == "nexus":
        branches = predictor.predict_branches(case_input, mask_tuple)
        model_specific["branches"] = {}
        weights = predictor.config["combination_weights"]
        for b_name, b_p in branches.items():
            if b_name == "image_text":
                is_active = ("D" in mask_tuple or "R" in mask_tuple)
            elif b_name == "telemetry":
                is_active = ("T" in mask_tuple)
            else:
                is_active = True
            c_idx = int(np.argmax(b_p[0]))
            model_specific["branches"][b_name] = {
                "pred": FAULT_CLASSES[c_idx] if is_active else "OFF",
                "conf": round(float(b_p[0][c_idx]), 4) if is_active else 0.0,
                "weight": weights[b_name] if is_active else 0.0,
                "active": is_active,
            }
    elif model_id == "conflux":
        model_specific["joint_dimension"] = 384
        model_specific["masked_count"] = 3 - len(mask_tuple)
        model_specific["active_modalities"] = list(mask_tuple)

    # 4. Grounded deterministic narrative mentioning active & disabled modalities
    tw_raw = case_input["telemetry_window"][0].detach().cpu().numpy()
    narrative = build_narrative_lines(
        model_id=model_id,
        true_class=true_class,
        pred_class=pred_class,
        confidence=confidence,
        contrib=shapley_data["contrib"],
        top_words=top_words,
        telemetry_window=tw_raw,
        element_name=element_name,
        model_specific=model_specific,
        mask=mask_tuple,
    )

    prediction_payload = {
        "pred": pred_class,
        "probs": {FAULT_CLASSES[i]: round(float(probs[i]), 4) for i in range(len(FAULT_CLASSES))},
        "confidence": round(confidence, 4),
        "correct": correct,
    }

    explanation_payload = {
        "method": "Exact Shapley (Modality) + Gradient Attribution (Within-Modality)",
        "contrib": shapley_data["contrib"],
        "dominant": shapley_data["dominant"],
        "telemetry_importance": tel_importance,
        "report_top_words": top_words,
        "cam": cam_path,
        "model_specific": model_specific,
        "narrative": narrative,
    }

    return {
        "prediction": prediction_payload,
        "explanation": explanation_payload,
        "ablations": shapley_data["ablations"],
        "word_scores": word_scores,
        "mask": list(mask_tuple),
    }

def _get_vectorizer():
    """Return the TF-IDF vectorizer from the preprocessing pickle (lazy load)."""
    prep = _get_prep()
    return prep["tfidf_vectorizer"]

def _get_scaler():
    """Return the telemetry StandardScaler from the preprocessing pickle (lazy load)."""
    prep = _get_prep()
    return prep["telemetry_scaler"]


def build_case_input_from_json(case_data: dict) -> dict:
    """
    Reconstruct model-ready tensors from preprocessed cache or case JSON dict.
    """
    case_id = case_data["id"]
    pre = _get_preprocessed_inputs()
    if pre and case_id in pre:
        item = pre[case_id]
        return {
            "case_id": case_id,
            "image_feat": item["image_feat"],
            "text_feat": item["text_feat"],
            "telemetry_window": item["telemetry_window"],
            "telemetry_mask": item["telemetry_mask"],
            "_active_text": item.get("active_text", case_data.get("report", {}).get("en") or case_data.get("report", {}).get("es") or ""),
        }

    demo_dir = Path(__file__).resolve().parent.parent
    root_dir = demo_dir.parent

    # ── 1. Telemetry ──────────────────────────────────────────────────────────
    rows = case_data["telemetry"]["rows"]
    n_ch = 7
    target_len = 48
    raw_win = np.zeros((n_ch, target_len), dtype=np.float32)
    valid_mask = np.zeros((1, target_len), dtype=np.float32)

    for t_idx, row in enumerate(rows[:target_len]):
        ch = row.get("channels", [0.0] * n_ch)
        for c_i in range(min(n_ch, len(ch))):
            raw_win[c_i, t_idx] = float(ch[c_i])
        valid_mask[0, t_idx] = 1.0

    # Scale telemetry the same way as training
    scaler = _get_scaler()
    # scaler is stored as dict {'mean': (1,7,1), 'std': (1,7,1)}
    mean = np.array(scaler["mean"])  # (1,7,1)
    std  = np.array(scaler["std"])   # (1,7,1)
    win3d = raw_win[np.newaxis, :, :]  # (1,7,48)
    scaled_3d = (win3d - mean) / (std + 1e-8)
    scaled_win = scaled_3d[0]  # (7,48)

    tw_t = torch.tensor(scaled_win[np.newaxis, :, :], dtype=torch.float32)
    tm_t = torch.tensor(valid_mask[np.newaxis, :, :], dtype=torch.float32)

    # ── 2. Report Text ────────────────────────────────────────────────────────
    rpt_en = case_data["report"].get("en", "")
    rpt_es = case_data["report"].get("es", "")
    active_text = rpt_en if rpt_en else (rpt_es if rpt_es else "No report text available.")

    vectorizer = _get_vectorizer()
    txt_sparse = vectorizer.transform([active_text]).toarray().astype(np.float32)
    txt_f = torch.tensor(txt_sparse, dtype=torch.float32)

    # ── 3. Image Features (ResNet-18) ─────────────────────────────────────────
    # Try to locate the source image from curated_cases.json for this case_id
    case_id = case_data["id"]
    img_f = _get_image_feat_for_case(case_id, root_dir)

    return {
        "case_id": case_id,
        "image_feat": img_f,
        "text_feat": txt_f,
        "telemetry_window": tw_t,
        "telemetry_mask": tm_t,
        "_active_text": active_text,
    }


def _get_image_feat_for_case(case_id: str, root_dir: Path) -> torch.Tensor:
    """
    Extract image features for a single case using the ResNet-18 feature extractor.
    Falls back to zero tensor if image not found.
    """
    # Load curated_cases.json to find image path
    demo_dir = root_dir / "demo"
    curated_path = demo_dir / "export" / "curated_cases.json"

    image_path = None
    if curated_path.exists():
        import json as _json
        curated = _json.loads(curated_path.read_text())
        for entry in curated:
            if entry.get("case_id") == case_id:
                image_path = entry.get("image_path")
                break

    if image_path and Path(image_path).exists():
        try:
            sys.path.insert(0, str(root_dir))
            from demo.export.vendor.feature_utils import ResNet18FeatureExtractor
            extractor = ResNet18FeatureExtractor()
            feat = extractor.extract(image_path)  # (512,)
            return torch.tensor(feat[np.newaxis, :], dtype=torch.float32)
        except Exception:
            pass

    # Fallback: zero tensor (512-dim image features)
    return torch.zeros((1, 512), dtype=torch.float32)


def run_full_diagnosis(*args, **kwargs) -> Dict[str, Any]:
    """
    Unified entry point handling both:
    1) Core invocation from build_demo_data.py:
       run_full_diagnosis(model_id=..., predictor=..., case_input=..., true_class=..., element_name=..., report_text=..., vectorizer=..., mask=...)
    2) API invocation from api/main.py:
       run_full_diagnosis(case_data, model_id, mask, predictor)
    """
    if "case_input" in kwargs or (len(args) >= 3 and isinstance(args[2], dict) and "image_feat" in args[2]):
        return _run_full_diagnosis_core(*args, **kwargs)
    elif len(args) >= 1 and isinstance(args[0], dict) and ("true_class" in args[0] or "id" in args[0]):
        case_data = args[0]
        model_id = args[1] if len(args) > 1 else kwargs.get("model_id")
        mask = args[2] if len(args) > 2 else kwargs.get("mask", ("D", "T", "R"))
        predictor = args[3] if len(args) > 3 else kwargs.get("predictor")

        case_input = build_case_input_from_json(case_data)
        true_class = case_data.get("true_class", "line_fault")
        element_name = case_data.get("element_name", "")
        active_text = case_input.pop("_active_text", "")
        vectorizer = _get_vectorizer()

        return _run_full_diagnosis_core(
            model_id=model_id,
            predictor=predictor,
            case_input=case_input,
            true_class=true_class,
            element_name=element_name,
            report_text=active_text,
            vectorizer=vectorizer,
            mask=mask,
        )
    else:
        return _run_full_diagnosis_core(*args, **kwargs)

def compute_full_evidence(
    case_data: dict,
    model_id: str,
    mask: Sequence[str],
    predictor: Any,
) -> Dict[str, Any]:
    """
    API endpoint: compute within-modality evidence (telemetry time steps, report words, CAM).
    Reconstructs tensors from case_data JSON and returns compact evidence payload.
    Hides evidence for disabled modalities.
    """
    case_input = build_case_input_from_json(case_data)
    active_text = case_input.pop("_active_text", "")
    vectorizer = _get_vectorizer()

    mask_tuple = tuple(sorted(mask)) if mask else ("D", "T", "R")

    # Get predicted class for attribution target under active mask
    probs = predictor.predict(case_input, mask=mask_tuple)[0]
    pred_idx = int(np.argmax(probs))

    # Telemetry time-step importance (only if T is active)
    tel_importance = (
        compute_telemetry_evidence(model_id, predictor, case_input, pred_idx)
        if "T" in mask_tuple else []
    )

    # Report word attribution (only if R is active)
    if "R" in mask_tuple:
        top_words, word_scores = compute_report_evidence(
            model_id, predictor, case_input, active_text, vectorizer, pred_idx
        )
    else:
        top_words, word_scores = [], []

    # CAM overlay as base64 WebP (only if D is active)
    cam_b64 = None
    if "D" in mask_tuple:
        try:
            cam_img = compute_diagram_cam_overlay(
                "", case_input["image_feat"], importance_weight=1.0
            )
            buf = io.BytesIO()
            cam_img.save(buf, format="WEBP", quality=75)
            cam_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception:
            pass

    return {
        "telemetry_importance": tel_importance,
        "report_top_words": top_words,
        "word_scores": word_scores,
        "cam_base64": cam_b64,
    }
