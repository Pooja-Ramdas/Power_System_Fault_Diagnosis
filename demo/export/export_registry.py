"""
Export and register the three trained multimodal fusion models:
- CONFLUX (Early Fusion)
- QUORUM (Late Fusion)
- NEXUS (Hybrid Fusion)
Writes to demo/artifacts/models/<model_id>/
"""

import os
import sys
import json
import shutil
import hashlib
import pickle
import numpy as np
import torch

# Ensure repository root is in python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT_DIR)

import dataset_utils as du
from demo.export.vendor.models import (
    EarlyFusionModel,
    ImageOnlyModel,
    TextOnlyModel,
    TelemetryOnlyModel,
    HybridImageTextModel,
)

ARTIFACTS_DIR = os.path.join(ROOT_DIR, "demo", "artifacts")
MODELS_DIR = os.path.join(ARTIFACTS_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# 1. Fit text featurizer & telemetry scaler on TRAIN ONLY (seed 42)
print("Scanning dataset to extract training preprocessing parameters...")
cases = du.scan_dataset()
train_cases, val_cases, test_cases = du.stratified_case_split(cases, seed=42)

tf = du.TextFeaturizer(use_pretrained_transformer=False, max_features=128)
tf.fit_transform(du.load_texts(train_cases))

tel_windows_all, tel_masks_all, n_channels = du.load_telemetry_windows(cases)
case_index = {c.case_id: i for i, c in enumerate(cases)}
tr_idx = [case_index[c.case_id] for c in train_cases]
tr_tw, tr_tm = tel_windows_all[tr_idx], tel_masks_all[tr_idx]
scaler = du.fit_telemetry_scaler(tr_tw, tr_tm)

# Save shared preprocessing bundle
shared_prep_path = os.path.join(ARTIFACTS_DIR, "preprocessing.pkl")
with open(shared_prep_path, "wb") as f:
    pickle.dump({
        "tfidf_vectorizer": tf.vectorizer,
        "telemetry_scaler": scaler,
        "fault_classes": du.FAULT_CLASSES,
        "telemetry_channels": du.TELEMETRY_CHANNELS,
    }, f)

# Also JSON-compatible scaler representation for transparency
shared_prep_json = os.path.join(ARTIFACTS_DIR, "preprocessing.json")
with open(shared_prep_json, "w", encoding="utf-8") as f:
    json.dump({
        "fault_classes": du.FAULT_CLASSES,
        "telemetry_channels": du.TELEMETRY_CHANNELS,
        "telemetry_scaler": {
            "mean": scaler["mean"].squeeze().tolist(),
            "std": scaler["std"].squeeze().tolist(),
        },
        "tfidf_vocabulary_size": len(tf.vectorizer.vocabulary_),
        "tfidf_ngram_range": list(tf.vectorizer.ngram_range),
    }, f, indent=2)

print("Preprocessing artifacts saved.")

# -------------------------------------------------------------
# 2. Export Model 1: CONFLUX (Early Fusion)
# -------------------------------------------------------------
conflux_dir = os.path.join(MODELS_DIR, "conflux")
os.makedirs(conflux_dir, exist_ok=True)
shutil.copyfile(
    os.path.join(ROOT_DIR, "final_results", "early_fusion", "early_fusion_model.pt"),
    os.path.join(conflux_dir, "model.pt")
)
shutil.copyfile(
    os.path.join(ROOT_DIR, "final_results", "early_fusion", "early_fusion_metrics.json"),
    os.path.join(conflux_dir, "metrics.json")
)
with open(os.path.join(conflux_dir, "config.json"), "w", encoding="utf-8") as f:
    json.dump({
        "model_id": "conflux",
        "display_name": "CONFLUX",
        "fusion_type": "EARLY FUSION",
        "tooltip": "Features from diagram, telemetry and report are merged into one joint representation at the input, before any decision is made.",
        "image_dim": 512,
        "text_dim": 128,
        "telemetry_channels": 7,
        "d_model": 128,
        "n_classes": 4,
        "class_order": du.FAULT_CLASSES,
        "decision_thresholds": None,
    }, f, indent=2)

with open(os.path.join(conflux_dir, "README.md"), "w", encoding="utf-8") as f:
    f.write("""# CONFLUX (Early Fusion)
Trained joint representation model fusing ResNet-18 diagram features, TF-IDF text features, and 1D-TCN telemetry features.
- Source Commit SHA: 87d90548
- Seed: 42
- Test Accuracy: 87.97%
- Test Macro-F1: 82.20%
""")

# -------------------------------------------------------------
# 3. Export Model 2: QUORUM (Late Fusion)
# -------------------------------------------------------------
quorum_dir = os.path.join(MODELS_DIR, "quorum")
os.makedirs(quorum_dir, exist_ok=True)

# Combine state_dicts into a unified dictionary or keep clean sub-models
shutil.copyfile(
    os.path.join(ROOT_DIR, "final_results", "late_fusion", "late_fusion_image_model.pt"),
    os.path.join(quorum_dir, "image_model.pt")
)
shutil.copyfile(
    os.path.join(ROOT_DIR, "final_results", "late_fusion", "late_fusion_text_model.pt"),
    os.path.join(quorum_dir, "text_model.pt")
)
shutil.copyfile(
    os.path.join(ROOT_DIR, "final_results", "late_fusion", "late_fusion_telemetry_model.pt"),
    os.path.join(quorum_dir, "telemetry_model.pt")
)

# Also create unified quorum_model.pt for single-file packaging
quorum_state = {
    "image_model": torch.load(os.path.join(quorum_dir, "image_model.pt"), map_location="cpu"),
    "text_model": torch.load(os.path.join(quorum_dir, "text_model.pt"), map_location="cpu"),
    "telemetry_model": torch.load(os.path.join(quorum_dir, "telemetry_model.pt"), map_location="cpu"),
}
torch.save(quorum_state, os.path.join(quorum_dir, "model.pt"))

with open(os.path.join(ROOT_DIR, "final_results", "late_fusion", "late_fusion_combination_weights.json")) as f:
    quorum_weights = json.load(f)

shutil.copyfile(
    os.path.join(ROOT_DIR, "final_results", "late_fusion", "late_fusion_combined_metrics.json"),
    os.path.join(quorum_dir, "metrics.json")
)
with open(os.path.join(quorum_dir, "config.json"), "w", encoding="utf-8") as f:
    json.dump({
        "model_id": "quorum",
        "display_name": "QUORUM",
        "fusion_type": "LATE FUSION",
        "tooltip": "Each modality has its own specialist model that reaches an independent verdict; a learned vote combines them.",
        "image_dim": 512,
        "text_dim": 128,
        "telemetry_channels": 7,
        "d_model": 128,
        "n_classes": 4,
        "class_order": du.FAULT_CLASSES,
        "combination_weights": {
            "image": quorum_weights["image"],
            "text": quorum_weights["text"],
            "telemetry": quorum_weights["telemetry"],
        },
    }, f, indent=2)

with open(os.path.join(quorum_dir, "README.md"), "w", encoding="utf-8") as f:
    f.write(f"""# QUORUM (Late Fusion)
Independent unimodal specialist branches combined via learned validation decision weights:
Image: {quorum_weights['image']:.2f}, Text: {quorum_weights['text']:.2f}, Telemetry: {quorum_weights['telemetry']:.2f}.
- Source Commit SHA: 87d90548
- Seed: 42
- Test Accuracy: 83.54%
- Test Macro-F1: 76.51%
""")

# -------------------------------------------------------------
# 4. Export Model 3: NEXUS (Hybrid Fusion)
# -------------------------------------------------------------
nexus_dir = os.path.join(MODELS_DIR, "nexus")
os.makedirs(nexus_dir, exist_ok=True)
shutil.copyfile(
    os.path.join(ROOT_DIR, "final_results", "hybrid_fusion", "hybrid_image_text_model.pt"),
    os.path.join(nexus_dir, "image_text_model.pt")
)
shutil.copyfile(
    os.path.join(ROOT_DIR, "final_results", "hybrid_fusion", "hybrid_telemetry_model.pt"),
    os.path.join(nexus_dir, "telemetry_model.pt")
)

nexus_state = {
    "image_text_model": torch.load(os.path.join(nexus_dir, "image_text_model.pt"), map_location="cpu"),
    "telemetry_model": torch.load(os.path.join(nexus_dir, "telemetry_model.pt"), map_location="cpu"),
}
torch.save(nexus_state, os.path.join(nexus_dir, "model.pt"))

with open(os.path.join(ROOT_DIR, "final_results", "hybrid_fusion", "hybrid_fusion_combination_weights.json")) as f:
    nexus_weights = json.load(f)

shutil.copyfile(
    os.path.join(ROOT_DIR, "final_results", "hybrid_fusion", "hybrid_fusion_combined_metrics.json"),
    os.path.join(nexus_dir, "metrics.json")
)
with open(os.path.join(nexus_dir, "config.json"), "w", encoding="utf-8") as f:
    json.dump({
        "model_id": "nexus",
        "display_name": "NEXUS",
        "fusion_type": "HYBRID FUSION",
        "tooltip": "Encoders exchange evidence through joint representation mid-network, then a decision-level stage combines with telemetry.",
        "image_dim": 512,
        "text_dim": 128,
        "telemetry_channels": 7,
        "d_model": 128,
        "n_classes": 4,
        "class_order": du.FAULT_CLASSES,
        "combination_weights": {
            "image_text": nexus_weights["image_text"],
            "telemetry": nexus_weights["telemetry"],
        },
    }, f, indent=2)

with open(os.path.join(nexus_dir, "README.md"), "w", encoding="utf-8") as f:
    f.write(f"""# NEXUS (Hybrid Fusion)
Feature-level fusion for Image + Text, combined at decision level with independent Telemetry branch.
Weights: Image+Text = {nexus_weights['image_text']:.2f}, Telemetry = {nexus_weights['telemetry']:.2f}.
- Source Commit SHA: 87d90548
- Seed: 42
- Test Accuracy: 82.28%
- Test Macro-F1: 73.03%
""")

# -------------------------------------------------------------
# 5. Build Manifest
# -------------------------------------------------------------
manifest = {
    "date_utc": "2026-10-25T00:00:00Z",
    "source_git_sha": "87d90548",
    "seed": 42,
    "torch_version": torch.__version__,
    "artifacts": {}
}

for root, _, files in os.walk(ARTIFACTS_DIR):
    for fname in sorted(files):
        fpath = os.path.join(root, fname)
        relpath = os.path.relpath(fpath, ARTIFACTS_DIR).replace("\\", "/")
        h = hashlib.sha256()
        with open(fpath, "rb") as fp:
            while chunk := fp.read(65536):
                h.update(chunk)
        size_bytes = os.path.getsize(fpath)
        manifest["artifacts"][relpath] = {
            "sha256": h.hexdigest(),
            "size_bytes": size_bytes,
        }

with open(os.path.join(ARTIFACTS_DIR, "manifest.json"), "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print(f"Model registry exported successfully to {MODELS_DIR} with manifest.")
