# PHASE 0 — RECON REPORT

**Date**: October 2026  
**Repository**: `Power_System_Fault_Diagnosis` (Group 13 - Multimodal Reasoning for Power-System Fault Diagnosis)  
**Branch**: `feat/demo-dashboard` (created from `main` at commit `87d90548`, tagged `pre-demo-backup`)

---

## 1. Repository Layout & Key Entry Points

- **Trained Model Code & Architectures**:
  - `early_fusion.py`: Single joint model (`EarlyFusionModel`) fusing diagram CNN features, text TF-IDF features, and telemetry TCN features via concatenation upstream of a unified classifier head.
  - `late_fusion.py`: Independent unimodal classifiers (`ImageOnly`, `TextOnly`, `TelemetryOnly`) combined via validation-tuned decision weights (`wi=0.40, wt=0.30, wl=0.30`).
  - `hybrid_fusion.py`: Two-stage architecture fusing image + text features at representation level (`ImageTextFusionModel`), combined with an independent telemetry TCN model at decision level with weights (`w_it=0.50, w_tel=0.50`).
- **Shared Utilities & Dataloaders**:
  - `dataset_utils.py`: Scans dataset for completeness (`scan_dataset()`), executes stratified split (`stratified_case_split()`), extracts frozen ResNet-18 image features (`get_image_features()`), fits TF-IDF text features (`TextFeaturizer`), processes 48-hour 7-channel hourly generation telemetry (`load_telemetry_windows()`), fits train-only telemetry scaler (`fit_telemetry_scaler()`), and defines TCN and MLP layers.
- **Evaluation & Benchmarking Scripts**:
  - `run_multi_seed_and_analysis.py`: Multi-seed evaluation (seeds 42, 13, 7), synthetic fallback auditing, ablation studies, and evaluation report generation.
- **Results & Checkpoints**:
  - `final_results/`: Contains evaluation metrics, confusion matrices, training history, and PyTorch model checkpoints (`early_fusion/early_fusion_model.pt`, `late_fusion/*.pt`, `hybrid_fusion/*.pt`).
  - `.first-level results/`: Archived earlier run on partial dataset (263 cases).

---

## 2. Model Architecture, Encoders, Preprocessing & Modality Masking

| Model ID | Architecture Class | Encoders Used | Fusion Mechanism | Input Preprocessing | Class Label Order | Missing/Masked Handling |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CONFLUX** (Early Fusion) | `EarlyFusionModel` | Diagram: ResNet-18 (512-d); Text: TF-IDF (128-d, english stop words, n-gram 1-2); Telemetry: 3-layer Dilated 1D TCN (128-d) | Concatenation of projections (`d_model=128 * 3 = 384`), followed by MLP (384 -> 96 -> 4) | Image: 224x224 RGB, ImageNet norm; Telemetry: 7 channels x 48 hours, mean/std scaled; Text: English report (fallback ES) | `['line_fault', 'breaker_fault', 'transformer_fault', 'busbar_fault']` | Inactive modality zeroed out at projection input or feature level |
| **QUORUM** (Late Fusion) | `ImageOnly`, `TextOnly`, `TelemetryOnly` | Three independent branches (ResNet-18 -> 128 -> 4, TF-IDF -> 128 -> 4, TCN -> 128 -> 4) | Decision-level weighted average of softmax probabilities: $0.40 \cdot P_{img} + 0.30 \cdot P_{txt} + 0.30 \cdot P_{tel}$ | Identical to Early Fusion | `['line_fault', 'breaker_fault', 'transformer_fault', 'busbar_fault']` | Inactive branch excluded from vote and weights re-normalized across active branches |
| **NEXUS** (Hybrid Fusion) | `ImageTextFusionModel`, `TelemetryOnly` | Image+Text feature fusion (ResNet-18 512-d + TF-IDF 128-d -> 256 -> 96 -> 4); Telemetry TCN (7-ch -> 128 -> 4) | Decision-level combination of fused Image+Text and Telemetry: $0.50 \cdot P_{it} + 0.50 \cdot P_{tel}$ | Identical to Early Fusion | `['line_fault', 'breaker_fault', 'transformer_fault', 'busbar_fault']` | Zeroing image or text within ImageText branch, or re-weighting branches if telemetry masked |

---

## 3. Saved Artifacts Inventory

All checkpoints exist, are healthy, and load cleanly:

| Artifact Path | Size (Bytes) | Model / Role | Timestamp |
| :--- | :--- | :--- | :--- |
| `final_results/early_fusion/early_fusion_model.pt` | 798,474 | CONFLUX (Early Fusion joint weights) | 2026-10-10 19:31 |
| `final_results/late_fusion/late_fusion_image_model.pt` | 318,336 | QUORUM Image Specialist | 2026-10-10 19:33 |
| `final_results/late_fusion/late_fusion_text_model.pt` | 121,716 | QUORUM Text Specialist | 2026-10-10 19:33 |
| `final_results/late_fusion/late_fusion_telemetry_model.pt` | 366,479 | QUORUM Telemetry Specialist | 2026-10-10 19:33 |
| `final_results/late_fusion/late_fusion_combination_weights.json` | 128 | QUORUM Decision Weights (`wi=0.4, wt=0.3, wl=0.3`) | 2026-10-10 19:33 |
| `final_results/hybrid_fusion/hybrid_image_text_model.pt` | 436,082 | NEXUS Image+Text Joint Branch | 2026-10-10 19:32 |
| `final_results/hybrid_fusion/hybrid_telemetry_model.pt` | 366,334 | NEXUS Telemetry Branch | 2026-10-10 19:32 |
| `final_results/hybrid_fusion/hybrid_fusion_combination_weights.json` | 192 | NEXUS Decision Weights (`w_it=0.5, w_tel=0.5`) | 2026-10-10 19:32 |

---

## 4. Train / Validation / Test Split Reproducibility

- Split implementation: `du.stratified_case_split(cases, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42)`.
- Verified exact counts:
  - Total complete cases: **1,048**
  - Train set: **733** cases
  - Validation set: **157** cases
  - Held-out test set: **158** cases
- Test set class distribution:
  - `line_fault`: 88 cases
  - `transformer_fault`: 27 cases
  - `breaker_fault`: 25 cases
  - `busbar_fault`: 18 cases
  - **Total**: 158 cases

---

## 5. Reported Test Metrics vs. Verified Reproducibility (HARD GATES)

| Model Name | Display ID | Metric | Reported (`final_results`) | Verified Reproduction | Delta | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Early Fusion** | `conflux` | Accuracy | **87.97%** (0.879747) | **87.97%** (0.879747) | 0.000% | **PASSED** |
| | | Macro-F1 | **82.20%** (0.822041) | **82.20%** (0.822041) | 0.000% | **PASSED** |
| **Late Fusion** | `quorum` | Accuracy | **83.54%** (0.835443) | **83.54%** (0.835443) | 0.000% | **PASSED** |
| | | Macro-F1 | **76.51%** (0.765136) | **76.51%** (0.765136) | 0.000% | **PASSED** |
| **Hybrid Fusion** | `nexus` | Accuracy | **82.28%** (0.822785) | **82.28%** (0.822785) | 0.000% | **PASSED** |
| | | Macro-F1 | **73.03%** (0.730288) | **73.03%** (0.730288) | 0.000% | **PASSED** |

- **HARD GATE A**: Checkpoints exist and reproduce reported numbers with 0.000% deviation.
- **HARD GATE B**: Exact test split reproduced with 100% fidelity.

---

## 6. Dataset Facts & Telemetry Specifications

- **Total cases in registry**: 1,113 scraped folders.
- **Eligible complete cases**: 1,048 cases.
- **Test set size**: 158 cases.
- **Real InfoTécnica diagram cases in test set**: **63 cases** have `image_is_real_infotecnica == True` or `diagram_source == 'official_pdf'`.
- **Telemetry Modality**:
  - Telemetry is **hourly generation power data (MWh)** across 7 generation types: `total`, `thermal`, `solar`, `hydro`, `wind`, `bess`, `other`.
  - **CRITICAL**: The telemetry is strictly generation data, **NOT** voltage or current. Units are MWh.
  - Window length: 48 hours centered around event window.
- **Report Text**:
  - Technical operator incident reports from CEN EAF (Estudio de Análisis de Falla). Available in English (`report_en.txt`) and original Spanish (`report_es.txt`).
  - Average word length: ~150-250 words per report.

---

## 7. Git & Environment Facts

- **Repo Size**: ~1.1 GB (due to cached xlsx files in Git LFS).
- **Git LFS Files**:
  - `cen_pipeline/_raw_cache/generation_by_plant_2016_2019.xlsx`
  - `cen_pipeline/_raw_cache/generation_by_plant_2020_2023.xlsx`
  - `cen_pipeline/_raw_cache/generation_by_plant_2024_2024.xlsx`
  - `cen_pipeline/_raw_cache/generation_by_plant_2025_2025.xlsx`
  - `cen_pipeline/_raw_cache/hourly_generation.xlsx`
- **Pre-existing Dirty Files**: `cen_pipeline/requirements.txt` (recorded in `demo/_recon/dirty_files.txt`, untouched).
- **Safety Tag & Branch**: `pre-demo-backup` created and pushed; active branch `feat/demo-dashboard`.
- **Manifest**: 1,652 tracked files hashed into `demo/_recon/tracked_files.sha256`.
- **Python Env**: Isolated `demo/.venv-export` created with system-site packages. Existing system packages verified unchanged.
