"""
run_multi_seed_and_analysis.py
Performs:
1. Multi-seed evaluation for Early, Hybrid, and Late fusion (reporting mean +/- std and best run)
2. Synthetic fallback audit (counts, percentages, per-class breakdown, charts)
3. Modality contribution & leave-one-out ablation analysis
4. Real vs synthetic-fallback image performance evaluation
5. Generates comprehensive report files inside final_results/
"""

import json
import os
import collections
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
import torch
from torch.utils.data import DataLoader, TensorDataset

import dataset_utils as du
import early_fusion
import late_fusion
import hybrid_fusion

OUT_DIR = "final_results"
os.makedirs(OUT_DIR, exist_ok=True)

# ==============================================================================
# PART 1: SYNTHETIC FALLBACK AUDIT
# ==============================================================================
print("=== PART 1: AUDITING SYNTHETIC FALLBACK ===")
all_case_dirs = sorted([d for d in os.listdir("cen_dataset") if d.startswith("CASE-")])
print(f"Total CASE folders examined: {len(all_case_dirs)}")

audit_records = []
for c_name in all_case_dirs:
    c_path = os.path.join("cen_dataset", c_name)
    m_path = os.path.join(c_path, "meta.json")
    if not os.path.exists(m_path):
        continue
    with open(m_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    
    diag_src = meta.get("diagram_source", "unknown")
    is_real = meta.get("image_is_real_infotecnica", False)
    fault_label = meta.get("fault_label", meta.get("label", "unknown"))
    
    # Classify category
    # official_pdf: actual CEN PDF diagram
    # official_archive_pending_conversion: official archive (.rar/.dwg) rendered via topology
    # synthetic_fallback: pure synthetic topology diagram
    if diag_src == "official_pdf" or is_real:
        data_origin = "Actual CEN Diagram (PDF)"
    elif diag_src == "official_archive_pending_conversion":
        data_origin = "Official Archive (Pending Conversion / Topology Rendered)"
    else:
        data_origin = "Pure Synthetic Fallback"
    
    audit_records.append({
        "case_id": c_name,
        "fault_label": fault_label,
        "diagram_source": diag_src,
        "is_real_infotecnica": is_real,
        "data_origin": data_origin,
    })

audit_df = pd.DataFrame(audit_records)
total_cases = len(audit_df)

# Summary table overall
origin_summary = audit_df["data_origin"].value_counts().reset_index()
origin_summary.columns = ["Data Origin", "Count"]
origin_summary["Percentage"] = (origin_summary["Count"] / total_cases) * 100

# Summary by fault class
class_origin_cross = pd.crosstab(
    audit_df["fault_label"],
    audit_df["data_origin"],
    margins=True,
    margins_name="Total"
)

class_origin_pct = pd.crosstab(
    audit_df["fault_label"],
    audit_df["data_origin"],
    normalize="index"
) * 100

# Save audit CSVs and JSON
origin_summary.to_csv(os.path.join(OUT_DIR, "synthetic_fallback_summary.csv"), index=False)
class_origin_cross.to_csv(os.path.join(OUT_DIR, "synthetic_fallback_by_class_counts.csv"))
class_origin_pct.to_csv(os.path.join(OUT_DIR, "synthetic_fallback_by_class_percentages.csv"))

audit_summary_dict = {
    "total_cases": total_cases,
    "overall": origin_summary.to_dict(orient="records"),
    "by_class_counts": class_origin_cross.to_dict(),
    "by_class_percentages": class_origin_pct.to_dict(),
}
with open(os.path.join(OUT_DIR, "synthetic_fallback_audit.json"), "w", encoding="utf-8") as f:
    json.dump(audit_summary_dict, f, indent=2)

# Chart 1: Overall Breakdown
plt.figure(figsize=(8, 5))
colors = ["#2b5c8f", "#d95f02", "#7570b3"]
bars = plt.bar(origin_summary["Data Origin"], origin_summary["Count"], color=colors, width=0.55)
plt.title("Image Modality Data Origin Audit (N=1113 Cases)", fontsize=13, weight="bold")
plt.ylabel("Number of Cases", fontsize=11)
plt.xticks(rotation=15, ha="right", fontsize=10)
for bar in bars:
    yval = bar.get_height()
    pct = (yval / total_cases) * 100
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 15, f"{yval}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=10, weight="bold")
plt.ylim(0, max(origin_summary["Count"]) * 1.25)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_synthetic_fallback_distribution.png"), dpi=150)
plt.close()

# Chart 2: Stacked Bar by Fault Class (excluding 'Total')
cross_no_total = class_origin_cross.drop("Total", errors="ignore")
plt.figure(figsize=(10, 6))
cross_no_total.plot(kind="bar", stacked=True, color=colors, figsize=(10, 6), edgecolor="black", linewidth=0.5)
plt.title("Image Data Origin Distribution Across Fault Classes", fontsize=13, weight="bold")
plt.xlabel("Fault Class", fontsize=11)
plt.ylabel("Case Count", fontsize=11)
plt.xticks(rotation=30, ha="right")
plt.legend(title="Image Origin", loc="upper right")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_synthetic_fallback_by_class.png"), dpi=150)
plt.close()

print(f"Synthetic fallback audit saved: {total_cases} cases audited.")


# ==============================================================================
# PART 2: MULTI-SEED EXPERIMENTS (Seeds 42, 13, 7)
# ==============================================================================
print("\n=== PART 2: MULTI-SEED RUNS & REPRODUCIBILITY ===")
seeds = [42, 13, 7]
multi_seed_results = {
    "early_fusion": [],
    "late_fusion": [],
    "hybrid_fusion": [],
}

cases = du.scan_dataset()
case_index = {c.case_id: i for i, c in enumerate(cases)}
img_feats_all = du.get_image_features(cases)
tel_windows_all, tel_masks_all, n_channels = du.load_telemetry_windows(cases)
class_to_idx = {c: i for i, c in enumerate(du.FAULT_CLASSES)}

for seed in seeds:
    print(f"\n--- Running Multi-Seed Evaluation with Seed = {seed} ---")
    du.set_seed(seed)
    train_cases, val_cases, test_cases = du.stratified_case_split(cases, seed=seed)
    weights = du.class_weights(train_cases)
    
    # Text
    tf = du.TextFeaturizer(use_pretrained_transformer=False)
    tr_txt = tf.fit_transform(du.load_texts(train_cases))
    va_txt = tf.transform(du.load_texts(val_cases))
    te_txt = tf.transform(du.load_texts(test_cases))
    
    # Telemetry
    def tel_sub(sub):
        idx = [case_index[c.case_id] for c in sub]
        return tel_windows_all[idx], tel_masks_all[idx]
    tr_tw, tr_tm = tel_sub(train_cases)
    va_tw, va_tm = tel_sub(val_cases)
    te_tw, te_tm = tel_sub(test_cases)
    scaler = du.fit_telemetry_scaler(tr_tw, tr_tm)
    tr_tw = du.apply_telemetry_scaler(tr_tw, scaler)
    va_tw = du.apply_telemetry_scaler(va_tw, scaler)
    te_tw = du.apply_telemetry_scaler(te_tw, scaler)
    
    # Images
    def img_sub(sub):
        return img_feats_all[[case_index[c.case_id] for c in sub]]
    tr_img = img_sub(train_cases)
    va_img = img_sub(val_cases)
    te_img = img_sub(test_cases)
    
    def y_t(sub):
        return torch.tensor([class_to_idx[c.label] for c in sub], dtype=torch.long)
    tr_y, va_y, te_y = y_t(train_cases), y_t(val_cases), y_t(test_cases)
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))
    
    # --- Early Fusion ---
    train_loader = DataLoader(
        TensorDataset(torch.tensor(tr_img, dtype=torch.float32),
                      torch.tensor(tr_txt, dtype=torch.float32),
                      torch.tensor(tr_tw, dtype=torch.float32),
                      torch.tensor(tr_tm, dtype=torch.float32), tr_y),
        batch_size=16, shuffle=True)
    val_loader = DataLoader(
        TensorDataset(torch.tensor(va_img, dtype=torch.float32),
                      torch.tensor(va_txt, dtype=torch.float32),
                      torch.tensor(va_tw, dtype=torch.float32),
                      torch.tensor(va_tm, dtype=torch.float32), va_y),
        batch_size=16)
    test_loader = DataLoader(
        TensorDataset(torch.tensor(te_img, dtype=torch.float32),
                      torch.tensor(te_txt, dtype=torch.float32),
                      torch.tensor(te_tw, dtype=torch.float32),
                      torch.tensor(te_tm, dtype=torch.float32), te_y),
        batch_size=16)
    
    def ef_batch(loader):
        def _g():
            for im, tx, tw, tm, y in loader:
                yield {"image_feat": im, "text_feat": tx, "telemetry_window": tw, "telemetry_mask": tm}, y
        return _g
    
    ef_model = early_fusion.build_early_fusion_model(
        image_dim=tr_img.shape[1], text_dim=tr_txt.shape[1],
        telemetry_channels=n_channels, n_classes=len(du.FAULT_CLASSES))
    ef_opt = torch.optim.Adam(ef_model.parameters(), lr=1e-3, weight_decay=1e-4)
    du.train_with_early_stopping(ef_model, ef_opt, loss_fn, ef_batch(train_loader), ef_batch(val_loader),
                                 max_epochs=60, patience=10, verbose_name=f"ef_seed{seed}")
    ef_model.eval()
    all_logits = []
    with torch.no_grad():
        for inputs, _ in ef_batch(test_loader)():
            all_logits.append(ef_model(**inputs))
    ef_probs = torch.softmax(torch.cat(all_logits, dim=0), dim=1).numpy()
    ef_preds = ef_probs.argmax(axis=1)
    te_true = te_y.tolist()
    
    ef_acc = accuracy_score(te_true, ef_preds)
    ef_f1 = f1_score(te_true, ef_preds, average="macro", zero_division=0)
    ef_wf1 = f1_score(te_true, ef_preds, average="weighted", zero_division=0)
    multi_seed_results["early_fusion"].append({
        "seed": seed, "accuracy": ef_acc, "macro_f1": ef_f1, "weighted_f1": ef_wf1
    })
    print(f"Early Fusion Seed {seed}: Acc={ef_acc:.4f}, Macro-F1={ef_f1:.4f}")
    
    # --- Late Fusion ---
    def train_unimodal(name, model, tr_x, va_x, te_x, in_keys):
        tr_l = DataLoader(TensorDataset(*tr_x, tr_y), batch_size=16, shuffle=True)
        va_l = DataLoader(TensorDataset(*va_x, va_y), batch_size=16)
        te_l = DataLoader(TensorDataset(*te_x, te_y), batch_size=16)
        def bg(l):
            def _g():
                for *xs, y in l:
                    yield dict(zip(in_keys, xs)), y
            return _g
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        du.train_with_early_stopping(model, opt, loss_fn, bg(tr_l), bg(va_l), max_epochs=60, patience=10, verbose_name=f"lf_{name}_{seed}")
        model.eval()
        def get_p(l):
            logs = []
            with torch.no_grad():
                for inp, _ in bg(l)():
                    logs.append(model(**inp))
            return torch.softmax(torch.cat(logs, dim=0), dim=1).numpy()
        return get_p(va_l), get_p(te_l)
    
    img_m = late_fusion.build_unimodal_classifier("image", image_dim=tr_img.shape[1], n_classes=len(du.FAULT_CLASSES))
    val_p_img, te_p_img = train_unimodal("img", img_m, (torch.tensor(tr_img, dtype=torch.float32),),
                                         (torch.tensor(va_img, dtype=torch.float32),),
                                         (torch.tensor(te_img, dtype=torch.float32),), ["image_feat"])
    
    txt_m = late_fusion.build_unimodal_classifier("text", text_dim=tr_txt.shape[1], n_classes=len(du.FAULT_CLASSES))
    val_p_txt, te_p_txt = train_unimodal("txt", txt_m, (torch.tensor(tr_txt, dtype=torch.float32),),
                                         (torch.tensor(va_txt, dtype=torch.float32),),
                                         (torch.tensor(te_txt, dtype=torch.float32),), ["text_feat"])
    
    tel_m = late_fusion.build_unimodal_classifier("telemetry", telemetry_channels=n_channels, n_classes=len(du.FAULT_CLASSES))
    val_p_tel, te_p_tel = train_unimodal("tel", tel_m,
                                         (torch.tensor(tr_tw, dtype=torch.float32), torch.tensor(tr_tm, dtype=torch.float32)),
                                         (torch.tensor(va_tw, dtype=torch.float32), torch.tensor(va_tm, dtype=torch.float32)),
                                         (torch.tensor(te_tw, dtype=torch.float32), torch.tensor(te_tm, dtype=torch.float32)),
                                         ["telemetry_window", "telemetry_mask"])
    
    best_lf_w, _ = late_fusion._search_best_weights([val_p_img, val_p_txt, val_p_tel], va_y.tolist())
    w_i, w_t, w_l = best_lf_w
    comb_lf_p = w_i * te_p_img + w_t * te_p_txt + w_l * te_p_tel
    lf_preds = comb_lf_p.argmax(axis=1)
    lf_acc = accuracy_score(te_true, lf_preds)
    lf_f1 = f1_score(te_true, lf_preds, average="macro", zero_division=0)
    lf_wf1 = f1_score(te_true, lf_preds, average="weighted", zero_division=0)
    multi_seed_results["late_fusion"].append({
        "seed": seed, "accuracy": lf_acc, "macro_f1": lf_f1, "weighted_f1": lf_wf1, "weights": [w_i, w_t, w_l]
    })
    print(f"Late Fusion Seed {seed}: Acc={lf_acc:.4f}, Macro-F1={lf_f1:.4f}, weights=({w_i:.2f}, {w_t:.2f}, {w_l:.2f})")
    
    # --- Hybrid Fusion ---
    it_model = hybrid_fusion.build_hybrid_image_text_model(tr_img.shape[1], tr_txt.shape[1], len(du.FAULT_CLASSES))
    tr_it_l = DataLoader(TensorDataset(torch.tensor(tr_img, dtype=torch.float32), torch.tensor(tr_txt, dtype=torch.float32), tr_y), batch_size=16, shuffle=True)
    va_it_l = DataLoader(TensorDataset(torch.tensor(va_img, dtype=torch.float32), torch.tensor(va_txt, dtype=torch.float32), va_y), batch_size=16)
    te_it_l = DataLoader(TensorDataset(torch.tensor(te_img, dtype=torch.float32), torch.tensor(te_txt, dtype=torch.float32), te_y), batch_size=16)
    def it_bg(l):
        def _g():
            for im, tx, y in l:
                yield {"image_feat": im, "text_feat": tx}, y
        return _g
    it_opt = torch.optim.Adam(it_model.parameters(), lr=1e-3, weight_decay=1e-4)
    du.train_with_early_stopping(it_model, it_opt, loss_fn, it_bg(tr_it_l), it_bg(va_it_l), max_epochs=60, patience=10, verbose_name=f"hf_it_{seed}")
    it_model.eval()
    def get_it_p(l):
        logs = []
        with torch.no_grad():
            for inp, _ in it_bg(l)():
                logs.append(it_model(**inp))
        return torch.softmax(torch.cat(logs, dim=0), dim=1).numpy()
    val_p_it = get_it_p(va_it_l)
    te_p_it = get_it_p(te_it_l)
    
    best_hf_w, _ = hybrid_fusion._search_best_weights([val_p_it, val_p_tel], va_y.tolist())
    w_it, w_tel = best_hf_w
    comb_hf_p = w_it * te_p_it + w_tel * te_p_tel
    hf_preds = comb_hf_p.argmax(axis=1)
    hf_acc = accuracy_score(te_true, hf_preds)
    hf_f1 = f1_score(te_true, hf_preds, average="macro", zero_division=0)
    hf_wf1 = f1_score(te_true, hf_preds, average="weighted", zero_division=0)
    multi_seed_results["hybrid_fusion"].append({
        "seed": seed, "accuracy": hf_acc, "macro_f1": hf_f1, "weighted_f1": hf_wf1, "weights": [w_it, w_tel]
    })
    print(f"Hybrid Fusion Seed {seed}: Acc={hf_acc:.4f}, Macro-F1={hf_f1:.4f}, weights=({w_it:.2f}, {w_tel:.2f})")

# Compute mean +/- std
multi_seed_stats = {}
for m_name, runs in multi_seed_results.items():
    accs = [r["accuracy"] for r in runs]
    f1s = [r["macro_f1"] for r in runs]
    wf1s = [r["weighted_f1"] for r in runs]
    best_idx = int(np.argmax(f1s))
    multi_seed_stats[m_name] = {
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "macro_f1_mean": float(np.mean(f1s)),
        "macro_f1_std": float(np.std(f1s)),
        "weighted_f1_mean": float(np.mean(wf1s)),
        "weighted_f1_std": float(np.std(wf1s)),
        "best_run": runs[best_idx],
        "all_runs": runs,
    }

with open(os.path.join(OUT_DIR, "multi_seed_evaluation_summary.json"), "w", encoding="utf-8") as f:
    json.dump(multi_seed_stats, f, indent=2)

print("\n--- MULTI-SEED SUMMARY STATS ---")
for m_name, st in multi_seed_stats.items():
    print(f"{m_name:<15}: Acc = {st['accuracy_mean']:.4f} +/- {st['accuracy_std']:.4f} | "
          f"Macro-F1 = {st['macro_f1_mean']:.4f} +/- {st['macro_f1_std']:.4f} | "
          f"Best Seed: {st['best_run']['seed']} (Macro-F1: {st['best_run']['macro_f1']:.4f})")


# ==============================================================================
# PART 3: MODALITY CONTRIBUTION & LEAVE-ONE-OUT ABLATIONS (Seed 42)
# ==============================================================================
print("\n=== PART 3: MODALITY CONTRIBUTION & ABLATION STUDY ===")
# Evaluated on held-out test set (seed 42, N=158)
# We test:
# 1. Image-only
# 2. Text-only
# 3. Telemetry-only
# 4. Image + Text (Leave-out Telemetry)
# 5. Image + Telemetry (Leave-out Text)
# 6. Text + Telemetry (Leave-out Image)
# 7. Full Multimodal (Early Fusion)

du.set_seed(42)
train_cases, val_cases, test_cases = du.stratified_case_split(cases, seed=42)
weights = du.class_weights(train_cases)
loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))

tf = du.TextFeaturizer(use_pretrained_transformer=False)
tr_txt = tf.fit_transform(du.load_texts(train_cases))
va_txt = tf.transform(du.load_texts(val_cases))
te_txt = tf.transform(du.load_texts(test_cases))

tr_tw, tr_tm = tel_sub(train_cases)
va_tw, va_tm = tel_sub(val_cases)
te_tw, te_tm = tel_sub(test_cases)
scaler = du.fit_telemetry_scaler(tr_tw, tr_tm)
tr_tw = du.apply_telemetry_scaler(tr_tw, scaler)
va_tw = du.apply_telemetry_scaler(va_tw, scaler)
te_tw = du.apply_telemetry_scaler(te_tw, scaler)

tr_img = img_sub(train_cases)
va_img = img_sub(val_cases)
te_img = img_sub(test_cases)

tr_y, va_y, te_y = y_t(train_cases), y_t(val_cases), y_t(test_cases)
te_true = te_y.tolist()

# Train Image+Telemetry joint branch
class ImageTelemetryModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.img_proj = du.ProjectionHead(tr_img.shape[1], 128)
        self.tel_enc = du.build_tcn(n_channels, 128)
        self.head = du.build_classifier_head(256, len(du.FAULT_CLASSES), hidden=96)
    def forward(self, img_feat, tel_w, tel_m):
        e1 = self.img_proj(img_feat)
        e2 = self.tel_enc(tel_w, tel_m)
        return self.head(torch.cat([e1, e2], dim=1))

it_tel_m = ImageTelemetryModel()
it_tr_l = DataLoader(TensorDataset(torch.tensor(tr_img, dtype=torch.float32), torch.tensor(tr_tw, dtype=torch.float32), torch.tensor(tr_tm, dtype=torch.float32), tr_y), batch_size=16, shuffle=True)
it_va_l = DataLoader(TensorDataset(torch.tensor(va_img, dtype=torch.float32), torch.tensor(va_tw, dtype=torch.float32), torch.tensor(va_tm, dtype=torch.float32), va_y), batch_size=16)
it_te_l = DataLoader(TensorDataset(torch.tensor(te_img, dtype=torch.float32), torch.tensor(te_tw, dtype=torch.float32), torch.tensor(te_tm, dtype=torch.float32), te_y), batch_size=16)
def it_bg_gen(l):
    def _g():
        for im, tw, tm, y in l:
            yield {"img_feat": im, "tel_w": tw, "tel_m": tm}, y
    return _g
it_opt = torch.optim.Adam(it_tel_m.parameters(), lr=1e-3, weight_decay=1e-4)
du.train_with_early_stopping(it_tel_m, it_opt, loss_fn, it_bg_gen(it_tr_l), it_bg_gen(it_va_l), max_epochs=60, patience=10, verbose_name="ablation_img_tel")
it_tel_m.eval()
it_logs = []
with torch.no_grad():
    for inp, _ in it_bg_gen(it_te_l)():
        it_logs.append(it_tel_m(**inp))
img_tel_preds = torch.cat(it_logs, dim=0).argmax(dim=1).cpu().numpy()

# Train Text+Telemetry joint branch
class TextTelemetryModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.txt_proj = du.ProjectionHead(tr_txt.shape[1], 128)
        self.tel_enc = du.build_tcn(n_channels, 128)
        self.head = du.build_classifier_head(256, len(du.FAULT_CLASSES), hidden=96)
    def forward(self, txt_feat, tel_w, tel_m):
        e1 = self.txt_proj(txt_feat)
        e2 = self.tel_enc(tel_w, tel_m)
        return self.head(torch.cat([e1, e2], dim=1))

tt_m = TextTelemetryModel()
tt_tr_l = DataLoader(TensorDataset(torch.tensor(tr_txt, dtype=torch.float32), torch.tensor(tr_tw, dtype=torch.float32), torch.tensor(tr_tm, dtype=torch.float32), tr_y), batch_size=16, shuffle=True)
tt_va_l = DataLoader(TensorDataset(torch.tensor(va_txt, dtype=torch.float32), torch.tensor(va_tw, dtype=torch.float32), torch.tensor(va_tm, dtype=torch.float32), va_y), batch_size=16)
tt_te_l = DataLoader(TensorDataset(torch.tensor(te_txt, dtype=torch.float32), torch.tensor(te_tw, dtype=torch.float32), torch.tensor(te_tm, dtype=torch.float32), te_y), batch_size=16)
def tt_bg_gen(l):
    def _g():
        for tx, tw, tm, y in l:
            yield {"txt_feat": tx, "tel_w": tw, "tel_m": tm}, y
    return _g
tt_opt = torch.optim.Adam(tt_m.parameters(), lr=1e-3, weight_decay=1e-4)
du.train_with_early_stopping(tt_m, tt_opt, loss_fn, tt_bg_gen(tt_tr_l), tt_bg_gen(tt_va_l), max_epochs=60, patience=10, verbose_name="ablation_txt_tel")
tt_m.eval()
tt_logs = []
with torch.no_grad():
    for inp, _ in tt_bg_gen(tt_te_l)():
        tt_logs.append(tt_m(**inp))
txt_tel_preds = torch.cat(tt_logs, dim=0).argmax(dim=1).cpu().numpy()

# Load individual unimodal and fusion results from final_results/
with open(os.path.join(OUT_DIR, "late_fusion", "late_fusion_image_only_metrics.json")) as f:
    lf_img = json.load(f)
with open(os.path.join(OUT_DIR, "late_fusion", "late_fusion_text_only_metrics.json")) as f:
    lf_txt = json.load(f)
with open(os.path.join(OUT_DIR, "late_fusion", "late_fusion_telemetry_only_metrics.json")) as f:
    lf_tel = json.load(f)
with open(os.path.join(OUT_DIR, "hybrid_fusion", "hybrid_fusion_image_text_only_metrics.json")) as f:
    hf_it = json.load(f)
with open(os.path.join(OUT_DIR, "early_fusion", "early_fusion_metrics.json")) as f:
    ef_full = json.load(f)
with open(os.path.join(OUT_DIR, "late_fusion", "late_fusion_combined_metrics.json")) as f:
    lf_comb = json.load(f)
with open(os.path.join(OUT_DIR, "hybrid_fusion", "hybrid_fusion_combined_metrics.json")) as f:
    hf_comb = json.load(f)

img_tel_acc = accuracy_score(te_true, img_tel_preds)
img_tel_f1 = f1_score(te_true, img_tel_preds, average="macro", zero_division=0)
img_tel_wf1 = f1_score(te_true, img_tel_preds, average="weighted", zero_division=0)

txt_tel_acc = accuracy_score(te_true, txt_tel_preds)
txt_tel_f1 = f1_score(te_true, txt_tel_preds, average="macro", zero_division=0)
txt_tel_wf1 = f1_score(te_true, txt_tel_preds, average="weighted", zero_division=0)

ablation_table = [
    {"Configuration": "Telemetry Only (1 modality)", "Modalities": "Telemetry", "Accuracy": lf_tel["accuracy"], "Macro-F1": lf_tel["macro_f1"], "Weighted-F1": lf_tel["weighted_f1"], "Type": "Unimodal"},
    {"Configuration": "Image Only (1 modality)", "Modalities": "Image", "Accuracy": lf_img["accuracy"], "Macro-F1": lf_img["macro_f1"], "Weighted-F1": lf_img["weighted_f1"], "Type": "Unimodal"},
    {"Configuration": "Text Only (1 modality)", "Modalities": "Text", "Accuracy": lf_txt["accuracy"], "Macro-F1": lf_txt["macro_f1"], "Weighted-F1": lf_txt["weighted_f1"], "Type": "Unimodal"},
    {"Configuration": "Image + Telemetry (Leave out Text)", "Modalities": "Image + Telemetry", "Accuracy": img_tel_acc, "Macro-F1": img_tel_f1, "Weighted-F1": img_tel_wf1, "Type": "Bimodal"},
    {"Configuration": "Text + Telemetry (Leave out Image)", "Modalities": "Text + Telemetry", "Accuracy": txt_tel_acc, "Macro-F1": txt_tel_f1, "Weighted-F1": txt_tel_wf1, "Type": "Bimodal"},
    {"Configuration": "Image + Text (Leave out Telemetry)", "Modalities": "Image + Text", "Accuracy": hf_it["accuracy"], "Macro-F1": hf_it["macro_f1"], "Weighted-F1": hf_it["weighted_f1"], "Type": "Bimodal"},
    {"Configuration": "Early Fusion (All 3 Modalities - Feature Level)", "Modalities": "Image + Text + Telemetry", "Accuracy": ef_full["accuracy"], "Macro-F1": ef_full["macro_f1"], "Weighted-F1": ef_full["weighted_f1"], "Type": "Trimodal Full"},
    {"Configuration": "Hybrid Fusion (All 3 Modalities - Image+Text + Tel)", "Modalities": "Image + Text + Telemetry", "Accuracy": hf_comb["accuracy"], "Macro-F1": hf_comb["macro_f1"], "Weighted-F1": hf_comb["weighted_f1"], "Type": "Trimodal Full"},
    {"Configuration": "Late Fusion (All 3 Modalities - Decision Level)", "Modalities": "Image + Text + Telemetry", "Accuracy": lf_comb["accuracy"], "Macro-F1": lf_comb["macro_f1"], "Weighted-F1": lf_comb["weighted_f1"], "Type": "Trimodal Full"},
]

ablation_df = pd.DataFrame(ablation_table)
ablation_df.to_csv(os.path.join(OUT_DIR, "modality_ablation_results.csv"), index=False)

with open(os.path.join(OUT_DIR, "modality_ablation_results.json"), "w", encoding="utf-8") as f:
    json.dump(ablation_table, f, indent=2)

# Chart 3: Modality Contribution Chart
plt.figure(figsize=(11, 6))
palette = {"Unimodal": "#e7298a", "Bimodal": "#1f78b4", "Trimodal Full": "#33a02c"}
ax = sns.barplot(data=ablation_df, x="Macro-F1", y="Configuration", hue="Type", dodge=False, palette=palette)
plt.title("Modality Contribution & Ablation Study (Macro-F1 Score, N=158 Test)", fontsize=13, weight="bold")
plt.xlabel("Macro-F1 Score", fontsize=11)
plt.xlim(0.0, 1.0)
for p in ax.patches:
    val = p.get_width()
    if val > 0.05:
        ax.annotate(f"{val:.3f}", (val + 0.015, p.get_y() + p.get_height() / 2), ha="left", va="center", fontsize=10, weight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_modality_contributions.png"), dpi=150)
plt.close()


# ==============================================================================
# PART 4: REAL-IMAGE VS SYNTHETIC-FALLBACK EVALUATION
# ==============================================================================
print("\n=== PART 4: REAL VS SYNTHETIC IMAGE PERFORMANCE COMPARISON ===")
# Evaluate Early Fusion predictions on test cases broken down by whether
# their image was an actual official PDF or synthetic/topology fallback.

test_case_ids = [c.case_id for c in test_cases]
test_meta_map = {}
for c in test_cases:
    with open(c.meta_path, "r", encoding="utf-8") as f:
        m = json.load(f)
    test_meta_map[c.case_id] = m

# Load early fusion test predictions
ef_test_loader = DataLoader(
    TensorDataset(torch.tensor(te_img, dtype=torch.float32),
                  torch.tensor(te_txt, dtype=torch.float32),
                  torch.tensor(te_tw, dtype=torch.float32),
                  torch.tensor(te_tm, dtype=torch.float32), te_y),
    batch_size=16)

# Load saved early fusion model
ef_saved = early_fusion.build_early_fusion_model(
    image_dim=te_img.shape[1], text_dim=te_txt.shape[1],
    telemetry_channels=n_channels, n_classes=len(du.FAULT_CLASSES))
ef_saved.load_state_dict(torch.load(os.path.join(OUT_DIR, "early_fusion", "early_fusion_model.pt")))
ef_saved.eval()

ef_test_logits = []
with torch.no_grad():
    for inp, _ in ef_batch(ef_test_loader)():
        ef_test_logits.append(ef_saved(**inp))
ef_test_preds = torch.cat(ef_test_logits, dim=0).argmax(dim=1).cpu().numpy()

# Categorize test cases: Real (official_pdf) vs Synthetic (fallback / archive)
is_real_mask = np.array([
    test_meta_map[cid].get("diagram_source") == "official_pdf" or test_meta_map[cid].get("image_is_real_infotecnica") is True
    for cid in test_case_ids
])

is_pure_synth_mask = np.array([
    test_meta_map[cid].get("diagram_source") == "synthetic_fallback"
    for cid in test_case_ids
])

y_test_arr = np.array(te_true)

def get_subset_metrics(mask, name):
    sub_y = y_test_arr[mask]
    sub_p = ef_test_preds[mask]
    n = len(sub_y)
    if n == 0:
        return {"Subset": name, "N": 0, "Accuracy": 0, "Macro-F1": 0, "Weighted-F1": 0}
    return {
        "Subset": name,
        "N": int(n),
        "Accuracy": float(accuracy_score(sub_y, sub_p)),
        "Macro-F1": float(f1_score(sub_y, sub_p, average="macro", zero_division=0)),
        "Weighted-F1": float(f1_score(sub_y, sub_p, average="weighted", zero_division=0)),
    }

real_synth_metrics = [
    get_subset_metrics(is_real_mask, "Actual CEN Data (Official PDF)"),
    get_subset_metrics(~is_real_mask, "Topology Rendered / Synthetic Fallback"),
    get_subset_metrics(is_pure_synth_mask, "Pure Synthetic Fallback (Only)"),
    get_subset_metrics(np.ones(len(y_test_arr), dtype=bool), "All Test Cases Combined"),
]

real_synth_df = pd.DataFrame(real_synth_metrics)
real_synth_df.to_csv(os.path.join(OUT_DIR, "real_vs_synthetic_performance.csv"), index=False)
with open(os.path.join(OUT_DIR, "real_vs_synthetic_performance.json"), "w", encoding="utf-8") as f:
    json.dump(real_synth_metrics, f, indent=2)

print(real_synth_df)

# Chart 4: Real vs Synthetic Performance
plt.figure(figsize=(9, 5))
bar_colors = ["#2b5c8f", "#e66101", "#fdb863", "#4dac26"]
x = np.arange(len(real_synth_df))
width = 0.35
plt.bar(x - width/2, real_synth_df["Accuracy"], width, label="Accuracy", color="#1f78b4")
plt.bar(x + width/2, real_synth_df["Macro-F1"], width, label="Macro-F1", color="#33a02c")
plt.xticks(x, [f"{row['Subset']}\n(N={row['N']})" for _, row in real_synth_df.iterrows()], fontsize=9)
plt.ylabel("Score", fontsize=11)
plt.ylim(0, 1.1)
plt.title("Early Fusion Performance on Real CEN Data vs Synthetic Fallback", fontsize=12, weight="bold")
plt.legend(loc="upper right")
for i, row in real_synth_df.iterrows():
    plt.text(i - width/2, row["Accuracy"] + 0.02, f"{row['Accuracy']:.3f}", ha="center", fontsize=8, weight="bold")
    plt.text(i + width/2, row["Macro-F1"] + 0.02, f"{row['Macro-F1']:.3f}", ha="center", fontsize=8, weight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_real_vs_synthetic_performance.png"), dpi=150)
plt.close()

print("\nAll analysis scripts and charts generated successfully.")
