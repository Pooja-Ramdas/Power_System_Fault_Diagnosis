"""
dataset_utils.py
================
Shared utilities for CEN multimodal power-system fault diagnosis:
- Dataset discovery and completeness scanning
- Stratified train/val/test splitting
- Class weighting
- Image, text, and telemetry feature extraction and caching
- Lightweight neural network modules (ProjectionHead, TCN, ClassifierHead)
- Early-stopping training loop
- Evaluation metrics and plotting
"""

import copy
import json
import os
import random
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from PIL import Image

# ---------------------------------------------------------------------------
# Default constants and paths
# ---------------------------------------------------------------------------
DATASET_ROOT = "cen_dataset"

FAULT_CLASSES = [
    "line_fault",
    "breaker_fault",
    "transformer_fault",
    "busbar_fault",
]

TELEMETRY_CHANNELS = [
    "total",
    "thermal",
    "solar",
    "hydro",
    "wind",
    "bess",
    "other",
]


@dataclass
class Case:
    case_id: str
    label: str
    case_dir: str
    image_path: str
    telemetry_path: str
    report_en_path: str
    report_es_path: str
    meta_path: str


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_dataset_root(root: str) -> str:
    if os.path.exists(root):
        return root
    alt = os.path.join("cen_pipeline", root)
    if os.path.exists(alt):
        return alt
    return root


# ---------------------------------------------------------------------------
# 1. Dataset scanning & completeness validation
# ---------------------------------------------------------------------------
def scan_dataset(dataset_root: str = DATASET_ROOT, force_rescan: bool = False) -> List[Case]:
    """
    Dynamically scan the dataset directory and return ONLY fully-developed, complete cases.
    A complete case must contain:
      - valid meta.json with fault label in FAULT_CLASSES
      - non-empty image.png
      - non-empty telemetry.csv
      - non-empty report_en.txt or report_es.txt
    """
    real_root = resolve_dataset_root(dataset_root)
    cache_path = os.path.join(real_root, ".completeness_cache.json")

    if not force_rescan and os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
            cases = []
            for item in cached_data:
                cdir = item["case_dir"]
                if not os.path.exists(cdir):
                    raise ValueError(f"Path no longer exists: {cdir}")
                cases.append(Case(
                    case_id=item["case_id"],
                    label=item["label"],
                    case_dir=cdir,
                    image_path=item["image_path"],
                    telemetry_path=item["telemetry_path"],
                    report_en_path=item["report_en_path"],
                    report_es_path=item["report_es_path"],
                    meta_path=item["meta_path"],
                ))
            print(f"[dataset_utils] Loaded {len(cases)} fully-developed cases from cache ({cache_path})")
            return cases
        except Exception:
            pass  # Fall through to dynamic rescan

    if not os.path.exists(real_root):
        raise FileNotFoundError(f"Dataset root '{dataset_root}' not found at '{real_root}'")

    case_dirs = sorted([
        os.path.join(real_root, d)
        for d in os.listdir(real_root)
        if d.startswith("CASE-") and os.path.isdir(os.path.join(real_root, d))
    ])

    complete_cases: List[Case] = []
    fault_classes_set = set(FAULT_CLASSES)

    for cdir in case_dirs:
        case_id = os.path.basename(cdir)
        meta_p = os.path.join(cdir, "meta.json")
        img_p = os.path.join(cdir, "image.png")
        tel_p = os.path.join(cdir, "telemetry.csv")
        rpt_en_p = os.path.join(cdir, "report_en.txt")
        rpt_es_p = os.path.join(cdir, "report_es.txt")

        # Modality completeness checks
        if not (os.path.exists(meta_p) and os.path.exists(img_p) and os.path.exists(tel_p)):
            continue
        if not (os.path.exists(rpt_en_p) or os.path.exists(rpt_es_p)):
            continue

        try:
            if os.path.getsize(img_p) < 100 or os.path.getsize(tel_p) < 50:
                continue

            with open(meta_p, "r", encoding="utf-8") as f:
                meta = json.load(f)
            label = meta.get("fault_label") or meta.get("fault_type") or meta.get("label")

            if label not in fault_classes_set:
                continue

            complete_cases.append(Case(
                case_id=case_id,
                label=label,
                case_dir=cdir,
                image_path=img_p,
                telemetry_path=tel_p,
                report_en_path=rpt_en_p if os.path.exists(rpt_en_p) else "",
                report_es_path=rpt_es_p if os.path.exists(rpt_es_p) else "",
                meta_path=meta_p,
            ))
        except Exception:
            continue

    print(f"[dataset_utils] Scan completed: found {len(complete_cases)} fully-developed cases "
          f"(out of {len(case_dirs)} total case folders in {dataset_root}).")

    # Save completeness cache
    try:
        cache_data = [
            {
                "case_id": c.case_id,
                "label": c.label,
                "case_dir": c.case_dir,
                "image_path": c.image_path,
                "telemetry_path": c.telemetry_path,
                "report_en_path": c.report_en_path,
                "report_es_path": c.report_es_path,
                "meta_path": c.meta_path,
            }
            for c in complete_cases
        ]
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2)
    except Exception as e:
        print(f"[dataset_utils] Warning: could not write completeness cache: {e}")

    return complete_cases


# ---------------------------------------------------------------------------
# 2. Stratified splitting & class weighting
# ---------------------------------------------------------------------------
def stratified_case_split(
    cases: List[Case],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Case], List[Case], List[Case]]:
    labels = [c.label for c in cases]
    temp_ratio = val_ratio + test_ratio

    train_cases, temp_cases = train_test_split(
        cases,
        test_size=temp_ratio,
        random_state=seed,
        stratify=labels,
    )

    temp_labels = [c.label for c in temp_cases]
    val_share_of_temp = val_ratio / temp_ratio

    val_cases, test_cases = train_test_split(
        temp_cases,
        test_size=(1.0 - val_share_of_temp),
        random_state=seed,
        stratify=temp_labels,
    )

    return train_cases, val_cases, test_cases


def class_weights(train_cases: List[Case]) -> List[float]:
    """Computes train-set inverse frequency class weights for loss weighting."""
    counts = Counter(c.label for c in train_cases)
    n_total = len(train_cases)
    n_classes = len(FAULT_CLASSES)
    weights = []
    for cls in FAULT_CLASSES:
        cnt = counts.get(cls, 1)
        w = n_total / (n_classes * cnt)
        weights.append(float(w))
    return weights


# ---------------------------------------------------------------------------
# 3. Image Feature Extraction & Caching
# ---------------------------------------------------------------------------
def get_image_features(cases: List[Case], cache_path: Optional[str] = None) -> np.ndarray:
    """
    Extract frozen CNN features (ResNet-18, 512-dim) for each case.
    Reuses cached features if available.
    """
    if not cases:
        return np.empty((0, 512), dtype=np.float32)

    root = os.path.dirname(cases[0].case_dir)
    if cache_path is None:
        cache_path = os.path.join(root, ".image_features_cache.npz")

    case_ids = [c.case_id for c in cases]

    if os.path.exists(cache_path):
        try:
            cached = np.load(cache_path, allow_pickle=True)
            cached_ids = list(cached["case_ids"])
            if cached_ids == case_ids:
                return cached["features"].astype(np.float32)
            # If subset or superset, map by ID
            id_to_idx = {cid: i for i, cid in enumerate(cached_ids)}
            if all(cid in id_to_idx for cid in case_ids):
                idx = [id_to_idx[cid] for cid in case_ids]
                return cached["features"][idx].astype(np.float32)
        except Exception:
            pass

    import torchvision.models as models
    import torchvision.transforms as transforms

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.fc = nn.Identity()
    model.eval().to(device)

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    features_list = []
    batch_size = 32

    with torch.no_grad():
        for i in range(0, len(cases), batch_size):
            batch_cases = cases[i : i + batch_size]
            imgs = []
            for c in batch_cases:
                try:
                    img = Image.open(c.image_path).convert("RGB")
                    imgs.append(preprocess(img))
                except Exception:
                    imgs.append(torch.zeros(3, 224, 224))
            batch_tensor = torch.stack(imgs).to(device)
            feats = model(batch_tensor).cpu().numpy()
            features_list.append(feats)

    features = np.concatenate(features_list, axis=0).astype(np.float32)

    try:
        np.savez_compressed(cache_path, case_ids=np.array(case_ids), features=features)
    except Exception as e:
        print(f"[dataset_utils] Warning: could not write image cache: {e}")

    return features


# ---------------------------------------------------------------------------
# 4. Text Loading & Featurization
# ---------------------------------------------------------------------------
def load_texts(cases: List[Case]) -> List[str]:
    texts = []
    for c in cases:
        text = ""
        if c.report_en_path and os.path.exists(c.report_en_path):
            with open(c.report_en_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read().strip()
        if not text and c.report_es_path and os.path.exists(c.report_es_path):
            with open(c.report_es_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read().strip()
        texts.append(text if text else "No report text available.")
    return texts


class TextFeaturizer:
    def __init__(self, use_pretrained_transformer: bool = False, max_features: int = 128):
        self.use_transformer = use_pretrained_transformer
        self.max_features = max_features
        self.vectorizer = None
        self.transformer_model = None

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        if self.use_transformer:
            from sentence_transformers import SentenceTransformer
            self.transformer_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
            embs = self.transformer_model.encode(texts, show_progress_bar=False)
            return embs.astype(np.float32)
        else:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                stop_words="english",
                ngram_range=(1, 2),
                sublinear_tf=True,
            )
            sparse_mat = self.vectorizer.fit_transform(texts)
            arr = sparse_mat.toarray().astype(np.float32)
            # Ensure fixed dimensionality
            if arr.shape[1] < self.max_features:
                pad = np.zeros((arr.shape[0], self.max_features - arr.shape[1]), dtype=np.float32)
                arr = np.hstack([arr, pad])
            return arr

    def transform(self, texts: List[str]) -> np.ndarray:
        if self.use_transformer:
            embs = self.transformer_model.encode(texts, show_progress_bar=False)
            return embs.astype(np.float32)
        else:
            sparse_mat = self.vectorizer.transform(texts)
            arr = sparse_mat.toarray().astype(np.float32)
            if arr.shape[1] < self.max_features:
                pad = np.zeros((arr.shape[0], self.max_features - arr.shape[1]), dtype=np.float32)
                arr = np.hstack([arr, pad])
            return arr


# ---------------------------------------------------------------------------
# 5. Telemetry Loading, Scaling, and Caching
# ---------------------------------------------------------------------------
def _extract_telemetry_window_single(csv_path: str, target_len: int = 48) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts a (C, T) window and (1, T) mask from a telemetry CSV.
    Channels: total, thermal, solar, hydro, wind, bess, other.
    """
    n_ch = len(TELEMETRY_CHANNELS)
    window = np.zeros((n_ch, target_len), dtype=np.float32)
    mask = np.zeros((1, target_len), dtype=np.float32)

    try:
        df = pd.read_csv(csv_path, usecols=["timestamp", "mwh", "generation_type"])
        df = df.dropna(subset=["timestamp", "mwh"])
        if len(df) == 0:
            return window, mask

        df["mwh"] = pd.to_numeric(df["mwh"], errors="coerce").fillna(0.0)
        df["gen_lower"] = df["generation_type"].fillna("").astype(str).str.lower()

        # Identify channels
        is_thermal = df["gen_lower"].str.contains("t[eé]rmica|gas|diesel|carb|cogen", regex=True)
        is_solar = df["gen_lower"].str.contains("solar", regex=True)
        is_hydro = df["gen_lower"].str.contains("hidr[aá]ulica|pasada|embalse", regex=True)
        is_wind = df["gen_lower"].str.contains("e[oó]lica", regex=True)
        is_bess = df["gen_lower"].str.contains("bess|bater", regex=True)
        is_other = ~(is_thermal | is_solar | is_hydro | is_wind | is_bess)

        df["ch_thermal"] = np.where(is_thermal, df["mwh"], 0.0)
        df["ch_solar"] = np.where(is_solar, df["mwh"], 0.0)
        df["ch_hydro"] = np.where(is_hydro, df["mwh"], 0.0)
        df["ch_wind"] = np.where(is_wind, df["mwh"], 0.0)
        df["ch_bess"] = np.where(is_bess, df["mwh"], 0.0)
        df["ch_other"] = np.where(is_other, df["mwh"], 0.0)
        df["ch_total"] = df["mwh"]

        grouped = df.groupby("timestamp")[
            ["ch_total", "ch_thermal", "ch_solar", "ch_hydro", "ch_wind", "ch_bess", "ch_other"]
        ].sum().reset_index().sort_values("timestamp")

        ts_len = len(grouped)
        if ts_len == 0:
            return window, mask

        vals = grouped[["ch_total", "ch_thermal", "ch_solar", "ch_hydro", "ch_wind", "ch_bess", "ch_other"]].values
        # vals shape: (ts_len, 7) -> transpose to (7, ts_len)
        vals_t = vals.T.astype(np.float32)

        if ts_len >= target_len:
            window = vals_t[:, :target_len]
            mask[:, :] = 1.0
        else:
            window[:, :ts_len] = vals_t
            mask[:, :ts_len] = 1.0

        return window, mask
    except Exception:
        return window, mask


def load_telemetry_windows(
    cases: List[Case],
    target_len: int = 48,
    cache_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Loads telemetry windows for all cases.
    Returns: (windows, masks, n_channels)
      windows: (N, n_channels, target_len)
      masks:   (N, 1, target_len)
      n_channels: 7
    """
    n_ch = len(TELEMETRY_CHANNELS)
    if not cases:
        return np.empty((0, n_ch, target_len), dtype=np.float32), np.empty((0, 1, target_len), dtype=np.float32), n_ch

    root = os.path.dirname(cases[0].case_dir)
    if cache_path is None:
        cache_path = os.path.join(root, ".telemetry_cache.npz")

    case_ids = [c.case_id for c in cases]

    if os.path.exists(cache_path):
        try:
            cached = np.load(cache_path, allow_pickle=True)
            cached_ids = list(cached["case_ids"])
            if cached_ids == case_ids:
                return (
                    cached["windows"].astype(np.float32),
                    cached["masks"].astype(np.float32),
                    n_ch,
                )
            id_to_idx = {cid: i for i, cid in enumerate(cached_ids)}
            if all(cid in id_to_idx for cid in case_ids):
                idx = [id_to_idx[cid] for cid in case_ids]
                return (
                    cached["windows"][idx].astype(np.float32),
                    cached["masks"][idx].astype(np.float32),
                    n_ch,
                )
        except Exception:
            pass

    windows_list = []
    masks_list = []

    for c in cases:
        win, msk = _extract_telemetry_window_single(c.telemetry_path, target_len=target_len)
        windows_list.append(win)
        masks_list.append(msk)

    windows_arr = np.stack(windows_list, axis=0).astype(np.float32)
    masks_arr = np.stack(masks_list, axis=0).astype(np.float32)

    try:
        np.savez_compressed(
            cache_path,
            case_ids=np.array(case_ids),
            windows=windows_arr,
            masks=masks_arr,
        )
    except Exception as e:
        print(f"[dataset_utils] Warning: could not write telemetry cache: {e}")

    return windows_arr, masks_arr, n_ch


def fit_telemetry_scaler(train_tel_w: np.ndarray, train_tel_m: np.ndarray) -> Dict[str, np.ndarray]:
    """Fits mean and std on TRAIN telemetry only, ignoring masked/padded hours."""
    # train_tel_w: (N, C, T), train_tel_m: (N, 1, T)
    weights = train_tel_m  # (N, 1, T)
    sum_w = weights.sum(axis=(0, 2), keepdims=True).clip(min=1.0)  # (1, 1, 1)

    mean = (train_tel_w * weights).sum(axis=(0, 2), keepdims=True) / sum_w  # (1, C, 1)
    variance = (((train_tel_w - mean) ** 2) * weights).sum(axis=(0, 2), keepdims=True) / sum_w
    std = np.sqrt(variance).clip(min=1e-3)  # (1, C, 1)

    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}


def apply_telemetry_scaler(tel_w: np.ndarray, scaler: Dict[str, np.ndarray]) -> np.ndarray:
    scaled = (tel_w - scaler["mean"]) / scaler["std"]
    return scaled.astype(np.float32)


# ---------------------------------------------------------------------------
# 6. Lightweight Neural Network Modules
# ---------------------------------------------------------------------------
class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TelemetryTCN(nn.Module):
    def __init__(self, in_channels: int, d_model: int = 128):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=2, dilation=2)
        self.norm2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, d_model, kernel_size=3, padding=1)
        self.norm3 = nn.BatchNorm1d(d_model)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, telemetry_window: torch.Tensor, telemetry_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # telemetry_window: (B, C, T)
        h = self.dropout(self.act(self.norm1(self.conv1(telemetry_window))))
        h = self.dropout(self.act(self.norm2(self.conv2(h))))
        h = self.act(self.norm3(self.conv3(h)))
        # Masked average pooling
        if telemetry_mask is not None:
            h = h * telemetry_mask
            denom = telemetry_mask.sum(dim=-1).clamp(min=1.0)
            pooled = h.sum(dim=-1) / denom
        else:
            pooled = h.mean(dim=-1)
        return pooled


def build_tcn(telemetry_channels: int, d_model: int = 128) -> nn.Module:
    return TelemetryTCN(telemetry_channels, d_model)


def build_classifier_head(in_dim: int, n_classes: int, hidden: int = 96) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden, n_classes),
    )


# ---------------------------------------------------------------------------
# 7. Training loop with early stopping
# ---------------------------------------------------------------------------
def train_with_early_stopping(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    train_loader_gen,
    val_loader_gen,
    max_epochs: int = 60,
    patience: int = 10,
    verbose_name: str = "model",
) -> Dict[str, List[float]]:
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_macro_f1": [],
    }

    best_val_loss = float("inf")
    best_weights = copy.deepcopy(model.state_dict())
    patience_counter = 0

    for epoch in range(1, max_epochs + 1):
        # --- Training ---
        model.train()
        train_loss_total = 0.0
        n_train_samples = 0
        train_correct = 0

        for inputs, targets in train_loader_gen():
            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            bs = targets.size(0)
            train_loss_total += loss.item() * bs
            train_correct += (outputs.argmax(dim=1) == targets).sum().item()
            n_train_samples += bs

        train_loss = train_loss_total / max(1, n_train_samples)
        train_acc = train_correct / max(1, n_train_samples)

        # --- Validation ---
        model.eval()
        val_loss_total = 0.0
        n_val_samples = 0
        val_preds, val_targets_list = [], []

        with torch.no_grad():
            for inputs, targets in val_loader_gen():
                outputs = model(**inputs)
                loss = loss_fn(outputs, targets)
                bs = targets.size(0)
                val_loss_total += loss.item() * bs
                val_preds.extend(outputs.argmax(dim=1).cpu().tolist())
                val_targets_list.extend(targets.cpu().tolist())
                n_val_samples += bs

        val_loss = val_loss_total / max(1, n_val_samples)
        val_acc = accuracy_score(val_targets_list, val_preds) if n_val_samples > 0 else 0.0
        val_macro_f1 = f1_score(val_targets_list, val_preds, average="macro", zero_division=0)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_macro_f1"].append(val_macro_f1)

        if epoch % 5 == 0 or epoch == 1 or epoch == max_epochs:
            print(f"[{verbose_name}] Epoch {epoch:2d}/{max_epochs} | "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.3f} | "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.3f}, Macro-F1: {val_macro_f1:.3f}")

        # Early stopping check
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[{verbose_name}] Early stopping triggered at epoch {epoch}. "
                      f"Restoring best model from validation loss {best_val_loss:.4f}.")
                break

    model.load_state_dict(best_weights)
    return history


# ---------------------------------------------------------------------------
# 8. Evaluation, Plotting, and Reporting
# ---------------------------------------------------------------------------
def plot_training_curves(history: Dict[str, List[float]], out_dir: str, prefix: str):
    os.makedirs(out_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, ax1 = plt.subplots(figsize=(8, 5))

    color = "tab:blue"
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color=color)
    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="tab:blue", linestyle="--")
    ax1.plot(epochs, history["val_loss"], label="Val Loss", color="tab:red")
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()
    color = "tab:green"
    ax2.set_ylabel("Score", color=color)
    ax2.plot(epochs, history["val_acc"], label="Val Accuracy", color="tab:green", alpha=0.7)
    ax2.plot(epochs, history["val_macro_f1"], label="Val Macro-F1", color="tab:purple", alpha=0.9)
    ax2.tick_params(axis="y", labelcolor=color)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.title(f"Training Curves - {prefix}")
    fig.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_training_curves.png"), dpi=150)
    plt.close()

    with open(os.path.join(out_dir, f"{prefix}_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def evaluate_and_plot(
    y_true: List[int],
    y_pred: np.ndarray,
    y_probs: Optional[np.ndarray],
    class_names: List[str],
    out_dir: str,
    prefix: str,
) -> Dict:
    os.makedirs(out_dir, exist_ok=True)

    y_pred_arr = np.asarray(y_pred)
    y_true_arr = np.asarray(y_true)

    acc = accuracy_score(y_true_arr, y_pred_arr)
    p_macro = precision_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
    r_macro = recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
    f1_macro = f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)

    p_weighted = precision_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)
    r_weighted = recall_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=list(range(len(class_names))))
    clf_report = classification_report(
        y_true_arr,
        y_pred_arr,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "accuracy": float(acc),
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(f1_macro),
        "weighted_precision": float(p_weighted),
        "weighted_recall": float(r_weighted),
        "weighted_f1": float(f1_weighted),
        "classification_report": clf_report,
        "confusion_matrix": cm.tolist(),
        "n_samples": len(y_true),
    }

    # Save metrics JSON
    with open(os.path.join(out_dir, f"{prefix}_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Plot Confusion Matrix
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(f"Confusion Matrix - {prefix}\nMacro-F1: {f1_macro:.3f} | Accuracy: {acc:.3f}")
    plt.ylabel("Ground Truth")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_confusion_matrix.png"), dpi=150)
    plt.close()

    print(f"\n--- [{prefix}] Evaluation Results (N={len(y_true)}) ---")
    print(f"Accuracy:         {acc:.4f}")
    print(f"Macro Precision:  {p_macro:.4f}")
    print(f"Macro Recall:     {r_macro:.4f}")
    print(f"Macro F1-Score:   {f1_macro:.4f}")
    print(f"Weighted F1-Score:{f1_weighted:.4f}")
    print(classification_report(y_true_arr, y_pred_arr, labels=list(range(len(class_names))),
                                target_names=class_names, zero_division=0))

    return metrics
