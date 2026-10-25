"""
Vendorized feature utilities for Multimodal Fault Console.
Source: dataset_utils.py
Source Git SHA: 87d90548
Extracts ResNet-18 image features, TF-IDF text features, and telemetry preprocessing.
"""

import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image

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


def get_image_transform():
    import torchvision.transforms as transforms
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


class ResNet18FeatureExtractor:
    def __init__(self):
        import torchvision.models as models
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        model.fc = nn.Identity()
        model.eval()
        self.model = model
        self.transform = get_image_transform()

    @torch.inference_mode()
    def extract_from_pil(self, img: Image.Image) -> np.ndarray:
        tensor = self.transform(img.convert("RGB")).unsqueeze(0)
        feat = self.model(tensor).squeeze(0).numpy()
        return feat.astype(np.float32)

    @torch.inference_mode()
    def extract_from_path(self, path: str) -> np.ndarray:
        img = Image.open(path)
        return self.extract_from_pil(img)


def extract_telemetry_window_single(csv_path: str, target_len: int = 48) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Extracts a (C, T) window, (1, T) mask, and raw grouped dataframe from telemetry CSV.
    """
    n_ch = len(TELEMETRY_CHANNELS)
    window = np.zeros((n_ch, target_len), dtype=np.float32)
    mask = np.zeros((1, target_len), dtype=np.float32)

    try:
        df = pd.read_csv(csv_path, usecols=["timestamp", "mwh", "generation_type"])
        df = df.dropna(subset=["timestamp", "mwh"])
        if len(df) == 0:
            return window, mask, pd.DataFrame()

        df["mwh"] = pd.to_numeric(df["mwh"], errors="coerce").fillna(0.0)
        df["gen_lower"] = df["generation_type"].fillna("").astype(str).str.lower()

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
            return window, mask, pd.DataFrame()

        vals = grouped[["ch_total", "ch_thermal", "ch_solar", "ch_hydro", "ch_wind", "ch_bess", "ch_other"]].values
        vals_t = vals.T.astype(np.float32)

        if ts_len >= target_len:
            window = vals_t[:, :target_len]
            mask[:, :] = 1.0
        else:
            window[:, :ts_len] = vals_t
            mask[:, :ts_len] = 1.0

        return window, mask, grouped
    except Exception:
        return window, mask, pd.DataFrame()
