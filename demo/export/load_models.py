"""
Model Loader and Multi-modal Predictor with Modality Masking.
Provides load(model_id) -> Predictor object with:
  predict(case_input, mask=('D', 'T', 'R')) -> np.ndarray (probabilities across 4 classes)
"""

import os
import json
import torch
import numpy as np
from typing import Dict, Any, Tuple, Union, Sequence

from demo.export.vendor.models import (
    EarlyFusionModel,
    ImageOnlyModel,
    TextOnlyModel,
    TelemetryOnlyModel,
    HybridImageTextModel,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
MODELS_DIR = os.path.join(ARTIFACTS_DIR, "models")


class ConfluxPredictor:
    """Early Fusion model predictor."""
    def __init__(self, model: EarlyFusionModel, config: Dict[str, Any]):
        self.model = model
        self.config = config
        self.model.eval()

    @torch.inference_mode()
    def predict(self, case_input: Dict[str, torch.Tensor],
                mask: Sequence[str] = ("D", "T", "R")) -> np.ndarray:
        mask_set = set(mask)
        # Diagram
        img = case_input["image_feat"]
        if "D" not in mask_set:
            img = torch.zeros_like(img)

        # Report Text
        txt = case_input["text_feat"]
        if "R" not in mask_set:
            txt = torch.zeros_like(txt)

        # Telemetry
        tw = case_input["telemetry_window"]
        tm = case_input.get("telemetry_mask", torch.ones_like(tw[:, :1, :]))
        if "T" not in mask_set:
            tw = torch.zeros_like(tw)
            tm = torch.zeros_like(tm)

        if not mask_set:
            return np.full((img.shape[0], self.config["n_classes"]), 1.0 / self.config["n_classes"], dtype=np.float32)

        logits = self.model(img, txt, tw, tm)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs


class QuorumPredictor:
    """Late Fusion model predictor with dynamic vote reweighting."""
    def __init__(self, img_m: ImageOnlyModel, txt_m: TextOnlyModel,
                 tel_m: TelemetryOnlyModel, config: Dict[str, Any]):
        self.img_m = img_m
        self.txt_m = txt_m
        self.tel_m = tel_m
        self.config = config
        self.weights = config["combination_weights"]
        self.img_m.eval()
        self.txt_m.eval()
        self.tel_m.eval()

    @torch.inference_mode()
    def predict_specialists(self, case_input: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
        p_img = torch.softmax(self.img_m(case_input["image_feat"]), dim=1).cpu().numpy()
        p_txt = torch.softmax(self.txt_m(case_input["text_feat"]), dim=1).cpu().numpy()
        tw = case_input["telemetry_window"]
        tm = case_input.get("telemetry_mask", torch.ones_like(tw[:, :1, :]))
        p_tel = torch.softmax(self.tel_m(tw, tm), dim=1).cpu().numpy()
        return {"image": p_img, "text": p_txt, "telemetry": p_tel}

    @torch.inference_mode()
    def predict(self, case_input: Dict[str, torch.Tensor],
                mask: Sequence[str] = ("D", "T", "R")) -> np.ndarray:
        mask_set = set(mask)
        if not mask_set:
            batch_size = case_input["image_feat"].shape[0]
            return np.full((batch_size, self.config["n_classes"]), 1.0 / self.config["n_classes"], dtype=np.float32)

        specialists = self.predict_specialists(case_input)
        active_weights = {}
        if "D" in mask_set:
            active_weights["image"] = self.weights["image"]
        if "R" in mask_set:
            active_weights["text"] = self.weights["text"]
        if "T" in mask_set:
            active_weights["telemetry"] = self.weights["telemetry"]

        total_weight = sum(active_weights.values())
        if total_weight <= 0:
            batch_size = case_input["image_feat"].shape[0]
            return np.full((batch_size, self.config["n_classes"]), 1.0 / self.config["n_classes"], dtype=np.float32)

        norm_weights = {k: v / total_weight for k, v in active_weights.items()}

        combined = np.zeros_like(next(iter(specialists.values())))
        for mod, w in norm_weights.items():
            combined += w * specialists[mod]

        return combined


class NexusPredictor:
    """Hybrid Fusion model predictor combining ImageText branch and Telemetry branch."""
    def __init__(self, it_m: HybridImageTextModel, tel_m: TelemetryOnlyModel, config: Dict[str, Any]):
        self.it_m = it_m
        self.tel_m = tel_m
        self.config = config
        self.weights = config["combination_weights"]
        self.it_m.eval()
        self.tel_m.eval()

    @torch.inference_mode()
    def predict_branches(self, case_input: Dict[str, torch.Tensor],
                         mask: Sequence[str] = ("D", "T", "R")) -> Dict[str, np.ndarray]:
        mask_set = set(mask)
        img = case_input["image_feat"]
        if "D" not in mask_set:
            img = torch.zeros_like(img)
        txt = case_input["text_feat"]
        if "R" not in mask_set:
            txt = torch.zeros_like(txt)

        p_it = torch.softmax(self.it_m(img, txt), dim=1).cpu().numpy()

        tw = case_input["telemetry_window"]
        tm = case_input.get("telemetry_mask", torch.ones_like(tw[:, :1, :]))
        p_tel = torch.softmax(self.tel_m(tw, tm), dim=1).cpu().numpy()

        return {"image_text": p_it, "telemetry": p_tel}

    @torch.inference_mode()
    def predict(self, case_input: Dict[str, torch.Tensor],
                mask: Sequence[str] = ("D", "T", "R")) -> np.ndarray:
        mask_set = set(mask)
        if not mask_set:
            batch_size = case_input["image_feat"].shape[0]
            return np.full((batch_size, self.config["n_classes"]), 1.0 / self.config["n_classes"], dtype=np.float32)

        has_it = "D" in mask_set or "R" in mask_set
        has_tel = "T" in mask_set

        branches = self.predict_branches(case_input, mask)

        if has_it and has_tel:
            w_it = self.weights["image_text"]
            w_tel = self.weights["telemetry"]
            total = w_it + w_tel
            return (w_it / total) * branches["image_text"] + (w_tel / total) * branches["telemetry"]
        elif has_it:
            return branches["image_text"]
        elif has_tel:
            return branches["telemetry"]
        else:
            batch_size = case_input["image_feat"].shape[0]
            return np.full((batch_size, self.config["n_classes"]), 1.0 / self.config["n_classes"], dtype=np.float32)


def load(model_id: str):
    """Load model by id: 'conflux', 'quorum', or 'nexus'."""
    model_dir = os.path.join(MODELS_DIR, model_id)
    with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as f:
        config = json.load(f)

    if model_id == "conflux":
        m = EarlyFusionModel(
            image_dim=config["image_dim"],
            text_dim=config["text_dim"],
            telemetry_channels=config["telemetry_channels"],
            n_classes=config["n_classes"],
            d_model=config["d_model"],
        )
        weights = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu")
        m.load_state_dict(weights)
        return ConfluxPredictor(m, config)

    elif model_id == "quorum":
        img_m = ImageOnlyModel(config["image_dim"], config["n_classes"], config["d_model"])
        txt_m = TextOnlyModel(config["text_dim"], config["n_classes"], config["d_model"])
        tel_m = TelemetryOnlyModel(config["telemetry_channels"], config["n_classes"], config["d_model"])

        state = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu")
        img_m.load_state_dict(state["image_model"])
        txt_m.load_state_dict(state["text_model"])
        tel_m.load_state_dict(state["telemetry_model"])
        return QuorumPredictor(img_m, txt_m, tel_m, config)

    elif model_id == "nexus":
        it_m = HybridImageTextModel(config["image_dim"], config["text_dim"], config["n_classes"], config["d_model"])
        tel_m = TelemetryOnlyModel(config["telemetry_channels"], config["n_classes"], config["d_model"])

        state = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu")
        it_m.load_state_dict(state["image_text_model"])
        tel_m.load_state_dict(state["telemetry_model"])
        return NexusPredictor(it_m, tel_m, config)

    else:
        raise ValueError(f"Unknown model_id: {model_id}. Expected 'conflux', 'quorum', or 'nexus'.")
