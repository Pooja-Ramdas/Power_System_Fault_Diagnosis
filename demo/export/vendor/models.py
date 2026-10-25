"""
Vendorized model architectures for Multimodal Fault Console.
Source: early_fusion.py, late_fusion.py, hybrid_fusion.py, dataset_utils.py
Source Git SHA: 87d90548
Preserves exact architecture and layer naming to load saved weights faithfully.
"""

from typing import Optional
import torch
import torch.nn as nn

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
    def __init__(self, in_channels: int = 7, d_model: int = 128):
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
        if telemetry_mask is not None:
            h = h * telemetry_mask
            denom = telemetry_mask.sum(dim=-1).clamp(min=1.0)
            pooled = h.sum(dim=-1) / denom
        else:
            pooled = h.mean(dim=-1)
        return pooled


def build_classifier_head(in_dim: int, n_classes: int = 4, hidden: int = 96) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden, n_classes),
    )


# 1. Early Fusion Model (CONFLUX)
class EarlyFusionModel(nn.Module):
    def __init__(self, image_dim: int = 512, text_dim: int = 128, telemetry_channels: int = 7,
                 n_classes: int = 4, d_model: int = 128):
        super().__init__()
        self.image_proj = ProjectionHead(image_dim, d_model)
        self.text_proj = ProjectionHead(text_dim, d_model)
        self.telemetry_encoder = TelemetryTCN(telemetry_channels, d_model)
        self.joint_classifier = build_classifier_head(d_model * 3, n_classes, hidden=96)

    def forward(self, image_feat: torch.Tensor, text_feat: torch.Tensor,
                telemetry_window: torch.Tensor, telemetry_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        img_emb = self.image_proj(image_feat)
        txt_emb = self.text_proj(text_feat)
        tel_emb = self.telemetry_encoder(telemetry_window, telemetry_mask)
        fused = torch.cat([img_emb, txt_emb, tel_emb], dim=1)
        return self.joint_classifier(fused)


# 2. Late Fusion Specialists (QUORUM)
class ImageOnlyModel(nn.Module):
    def __init__(self, image_dim: int = 512, n_classes: int = 4, d_model: int = 128):
        super().__init__()
        self.proj = ProjectionHead(image_dim, d_model)
        self.head = build_classifier_head(d_model, n_classes)

    def forward(self, image_feat: torch.Tensor) -> torch.Tensor:
        return self.head(self.proj(image_feat))


class TextOnlyModel(nn.Module):
    def __init__(self, text_dim: int = 128, n_classes: int = 4, d_model: int = 128):
        super().__init__()
        self.proj = ProjectionHead(text_dim, d_model)
        self.head = build_classifier_head(d_model, n_classes)

    def forward(self, text_feat: torch.Tensor) -> torch.Tensor:
        return self.head(self.proj(text_feat))


class TelemetryOnlyModel(nn.Module):
    def __init__(self, telemetry_channels: int = 7, n_classes: int = 4, d_model: int = 128):
        super().__init__()
        self.encoder = TelemetryTCN(telemetry_channels, d_model)
        self.head = build_classifier_head(d_model, n_classes)

    def forward(self, telemetry_window: torch.Tensor, telemetry_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.head(self.encoder(telemetry_window, telemetry_mask))


# 3. Hybrid Fusion Models (NEXUS)
class HybridImageTextModel(nn.Module):
    def __init__(self, image_dim: int = 512, text_dim: int = 128, n_classes: int = 4, d_model: int = 128):
        super().__init__()
        self.image_proj = ProjectionHead(image_dim, d_model)
        self.text_proj = ProjectionHead(text_dim, d_model)
        self.joint_classifier = build_classifier_head(d_model * 2, n_classes, hidden=96)

    def forward(self, image_feat: torch.Tensor, text_feat: torch.Tensor) -> torch.Tensor:
        img_emb = self.image_proj(image_feat)
        txt_emb = self.text_proj(text_feat)
        fused = torch.cat([img_emb, txt_emb], dim=1)
        return self.joint_classifier(fused)
