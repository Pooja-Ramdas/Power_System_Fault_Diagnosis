"""
early_fusion.py
================
EARLY FUSION for the CEN multimodal power-system fault diagnosis project.

Definition used here (and why it is correct "early" fusion, not late or
hybrid): every modality is converted into a fixed-length embedding
(image -> frozen CNN features, text -> TF-IDF/SVD features, telemetry ->
trainable TCN features), and those three embeddings are CONCATENATED into a
single vector *before* any classification decision is made. A single joint
classifier head then makes ONE decision from that combined vector. There is
only one loss, one prediction, one confidence score per case -- fusion
happens at the representation level, upstream of any per-modality output.

Run:
    python early_fusion.py --dataset-root cen_dataset --out-dir results/early_fusion

See the bottom of this file / the accompanying prompt for full instructions.
"""

import argparse
import os

import numpy as np

import dataset_utils as du


def build_early_fusion_model(image_dim: int, text_dim: int, telemetry_channels: int,
                              n_classes: int, d_model: int = 128):
    import torch
    import torch.nn as nn

    class EarlyFusionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.image_proj = du.ProjectionHead(image_dim, d_model)
            self.text_proj = du.ProjectionHead(text_dim, d_model)
            self.telemetry_encoder = du.build_tcn(telemetry_channels, d_model)
            # single joint classifier over the CONCATENATED representation
            self.joint_classifier = du.build_classifier_head(d_model * 3, n_classes, hidden=96)

        def forward(self, image_feat, text_feat, telemetry_window, telemetry_mask):
            img_emb = self.image_proj(image_feat)
            txt_emb = self.text_proj(text_feat)
            tel_emb = self.telemetry_encoder(telemetry_window, telemetry_mask)
            fused = torch.cat([img_emb, txt_emb, tel_emb], dim=1)   # <-- the fusion step
            return self.joint_classifier(fused)

    return EarlyFusionModel()


def main():
    parser = argparse.ArgumentParser(description="Early fusion (image+text+telemetry) fault classifier")
    parser.add_argument("--dataset-root", default=du.DATASET_ROOT)
    parser.add_argument("--out-dir", default="results/early_fusion")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--use-transformer-text", action="store_true",
                         help="Use a pretrained multilingual sentence-transformer for text "
                              "instead of the fast TF-IDF default (needs internet + is slower).")
    parser.add_argument("--force-rescan", action="store_true",
                         help="Ignore the completeness cache and re-check every case folder.")
    args = parser.parse_args()

    du.set_seed()
    import torch
    from torch.utils.data import TensorDataset, DataLoader

    # ---- 1. which cases are fully developed, right now, and FAST ----
    du.DATASET_ROOT = args.dataset_root
    cases = du.scan_dataset(dataset_root=args.dataset_root, force_rescan=args.force_rescan)
    if len(cases) < 20:
        raise SystemExit(f"Only {len(cases)} fully-developed cases found -- need at least "
                          f"~20 to train anything meaningful. Let more cases finish processing.")

    train_cases, val_cases, test_cases = du.stratified_case_split(cases)
    weights = du.class_weights(train_cases)
    print(f"[early_fusion] class weights (train-set inverse frequency): {weights}")

    # ---- 2. modality features ----
    print("[early_fusion] extracting/loading image features (frozen CNN, cached)...")
    img_feats_all = du.get_image_features(cases)
    case_index = {c.case_id: i for i, c in enumerate(cases)}

    def img_subset(subset):
        return img_feats_all[[case_index[c.case_id] for c in subset]]

    text_featurizer = du.TextFeaturizer(use_pretrained_transformer=args.use_transformer_text)
    train_text = text_featurizer.fit_transform(du.load_texts(train_cases))   # fit on TRAIN ONLY
    val_text = text_featurizer.transform(du.load_texts(val_cases))
    test_text = text_featurizer.transform(du.load_texts(test_cases))

    tel_windows_all, tel_masks_all, n_channels = du.load_telemetry_windows(cases)

    def tel_subset(subset):
        idx = [case_index[c.case_id] for c in subset]
        return tel_windows_all[idx], tel_masks_all[idx]

    train_tel_w, train_tel_m = tel_subset(train_cases)
    val_tel_w, val_tel_m = tel_subset(val_cases)
    test_tel_w, test_tel_m = tel_subset(test_cases)

    scaler = du.fit_telemetry_scaler(train_tel_w, train_tel_m)   # fit on TRAIN ONLY
    train_tel_w = du.apply_telemetry_scaler(train_tel_w, scaler)
    val_tel_w = du.apply_telemetry_scaler(val_tel_w, scaler)
    test_tel_w = du.apply_telemetry_scaler(test_tel_w, scaler)

    class_to_idx = {c: i for i, c in enumerate(du.FAULT_CLASSES)}

    def to_tensors(subset, img, text, tel_w, tel_m):
        y = torch.tensor([class_to_idx[c.label] for c in subset], dtype=torch.long)
        return (torch.tensor(img, dtype=torch.float32),
                torch.tensor(text, dtype=torch.float32),
                torch.tensor(tel_w, dtype=torch.float32),
                torch.tensor(tel_m, dtype=torch.float32), y)

    train_t = to_tensors(train_cases, img_subset(train_cases), train_text, train_tel_w, train_tel_m)
    val_t = to_tensors(val_cases, img_subset(val_cases), val_text, val_tel_w, val_tel_m)
    test_t = to_tensors(test_cases, img_subset(test_cases), test_text, test_tel_w, test_tel_m)

    train_loader = DataLoader(TensorDataset(*train_t), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(*val_t), batch_size=args.batch_size)
    test_loader = DataLoader(TensorDataset(*test_t), batch_size=args.batch_size)

    def batches(loader):
        def _gen():
            for img, txt, tw, tm, y in loader:
                yield {"image_feat": img, "text_feat": txt,
                       "telemetry_window": tw, "telemetry_mask": tm}, y
        return _gen

    # ---- 3. model ----
    model = build_early_fusion_model(
        image_dim=train_t[0].shape[1], text_dim=train_t[1].shape[1],
        telemetry_channels=n_channels, n_classes=len(du.FAULT_CLASSES))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))

    print("[early_fusion] training joint model (single loss over the concatenated embedding)...")
    history = du.train_with_early_stopping(
        model, optimizer, loss_fn, batches(train_loader), batches(val_loader),
        max_epochs=args.epochs, patience=args.patience, verbose_name="early_fusion")

    os.makedirs(args.out_dir, exist_ok=True)
    du.plot_training_curves(history, args.out_dir, "early_fusion")

    # ---- 4. test-set evaluation ----
    model.eval()
    all_logits, all_true = [], []
    with torch.no_grad():
        for inputs, y in batches(test_loader)():
            all_logits.append(model(**inputs))
            all_true.extend(y.tolist())
    logits = torch.cat(all_logits, dim=0)
    probs = torch.softmax(logits, dim=1).numpy()
    preds = probs.argmax(axis=1)

    du.evaluate_and_plot(all_true, preds, probs, du.FAULT_CLASSES, args.out_dir, "early_fusion")
    torch.save(model.state_dict(), os.path.join(args.out_dir, "early_fusion_model.pt"))
    print(f"\n[early_fusion] done. Metrics, plots and weights saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
