"""
hybrid_fusion.py
================
HYBRID FUSION for the CEN multimodal power-system fault diagnosis project.

Definition used here:
    1. IMAGE + TEXT are fused at the representation level:
       image and text embeddings are projected, concatenated, and passed
       through ONE joint classifier.
    2. TELEMETRY is kept as an independent branch with its own encoder,
       classifier, loss, optimizer, and training loop.
    3. After both branches are trained, their probability vectors are
       combined at the decision level using validation-tuned weights.

This is hybrid fusion because it combines two fusion levels:
    - early/feature-level fusion: image + text
    - late/decision-level fusion: fused image-text branch + telemetry

This implementation is intentionally based on the same preprocessing,
feature extraction, TCN, classifier heads, train/validation/test split,
early stopping, metrics, and plotting utilities used by early_fusion.py
and late_fusion.py, so the three experiments remain comparable.

Run:
    python hybrid_fusion.py --dataset-root cen_dataset --out-dir results/hybrid_fusion
"""

import argparse
import json
import os

import numpy as np

import dataset_utils as du


def build_hybrid_image_text_model(
    image_dim: int,
    text_dim: int,
    n_classes: int,
    d_model: int = 128,
):
    import torch
    import torch.nn as nn

    class ImageTextFusionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.image_proj = du.ProjectionHead(image_dim, d_model)
            self.text_proj = du.ProjectionHead(text_dim, d_model)

            # Feature-level fusion: concatenate image + text embeddings
            self.joint_classifier = du.build_classifier_head(
                d_model * 2, n_classes, hidden=96
            )

        def forward(self, image_feat, text_feat):
            img_emb = self.image_proj(image_feat)
            txt_emb = self.text_proj(text_feat)

            fused = torch.cat([img_emb, txt_emb], dim=1)
            return self.joint_classifier(fused)

    return ImageTextFusionModel()


def build_telemetry_model(
    telemetry_channels: int,
    n_classes: int,
    d_model: int = 128,
):
    import torch.nn as nn

    class TelemetryOnly(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = du.build_tcn(telemetry_channels, d_model)
            self.head = du.build_classifier_head(d_model, n_classes)

        def forward(self, telemetry_window, telemetry_mask):
            return self.head(self.encoder(telemetry_window, telemetry_mask))

    return TelemetryOnly()


def _search_best_weights(probs_list, y_true, n_steps: int = 11):
    """
    Find the image-text / telemetry probability weights that maximize
    validation macro-F1.

    n_steps=11 gives weights in increments of 0.1:
        w_it + w_tel = 1
    This is deliberately small and fast because there are only two branches.
    """
    from sklearn.metrics import f1_score

    best_f1 = -1.0
    best_weights = (0.5, 0.5)

    for w_it in np.linspace(0.0, 1.0, n_steps):
        w_tel = 1.0 - w_it

        combined = (
            w_it * probs_list[0]
            + w_tel * probs_list[1]
        )

        preds = combined.argmax(axis=1)
        f1 = f1_score(
            y_true,
            preds,
            average="macro",
            zero_division=0,
        )

        if f1 > best_f1:
            best_f1 = f1
            best_weights = (float(w_it), float(w_tel))

    return best_weights, best_f1


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid fusion (image+text representation fusion + telemetry decision fusion)"
    )
    parser.add_argument("--dataset-root", default=du.DATASET_ROOT)
    parser.add_argument("--out-dir", default="results/hybrid_fusion")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--use-transformer-text",
        action="store_true",
        help="Use a pretrained multilingual sentence-transformer for text "
             "instead of the fast TF-IDF default (needs internet + is slower).",
    )
    parser.add_argument(
        "--force-rescan",
        action="store_true",
        help="Ignore the completeness cache and re-check every case folder.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    du.set_seed(args.seed)

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    # ---------------------------------------------------------------
    # 1. Discover fully-developed cases dynamically
    # ---------------------------------------------------------------
    du.DATASET_ROOT = args.dataset_root

    cases = du.scan_dataset(
        dataset_root=args.dataset_root,
        force_rescan=args.force_rescan,
    )

    if len(cases) < 20:
        raise SystemExit(
            f"Only {len(cases)} fully-developed cases found -- need at least "
            f"~20 to train anything meaningful. Let more cases finish processing."
        )

    print(f"[hybrid_fusion] fully-developed cases available: {len(cases)}")

    train_cases, val_cases, test_cases = du.stratified_case_split(cases, seed=args.seed)
    weights = du.class_weights(train_cases)
    class_to_idx = {c: i for i, c in enumerate(du.FAULT_CLASSES)}

    print(f"[hybrid_fusion] train={len(train_cases)}, "
          f"val={len(val_cases)}, test={len(test_cases)}")
    print(f"[hybrid_fusion] class weights: {weights}")

    # ---------------------------------------------------------------
    # 2. Shared feature extraction
    #    Keep this aligned with early_fusion.py / late_fusion.py
    # ---------------------------------------------------------------
    print("[hybrid_fusion] extracting/loading image features (cached)...")

    img_feats_all = du.get_image_features(cases)
    case_index = {c.case_id: i for i, c in enumerate(cases)}

    def img_subset(subset):
        return img_feats_all[
            [case_index[c.case_id] for c in subset]
        ]

    print("[hybrid_fusion] extracting text features...")

    text_featurizer = du.TextFeaturizer(
        use_pretrained_transformer=args.use_transformer_text
    )

    # Fit text representation ONLY on training data
    train_text = text_featurizer.fit_transform(
        du.load_texts(train_cases)
    )
    val_text = text_featurizer.transform(
        du.load_texts(val_cases)
    )
    test_text = text_featurizer.transform(
        du.load_texts(test_cases)
    )

    print("[hybrid_fusion] loading telemetry windows...")

    tel_windows_all, tel_masks_all, n_channels = du.load_telemetry_windows(cases)

    def tel_subset(subset):
        idx = [case_index[c.case_id] for c in subset]
        return (
            tel_windows_all[idx],
            tel_masks_all[idx],
        )

    train_tel_w, train_tel_m = tel_subset(train_cases)
    val_tel_w, val_tel_m = tel_subset(val_cases)
    test_tel_w, test_tel_m = tel_subset(test_cases)

    # Fit telemetry scaler ONLY on training data
    scaler = du.fit_telemetry_scaler(
        train_tel_w,
        train_tel_m,
    )

    train_tel_w = du.apply_telemetry_scaler(
        train_tel_w,
        scaler,
    )
    val_tel_w = du.apply_telemetry_scaler(
        val_tel_w,
        scaler,
    )
    test_tel_w = du.apply_telemetry_scaler(
        test_tel_w,
        scaler,
    )

    def y_tensor(subset):
        return torch.tensor(
            [class_to_idx[c.label] for c in subset],
            dtype=torch.long,
        )

    train_y = y_tensor(train_cases)
    val_y = y_tensor(val_cases)
    test_y = y_tensor(test_cases)

    os.makedirs(args.out_dir, exist_ok=True)

    loss_fn = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(
            weights,
            dtype=torch.float32,
        )
    )

    # ---------------------------------------------------------------
    # 3. IMAGE + TEXT branch
    #    Feature-level fusion happens INSIDE this model.
    # ---------------------------------------------------------------
    image_text_model = build_hybrid_image_text_model(
        image_dim=img_feats_all.shape[1],
        text_dim=train_text.shape[1],
        n_classes=len(du.FAULT_CLASSES),
    )

    train_it = (
        torch.tensor(
            img_subset(train_cases),
            dtype=torch.float32,
        ),
        torch.tensor(
            train_text,
            dtype=torch.float32,
        ),
        train_y,
    )

    val_it = (
        torch.tensor(
            img_subset(val_cases),
            dtype=torch.float32,
        ),
        torch.tensor(
            val_text,
            dtype=torch.float32,
        ),
        val_y,
    )

    test_it = (
        torch.tensor(
            img_subset(test_cases),
            dtype=torch.float32,
        ),
        torch.tensor(
            test_text,
            dtype=torch.float32,
        ),
        test_y,
    )

    train_it_loader = DataLoader(
        TensorDataset(*train_it),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_it_loader = DataLoader(
        TensorDataset(*val_it),
        batch_size=args.batch_size,
    )
    test_it_loader = DataLoader(
        TensorDataset(*test_it),
        batch_size=args.batch_size,
    )

    def image_text_batches(loader):
        def _gen():
            for img, txt, y in loader:
                yield {
                    "image_feat": img,
                    "text_feat": txt,
                }, y

        return _gen

    it_optimizer = torch.optim.Adam(
        image_text_model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    print(
        "\n[hybrid_fusion] training IMAGE+TEXT branch "
        "(feature-level fusion, single joint loss)..."
    )

    it_history = du.train_with_early_stopping(
        image_text_model,
        it_optimizer,
        loss_fn,
        image_text_batches(train_it_loader),
        image_text_batches(val_it_loader),
        max_epochs=args.epochs,
        patience=args.patience,
        verbose_name="hybrid_fusion/image_text",
    )

    du.plot_training_curves(
        it_history,
        args.out_dir,
        "hybrid_fusion_image_text",
    )

    # Get validation and test probabilities from image-text branch
    image_text_model.eval()

    def get_image_text_probs(loader):
        all_logits = []
        all_true = []

        with torch.no_grad():
            for inputs, y in image_text_batches(loader)():
                all_logits.append(
                    image_text_model(**inputs)
                )
                all_true.extend(y.tolist())

        probs = torch.softmax(
            torch.cat(all_logits, dim=0),
            dim=1,
        ).numpy()

        return probs, all_true

    val_p_it, _ = get_image_text_probs(val_it_loader)
    test_p_it, test_true = get_image_text_probs(test_it_loader)

    # Ablation / intermediate result for image+text branch
    du.evaluate_and_plot(
        test_true,
        test_p_it.argmax(axis=1),
        test_p_it,
        du.FAULT_CLASSES,
        args.out_dir,
        "hybrid_fusion_image_text_only",
    )

    # ---------------------------------------------------------------
    # 4. TELEMETRY branch
    #    Independent decision-level branch.
    # ---------------------------------------------------------------
    telemetry_model = build_telemetry_model(
        telemetry_channels=n_channels,
        n_classes=len(du.FAULT_CLASSES),
    )

    train_tel = (
        torch.tensor(train_tel_w, dtype=torch.float32),
        torch.tensor(train_tel_m, dtype=torch.float32),
        train_y,
    )

    val_tel = (
        torch.tensor(val_tel_w, dtype=torch.float32),
        torch.tensor(val_tel_m, dtype=torch.float32),
        val_y,
    )

    test_tel = (
        torch.tensor(test_tel_w, dtype=torch.float32),
        torch.tensor(test_tel_m, dtype=torch.float32),
        test_y,
    )

    train_tel_loader = DataLoader(
        TensorDataset(*train_tel),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_tel_loader = DataLoader(
        TensorDataset(*val_tel),
        batch_size=args.batch_size,
    )
    test_tel_loader = DataLoader(
        TensorDataset(*test_tel),
        batch_size=args.batch_size,
    )

    def telemetry_batches(loader):
        def _gen():
            for tw, tm, y in loader:
                yield {
                    "telemetry_window": tw,
                    "telemetry_mask": tm,
                }, y

        return _gen

    tel_optimizer = torch.optim.Adam(
        telemetry_model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    print(
        "\n[hybrid_fusion] training TELEMETRY branch "
        "(independent branch, own loss)..."
    )

    tel_history = du.train_with_early_stopping(
        telemetry_model,
        tel_optimizer,
        loss_fn,
        telemetry_batches(train_tel_loader),
        telemetry_batches(val_tel_loader),
        max_epochs=args.epochs,
        patience=args.patience,
        verbose_name="hybrid_fusion/telemetry",
    )

    du.plot_training_curves(
        tel_history,
        args.out_dir,
        "hybrid_fusion_telemetry",
    )

    telemetry_model.eval()

    def get_telemetry_probs(loader):
        all_logits = []
        all_true = []

        with torch.no_grad():
            for inputs, y in telemetry_batches(loader)():
                all_logits.append(
                    telemetry_model(**inputs)
                )
                all_true.extend(y.tolist())

        probs = torch.softmax(
            torch.cat(all_logits, dim=0),
            dim=1,
        ).numpy()

        return probs, all_true

    val_p_tel, _ = get_telemetry_probs(val_tel_loader)
    test_p_tel, _ = get_telemetry_probs(test_tel_loader)

    # Telemetry-only ablation
    du.evaluate_and_plot(
        test_true,
        test_p_tel.argmax(axis=1),
        test_p_tel,
        du.FAULT_CLASSES,
        args.out_dir,
        "hybrid_fusion_telemetry_only",
    )

    # ---------------------------------------------------------------
    # 5. DECISION-LEVEL FUSION
    #    Combine IMAGE+TEXT and TELEMETRY probabilities.
    #    Weights are tuned ONLY on validation data.
    # ---------------------------------------------------------------
    val_true = val_y.tolist()

    best_weights, best_val_f1 = _search_best_weights(
        [val_p_it, val_p_tel],
        val_true,
    )

    w_it, w_tel = best_weights

    print(
        f"\n[hybrid_fusion] best validation decision weights: "
        f"image+text={w_it:.2f}, telemetry={w_tel:.2f} "
        f"(val macro_f1={best_val_f1:.4f})"
    )

    combined_test_probs = (
        w_it * test_p_it
        + w_tel * test_p_tel
    )

    combined_preds = combined_test_probs.argmax(axis=1)

    # Final hybrid evaluation
    du.evaluate_and_plot(
        test_true,
        combined_preds,
        combined_test_probs,
        du.FAULT_CLASSES,
        args.out_dir,
        "hybrid_fusion_combined",
    )

    # ---------------------------------------------------------------
    # 6. Save weights and models
    # ---------------------------------------------------------------
    torch.save(
        image_text_model.state_dict(),
        os.path.join(
            args.out_dir,
            "hybrid_image_text_model.pt",
        ),
    )

    torch.save(
        telemetry_model.state_dict(),
        os.path.join(
            args.out_dir,
            "hybrid_telemetry_model.pt",
        ),
    )

    with open(
        os.path.join(
            args.out_dir,
            "hybrid_fusion_combination_weights.json",
        ),
        "w",
    ) as f:
        json.dump(
            {
                "image_text": w_it,
                "telemetry": w_tel,
                "val_macro_f1": best_val_f1,
                "fully_developed_cases": len(cases),
                "train_cases": len(train_cases),
                "validation_cases": len(val_cases),
                "test_cases": len(test_cases),
            },
            f,
            indent=2,
        )

    print(
        f"\n[hybrid_fusion] done. "
        f"Metrics, plots, models and fusion weights saved to: {args.out_dir}"
    )


if __name__ == "__main__":
    main()
