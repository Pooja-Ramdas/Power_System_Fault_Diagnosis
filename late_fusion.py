"""
late_fusion.py
===============
LATE FUSION for the CEN multimodal power-system fault diagnosis project.

Definition used here (and why it is correct "late" fusion, not early or
hybrid): three COMPLETELY INDEPENDENT unimodal classifiers are trained --
one on image features only, one on text features only, one on telemetry
only -- each with its OWN loss, its OWN optimizer, its OWN training loop.
None of them ever sees another modality's representation during training.
Only AFTER each is fully trained do we combine their independent
predictions (softmax probability vectors) at the decision level, via a
weighted average whose weights are tuned on the validation set. This also
gives you, for free, the per-modality ablation results the project report
asks for (Ch. 4.3: "train and test ... telemetry-only ...").

Run:
    python late_fusion.py --dataset-root cen_dataset --out-dir results/late_fusion
"""

import argparse
import itertools
import os

import numpy as np

import dataset_utils as du


def build_unimodal_classifier(input_kind: str, image_dim=None, text_dim=None,
                               telemetry_channels=None, n_classes=4, d_model=128):
    import torch.nn as nn

    class ImageOnly(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = du.ProjectionHead(image_dim, d_model)
            self.head = du.build_classifier_head(d_model, n_classes)

        def forward(self, image_feat):
            return self.head(self.proj(image_feat))

    class TextOnly(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = du.ProjectionHead(text_dim, d_model)
            self.head = du.build_classifier_head(d_model, n_classes)

        def forward(self, text_feat):
            return self.head(self.proj(text_feat))

    class TelemetryOnly(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = du.build_tcn(telemetry_channels, d_model)
            self.head = du.build_classifier_head(d_model, n_classes)

        def forward(self, telemetry_window, telemetry_mask):
            return self.head(self.encoder(telemetry_window, telemetry_mask))

    return {"image": ImageOnly, "text": TextOnly, "telemetry": TelemetryOnly}[input_kind]()


def _search_best_weights(probs_list, y_true, n_steps: int = 11):
    """Grid-search convex-combination weights (image, text, telemetry) that
    maximise macro-F1 on the validation set. n_steps=11 -> weights in
    increments of 0.1; coarse but fast and sufficient for 3 modalities."""
    from sklearn.metrics import f1_score

    best_f1, best_weights = -1.0, (1 / 3, 1 / 3, 1 / 3)
    grid = np.linspace(0, 1, n_steps)
    for wi in grid:
        for wt in grid:
            wl = 1.0 - wi - wt
            if wl < -1e-9 or wl > 1.0 + 1e-9:
                continue
            wl = max(0.0, wl)
            combined = wi * probs_list[0] + wt * probs_list[1] + wl * probs_list[2]
            preds = combined.argmax(axis=1)
            f1 = f1_score(y_true, preds, average="macro", zero_division=0)
            if f1 > best_f1:
                best_f1, best_weights = f1, (wi, wt, wl)
    return best_weights, best_f1


def main():
    parser = argparse.ArgumentParser(description="Late fusion (decision-level) fault classifier")
    parser.add_argument("--dataset-root", default=du.DATASET_ROOT)
    parser.add_argument("--out-dir", default="results/late_fusion")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--use-transformer-text", action="store_true")
    parser.add_argument("--force-rescan", action="store_true")
    args = parser.parse_args()

    du.set_seed()
    import torch
    from torch.utils.data import TensorDataset, DataLoader

    cases = du.scan_dataset(dataset_root=args.dataset_root, force_rescan=args.force_rescan)
    if len(cases) < 20:
        raise SystemExit(f"Only {len(cases)} fully-developed cases found -- need at least ~20.")

    train_cases, val_cases, test_cases = du.stratified_case_split(cases)
    weights = du.class_weights(train_cases)
    class_to_idx = {c: i for i, c in enumerate(du.FAULT_CLASSES)}

    # ---- shared feature extraction (identical to early_fusion.py, for a fair comparison) ----
    img_feats_all = du.get_image_features(cases)
    case_index = {c.case_id: i for i, c in enumerate(cases)}
    img_subset = lambda subset: img_feats_all[[case_index[c.case_id] for c in subset]]  # noqa: E731

    text_featurizer = du.TextFeaturizer(use_pretrained_transformer=args.use_transformer_text)
    train_text = text_featurizer.fit_transform(du.load_texts(train_cases))
    val_text = text_featurizer.transform(du.load_texts(val_cases))
    test_text = text_featurizer.transform(du.load_texts(test_cases))

    tel_windows_all, tel_masks_all, n_channels = du.load_telemetry_windows(cases)
    tel_subset = lambda subset: (  # noqa: E731
        tel_windows_all[[case_index[c.case_id] for c in subset]],
        tel_masks_all[[case_index[c.case_id] for c in subset]])
    train_tel_w, train_tel_m = tel_subset(train_cases)
    val_tel_w, val_tel_m = tel_subset(val_cases)
    test_tel_w, test_tel_m = tel_subset(test_cases)
    scaler = du.fit_telemetry_scaler(train_tel_w, train_tel_m)
    train_tel_w = du.apply_telemetry_scaler(train_tel_w, scaler)
    val_tel_w = du.apply_telemetry_scaler(val_tel_w, scaler)
    test_tel_w = du.apply_telemetry_scaler(test_tel_w, scaler)

    def y_tensor(subset):
        return torch.tensor([class_to_idx[c.label] for c in subset], dtype=torch.long)

    train_y, val_y, test_y = y_tensor(train_cases), y_tensor(val_cases), y_tensor(test_cases)

    os.makedirs(args.out_dir, exist_ok=True)
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))

    modality_probs = {}   # name -> (val_probs, test_probs)

    # ---------------------------------------------------------------
    # helper to train ONE independent unimodal branch end-to-end
    # ---------------------------------------------------------------
    def train_branch(name, model, train_tensors, val_tensors, test_tensors, input_keys):
        train_loader = DataLoader(TensorDataset(*train_tensors, train_y),
                                   batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(*val_tensors, val_y), batch_size=args.batch_size)
        test_loader = DataLoader(TensorDataset(*test_tensors, test_y), batch_size=args.batch_size)

        def batches(loader):
            def _gen():
                for *xs, y in loader:
                    yield dict(zip(input_keys, xs)), y
            return _gen

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        print(f"\n[late_fusion] training INDEPENDENT '{name}' branch (own loss, own optimizer)...")
        history = du.train_with_early_stopping(
            model, optimizer, loss_fn, batches(train_loader), batches(val_loader),
            max_epochs=args.epochs, patience=args.patience, verbose_name=f"late_fusion/{name}")
        du.plot_training_curves(history, args.out_dir, f"late_fusion_{name}_only")

        model.eval()
        def get_probs(loader):
            all_logits, all_true = [], []
            with torch.no_grad():
                for inputs, y in batches(loader)():
                    all_logits.append(model(**inputs))
                    all_true.extend(y.tolist())
            probs = torch.softmax(torch.cat(all_logits, dim=0), dim=1).numpy()
            return probs, all_true

        val_probs, _ = get_probs(val_loader)
        test_probs, test_true = get_probs(test_loader)

        # per-modality ablation result (matches report's planned unimodal ablations)
        du.evaluate_and_plot(test_true, test_probs.argmax(axis=1), test_probs,
                              du.FAULT_CLASSES, args.out_dir, f"late_fusion_{name}_only")
        return val_probs, test_probs, test_true

    image_model = build_unimodal_classifier("image", image_dim=img_feats_all.shape[1],
                                             n_classes=len(du.FAULT_CLASSES))
    val_p_img, test_p_img, test_true = train_branch(
        "image", image_model, (torch.tensor(img_subset(train_cases), dtype=torch.float32),),
        (torch.tensor(img_subset(val_cases), dtype=torch.float32),),
        (torch.tensor(img_subset(test_cases), dtype=torch.float32),), ["image_feat"])

    text_model = build_unimodal_classifier("text", text_dim=train_text.shape[1],
                                            n_classes=len(du.FAULT_CLASSES))
    val_p_txt, test_p_txt, _ = train_branch(
        "text", text_model, (torch.tensor(train_text, dtype=torch.float32),),
        (torch.tensor(val_text, dtype=torch.float32),),
        (torch.tensor(test_text, dtype=torch.float32),), ["text_feat"])

    telemetry_model = build_unimodal_classifier("telemetry", telemetry_channels=n_channels,
                                                 n_classes=len(du.FAULT_CLASSES))
    val_p_tel, test_p_tel, _ = train_branch(
        "telemetry", telemetry_model,
        (torch.tensor(train_tel_w, dtype=torch.float32), torch.tensor(train_tel_m, dtype=torch.float32)),
        (torch.tensor(val_tel_w, dtype=torch.float32), torch.tensor(val_tel_m, dtype=torch.float32)),
        (torch.tensor(test_tel_w, dtype=torch.float32), torch.tensor(test_tel_m, dtype=torch.float32)),
        ["telemetry_window", "telemetry_mask"])

    # ---- decision-level combination, weights tuned on the VALIDATION set ----
    val_true = val_y.tolist()
    best_weights, best_val_f1 = _search_best_weights([val_p_img, val_p_txt, val_p_tel], val_true)
    wi, wt, wl = best_weights
    print(f"\n[late_fusion] best validation weighted-average weights: "
          f"image={wi:.2f} text={wt:.2f} telemetry={wl:.2f}  (val macro_f1={best_val_f1:.4f})")

    combined_test_probs = wi * test_p_img + wt * test_p_txt + wl * test_p_tel
    combined_preds = combined_test_probs.argmax(axis=1)

    du.evaluate_and_plot(test_true, combined_preds, combined_test_probs,
                          du.FAULT_CLASSES, args.out_dir, "late_fusion_combined")

    # majority-vote as a simple, parameter-free sanity check / extra comparison point
    from scipy import stats  # scipy ships with scikit-learn's deps
    votes = np.stack([test_p_img.argmax(1), test_p_txt.argmax(1), test_p_tel.argmax(1)], axis=1)
    majority_preds = stats.mode(votes, axis=1, keepdims=False).mode
    du.evaluate_and_plot(test_true, majority_preds, None,
                          du.FAULT_CLASSES, args.out_dir, "late_fusion_majority_vote")

    torch.save(image_model.state_dict(), os.path.join(args.out_dir, "late_fusion_image_model.pt"))
    torch.save(text_model.state_dict(), os.path.join(args.out_dir, "late_fusion_text_model.pt"))
    torch.save(telemetry_model.state_dict(), os.path.join(args.out_dir, "late_fusion_telemetry_model.pt"))

    import json
    with open(os.path.join(args.out_dir, "late_fusion_combination_weights.json"), "w") as f:
        json.dump({"image": wi, "text": wt, "telemetry": wl, "val_macro_f1": best_val_f1}, f, indent=2)

    print(f"\n[late_fusion] done. Per-modality ablations + combined result saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
