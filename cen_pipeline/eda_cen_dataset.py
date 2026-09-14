"""
Exploratory Data Analysis - CEN Multimodal Fault Diagnosis Dataset
Group 13 | Multimodal Reasoning for Power-System Fault Diagnosis

Run this against the `cen_dataset/` folder produced by the pipeline
(build_dataset.py). It walks every CASE-XXXX subfolder it finds, so it
works unchanged whether you currently have 100 cases or the full set
later on.

Expected per-case structure:
    cen_dataset/
        CASE-0001/
            image.png
            telemetry.csv
            report_es.txt
            report_en.txt
            meta.json
        CASE-0002/
            ...
        manifest.csv   (optional - used if present, not required)

Usage:
    python eda_cen_dataset.py --data_dir ./cen_dataset --out_dir ./eda_outputs

Outputs 9 figures (fig1_...png ... fig9_...png) into out_dir, plus a
printed console summary of key statistics for each modality.
"""

import argparse
import json
import os
import re
from collections import Counter
from glob import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

sns.set_theme(style="whitegrid")

STOPWORDS = set("""
the a an of to in on at for and or is was were are be been being with
by from as that this these those it its it's their his her they he she
we you your our not no de la el en y a que se por con para del las los
un una al lo su como más pero si ya fue son al""".split())


def find_cases(data_dir):
    cases = sorted(glob(os.path.join(data_dir, "CASE-*")))
    cases = [c for c in cases if os.path.isdir(c)]
    return cases


def load_meta(case_dir):
    path = os.path.join(case_dir, "meta.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def guess_fault_label(meta):
    if not meta:
        return None
    for key in ("fault_type", "fault_type_label", "label", "fault_class", "tipo_falla"):
        if key in meta and meta[key]:
            return str(meta[key])
    return None


def guess_timestamp(meta):
    if not meta:
        return None
    for key in ("fault_timestamp", "timestamp", "event_timestamp", "fecha"):
        if key in meta and meta[key]:
            return meta[key]
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./cen_dataset")
    parser.add_argument("--out_dir", default="./eda_outputs")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cases = find_cases(args.data_dir)
    n_cases = len(cases)
    print(f"Found {n_cases} case folders under {args.data_dir}\n")
    if n_cases == 0:
        print("No CASE-* folders found. Check --data_dir and rerun.")
        return

    # ------------------------------------------------------------------
    # Fig 1: Modality completeness per case
    # ------------------------------------------------------------------
    files_to_check = {
        "image.png": "Image",
        "telemetry.csv": "Telemetry",
        "report_en.txt": "Report (EN)",
        "meta.json": "Metadata",
    }
    presence_counts = {label: 0 for label in files_to_check.values()}
    for c in cases:
        for fname, label in files_to_check.items():
            if os.path.exists(os.path.join(c, fname)):
                presence_counts[label] += 1

    plt.figure(figsize=(7, 5))
    labels = list(presence_counts.keys())
    values = list(presence_counts.values())
    bars = plt.bar(labels, values, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"])
    plt.ylim(0, n_cases + max(2, int(n_cases * 0.1)))
    plt.ylabel("Number of cases with file present")
    plt.title(f"Modality Completeness Across {n_cases} Processed Cases")
    for b, v in zip(bars, values):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.5, str(v), ha="center")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "fig1_modality_completeness.png"), dpi=150)
    plt.close()
    print("Fig 1 - Modality completeness:", presence_counts)

    # ------------------------------------------------------------------
    # Fig 2: Fault type / label distribution
    # ------------------------------------------------------------------
    labels_list = []
    timestamps = []
    for c in cases:
        meta = load_meta(c)
        lbl = guess_fault_label(meta)
        if lbl:
            labels_list.append(lbl)
        ts = guess_timestamp(meta)
        if ts:
            timestamps.append(ts)

    if labels_list:
        label_counts = Counter(labels_list)
        plt.figure(figsize=(8, 5))
        keys = list(label_counts.keys())
        vals = [label_counts[k] for k in keys]
        sns.barplot(x=vals, y=keys, orient="h", palette="viridis")
        plt.xlabel("Number of cases")
        plt.title("Fault Type / Label Distribution")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "fig2_fault_label_distribution.png"), dpi=150)
        plt.close()
        print("Fig 2 - Fault label counts:", dict(label_counts))
    else:
        print("Fig 2 - Skipped: no fault-type field found in meta.json files "
              "(update guess_fault_label() with your actual key name).")

    # ------------------------------------------------------------------
    # Load telemetry into one long dataframe for figs 3-5
    # ------------------------------------------------------------------
    telemetry_frames = []
    for c in cases:
        tpath = os.path.join(c, "telemetry.csv")
        if os.path.exists(tpath):
            try:
                df = pd.read_csv(tpath)
                df["case_id"] = os.path.basename(c)
                telemetry_frames.append(df)
            except Exception as e:
                print(f"  Warning: could not read {tpath}: {e}")

    if telemetry_frames:
        telemetry_all = pd.concat(telemetry_frames, ignore_index=True, sort=False)
        numeric_cols = telemetry_all.select_dtypes(include="number").columns.tolist()
    else:
        telemetry_all = pd.DataFrame()
        numeric_cols = []

    # ------------------------------------------------------------------
    # Fig 3: Missing-value heatmap across telemetry columns
    # ------------------------------------------------------------------
    if not telemetry_all.empty:
        missing_by_case = (
            telemetry_all.drop(columns=["case_id"])
            .isna()
            .groupby(telemetry_all["case_id"])
            .mean()
        )
        plt.figure(figsize=(10, max(4, len(missing_by_case) * 0.12)))
        sns.heatmap(missing_by_case, cmap="rocket_r", cbar_kws={"label": "Fraction missing"})
        plt.title("Missing-Value Fraction by Case and Telemetry Column")
        plt.xlabel("Telemetry column")
        plt.ylabel("Case")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "fig3_telemetry_missingness.png"), dpi=150)
        plt.close()
        print("Fig 3 - Telemetry missingness heatmap saved. Columns:", numeric_cols)
        print("Telemetry summary stats:\n", telemetry_all[numeric_cols].describe())
    else:
        print("Fig 3 - Skipped: no telemetry.csv files could be loaded.")

    # ------------------------------------------------------------------
    # Fig 4: Sample time-series window for one representative case
    # ------------------------------------------------------------------
    if telemetry_frames:
        sample_df = telemetry_frames[0]
        time_col = None
        for cand in ("timestamp", "datetime", "hour", "fecha_hora", "date"):
            if cand in sample_df.columns:
                time_col = cand
                break
        value_col = None
        for cand in numeric_cols:
            if cand not in ("case_id",):
                value_col = cand
                break
        if value_col:
            plt.figure(figsize=(9, 4))
            x = sample_df[time_col] if time_col else range(len(sample_df))
            plt.plot(x, sample_df[value_col], marker="o", linewidth=1.2, markersize=3)
            plt.xticks(rotation=45, ha="right")
            plt.xlabel(time_col if time_col else "Sample index")
            plt.ylabel(value_col)
            plt.title(f"Sample Telemetry Window - {os.path.basename(cases[0])} ({value_col})")
            plt.tight_layout()
            plt.savefig(os.path.join(args.out_dir, "fig4_sample_telemetry_window.png"), dpi=150)
            plt.close()
            print(f"Fig 4 - Sample window plotted for column '{value_col}'.")
        else:
            print("Fig 4 - Skipped: no numeric telemetry column found to plot.")
    else:
        print("Fig 4 - Skipped: no telemetry available.")

    # ------------------------------------------------------------------
    # Fig 5: Telemetry feature correlation heatmap
    # ------------------------------------------------------------------
    if not telemetry_all.empty and len(numeric_cols) >= 2:
        corr = telemetry_all[numeric_cols].corr()
        plt.figure(figsize=(max(6, len(numeric_cols) * 0.8), max(5, len(numeric_cols) * 0.7)))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True)
        plt.title("Telemetry Feature Correlation Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "fig5_telemetry_correlation.png"), dpi=150)
        plt.close()
        print("Fig 5 - Correlation matrix saved.")
    else:
        print("Fig 5 - Skipped: fewer than 2 numeric telemetry columns found.")

    # ------------------------------------------------------------------
    # Fig 6: Report text length distribution
    # ------------------------------------------------------------------
    word_counts = []
    all_words = []
    for c in cases:
        rpath = os.path.join(c, "report_en.txt")
        if os.path.exists(rpath):
            with open(rpath, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            words = re.findall(r"[A-Za-z']+", text.lower())
            word_counts.append(len(words))
            all_words.extend(w for w in words if w not in STOPWORDS and len(w) > 2)

    if word_counts:
        plt.figure(figsize=(7, 5))
        plt.hist(word_counts, bins=min(20, max(5, n_cases // 4)), color="#4C72B0", edgecolor="white")
        plt.xlabel("Word count per translated report")
        plt.ylabel("Number of cases")
        plt.title("Report Text Length Distribution (report_en.txt)")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "fig6_report_length_distribution.png"), dpi=150)
        plt.close()
        print(f"Fig 6 - Report length stats: min={min(word_counts)}, "
              f"max={max(word_counts)}, mean={sum(word_counts)/len(word_counts):.1f}")
    else:
        print("Fig 6 - Skipped: no report_en.txt files found.")

    # ------------------------------------------------------------------
    # Fig 7: Top word frequency across reports
    # ------------------------------------------------------------------
    if all_words:
        top_words = Counter(all_words).most_common(20)
        words, counts = zip(*top_words)
        plt.figure(figsize=(8, 7))
        sns.barplot(x=list(counts), y=list(words), palette="crest")
        plt.xlabel("Frequency")
        plt.title("Top 20 Words Across Translated Operator Reports")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "fig7_report_word_frequency.png"), dpi=150)
        plt.close()
        print("Fig 7 - Top words:", top_words[:10])
    else:
        print("Fig 7 - Skipped: no words extracted from reports.")

    # ------------------------------------------------------------------
    # Fig 8: Image resolution and aspect ratio distribution
    # ------------------------------------------------------------------
    if HAS_PIL:
        widths, heights = [], []
        for c in cases:
            ipath = os.path.join(c, "image.png")
            if os.path.exists(ipath):
                try:
                    with Image.open(ipath) as img:
                        w, h = img.size
                        widths.append(w)
                        heights.append(h)
                except Exception as e:
                    print(f"  Warning: could not read image {ipath}: {e}")

        if widths:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes[0].scatter(widths, heights, alpha=0.6, color="#C44E52")
            axes[0].set_xlabel("Width (px)")
            axes[0].set_ylabel("Height (px)")
            axes[0].set_title("Image Resolution Scatter")

            aspect_ratios = [w / h for w, h in zip(widths, heights)]
            axes[1].hist(aspect_ratios, bins=15, color="#55A868", edgecolor="white")
            axes[1].set_xlabel("Aspect ratio (width / height)")
            axes[1].set_ylabel("Number of images")
            axes[1].set_title("Image Aspect Ratio Distribution")

            plt.tight_layout()
            plt.savefig(os.path.join(args.out_dir, "fig8_image_resolution_aspect_ratio.png"), dpi=150)
            plt.close()
            print(f"Fig 8 - Images analyzed: {len(widths)}. "
                  f"Width range: {min(widths)}-{max(widths)}px, "
                  f"Height range: {min(heights)}-{max(heights)}px")
        else:
            print("Fig 8 - Skipped: no readable image.png files found.")
    else:
        print("Fig 8 - Skipped: Pillow (PIL) not installed. "
              "Install with: pip install pillow --break-system-packages")

    # ------------------------------------------------------------------
    # Fig 9: Event timestamp distribution
    # ------------------------------------------------------------------
    if timestamps:
        parsed = pd.to_datetime(pd.Series(timestamps), errors="coerce")
        parsed = parsed.dropna()
        if not parsed.empty:
            plt.figure(figsize=(9, 5))
            parsed.dt.to_period("M").value_counts().sort_index().plot(kind="bar", color="#8172B2")
            plt.xlabel("Month")
            plt.ylabel("Number of fault events")
            plt.title("Event Timestamp Distribution Across Processed Cases")
            plt.tight_layout()
            plt.savefig(os.path.join(args.out_dir, "fig9_event_timestamp_distribution.png"), dpi=150)
            plt.close()
            print(f"Fig 9 - Timestamps parsed: {len(parsed)} / {len(timestamps)}")
        else:
            print("Fig 9 - Skipped: timestamps found but none could be parsed as dates.")
    else:
        print("Fig 9 - Skipped: no timestamp field found in meta.json files "
              "(update guess_timestamp() with your actual key name).")

    print(f"\nAll available figures written to: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
