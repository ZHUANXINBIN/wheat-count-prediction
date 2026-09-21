import os
import sys
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import (WheatDataset, G_CONFIGS, load_norm_stats,
                     load_split_and_labels, preload_all_channels)
from model import build_model, count_parameters
from trainer import train_one_fold

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOM_DIR     = os.path.join(BASE_DIR, "0_Data", "1_DOM")
INDICES_DIR = os.path.join(BASE_DIR, "0_Data", "indices")
NORM_STATS  = os.path.join(BASE_DIR, "0_Data", "norm_stats.json")
SPLIT_CSV   = os.path.join(BASE_DIR, "0_Data", "dataset_split_index.csv")
LABEL_CSV   = os.path.join(BASE_DIR, "0_Data", "wheat_count_labels.csv")
OUT_DIR     = os.path.join(BASE_DIR, "1_G_ablation", "results")

N_FOLDS      = 5
MAX_EPOCHS   = 150
PATIENCE     = 20
LR           = 1e-4
WEIGHT_DECAY = 1e-4
BATCH_SIZE   = 32
SEED         = 42

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    start_total = time.time()

    print("=" * 65)
    print("G-Group Ablation Study - Input Channel Comparison")
    print(f"Device : {DEVICE}")
    print(f"Folds  : {N_FOLDS}   MaxEpochs: {MAX_EPOCHS}   Patience: {PATIENCE}")
    print(f"LR     : {LR}        BatchSize: {BATCH_SIZE}")
    print("=" * 65)

    norm_stats                      = load_norm_stats(NORM_STATS)
    train_stems, test_stems, labels = load_split_and_labels(SPLIT_CSV, LABEL_CSV)

    all_stems = train_stems + test_stems
    preload_all_channels(all_stems, DOM_DIR, INDICES_DIR, verbose=True)

    train_stems = np.array(train_stems)

    print(f"\nTrainVal samples : {len(train_stems)}")
    print(f"Test samples     : {len(test_stems)} (sealed, not used here)")

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_indices = list(kf.split(train_stems))
    print(f"\n5-Fold CV indices generated (seed={SEED}), shared across all G-groups.\n")

    all_results = []
    summary     = []

    for g_name, ch_list in G_CONFIGS.items():
        n_ch   = len(ch_list)
        params = count_parameters(build_model(in_channels=n_ch, pretrained=False))
        start_g = time.time()

        print("-" * 65)
        print(f"[{g_name}]  Channels={n_ch}  ({', '.join(ch_list)})")
        print(f"         Params={params/1e6:.1f}M")
        print("-" * 65)

        fold_r2s   = []
        fold_rmses = []

        for fold_id, (tr_idx, val_idx) in enumerate(fold_indices, start=1):
            fold_start = time.time()

            tr_stems  = train_stems[tr_idx].tolist()
            val_stems = train_stems[val_idx].tolist()

            train_ds = WheatDataset(tr_stems,  labels, ch_list, norm_stats, augment=True)
            val_ds   = WheatDataset(val_stems, labels, ch_list, norm_stats, augment=False)

            train_loader = DataLoader(
                train_ds, batch_size=BATCH_SIZE, shuffle=True,
                num_workers=0, pin_memory=True,
            )
            val_loader = DataLoader(
                val_ds, batch_size=BATCH_SIZE, shuffle=False,
                num_workers=0, pin_memory=True,
            )

            model = build_model(in_channels=n_ch, pretrained=True)

            best_r2, best_rmse = train_one_fold(
                model        = model,
                train_loader = train_loader,
                val_loader   = val_loader,
                device       = DEVICE,
                max_epochs   = MAX_EPOCHS,
                lr           = LR,
                weight_decay = WEIGHT_DECAY,
                patience     = PATIENCE,
                verbose      = True,
                fold_id      = fold_id,
            )

            fold_r2s.append(best_r2)
            fold_rmses.append(best_rmse)
            fold_time = time.time() - fold_start

            all_results.append({
                "group":    g_name,
                "n_ch":     n_ch,
                "fold":     fold_id,
                "val_r2":   round(best_r2,   4),
                "val_rmse": round(best_rmse,  2),
                "time_s":   round(fold_time,  1),
            })

            print(f"  --> [{g_name}] Fold {fold_id} done: "
                  f"R2={best_r2:.4f}, RMSE={best_rmse:.2f}, "
                  f"time={fold_time/60:.1f}min\n")

        mean_r2   = float(np.mean(fold_r2s))
        std_r2    = float(np.std(fold_r2s))
        mean_rmse = float(np.mean(fold_rmses))
        std_rmse  = float(np.std(fold_rmses))
        g_time    = time.time() - start_g

        summary.append({
            "group":      g_name,
            "n_channels": n_ch,
            "channels":   " + ".join(ch_list),
            "mean_r2":    round(mean_r2,   4),
            "std_r2":     round(std_r2,    4),
            "mean_rmse":  round(mean_rmse,  2),
            "std_rmse":   round(std_rmse,   2),
            "params_M":   round(params/1e6, 1),
            "time_min":   round(g_time/60,  1),
        })

        print(f"  ==> [{g_name}] Summary: "
              f"R2={mean_r2:.4f}+/-{std_r2:.4f}, "
              f"RMSE={mean_rmse:.2f}+/-{std_rmse:.2f}, "
              f"total={g_time/60:.1f}min\n")

    total_time = time.time() - start_total

    df_all     = pd.DataFrame(all_results)
    df_summary = pd.DataFrame(summary)
    df_summary = df_summary.sort_values("mean_r2", ascending=False).reset_index(drop=True)
    best_group = df_summary.iloc[0]["group"]

    df_all.to_csv(    os.path.join(OUT_DIR, "g_ablation_all_folds.csv"), index=False, encoding="utf-8-sig")
    df_summary.to_csv(os.path.join(OUT_DIR, "g_ablation_summary.csv"),   index=False, encoding="utf-8-sig")

    _plot_bar_chart(df_summary, best_group, OUT_DIR)
    _write_report(df_summary, df_all, best_group, total_time, OUT_DIR)

    best_r2_val = df_summary.iloc[0]["mean_r2"]
    best_std_val = df_summary.iloc[0]["std_r2"]

    print("=" * 65)
    print(f"All experiments completed. Total time: {total_time/3600:.2f}h")
    print(f"Best group: {best_group}  (R2={best_r2_val:.4f}+/-{best_std_val:.4f})")
    print(f"Results saved to: {OUT_DIR}")
    print("=" * 65)


def _plot_bar_chart(df_summary, best_group, out_dir):
    order   = ["G0", "G1", "G2", "G3", "G4", "G5", "G6"]
    df_plot = df_summary.set_index("group").reindex(order).reset_index()

    groups = df_plot["group"].tolist()
    means  = df_plot["mean_r2"].tolist()
    stds   = df_plot["std_r2"].tolist()
    n_chs  = df_plot["n_channels"].tolist()

    colors = ["#2ecc71" if g == best_group else "#3498db" for g in groups]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(
        groups, means, yerr=stds,
        color=colors, edgecolor="black", linewidth=0.8,
        capsize=6, error_kw={"elinewidth": 1.5, "ecolor": "#2c3e50"},
        width=0.6, zorder=3,
    )

    for bar, mean, std, n_ch in zip(bars, means, stds, n_chs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + 0.005,
            f"R2={mean:.4f}\n(C={n_ch})",
            ha="center", va="bottom", fontsize=8.5, color="#2c3e50"
        )

    best_idx = groups.index(best_group)
    ax.text(
        bars[best_idx].get_x() + bars[best_idx].get_width() / 2,
        0.01, "BEST",
        ha="center", va="bottom", fontsize=9,
        fontweight="bold", color="white"
    )

    ax.set_xlabel("Input Channel Group", fontsize=12)
    ax.set_ylabel("Val R2 (5-Fold CV Mean +/- Std)", fontsize=12)
    ax.set_title(
        "G-Group Ablation: Input Channel Combination vs Validation R2\n"
        "(ResNet-50, 5-Fold CV, Fixed Seed=42)",
        fontsize=12, fontweight="bold"
    )
    ax.set_ylim(min(0, min(means) - max(stds) - 0.05),
                max(means) + max(stds) + 0.08)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#2ecc71", edgecolor="black", label=f"Best Group ({best_group})"),
        Patch(facecolor="#3498db", edgecolor="black", label="Other Groups"),
    ], loc="lower right", fontsize=10)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "g_ablation_chart.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Chart saved: {out_path}")


def _write_report(df_summary, df_all, best_group, total_time, out_dir):
    report_path = os.path.join(out_dir, "g_ablation_report.txt")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("G-GROUP ABLATION STUDY REPORT\n")
        f.write("Input Channel Combination vs Validation Performance\n")
        f.write("=" * 70 + "\n\n")

        f.write("[ EXPERIMENT CONFIGURATION ]\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Model          : ResNet-50 (CNN-MLP regression head)\n")
        f.write(f"  CV Strategy    : {N_FOLDS}-Fold Cross Validation (seed={SEED})\n")
        f.write(f"  TrainVal Size  : 414 samples\n")
        f.write(f"  Max Epochs     : {MAX_EPOCHS}\n")
        f.write(f"  Early Stopping : patience={PATIENCE}\n")
        f.write(f"  Optimizer      : AdamW (lr={LR}, wd={WEIGHT_DECAY})\n")
        f.write(f"  Scheduler      : CosineAnnealingLR\n")
        f.write(f"  Augmentation   : A1 (HFlip + VFlip + Rot90)\n")
        f.write(f"  Batch Size     : {BATCH_SIZE}\n")
        f.write(f"  Data Loading   : Full RAM preload (zero disk I/O during training)\n")
        f.write(f"  Total Time     : {total_time/3600:.2f} hours\n\n")

        f.write("[ G-GROUP DEFINITIONS ]\n")
        f.write("-" * 40 + "\n")
        g_desc = {
            "G0": "7 original bands (pure baseline)",
            "G1": "7 bands + NDRE + CIre (red-edge indices)",
            "G2": "7 bands + CIgreen + GRVI (green indices)",
            "G3": "7 bands + NDVI + GNDVI + NDRE (three-mechanism combination)",
            "G4": "7 bands + NDVI + NDRE + CIre (red-edge enhanced)",
            "G5": "7 bands + GNDVI + CIgreen + GRVI (ML Top-3 guided)",
            "G6": "7 bands + all 6 indices (upper bound test)",
        }
        for g, desc in g_desc.items():
            row = df_summary[df_summary["group"] == g]
            if not row.empty:
                n_ch = int(row.iloc[0]["n_channels"])
                f.write(f"  {g} (C={n_ch:2d}): {desc}\n")
        f.write("\n")

        f.write("[ SUMMARY RESULTS (sorted by Mean Val R2) ]\n")
        f.write("-" * 70 + "\n")
        header_group = "Group"
        header_c = "C"
        header_mean_r2 = "Mean R2"
        header_std_r2 = "Std R2"
        header_mean_rmse = "Mean RMSE"
        header_std_rmse = "Std RMSE"
        header_time = "Time(min)"
        f.write(f"  {header_group:<6} {header_c:>3}  {header_mean_r2:>9}  {header_std_r2:>8}  "
                f"{header_mean_rmse:>10}  {header_std_rmse:>9}  {header_time:>10}\n")
        f.write("  " + "-" * 66 + "\n")
        for _, row in df_summary.iterrows():
            mark = " <-- BEST" if row["group"] == best_group else ""
            group_val = row["group"]
            n_channels_val = int(row["n_channels"])
            mean_r2_val = row["mean_r2"]
            std_r2_val = row["std_r2"]
            mean_rmse_val = row["mean_rmse"]
            std_rmse_val = row["std_rmse"]
            time_min_val = row["time_min"]
            f.write(
                f"  {group_val:<6} {n_channels_val:>3}  "
                f"{mean_r2_val:>9.4f}  {std_r2_val:>8.4f}  "
                f"{mean_rmse_val:>10.2f}  {std_rmse_val:>9.2f}  "
                f"{time_min_val:>10.1f}{mark}\n"
            )
        f.write("\n")

        f.write("[ PER-FOLD DETAILED RESULTS ]\n")
        f.write("-" * 70 + "\n")
        for g_name in ["G0", "G1", "G2", "G3", "G4", "G5", "G6"]:
            rows = df_all[df_all["group"] == g_name]
            if rows.empty:
                continue
            f.write(f"\n  {g_name}:\n")
            header_fold = "Fold"
            header_val_r2 = "Val R2"
            header_val_rmse = "Val RMSE"
            header_time_s = "Time(s)"
            f.write(f"    {header_fold:>5}  {header_val_r2:>9}  {header_val_rmse:>10}  {header_time_s:>9}\n")
            f.write("    " + "-" * 38 + "\n")
            for _, r in rows.iterrows():
                fold_val = int(r["fold"])
                val_r2_val = r["val_r2"]
                val_rmse_val = r["val_rmse"]
                time_s_val = r["time_s"]
                f.write(f"    {fold_val:>5}  {val_r2_val:>9.4f}  "
                        f"{val_rmse_val:>10.2f}  {time_s_val:>9.1f}\n")
            r2s = rows["val_r2"].tolist()
            mean_label = "Mean"
            std_label = "Std"
            f.write(f"    {mean_label:>5}  {np.mean(r2s):>9.4f}  {np.mean(rows['val_rmse']):>10.2f}\n")
            f.write(f"    {std_label:>5}  {np.std(r2s):>9.4f}  {np.std(rows['val_rmse']):>10.2f}\n")

        best_row  = df_summary[df_summary["group"] == best_group].iloc[0]
        worst_row = df_summary.iloc[-1]
        g0_r2     = df_summary[df_summary["group"] == "G0"]["mean_r2"].values[0]

        f.write("\n\n[ CONCLUSION ]\n")
        f.write("-" * 70 + "\n")
        best_group_ch = int(best_row["n_channels"])
        best_mean_r2 = best_row["mean_r2"]
        best_std_r2 = best_row["std_r2"]
        best_mean_rmse = best_row["mean_rmse"]
        best_std_rmse = best_row["std_rmse"]
        worst_group_name = worst_row["group"]
        worst_group_ch = int(worst_row["n_channels"])
        worst_mean_r2 = worst_row["mean_r2"]
        worst_std_r2 = worst_row["std_r2"]

        f.write(
            f"  Best group   : {best_group} (C={best_group_ch})\n"
            f"  Best R2      : {best_mean_r2:.4f} +/- {best_std_r2:.4f}\n"
            f"  Best RMSE    : {best_mean_rmse:.2f} +/- {best_std_rmse:.2f}\n\n"
            f"  Worst group  : {worst_group_name} (C={worst_group_ch})\n"
            f"  Worst R2     : {worst_mean_r2:.4f} +/- {worst_std_r2:.4f}\n\n"
            f"  R2 gain over G0 (baseline):\n"
        )
        for _, row in df_summary.iterrows():
            delta = row["mean_r2"] - g0_r2
            sign  = "+" if delta >= 0 else ""
            f.write(f"    {row['group']}: {sign}{delta:.4f}\n")

        f.write(
            f"\n  Recommended next step:\n"
            f"    Use {best_group} as the fixed input configuration (G_BEST)\n"
            f"    for all subsequent architecture comparison experiments.\n"
        )
        f.write("\n" + "=" * 70 + "\n")

    print(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()