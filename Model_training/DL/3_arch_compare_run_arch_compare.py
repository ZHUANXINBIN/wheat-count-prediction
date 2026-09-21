import os
import sys
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "1_G_ablation"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import (G_CONFIGS, load_norm_stats, load_split_and_labels,
                     preload_all_channels, PRELOAD_CACHE, augment_a1)
from models import build_arch, count_parameters, ARCH_REGISTRY
from measure_inference import measure_all_archs

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOM_DIR     = os.path.join(BASE_DIR, "0_Data", "1_DOM")
INDICES_DIR = os.path.join(BASE_DIR, "0_Data", "indices")
NORM_STATS  = os.path.join(BASE_DIR, "0_Data", "norm_stats.json")
SPLIT_CSV   = os.path.join(BASE_DIR, "0_Data", "dataset_split_index.csv")
LABEL_CSV   = os.path.join(BASE_DIR, "0_Data", "wheat_count_labels.csv")
OUT_DIR     = os.path.join(BASE_DIR, "3_arch_compare", "results")

G_BEST      = "G3"
A_BEST_DESC = "A3: HFlip + VFlip + Rot90 + ChannelDropout(p=0.1)"

N_FOLDS      = 5
MAX_EPOCHS   = 150
PATIENCE     = 20
LR           = 1e-4
WEIGHT_DECAY = 1e-4
CLIP_NORM    = 5.0
BATCH_SIZE   = 32
SEED         = 42

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ARCH_ORDER = ["resnet50", "densenet121", "efficientnet_b0",
              "mobilenetv3_s", "convnext_tiny", "efficientvit_b0"]

SKIP_ARCHS = []
PREV_ALL_FOLDS_CSV = os.path.join(OUT_DIR, "arch_compare_all_folds.csv")


def augment_a3(image: np.ndarray) -> np.ndarray:
    image = augment_a1(image)
    if random.random() < 0.1:
        ch = random.randint(0, image.shape[0] - 1)
        image = image.copy()
        image[ch] = 0.0
    return image


class ArchDataset(Dataset):
    def __init__(self, stems, labels, channel_list, norm_stats, augment_fn=None):
        assert len(PRELOAD_CACHE) > 0, \
            "[ArchDataset] PRELOAD_CACHE is empty! Call preload_all_channels() first."

        self.augment_fn = augment_fn

        means = np.array([norm_stats[ch]["mean"] for ch in channel_list],
                         dtype=np.float32).reshape(-1, 1, 1)
        stds  = np.array([norm_stats[ch]["std"]  for ch in channel_list],
                         dtype=np.float32).reshape(-1, 1, 1)
        stds  = np.where(stds == 0, 1.0, stds)

        self.images     = []
        self.label_list = []
        for stem in stems:
            ch_arrays = [PRELOAD_CACHE[stem][ch] for ch in channel_list]
            image = np.stack(ch_arrays, axis=0)
            image = (image - means) / stds
            self.images.append(image)
            self.label_list.append(float(labels[stem]))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].copy()
        if self.augment_fn is not None:
            image = self.augment_fn(image)
        image_tensor = torch.from_numpy(np.ascontiguousarray(image))
        label_tensor = torch.tensor(self.label_list[idx], dtype=torch.float32)
        return image_tensor, label_tensor


def train_one_fold(model, train_loader, val_loader, device,
                   max_epochs, lr, weight_decay, patience,
                   clip_norm=5.0, verbose=True, fold_id=None):
    model     = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=lr * 1e-3)
    criterion = nn.MSELoss()

    best_r2        = -np.inf
    best_rmse      = np.inf
    no_improve_cnt = 0
    fold_str       = f"Fold {fold_id}" if fold_id else "Fold"

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses = []
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            preds = model(images)
            loss  = criterion(preds, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()

        model.eval()
        preds_all, labels_all = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                preds  = model(images)
                preds_all.append(preds.cpu().numpy())
                labels_all.append(labels.numpy())

        preds_all  = np.concatenate(preds_all)
        labels_all = np.concatenate(labels_all)
        val_r2     = r2_score(labels_all, preds_all)
        val_rmse   = float(np.sqrt(np.mean((preds_all - labels_all) ** 2)))

        if val_r2 > best_r2:
            best_r2        = val_r2
            best_rmse      = val_rmse
            no_improve_cnt = 0
        else:
            no_improve_cnt += 1

        if verbose:
            current_lr   = scheduler.get_last_lr()[0]
            improve_mark = "*" if no_improve_cnt == 0 else " "
            print(f"  [{fold_str}] Epoch {epoch:3d}/{max_epochs} | "
                  f"TrainLoss={np.mean(train_losses):8.1f} | "
                  f"ValR2={val_r2:.4f} {improve_mark} | "
                  f"ValRMSE={val_rmse:6.2f} | "
                  f"LR={current_lr:.2e} | "
                  f"NoImprove={no_improve_cnt}/{patience}")

        if no_improve_cnt >= patience:
            if verbose:
                print(f"  [{fold_str}] Early stopping at epoch {epoch}. "
                      f"Best R2={best_r2:.4f}, RMSE={best_rmse:.2f}")
            break

    if verbose and no_improve_cnt < patience:
        print(f"  [{fold_str}] Training completed. "
              f"Best R2={best_r2:.4f}, RMSE={best_rmse:.2f}")

    return best_r2, best_rmse


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    start_total = time.time()

    skip_display = SKIP_ARCHS if SKIP_ARCHS else "None (full rerun)"

    print("=" * 65)
    print("Architecture Comparison Experiment v3 (Train from Scratch)")
    print(f"Fixed Input    : {G_BEST} (10 channels)")
    print(f"Fixed Aug      : {A_BEST_DESC}")
    print(f"Device         : {DEVICE}")
    print(f"Folds          : {N_FOLDS}   MaxEpochs: {MAX_EPOCHS}   Patience: {PATIENCE}")
    print(f"LR             : {LR}   WD: {WEIGHT_DECAY}   ClipNorm: {CLIP_NORM}")
    print(f"BatchSize      : {BATCH_SIZE}   Pretrained: False (all from scratch)")
    print(f"Skip archs     : {skip_display}")
    print("=" * 65)

    norm_stats                      = load_norm_stats(NORM_STATS)
    train_stems, test_stems, labels = load_split_and_labels(SPLIT_CSV, LABEL_CSV)
    all_stems = train_stems + test_stems
    preload_all_channels(all_stems, DOM_DIR, INDICES_DIR, verbose=True)

    train_stems = np.array(train_stems)
    ch_list     = G_CONFIGS[G_BEST]

    print(f"\nTrainVal: {len(train_stems)} | Test: {len(test_stems)} (sealed) | "
          f"Channels: {len(ch_list)}\n")

    kf           = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_indices = list(kf.split(train_stems))
    print(f"5-Fold CV indices generated (seed={SEED}).\n")

    prev_results = []
    if SKIP_ARCHS and os.path.exists(PREV_ALL_FOLDS_CSV):
        df_prev = pd.read_csv(PREV_ALL_FOLDS_CSV)
        for arch_name in list(SKIP_ARCHS):
            rows = df_prev[df_prev["arch"] == arch_name]
            if not rows.empty:
                prev_results.extend(rows.to_dict("records"))
                print(f"[SKIP] {arch_name}: loaded {len(rows)} fold results from previous run.")
            else:
                print(f"[WARN] {arch_name} not found in previous results, will re-run.")
                SKIP_ARCHS.remove(arch_name)
        print()
    elif SKIP_ARCHS:
        print(f"[WARN] Previous results file not found. SKIP_ARCHS ignored, full rerun.\n")
        SKIP_ARCHS.clear()

    new_results = []
    summary     = []

    for arch_name in ARCH_ORDER:
        params = count_parameters(build_arch(arch_name, in_channels=len(ch_list)))

        if arch_name in SKIP_ARCHS:
            print("-" * 65)
            print(f"[{arch_name}]  SKIPPED (using previous results)")
            print("-" * 65)
            fold_rows  = [r for r in prev_results if r["arch"] == arch_name]
            fold_r2s   = [r["val_r2"]   for r in fold_rows]
            fold_rmses = [r["val_rmse"] for r in fold_rows]
            summary.append({
                "arch":      arch_name,
                "params_M":  round(params / 1e6, 2),
                "mean_r2":   round(float(np.mean(fold_r2s)),   4),
                "std_r2":    round(float(np.std(fold_r2s)),    4),
                "mean_rmse": round(float(np.mean(fold_rmses)), 2),
                "std_rmse":  round(float(np.std(fold_rmses)),  2),
                "time_min":  round(sum(r.get("time_s", 0) for r in fold_rows) / 60, 1),
            })
            mean_r2_skip = np.mean(fold_r2s)
            std_r2_skip = np.std(fold_r2s)
            print(f"  ==> [{arch_name}] R2={mean_r2_skip:.4f}+-"
                  f"{std_r2_skip:.4f} (from previous run)\n")
            continue

        start_a = time.time()
        print("-" * 65)
        print(f"[{arch_name}]  params={params/1e6:.1f}M  "
              f"lr={LR:.0e}  patience={PATIENCE}  clip_norm={CLIP_NORM}")
        print("-" * 65)

        fold_r2s   = []
        fold_rmses = []

        for fold_id, (tr_idx, val_idx) in enumerate(fold_indices, start=1):
            fold_start = time.time()

            tr_stems  = train_stems[tr_idx].tolist()
            val_stems = train_stems[val_idx].tolist()

            train_ds = ArchDataset(tr_stems,  labels, ch_list, norm_stats,
                                   augment_fn=augment_a3)
            val_ds   = ArchDataset(val_stems, labels, ch_list, norm_stats,
                                   augment_fn=None)

            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                                      shuffle=True,  num_workers=0, pin_memory=True)
            val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                                      shuffle=False, num_workers=0, pin_memory=True)

            model = build_arch(arch_name, in_channels=len(ch_list))

            best_r2, best_rmse = train_one_fold(
                model        = model,
                train_loader = train_loader,
                val_loader   = val_loader,
                device       = DEVICE,
                max_epochs   = MAX_EPOCHS,
                lr           = LR,
                weight_decay = WEIGHT_DECAY,
                patience     = PATIENCE,
                clip_norm    = CLIP_NORM,
                verbose      = True,
                fold_id      = fold_id,
            )

            fold_r2s.append(best_r2)
            fold_rmses.append(best_rmse)
            fold_time = time.time() - fold_start

            new_results.append({
                "arch":     arch_name,
                "fold":     fold_id,
                "val_r2":   round(best_r2,   4),
                "val_rmse": round(best_rmse,  2),
                "time_s":   round(fold_time,  1),
            })

            print(f"  --> [{arch_name}] Fold {fold_id} done: "
                  f"R2={best_r2:.4f}, RMSE={best_rmse:.2f}, "
                  f"time={fold_time/60:.1f}min\n")

        mean_r2   = float(np.mean(fold_r2s))
        std_r2    = float(np.std(fold_r2s))
        mean_rmse = float(np.mean(fold_rmses))
        std_rmse  = float(np.std(fold_rmses))
        a_time    = time.time() - start_a

        summary.append({
            "arch":      arch_name,
            "params_M":  round(params / 1e6, 2),
            "mean_r2":   round(mean_r2,   4),
            "std_r2":    round(std_r2,    4),
            "mean_rmse": round(mean_rmse,  2),
            "std_rmse":  round(std_rmse,   2),
            "time_min":  round(a_time / 60, 1),
        })

        print(f"  ==> [{arch_name}] Summary: "
              f"R2={mean_r2:.4f}+-{std_r2:.4f}, "
              f"RMSE={mean_rmse:.2f}+-{std_rmse:.2f}, "
              f"total={a_time/60:.1f}min\n")

    total_time = time.time() - start_total

    df_all            = pd.DataFrame(prev_results + new_results)
    df_summary        = pd.DataFrame(summary)
    df_summary_sorted = df_summary.sort_values("mean_r2", ascending=False).reset_index(drop=True)
    best_arch         = df_summary_sorted.iloc[0]["arch"]
    df_summary_ordered = df_summary.set_index("arch").reindex(ARCH_ORDER).reset_index()

    print("\nMeasuring inference time for all architectures...")
    df_inference = measure_all_archs(
        in_channels  = len(ch_list),
        device       = DEVICE,
        batch_size   = BATCH_SIZE,
        warmup_runs  = 10,
        measure_runs = 100,
        pretrained   = False,
    )
    for df in [df_summary_ordered, df_summary_sorted]:
        df.drop(columns=["ms_per_batch", "ms_per_sample"], errors="ignore", inplace=True)
    df_summary_ordered = df_summary_ordered.merge(
        df_inference[["arch", "ms_per_batch", "ms_per_sample"]], on="arch", how="left")
    df_summary_sorted = df_summary_sorted.merge(
        df_inference[["arch", "ms_per_batch", "ms_per_sample"]], on="arch", how="left")

    df_all.to_csv(os.path.join(OUT_DIR, "arch_compare_all_folds.csv"),
                  index=False, encoding="utf-8-sig")
    df_summary_ordered.to_csv(os.path.join(OUT_DIR, "arch_compare_summary.csv"),
                               index=False, encoding="utf-8-sig")

    _plot_bar_chart(df_summary_sorted, best_arch, OUT_DIR)
    _plot_efficiency_scatter(df_summary_sorted, best_arch, OUT_DIR)
    _write_report(df_summary_sorted, df_all, best_arch, total_time, OUT_DIR)

    best_mean_r2 = df_summary_sorted.iloc[0]["mean_r2"]
    best_std_r2 = df_summary_sorted.iloc[0]["std_r2"]

    print("=" * 65)
    print(f"All experiments completed. Total time: {total_time/3600:.2f}h")
    print(f"Best arch: {best_arch}  (R2={best_mean_r2:.4f}+-{best_std_r2:.4f})")
    print(f"Results saved to: {OUT_DIR}")
    print("=" * 65)


def _plot_bar_chart(df_summary, best_arch, out_dir):
    df_plot = df_summary.set_index("arch").reindex(ARCH_ORDER).reset_index()

    archs  = df_plot["arch"].tolist()
    means  = df_plot["mean_r2"].tolist()
    stds   = df_plot["std_r2"].tolist()
    colors = ["#2ecc71" if a == best_arch else "#3498db" for a in archs]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(archs, means, yerr=stds,
                  color=colors, edgecolor="black", linewidth=0.8,
                  capsize=6, error_kw={"elinewidth": 1.5, "ecolor": "#2c3e50"},
                  width=0.6, zorder=3)

    for bar, mean, std, row in zip(bars, means, stds, df_plot.itertuples()):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + 0.004,
                f"R2={mean:.4f}\n({row.params_M:.1f}M)",
                ha="center", va="bottom", fontsize=8, color="#2c3e50")

    best_idx = archs.index(best_arch)
    ax.text(bars[best_idx].get_x() + bars[best_idx].get_width() / 2,
            0.01, "BEST", ha="center", va="bottom",
            fontsize=9, fontweight="bold", color="white")

    ax.set_xlabel("Architecture", fontsize=12)
    ax.set_ylabel("Val R2 (5-Fold CV Mean +/- Std)", fontsize=12)
    ax.set_title(
        f"Architecture Comparison: Val R2\n"
        f"(Fixed: {G_BEST} + A3, Train from Scratch, 5-Fold CV, seed=42)",
        fontsize=12, fontweight="bold"
    )
    valid_means = [m for m in means if m is not None and not np.isnan(m)]
    valid_stds  = [s for s in stds  if s is not None and not np.isnan(s)]
    ax.set_ylim(min(0, min(valid_means) - max(valid_stds) - 0.05),
                max(valid_means) + max(valid_stds) + 0.08)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(handles=[
        Patch(facecolor="#2ecc71", edgecolor="black", label=f"Best ({best_arch})"),
        Patch(facecolor="#3498db", edgecolor="black", label="Others"),
    ], loc="lower right", fontsize=10)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "arch_compare_chart.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Chart saved: {out_path}")


def _plot_efficiency_scatter(df_summary, best_arch, out_dir):
    fig, ax = plt.subplots(figsize=(10, 6))

    for _, row in df_summary.iterrows():
        if pd.isna(row["mean_r2"]):
            continue
        color  = "#2ecc71" if row["arch"] == best_arch else "#3498db"
        marker = "*" if row["arch"] == best_arch else "o"
        size   = 220 if row["arch"] == best_arch else 120

        ax.scatter(row["params_M"], row["mean_r2"],
                   c=color, marker=marker, s=size,
                   edgecolors="black", linewidths=0.8, zorder=3)
        ax.errorbar(row["params_M"], row["mean_r2"],
                    yerr=row["std_r2"],
                    fmt="none", ecolor="#7f8c8d",
                    elinewidth=1.2, capsize=4, zorder=2)
        ax.annotate(
            f"{row['arch']}\n{row['ms_per_batch']:.1f}ms/batch",
            xy=(row["params_M"], row["mean_r2"]),
            xytext=(row["params_M"] + 0.5, row["mean_r2"]),
            fontsize=8, color="#2c3e50",
            arrowprops=dict(arrowstyle="-", color="#bdc3c7", lw=0.8),
        )

    ax.set_xlabel("Number of Parameters (M)", fontsize=12)
    ax.set_ylabel("Val R2 (5-Fold CV Mean +/- Std)", fontsize=12)
    ax.set_title(
        "Architecture Efficiency: Parameters vs Val R2\n"
        "(Labels show inference time ms/batch on GPU)",
        fontsize=12, fontweight="bold"
    )
    ax.xaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(handles=[
        plt.scatter([], [], c="#2ecc71", marker="*", s=150,
                    edgecolors="black", label=f"Best ({best_arch})"),
        plt.scatter([], [], c="#3498db", marker="o", s=80,
                    edgecolors="black", label="Others"),
    ], loc="lower right", fontsize=10)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "arch_efficiency_chart.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Efficiency chart saved: {out_path}")


def _write_report(df_summary, df_all, best_arch, total_time, out_dir):
    report_path = os.path.join(out_dir, "arch_compare_report.txt")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("ARCHITECTURE COMPARISON EXPERIMENT REPORT (v3)\n")
        f.write(f"Fixed: {G_BEST} (10 channels) + A3 augmentation\n")
        f.write("=" * 70 + "\n\n")

        f.write("[ EXPERIMENT CONFIGURATION ]\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Input Group    : {G_BEST} (NDVI + GNDVI + NDRE + 7 raw bands)\n")
        f.write(f"  Augmentation   : {A_BEST_DESC}\n")
        f.write(f"  CV Strategy    : {N_FOLDS}-Fold Cross Validation (seed={SEED})\n")
        f.write(f"  TrainVal Size  : 414 samples\n")
        f.write(f"  Pretrained     : False (all architectures train from scratch)\n")
        f.write(f"  Max Epochs     : {MAX_EPOCHS}\n")
        f.write(f"  Early Stopping : patience={PATIENCE} (unified)\n")
        f.write(f"  Optimizer      : AdamW (lr={LR}, wd={WEIGHT_DECAY})\n")
        f.write(f"  Scheduler      : CosineAnnealingLR\n")
        f.write(f"  Clip Norm      : {CLIP_NORM} (unified)\n")
        f.write(f"  Batch Size     : {BATCH_SIZE}\n")
        f.write(f"  Total Time     : {total_time/3600:.2f} hours\n\n")

        f.write("[ SUMMARY RESULTS (sorted by Mean Val R2) ]\n")
        f.write("-" * 70 + "\n")
        header_arch = "Arch"
        header_params = "Params(M)"
        header_mean_r2 = "Mean R2"
        header_std_r2 = "Std R2"
        header_rmse = "RMSE"
        header_ms = "ms/batch"
        header_time = "Time(min)"
        f.write(f"  {header_arch:<20} {header_params:>9}  {header_mean_r2:>9}  {header_std_r2:>8}  "
                f"{header_rmse:>8}  {header_ms:>9}  {header_time:>10}\n")
        f.write("  " + "-" * 68 + "\n")
        for _, row in df_summary.iterrows():
            mark = " <-- BEST" if row["arch"] == best_arch else ""
            arch_val = row["arch"]
            params_val = row["params_M"]
            mean_r2_val = row["mean_r2"]
            std_r2_val = row["std_r2"]
            mean_rmse_val = row["mean_rmse"]
            ms_batch_val = row["ms_per_batch"]
            time_min_val = row["time_min"]
            f.write(
                f"  {arch_val:<20} {params_val:>9.2f}  "
                f"{mean_r2_val:>9.4f}  {std_r2_val:>8.4f}  "
                f"{mean_rmse_val:>8.2f}  {ms_batch_val:>9.2f}  "
                f"{time_min_val:>10.1f}{mark}\n"
            )
        f.write("\n")

        f.write("[ PER-FOLD DETAILED RESULTS ]\n")
        f.write("-" * 70 + "\n")
        for arch_name in ARCH_ORDER:
            rows = df_all[df_all["arch"] == arch_name]
            if rows.empty:
                continue
            skp_tag = "  (from previous run)" if arch_name in SKIP_ARCHS else ""
            f.write(f"\n  {arch_name}{skp_tag}:\n")
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
                time_s_val = r.get("time_s", 0)
                f.write(f"    {fold_val:>5}  {val_r2_val:>9.4f}  "
                        f"{val_rmse_val:>10.2f}  {time_s_val:>9.1f}\n")
            r2s = rows["val_r2"].tolist()
            mean_label = "Mean"
            std_label = "Std"
            f.write(f"    {mean_label:>5}  {np.mean(r2s):>9.4f}  "
                    f"{np.mean(rows['val_rmse']):>10.2f}\n")
            f.write(f"    {std_label:>5}  {np.std(r2s):>9.4f}  "
                    f"{np.std(rows['val_rmse']):>10.2f}\n")

        best_row      = df_summary[df_summary["arch"] == best_arch].iloc[0]
        worst_row     = df_summary.iloc[-1]
        baseline_rows = df_summary[df_summary["arch"] == "resnet50"]
        baseline      = baseline_rows["mean_r2"].values[0] \
                        if not baseline_rows.empty else 0.0

        f.write("\n\n[ CONCLUSION ]\n")
        f.write("-" * 70 + "\n")
        best_params_val = best_row["params_M"]
        best_mean_r2_val = best_row["mean_r2"]
        best_std_r2_val = best_row["std_r2"]
        best_mean_rmse_val = best_row["mean_rmse"]
        best_std_rmse_val = best_row["std_rmse"]
        best_ms_batch_val = best_row["ms_per_batch"]
        worst_arch_val = worst_row["arch"]
        worst_mean_r2_val = worst_row["mean_r2"]
        worst_std_r2_val = worst_row["std_r2"]

        f.write(
            f"  Best arch    : {best_arch}\n"
            f"  Params       : {best_params_val:.2f}M\n"
            f"  Best R2      : {best_mean_r2_val:.4f} +/- {best_std_r2_val:.4f}\n"
            f"  Best RMSE    : {best_mean_rmse_val:.2f} +/- {best_std_rmse_val:.2f}\n"
            f"  Inference    : {best_ms_batch_val:.2f} ms/batch\n\n"
            f"  Worst arch   : {worst_arch_val}\n"
            f"  Worst R2     : {worst_mean_r2_val:.4f} +/- {worst_std_r2_val:.4f}\n\n"
            f"  R2 gain over ResNet-50 (baseline):\n"
        )
        for _, row in df_summary.iterrows():
            delta = row["mean_r2"] - baseline
            sign  = "+" if delta >= 0 else ""
            f.write(f"    {row['arch']:<20}: {sign}{delta:.4f}\n")

        f.write(
            f"\n  Recommended next step:\n"
            f"    Use {best_arch} as Best_Arch for stability analysis\n"
            f"    and final test set evaluation.\n"
        )
        f.write("\n" + "=" * 70 + "\n")

    print(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()