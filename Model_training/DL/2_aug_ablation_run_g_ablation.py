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
from dataset import (WheatDataset, G_CONFIGS, load_norm_stats,
                     load_split_and_labels, preload_all_channels,
                     augment_a1, PRELOAD_CACHE)
from model import build_model, count_parameters

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
OUT_DIR     = os.path.join(BASE_DIR, "2_aug_ablation", "results")

G_BEST       = "G3"
N_FOLDS      = 5
MAX_EPOCHS   = 150
PATIENCE     = 20
LR           = 1e-4
WEIGHT_DECAY = 1e-4
BATCH_SIZE   = 32
SEED         = 42

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def augment_a0(image):
    return image

def augment_a2(image):
    image = augment_a1(image)
    noise = np.random.normal(0, 0.05, image.shape).astype(np.float32)
    return image + noise

def augment_a3(image):
    image = augment_a1(image)
    if random.random() < 0.1:
        ch = random.randint(0, image.shape[0] - 1)
        image = image.copy()
        image[ch] = 0.0
    return image

def augment_a4(image):
    image = augment_a1(image)
    noise = np.random.normal(0, 0.05, image.shape).astype(np.float32)
    image = image + noise
    if random.random() < 0.1:
        ch = random.randint(0, image.shape[0] - 1)
        image = image.copy()
        image[ch] = 0.0
    return image

def augment_a5(image):
    return augment_a1(image)

def augment_a6(image):
    return augment_a4(image)


AUG_STRATEGIES = [
    {
        "name":        "A0",
        "description": "No augmentation (control baseline)",
        "augment_fn":  augment_a0,
        "use_mixup":   False,
        "mixup_alpha": 0.0,
    },
    {
        "name":        "A1",
        "description": "HFlip + VFlip + Rot90 (geometric baseline)",
        "augment_fn":  augment_a1,
        "use_mixup":   False,
        "mixup_alpha": 0.0,
    },
    {
        "name":        "A2",
        "description": "A1 + Gaussian noise (sigma=0.05)",
        "augment_fn":  augment_a2,
        "use_mixup":   False,
        "mixup_alpha": 0.0,
    },
    {
        "name":        "A3",
        "description": "A1 + Channel Dropout (p=0.1)",
        "augment_fn":  augment_a3,
        "use_mixup":   False,
        "mixup_alpha": 0.0,
    },
    {
        "name":        "A4",
        "description": "A1 + Gaussian noise + Channel Dropout",
        "augment_fn":  augment_a4,
        "use_mixup":   False,
        "mixup_alpha": 0.0,
    },
    {
        "name":        "A5",
        "description": "A1 + Mixup (alpha=0.2)",
        "augment_fn":  augment_a5,
        "use_mixup":   True,
        "mixup_alpha": 0.2,
    },
    {
        "name":        "A6",
        "description": "A4 + Mixup (alpha=0.2, strongest combination)",
        "augment_fn":  augment_a6,
        "use_mixup":   True,
        "mixup_alpha": 0.2,
    },
]


class AugWheatDataset(Dataset):
    def __init__(self, stems, labels, channel_list, norm_stats,
                 augment_fn=None):
        assert len(PRELOAD_CACHE) > 0, \
            "[AugWheatDataset] PRELOAD_CACHE is empty! Call preload_all_channels() first."

        self.augment_fn = augment_fn

        means = np.array([norm_stats[ch]["mean"] for ch in channel_list],
                         dtype=np.float32).reshape(-1, 1, 1)
        stds  = np.array([norm_stats[ch]["std"]  for ch in channel_list],
                         dtype=np.float32).reshape(-1, 1, 1)
        stds  = np.where(stds == 0, 1.0, stds)

        self.images      = []
        self.label_list  = []

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


def mixup_batch(images, labels, alpha):
    lam    = float(np.random.beta(alpha, alpha))
    idx    = torch.randperm(images.size(0), device=images.device)
    mixed_images = lam * images + (1 - lam) * images[idx]
    mixed_labels = lam * labels + (1 - lam) * labels[idx]
    return mixed_images, mixed_labels


def train_one_fold_aug(model, train_loader, val_loader, device,
                       use_mixup=False, mixup_alpha=0.2,
                       max_epochs=150, lr=1e-4, weight_decay=1e-4,
                       patience=20, verbose=True, fold_id=None):
    model     = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=lr * 1e-3)
    criterion = nn.MSELoss()

    best_r2        = -np.inf
    best_rmse      = np.inf
    no_improve_cnt = 0
    fold_str       = f"Fold {fold_id}" if fold_id is not None else "Fold"

    for epoch in range(1, max_epochs + 1):

        model.train()
        train_losses = []

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if use_mixup:
                images, labels = mixup_batch(images, labels, mixup_alpha)

            optimizer.zero_grad()
            preds = model(images)
            loss  = criterion(preds, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
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
                      f"Best Val R2={best_r2:.4f}, RMSE={best_rmse:.2f}")
            break

    if verbose and no_improve_cnt < patience:
        print(f"  [{fold_str}] Training completed. "
              f"Best Val R2={best_r2:.4f}, RMSE={best_rmse:.2f}")

    return best_r2, best_rmse


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    start_total = time.time()

    print("=" * 65)
    print("Augmentation Strategy Ablation Study")
    print(f"Fixed: ResNet-50 + {G_BEST} ({len(G_CONFIGS[G_BEST])} channels)")
    print(f"Device : {DEVICE}")
    print(f"Folds  : {N_FOLDS}   MaxEpochs: {MAX_EPOCHS}   Patience: {PATIENCE}")
    print(f"LR     : {LR}        BatchSize: {BATCH_SIZE}")
    print("=" * 65)

    norm_stats                      = load_norm_stats(NORM_STATS)
    train_stems, test_stems, labels = load_split_and_labels(SPLIT_CSV, LABEL_CSV)
    all_stems = train_stems + test_stems
    preload_all_channels(all_stems, DOM_DIR, INDICES_DIR, verbose=True)

    train_stems = np.array(train_stems)
    ch_list     = G_CONFIGS[G_BEST]
    n_ch        = len(ch_list)
    params      = count_parameters(build_model(in_channels=n_ch, pretrained=False))

    print(f"\nTrainVal: {len(train_stems)} samples | Channels: {n_ch} | Params: {params/1e6:.1f}M")

    kf           = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_indices = list(kf.split(train_stems))
    print(f"5-Fold CV indices generated (seed={SEED}).\n")

    all_results = []
    summary     = []

    for strategy in AUG_STRATEGIES:
        a_name     = strategy["name"]
        a_desc     = strategy["description"]
        augment_fn = strategy["augment_fn"]
        use_mixup  = strategy["use_mixup"]
        mixup_alpha= strategy["mixup_alpha"]
        start_a    = time.time()

        print("-" * 65)
        print(f"[{a_name}]  {a_desc}")
        mixup_status = ("ON (alpha=%.1f)" % mixup_alpha) if use_mixup else "OFF"
        print(f"         Mixup: {mixup_status}")
        print("-" * 65)

        fold_r2s   = []
        fold_rmses = []

        for fold_id, (tr_idx, val_idx) in enumerate(fold_indices, start=1):
            fold_start = time.time()

            tr_stems  = train_stems[tr_idx].tolist()
            val_stems = train_stems[val_idx].tolist()

            train_ds = AugWheatDataset(tr_stems,  labels, ch_list, norm_stats,
                                       augment_fn=augment_fn)
            val_ds   = AugWheatDataset(val_stems, labels, ch_list, norm_stats,
                                       augment_fn=None)

            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                                      shuffle=True,  num_workers=0, pin_memory=True)
            val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                                      shuffle=False, num_workers=0, pin_memory=True)

            model = build_model(in_channels=n_ch, pretrained=True)

            best_r2, best_rmse = train_one_fold_aug(
                model        = model,
                train_loader = train_loader,
                val_loader   = val_loader,
                device       = DEVICE,
                use_mixup    = use_mixup,
                mixup_alpha  = mixup_alpha,
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
                "strategy":   a_name,
                "description":a_desc,
                "fold":       fold_id,
                "val_r2":     round(best_r2,   4),
                "val_rmse":   round(best_rmse,  2),
                "time_s":     round(fold_time,  1),
            })

            print(f"  --> [{a_name}] Fold {fold_id} done: "
                  f"R2={best_r2:.4f}, RMSE={best_rmse:.2f}, "
                  f"time={fold_time/60:.1f}min\n")

        mean_r2   = float(np.mean(fold_r2s))
        std_r2    = float(np.std(fold_r2s))
        mean_rmse = float(np.mean(fold_rmses))
        std_rmse  = float(np.std(fold_rmses))
        a_time    = time.time() - start_a

        summary.append({
            "strategy":    a_name,
            "description": a_desc,
            "use_mixup":   use_mixup,
            "mean_r2":     round(mean_r2,   4),
            "std_r2":      round(std_r2,    4),
            "mean_rmse":   round(mean_rmse,  2),
            "std_rmse":    round(std_rmse,   2),
            "time_min":    round(a_time/60,  1),
        })

        print(f"  ==> [{a_name}] Summary: "
              f"R2={mean_r2:.4f}+-{std_r2:.4f}, "
              f"RMSE={mean_rmse:.2f}+-{std_rmse:.2f}, "
              f"total={a_time/60:.1f}min\n")

    total_time = time.time() - start_total

    df_all     = pd.DataFrame(all_results)
    df_summary = pd.DataFrame(summary)
    df_summary = df_summary.sort_values("mean_r2", ascending=False).reset_index(drop=True)
    best_strat = df_summary.iloc[0]["strategy"]

    df_all.to_csv(    os.path.join(OUT_DIR, "a_ablation_all_folds.csv"), index=False, encoding="utf-8-sig")
    df_summary.to_csv(os.path.join(OUT_DIR, "a_ablation_summary.csv"),   index=False, encoding="utf-8-sig")

    _plot_bar_chart(df_summary, best_strat, OUT_DIR)
    _write_report(df_summary, df_all, best_strat, total_time, OUT_DIR)

    best_mean_r2_val = df_summary.iloc[0]["mean_r2"]
    best_std_r2_val = df_summary.iloc[0]["std_r2"]

    print("=" * 65)
    print(f"All experiments completed. Total time: {total_time/3600:.2f}h")
    print(f"Best strategy: {best_strat}  (R2={best_mean_r2_val:.4f}+-{best_std_r2_val:.4f})")
    print(f"Results saved to: {OUT_DIR}")
    print("=" * 65)


def _plot_bar_chart(df_summary, best_strat, out_dir):
    order   = ["A0", "A1", "A2", "A3", "A4", "A5", "A6"]
    df_plot = df_summary.set_index("strategy").reindex(order).reset_index()

    strategies = df_plot["strategy"].tolist()
    means      = df_plot["mean_r2"].tolist()
    stds       = df_plot["std_r2"].tolist()

    colors = ["#2ecc71" if s == best_strat else "#e67e22"
              if df_plot.loc[df_plot["strategy"] == s, "use_mixup"].values[0]
              else "#3498db"
              for s in strategies]

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(
        strategies, means, yerr=stds,
        color=colors, edgecolor="black", linewidth=0.8,
        capsize=6, error_kw={"elinewidth": 1.5, "ecolor": "#2c3e50"},
        width=0.6, zorder=3,
    )

    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + 0.004,
            f"R2={mean:.4f}",
            ha="center", va="bottom", fontsize=8.5, color="#2c3e50"
        )

    best_idx = strategies.index(best_strat)
    ax.text(
        bars[best_idx].get_x() + bars[best_idx].get_width() / 2,
        0.01, "BEST",
        ha="center", va="bottom", fontsize=9,
        fontweight="bold", color="white"
    )

    ax.set_xlabel("Augmentation Strategy", fontsize=12)
    ax.set_ylabel("Val R2 (5-Fold CV Mean +/- Std)", fontsize=12)
    ax.set_title(
        "Augmentation Ablation: Strategy vs Validation R2\n"
        f"(ResNet-50 + {G_BEST}, 5-Fold CV, Fixed Seed=42)",
        fontsize=12, fontweight="bold"
    )
    ax.set_ylim(min(0, min(means) - max(stds) - 0.05),
                max(means) + max(stds) + 0.07)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)

    ax.legend(handles=[
        Patch(facecolor="#2ecc71", edgecolor="black", label=f"Best ({best_strat})"),
        Patch(facecolor="#3498db", edgecolor="black", label="No Mixup"),
        Patch(facecolor="#e67e22", edgecolor="black", label="With Mixup"),
    ], loc="lower right", fontsize=10)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "a_ablation_chart.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Chart saved: {out_path}")


def _write_report(df_summary, df_all, best_strat, total_time, out_dir):
    report_path = os.path.join(out_dir, "a_ablation_report.txt")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("AUGMENTATION STRATEGY ABLATION STUDY REPORT\n")
        f.write("Fixed: ResNet-50 + G3 (10 channels)\n")
        f.write("=" * 70 + "\n\n")

        f.write("[ EXPERIMENT CONFIGURATION ]\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Model          : ResNet-50 (CNN-MLP regression head)\n")
        f.write(f"  Input Group    : {G_BEST} (10 channels)\n")
        f.write(f"  CV Strategy    : {N_FOLDS}-Fold Cross Validation (seed={SEED})\n")
        f.write(f"  TrainVal Size  : 414 samples\n")
        f.write(f"  Max Epochs     : {MAX_EPOCHS}\n")
        f.write(f"  Early Stopping : patience={PATIENCE}\n")
        f.write(f"  Optimizer      : AdamW (lr={LR}, wd={WEIGHT_DECAY})\n")
        f.write(f"  Scheduler      : CosineAnnealingLR\n")
        f.write(f"  Batch Size     : {BATCH_SIZE}\n")
        f.write(f"  Total Time     : {total_time/3600:.2f} hours\n\n")

        f.write("[ STRATEGY DEFINITIONS ]\n")
        f.write("-" * 40 + "\n")
        for s in AUG_STRATEGIES:
            s_name = s["name"]
            s_desc = s["description"]
            f.write(f"  {s_name}: {s_desc}\n")
        f.write("\n")

        f.write("[ SUMMARY RESULTS (sorted by Mean Val R2) ]\n")
        f.write("-" * 70 + "\n")
        header_strat = "Strat"
        header_mean_r2 = "Mean R2"
        header_std_r2 = "Std R2"
        header_mean_rmse = "Mean RMSE"
        header_std_rmse = "Std RMSE"
        header_time = "Time(min)"
        f.write(f"  {header_strat:<6} {header_mean_r2:>9}  {header_std_r2:>8}  "
                f"{header_mean_rmse:>10}  {header_std_rmse:>9}  {header_time:>10}\n")
        f.write("  " + "-" * 60 + "\n")
        for _, row in df_summary.iterrows():
            mark = " <-- BEST" if row["strategy"] == best_strat else ""
            strat_val = row["strategy"]
            mean_r2_val = row["mean_r2"]
            std_r2_val = row["std_r2"]
            mean_rmse_val = row["mean_rmse"]
            std_rmse_val = row["std_rmse"]
            time_min_val = row["time_min"]
            f.write(
                f"  {strat_val:<6} {mean_r2_val:>9.4f}  {std_r2_val:>8.4f}  "
                f"{mean_rmse_val:>10.2f}  {std_rmse_val:>9.2f}  "
                f"{time_min_val:>10.1f}{mark}\n"
            )
        f.write("\n")

        f.write("[ PER-FOLD DETAILED RESULTS ]\n")
        f.write("-" * 70 + "\n")
        for s in AUG_STRATEGIES:
            a_name = s["name"]
            s_desc = s["description"]
            rows   = df_all[df_all["strategy"] == a_name]
            if rows.empty:
                continue
            f.write(f"\n  {a_name} ({s_desc}):\n")
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

        best_row  = df_summary[df_summary["strategy"] == best_strat].iloc[0]
        worst_row = df_summary.iloc[-1]
        a1_r2     = df_summary[df_summary["strategy"] == "A1"]["mean_r2"].values[0]

        f.write("\n\n[ CONCLUSION ]\n")
        f.write("-" * 70 + "\n")
        best_desc_val = best_row["description"]
        best_mean_r2_val2 = best_row["mean_r2"]
        best_std_r2_val2 = best_row["std_r2"]
        best_mean_rmse_val = best_row["mean_rmse"]
        best_std_rmse_val = best_row["std_rmse"]
        worst_strat_val = worst_row["strategy"]
        worst_mean_r2_val = worst_row["mean_r2"]
        worst_std_r2_val = worst_row["std_r2"]

        f.write(
            f"  Best strategy  : {best_strat}\n"
            f"  Description    : {best_desc_val}\n"
            f"  Best R2        : {best_mean_r2_val2:.4f} +/- {best_std_r2_val2:.4f}\n"
            f"  Best RMSE      : {best_mean_rmse_val:.2f} +/- {best_std_rmse_val:.2f}\n\n"
            f"  Worst strategy : {worst_strat_val}\n"
            f"  Worst R2       : {worst_mean_r2_val:.4f} +/- {worst_std_r2_val:.4f}\n\n"
            f"  R2 gain over A1 (geometric baseline):\n"
        )
        for _, row in df_summary.iterrows():
            delta = row["mean_r2"] - a1_r2
            sign  = "+" if delta >= 0 else ""
            f.write(f"    {row['strategy']}: {sign}{delta:.4f}  ({row['description']})\n")

        f.write(
            f"\n  Recommended next step:\n"
            f"    Use {best_strat} as the fixed augmentation strategy (A_BEST)\n"
            f"    for all subsequent architecture comparison experiments.\n"
        )
        f.write("\n" + "=" * 70 + "\n")

    print(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()