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

THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
ARCH_DIR    = os.path.dirname(THIS_DIR)
BASE_DIR    = os.path.dirname(ARCH_DIR)
DATA_DIR    = os.path.join(BASE_DIR, "0_Data")

DOM_DIR     = os.path.join(DATA_DIR, "1_DOM")
INDICES_DIR = os.path.join(DATA_DIR, "indices")
NORM_STATS  = os.path.join(DATA_DIR, "norm_stats.json")
SPLIT_CSV   = os.path.join(DATA_DIR, "dataset_split_index.csv")
LABEL_CSV   = os.path.join(DATA_DIR, "wheat_count_labels.csv")
OUT_DIR     = os.path.join(THIS_DIR, "results")

sys.path.insert(0, os.path.join(BASE_DIR, "1_G_ablation"))
sys.path.insert(0, THIS_DIR)

from dataset import (G_CONFIGS, load_norm_stats, load_split_and_labels,
                     preload_all_channels, PRELOAD_CACHE, augment_a1)
from densenet_model import build_densenet, count_parameters, DENSENET_REGISTRY

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

G_BEST       = "G3"
A_BEST_DESC  = "A3: HFlip + VFlip + Rot90 + ChannelDropout(p=0.1)"

N_FOLDS      = 5
MAX_EPOCHS   = 150
PATIENCE     = 20
LR           = 1e-4
WEIGHT_DECAY = 1e-4
CLIP_NORM    = 5.0
BATCH_SIZE   = 32
SEED         = 42

USE_WARMUP    = False
WARMUP_EPOCHS = 10

MODEL_ORDER = ["densenet121", "densenet169", "densenet201"]

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def augment_a3(image: np.ndarray) -> np.ndarray:
    image = augment_a1(image)
    if random.random() < 0.1:
        ch    = random.randint(0, image.shape[0] - 1)
        image = image.copy()
        image[ch] = 0.0
    return image


class DenseNetDataset(Dataset):
    def __init__(self, stems, labels, channel_list, norm_stats, augment_fn=None):
        assert len(PRELOAD_CACHE) > 0, \
            "[DenseNetDataset] PRELOAD_CACHE is empty! Call preload_all_channels() first."

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
            image     = np.stack(ch_arrays, axis=0)
            image     = (image - means) / stds
            self.images.append(image)
            self.label_list.append(float(labels[stem]))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].copy()
        if self.augment_fn is not None:
            image = self.augment_fn(image)
        return (torch.from_numpy(np.ascontiguousarray(image)),
                torch.tensor(self.label_list[idx], dtype=torch.float32))


class WarmupCosineScheduler:
    def __init__(self, optimizer, lr_max: float, warmup_epochs: int,
                 total_epochs: int):
        self.optimizer     = optimizer
        self.lr_max        = lr_max
        self.lr_start      = lr_max / 10.0
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
        self.cosine_epochs = total_epochs - warmup_epochs
        self._step(epoch=1)

    def step(self, epoch: int):
        self._step(epoch)

    def _step(self, epoch: int):
        if epoch <= self.warmup_epochs:
            frac = (epoch - 1) / max(self.warmup_epochs - 1, 1)
            lr   = self.lr_start + frac * (self.lr_max - self.lr_start)
        else:
            t    = epoch - self.warmup_epochs
            T    = self.cosine_epochs
            eta_min = self.lr_max * 1e-3
            lr   = eta_min + 0.5 * (self.lr_max - eta_min) * (
                1 + np.cos(np.pi * t / T))
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def get_last_lr(self) -> list:
        return [pg["lr"] for pg in self.optimizer.param_groups]


def train_one_fold(model, train_loader, val_loader, device,
                   max_epochs, lr, weight_decay, patience,
                   clip_norm, use_warmup, warmup_epochs,
                   verbose=True, fold_id=None):

    model     = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)

    if use_warmup:
        scheduler = WarmupCosineScheduler(
            optimizer, lr_max=lr,
            warmup_epochs=warmup_epochs, total_epochs=max_epochs)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs, eta_min=lr * 1e-3)

    criterion      = nn.MSELoss()
    best_r2        = -np.inf
    best_rmse      = np.inf
    no_improve_cnt = 0
    best_epoch     = 0
    fold_str       = f"Fold {fold_id}" if fold_id else "Fold"

    for epoch in range(1, max_epochs + 1):

        if use_warmup:
            scheduler.step(epoch)

        model.train()
        train_losses = []
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            preds  = model(images)
            loss   = criterion(preds, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            optimizer.step()
            train_losses.append(loss.item())

        if not use_warmup:
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
            best_epoch     = epoch
            no_improve_cnt = 0
        else:
            no_improve_cnt += 1

        if verbose:
            current_lr = scheduler.get_last_lr()[0]
            warmup_tag = f" [WU{epoch}/{warmup_epochs}]" \
                if use_warmup and epoch <= warmup_epochs else ""
            mark = "*" if no_improve_cnt == 0 else " "
            print(f"  [{fold_str}] Epoch {epoch:3d}/{max_epochs}{warmup_tag} | "
                  f"TrainLoss={np.mean(train_losses):8.1f} | "
                  f"ValR2={val_r2:.4f} {mark} | "
                  f"ValRMSE={val_rmse:6.2f} | "
                  f"LR={current_lr:.2e} | "
                  f"NoImprove={no_improve_cnt}/{patience}")

        if no_improve_cnt >= patience:
            if verbose:
                print(f"  [{fold_str}] Early stop @ epoch {epoch}. "
                      f"BestR2={best_r2:.4f} @ epoch {best_epoch}")
            break

    if verbose and no_improve_cnt < patience:
        print(f"  [{fold_str}] Finished {max_epochs} epochs. "
              f"BestR2={best_r2:.4f} @ epoch {best_epoch}")

    return best_r2, best_rmse, best_epoch


def measure_inference(model_name, in_channels, device, batch_size,
                      warmup_runs=10, measure_runs=100):
    model = build_densenet(model_name, in_channels=in_channels).to(device)
    model.eval()
    dummy = torch.randn(batch_size, in_channels, 64, 64).to(device)

    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(measure_runs):
            t0 = time.perf_counter()
            _  = model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    ms_batch  = float(np.mean(times))
    ms_sample = ms_batch / batch_size
    ms_std    = float(np.std(times))
    print(f"  [{model_name}] {ms_batch:.2f} ms/batch "
          f"({ms_sample:.3f} ms/sample) +/- {ms_std:.3f} ms")
    del model
    return ms_batch, ms_sample


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    start_total = time.time()

    warmup_str = (f"True (linear {WARMUP_EPOCHS} epochs, lr: {LR/10:.0e}->{LR:.0e})"
                  if USE_WARMUP else "False")

    print("=" * 65)
    print("DenseNet Size Ablation Experiment")
    print(f"Models         : {MODEL_ORDER}")
    print(f"Fixed Input    : {G_BEST} (10 channels)")
    print(f"Fixed Aug      : {A_BEST_DESC}")
    print(f"Device         : {DEVICE}")
    print(f"Folds/MaxEpoch : {N_FOLDS} / {MAX_EPOCHS}   Patience: {PATIENCE}")
    print(f"LR / WD        : {LR} / {WEIGHT_DECAY}   ClipNorm: {CLIP_NORM}")
    print(f"BatchSize      : {BATCH_SIZE}   Pretrained: False")
    print(f"USE_WARMUP     : {warmup_str}")
    print(f"Output dir     : {OUT_DIR}")
    print("=" * 65)

    norm_stats                      = load_norm_stats(NORM_STATS)
    train_stems, test_stems, labels = load_split_and_labels(SPLIT_CSV, LABEL_CSV)
    all_stems                       = train_stems + test_stems
    preload_all_channels(all_stems, DOM_DIR, INDICES_DIR, verbose=True)

    train_stems = np.array(train_stems)
    ch_list     = G_CONFIGS[G_BEST]
    in_channels = len(ch_list)

    print(f"\nTrainVal: {len(train_stems)} | Test: {len(test_stems)} (sealed) | "
          f"Channels: {in_channels}\n")

    kf           = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_indices = list(kf.split(train_stems))
    print(f"5-Fold CV indices generated (seed={SEED}, same as ResNet experiment).\n")

    all_results = []
    summary     = []

    for model_name in MODEL_ORDER:
        params  = count_parameters(build_densenet(model_name, in_channels=in_channels))
        start_m = time.time()

        print("-" * 65)
        print(f"[{model_name}]  params={params/1e6:.1f}M  "
              f"warmup={USE_WARMUP}  patience={PATIENCE}")
        print("-" * 65)

        fold_r2s, fold_rmses, fold_epochs = [], [], []

        for fold_id, (tr_idx, val_idx) in enumerate(fold_indices, start=1):
            fold_start = time.time()

            tr_stems  = train_stems[tr_idx].tolist()
            val_stems = train_stems[val_idx].tolist()

            train_ds = DenseNetDataset(tr_stems,  labels, ch_list, norm_stats,
                                       augment_fn=augment_a3)
            val_ds   = DenseNetDataset(val_stems, labels, ch_list, norm_stats,
                                       augment_fn=None)

            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                                      shuffle=True,  num_workers=0, pin_memory=True)
            val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                                      shuffle=False, num_workers=0, pin_memory=True)

            model = build_densenet(model_name, in_channels=in_channels)

            best_r2, best_rmse, best_epoch = train_one_fold(
                model         = model,
                train_loader  = train_loader,
                val_loader    = val_loader,
                device        = DEVICE,
                max_epochs    = MAX_EPOCHS,
                lr            = LR,
                weight_decay  = WEIGHT_DECAY,
                patience      = PATIENCE,
                clip_norm     = CLIP_NORM,
                use_warmup    = USE_WARMUP,
                warmup_epochs = WARMUP_EPOCHS,
                verbose       = True,
                fold_id       = fold_id,
            )

            fold_r2s.append(best_r2)
            fold_rmses.append(best_rmse)
            fold_epochs.append(best_epoch)
            fold_time = time.time() - fold_start

            all_results.append({
                "model":      model_name,
                "fold":       fold_id,
                "val_r2":     round(best_r2,   4),
                "val_rmse":   round(best_rmse,  2),
                "best_epoch": best_epoch,
                "time_s":     round(fold_time,  1),
            })

            print(f"  --> [{model_name}] Fold {fold_id}: "
                  f"R2={best_r2:.4f}, RMSE={best_rmse:.2f}, "
                  f"best_epoch={best_epoch}, time={fold_time/60:.1f}min\n")

        m_time = time.time() - start_m
        summary.append({
            "model":           model_name,
            "params_M":        round(params / 1e6, 2),
            "mean_r2":         round(float(np.mean(fold_r2s)),   4),
            "std_r2":          round(float(np.std(fold_r2s)),    4),
            "mean_rmse":       round(float(np.mean(fold_rmses)), 2),
            "std_rmse":        round(float(np.std(fold_rmses)),  2),
            "mean_best_epoch": round(float(np.mean(fold_epochs)),1),
            "time_min":        round(m_time / 60, 1),
        })
        mean_r2_disp = np.mean(fold_r2s)
        std_r2_disp = np.std(fold_r2s)
        mean_rmse_disp = np.mean(fold_rmses)
        std_rmse_disp = np.std(fold_rmses)
        print(f"  ==> [{model_name}] "
              f"R2={mean_r2_disp:.4f}+-{std_r2_disp:.4f}, "
              f"RMSE={mean_rmse_disp:.2f}+-{std_rmse_disp:.2f}, "
              f"mean_best_epoch={np.mean(fold_epochs):.1f}, "
              f"total={m_time/60:.1f}min\n")

    total_time = time.time() - start_total

    print("\nMeasuring inference time...")
    infer_rows = []
    for model_name in MODEL_ORDER:
        ms_batch, ms_sample = measure_inference(
            model_name, in_channels, DEVICE, BATCH_SIZE)
        infer_rows.append({
            "model":          model_name,
            "ms_per_batch":   round(ms_batch,  2),
            "ms_per_sample":  round(ms_sample, 3),
        })
    df_infer = pd.DataFrame(infer_rows)

    df_all     = pd.DataFrame(all_results)
    df_summary = pd.DataFrame(summary)
    df_summary = df_summary.merge(
        df_infer[["model", "ms_per_batch", "ms_per_sample"]],
        on="model", how="left"
    )
    df_summary_sorted = df_summary.sort_values(
        "mean_r2", ascending=False).reset_index(drop=True)
    best_model = df_summary_sorted.iloc[0]["model"]

    df_all.to_csv(os.path.join(OUT_DIR, "densenet_all_folds.csv"),
                  index=False, encoding="utf-8-sig")
    df_summary.set_index("model").reindex(MODEL_ORDER).reset_index().to_csv(
        os.path.join(OUT_DIR, "densenet_summary.csv"),
        index=False, encoding="utf-8-sig")

    _plot_bar_chart(df_summary_sorted, best_model, OUT_DIR)
    _plot_epoch_chart(df_summary_sorted, best_model, OUT_DIR)
    _plot_fold_stability(df_all, MODEL_ORDER, OUT_DIR)
    _write_report(df_summary_sorted, df_all, best_model, total_time, OUT_DIR)

    best_mean_r2 = df_summary_sorted.iloc[0]["mean_r2"]
    best_std_r2 = df_summary_sorted.iloc[0]["std_r2"]

    print("=" * 65)
    print(f"All done. Total time: {total_time/3600:.2f}h")
    print(f"Best model: {best_model}  "
          f"(R2={best_mean_r2:.4f}+-{best_std_r2:.4f})")
    print(f"Results -> {OUT_DIR}")
    print("=" * 65)


def _plot_bar_chart(df_summary, best_model, out_dir):
    df_plot = df_summary.set_index("model").reindex(MODEL_ORDER).reset_index()
    archs   = df_plot["model"].tolist()
    means   = df_plot["mean_r2"].tolist()
    stds    = df_plot["std_r2"].tolist()
    colors  = ["#2ecc71" if m == best_model else "#3498db" for m in archs]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(archs, means, yerr=stds,
                  color=colors, edgecolor="black", linewidth=0.8,
                  capsize=6, error_kw={"elinewidth": 1.5, "ecolor": "#2c3e50"},
                  width=0.5, zorder=3)

    for bar, mean, std, row in zip(bars, means, stds, df_plot.itertuples()):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + 0.004,
                f"R2={mean:.4f}\n({row.params_M:.1f}M params)\n"
                f"epoch~{row.mean_best_epoch:.0f}",
                ha="center", va="bottom", fontsize=8, color="#2c3e50")

    best_idx = archs.index(best_model)
    ax.text(bars[best_idx].get_x() + bars[best_idx].get_width() / 2,
            0.01, "BEST", ha="center", va="bottom",
            fontsize=9, fontweight="bold", color="white")

    ax.set_xlabel("DenseNet Variant", fontsize=12)
    ax.set_ylabel("Val R2 (5-Fold CV Mean +/- Std)", fontsize=12)
    warmup_note = f"USE_WARMUP=True ({WARMUP_EPOCHS} ep)" if USE_WARMUP else "USE_WARMUP=False"
    ax.set_title(
        f"DenseNet Size Ablation: Val R2\n"
        f"(Fixed: G3 + A3, Train from Scratch, 5-Fold CV, seed=42, {warmup_note})",
        fontsize=10, fontweight="bold"
    )
    valid_means = [m for m in means if not np.isnan(m)]
    valid_stds  = [s for s in stds  if not np.isnan(s)]
    ax.set_ylim(min(0, min(valid_means) - max(valid_stds) - 0.05),
                max(valid_means) + max(valid_stds) + 0.12)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(handles=[
        Patch(facecolor="#2ecc71", edgecolor="black", label=f"Best ({best_model})"),
        Patch(facecolor="#3498db", edgecolor="black", label="Others"),
    ], loc="lower right", fontsize=10)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "densenet_r2_chart.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Chart saved: {out_path}")


def _plot_epoch_chart(df_summary, best_model, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))

    for _, row in df_summary.iterrows():
        color  = "#2ecc71" if row["model"] == best_model else "#3498db"
        marker = "*" if row["model"] == best_model else "o"
        size   = 220 if row["model"] == best_model else 120
        ax.scatter(row["params_M"], row["mean_best_epoch"],
                   c=color, marker=marker, s=size,
                   edgecolors="black", linewidths=0.8, zorder=3)
        ax.annotate(
            f"{row['model']}\nR2={row['mean_r2']:.4f}",
            xy=(row["params_M"], row["mean_best_epoch"]),
            xytext=(row["params_M"] + 0.3, row["mean_best_epoch"] + 1.5),
            fontsize=8, color="#2c3e50",
            arrowprops=dict(arrowstyle="-", color="#bdc3c7", lw=0.8),
        )

    ax.axhline(y=MAX_EPOCHS, color="#e74c3c", linestyle="--",
               linewidth=1.2, alpha=0.7, label=f"Max epochs ({MAX_EPOCHS})")
    if USE_WARMUP:
        ax.axhline(y=WARMUP_EPOCHS, color="#f39c12", linestyle=":",
                   linewidth=1.0, alpha=0.7, label=f"Warmup end ({WARMUP_EPOCHS})")
    ax.set_xlabel("Number of Parameters (M)", fontsize=12)
    ax.set_ylabel("Mean Best Epoch (5-Fold Avg)", fontsize=12)
    ax.set_title(
        "Convergence Analysis: Params vs Mean Best Epoch\n"
        "(Near max -> still converging; Near warmup line -> possible instability)",
        fontsize=10, fontweight="bold"
    )
    ax.xaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "densenet_epoch_chart.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Epoch chart saved: {out_path}")


def _plot_fold_stability(df_all, model_order, out_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    colors  = ["#2ecc71", "#3498db", "#e74c3c"]
    markers = ["o", "s", "^"]

    for i, model_name in enumerate(model_order):
        rows   = df_all[df_all["model"] == model_name].sort_values("fold")
        folds  = rows["fold"].tolist()
        r2s    = rows["val_r2"].tolist()
        mean_r2 = np.mean(r2s)
        ax.plot(folds, r2s,
                color=colors[i], marker=markers[i], linewidth=1.8,
                markersize=7, label=f"{model_name} (mean={mean_r2:.4f})",
                zorder=3)
        for fold, r2 in zip(folds, r2s):
            ax.annotate(f"{r2:.3f}", xy=(fold, r2),
                        xytext=(0, 7), textcoords="offset points",
                        ha="center", fontsize=7, color=colors[i])

    ax.set_xlabel("Fold", fontsize=12)
    ax.set_ylabel("Val R2", fontsize=12)
    ax.set_xticks(list(range(1, N_FOLDS + 1)))
    ax.set_title(
        "Per-Fold Val R2 - Stability Diagnosis\n"
        "(Low outlier folds -> likely BN init instability from Pre-activation structure)",
        fontsize=10, fontweight="bold"
    )
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "densenet_fold_stability.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Fold stability chart saved: {out_path}")


def _write_report(df_summary, df_all, best_model, total_time, out_dir):
    report_path = os.path.join(out_dir, "densenet_compare_report.txt")

    warmup_cfg = (f"True - linear warmup {WARMUP_EPOCHS} epochs "
                  f"(lr: {LR/10:.0e} -> {LR:.0e})")  \
                 if USE_WARMUP else "False"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("DENSENET SIZE ABLATION EXPERIMENT REPORT\n")
        f.write("=" * 70 + "\n\n")

        f.write("[ EXPERIMENT CONFIGURATION ]\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Models         : DenseNet-121 / DenseNet-169 / DenseNet-201\n")
        f.write(f"  Input Group    : {G_BEST} (10ch)\n")
        f.write(f"  Augmentation   : {A_BEST_DESC}\n")
        f.write(f"  CV Strategy    : {N_FOLDS}-Fold CV (seed={SEED})\n")
        f.write(f"  TrainVal Size  : 414 samples\n")
        f.write(f"  Pretrained     : False (all from scratch)\n")
        f.write(f"  Max Epochs     : {MAX_EPOCHS}\n")
        f.write(f"  Early Stopping : patience={PATIENCE}\n")
        f.write(f"  Optimizer      : AdamW (lr={LR}, wd={WEIGHT_DECAY})\n")
        f.write(f"  USE_WARMUP     : {warmup_cfg}\n")
        f.write(f"  Scheduler      : Warmup->CosineAnnealingLR\n")
        f.write(f"  Clip Norm      : {CLIP_NORM}\n")
        f.write(f"  Batch Size     : {BATCH_SIZE}\n")
        f.write(f"  Total Time     : {total_time/3600:.2f} hours\n\n")

        f.write("[ ARCHITECTURE NOTES ]\n")
        f.write("-" * 40 + "\n")
        f.write("  DenseNet-121 : growth_rate=32, blocks=[6-12-24-16], "
                "feat=1024, ~8.0M\n")
        f.write("               regressor: 1024->256->1\n")
        f.write("  DenseNet-169 : growth_rate=32, blocks=[6-12-32-32], "
                "feat=1664, ~14.3M\n")
        f.write("               regressor: 1664->256->1\n")
        f.write("  DenseNet-201 : growth_rate=32, blocks=[6-12-48-32], "
                "feat=1920, ~20.1M\n")
        f.write("               regressor: 1920->256->1\n")
        f.write("  Pre-activation (BN->ReLU->Conv) - forward must apply\n")
        f.write("    F.relu() after features() before avgpool!\n\n")

        f.write("[ SUMMARY RESULTS (sorted by Mean Val R2) ]\n")
        f.write("-" * 70 + "\n")
        header_model = "Model"
        header_params = "Params(M)"
        header_r2 = "Mean R2"
        header_std = "Std R2"
        header_rmse = "RMSE"
        header_epoch = "Epoch"
        header_ms = "ms/batch"
        header_time = "Time(min)"
        f.write(f"  {header_model:<16} {header_params:>9}  {header_r2:>8}  {header_std:>7}  "
                f"{header_rmse:>7}  {header_epoch:>6}  {header_ms:>9}  {header_time:>10}\n")
        f.write("  " + "-" * 67 + "\n")
        for _, row in df_summary.iterrows():
            mark = " <-- BEST" if row["model"] == best_model else ""
            model_val = row["model"]
            params_val = row["params_M"]
            mean_r2_val = row["mean_r2"]
            std_r2_val = row["std_r2"]
            mean_rmse_val = row["mean_rmse"]
            mean_best_epoch_val = row["mean_best_epoch"]
            ms_batch_val = row["ms_per_batch"]
            time_min_val = row["time_min"]
            f.write(
                f"  {model_val:<16} {params_val:>9.2f}  "
                f"{mean_r2_val:>8.4f}  {std_r2_val:>7.4f}  "
                f"{mean_rmse_val:>7.2f}  {mean_best_epoch_val:>6.1f}  "
                f"{ms_batch_val:>9.2f}  {time_min_val:>10.1f}{mark}\n"
            )
        f.write("\n")

        f.write("[ PER-FOLD DETAILED RESULTS ]\n")
        f.write("-" * 70 + "\n")
        for model_name in MODEL_ORDER:
            rows = df_all[df_all["model"] == model_name]
            if rows.empty:
                continue
            f.write(f"\n  {model_name}:\n")
            header_fold = "Fold"
            header_val_r2 = "Val R2"
            header_val_rmse = "Val RMSE"
            header_best_epoch = "BestEpoch"
            header_time_s = "Time(s)"
            f.write(f"    {header_fold:>5}  {header_val_r2:>8}  {header_val_rmse:>10}  "
                    f"{header_best_epoch:>10}  {header_time_s:>9}\n")
            f.write("    " + "-" * 45 + "\n")
            for _, r in rows.iterrows():
                fold_val = int(r["fold"])
                val_r2_val = r["val_r2"]
                val_rmse_val = r["val_rmse"]
                best_epoch_val = int(r["best_epoch"])
                time_s_val = r["time_s"]
                f.write(f"    {fold_val:>5}  {val_r2_val:>8.4f}  "
                        f"{val_rmse_val:>10.2f}  {best_epoch_val:>10}  "
                        f"{time_s_val:>9.1f}\n")
            r2s = rows["val_r2"].tolist()
            ep  = rows["best_epoch"].tolist()
            mean_label = "Mean"
            std_label = "Std"
            f.write(f"    {mean_label:>5}  {np.mean(r2s):>8.4f}  "
                    f"{np.mean(rows['val_rmse']):>10.2f}  "
                    f"{np.mean(ep):>10.1f}\n")
            f.write(f"    {std_label:>5}  {np.std(r2s):>8.4f}  "
                    f"{np.std(rows['val_rmse']):>10.2f}  "
                    f"{np.std(ep):>10.1f}\n")

        f.write("\n\n[ WARMUP EFFECT ANALYSIS ]\n")
        f.write("-" * 70 + "\n")
        f.write(f"  USE_WARMUP = {USE_WARMUP}\n\n")
        if USE_WARMUP:
            f.write(
                f"  Warmup was ENABLED ({WARMUP_EPOCHS} epochs, "
                f"lr: {LR/10:.0e} -> {LR:.0e}).\n"
                f"  To assess its effect, check the Std R2 column above:\n\n"
                f"  Interpretation guide:\n"
                f"    Std R2 < 0.04  -> training stable (warmup helped or not needed)\n"
                f"    Std R2 0.04-0.08 -> moderate instability (warmup partially helped)\n"
                f"    Std R2 > 0.08  -> significant instability remains\n\n"
                f"  Compare with arch_compare DenseNet-121 result (no warmup): Std~0.11\n"
                f"  If current Std is noticeably lower -> warmup is effective for this task.\n"
            )
        else:
            f.write(
                f"  Warmup was DISABLED. Re-run with USE_WARMUP=True to compare.\n"
            )

        RESNET50_R2   = 0.8219
        RESNET50_STD  = 0.0408

        f.write("\n\n[ COMPARISON WITH RESNET BASELINE ]\n")
        f.write("-" * 70 + "\n")
        f.write(f"  ResNet-50 baseline: R2={RESNET50_R2:.4f} +/- {RESNET50_STD:.4f} "
                f"(24.1M params)\n\n")
        header_model2 = "Model"
        header_params2 = "Params(M)"
        header_r2_2 = "Mean R2"
        header_delta = "Delta R2 vs ResNet-50"
        header_eff = "Param efficiency"
        f.write(f"  {header_model2:<16} {header_params2:>9}  {header_r2_2:>8}  "
                f"{header_delta:>18}  {header_eff:>18}\n")
        f.write("  " + "-" * 67 + "\n")
        for _, row in df_summary.set_index("model").reindex(MODEL_ORDER).reset_index().iterrows():
            delta       = row["mean_r2"] - RESNET50_R2
            sign        = "+" if delta >= 0 else ""
            efficiency  = row["mean_r2"] / row["params_M"]
            model_val2 = row["model"]
            params_val2 = row["params_M"]
            mean_r2_val2 = row["mean_r2"]
            f.write(
                f"  {model_val2:<16} {params_val2:>9.2f}  "
                f"{mean_r2_val2:>8.4f}  "
                f"{sign}{delta:>17.4f}  "
                f"{efficiency:>18.4f}\n"
            )

        best_row    = df_summary[df_summary["model"] == best_model].iloc[0]
        worst_row   = df_summary.iloc[-1]

        f.write("\n\n[ CONCLUSION ]\n")
        f.write("-" * 70 + "\n")
        best_params_val = best_row["params_M"]
        best_mean_r2_val = best_row["mean_r2"]
        best_std_r2_val = best_row["std_r2"]
        best_mean_rmse_val = best_row["mean_rmse"]
        best_std_rmse_val = best_row["std_rmse"]
        best_mean_best_epoch_val = best_row["mean_best_epoch"]
        best_ms_batch_val = best_row["ms_per_batch"]
        worst_model_val = worst_row["model"]
        worst_mean_r2_val = worst_row["mean_r2"]
        worst_std_r2_val = worst_row["std_r2"]

        f.write(
            f"  Best model   : {best_model}\n"
            f"  Params       : {best_params_val:.2f}M\n"
            f"  Best R2      : {best_mean_r2_val:.4f} +/- {best_std_r2_val:.4f}\n"
            f"  Best RMSE    : {best_mean_rmse_val:.2f} +/- {best_std_rmse_val:.2f}\n"
            f"  Mean BestEp  : {best_mean_best_epoch_val:.1f} / {MAX_EPOCHS}\n"
            f"  Inference    : {best_ms_batch_val:.2f} ms/batch\n\n"
            f"  Worst model  : {worst_model_val}\n"
            f"  Worst R2     : {worst_mean_r2_val:.4f} +/- {worst_std_r2_val:.4f}\n\n"
        )

        f.write("  Convergence interpretation:\n")
        for _, row in df_summary.iterrows():
            ep = row["mean_best_epoch"]
            if ep >= MAX_EPOCHS * 0.9:
                note = "-> still converging, may benefit from more epochs"
            elif ep <= WARMUP_EPOCHS * 2:
                note = "-> WARNING: best_epoch near warmup end, likely BN instability"
            elif ep <= MAX_EPOCHS * 0.3:
                note = "-> early convergence, check for overfitting"
            else:
                note = "-> healthy convergence range"
            f.write(f"    {row['model']:<16}: epoch~{ep:.1f}  {note}\n")

        f.write(
            f"\n  Recommended next step:\n"
            f"    Use {best_model} as the confirmed DenseNet baseline.\n"
            f"    Proceed to next architecture family comparison.\n"
        )
        f.write("\n" + "=" * 70 + "\n")

    print(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()
