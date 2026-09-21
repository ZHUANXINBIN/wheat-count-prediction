import os
import sys
import copy
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
BASE_DIR    = os.path.dirname(THIS_DIR)
DATA_DIR    = os.path.join(BASE_DIR, "0_Data")

DOM_DIR     = os.path.join(DATA_DIR, "1_DOM")
INDICES_DIR = os.path.join(DATA_DIR, "indices")
NORM_STATS  = os.path.join(DATA_DIR, "norm_stats.json")
SPLIT_CSV   = os.path.join(DATA_DIR, "dataset_split_index.csv")
LABEL_CSV   = os.path.join(DATA_DIR, "wheat_count_labels.csv")
OUT_DIR     = os.path.join(THIS_DIR, "results")

sys.path.insert(0, os.path.join(BASE_DIR, "1_G_ablation"))
sys.path.insert(0, os.path.join(BASE_DIR, "3_arch_compare", "1_ResNet"))
sys.path.insert(0, os.path.join(BASE_DIR, "3_arch_compare", "2_DenseNet-121"))
sys.path.insert(0, os.path.join(BASE_DIR, "3_arch_compare", "5_ConvNeXt-Tiny"))

from dataset import (G_CONFIGS, load_norm_stats, load_split_and_labels,
                     preload_all_channels, PRELOAD_CACHE, augment_a1)
from resnet_model   import build_resnet
from densenet_model import build_densenet
from convnext_model import build_convnext

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

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TARGET_MODELS = [
    ("ResNet-50",     build_resnet,    "resnet50",      24.1),
    ("DenseNet-121",  build_densenet,  "densenet121",    8.0),
    ("ConvNeXt-Tiny", build_convnext,  "convnext_tiny", 28.6),
]

WARMUP_CONFIG = {
    "resnet50":     {"use_warmup": False, "warmup_epochs": 0},
    "densenet121":  {"use_warmup": True,  "warmup_epochs": 10},
    "convnext_tiny":{"use_warmup": False, "warmup_epochs": 0},
}


def augment_a3(image: np.ndarray) -> np.ndarray:
    image = augment_a1(image)
    if random.random() < 0.1:
        ch    = random.randint(0, image.shape[0] - 1)
        image = image.copy()
        image[ch] = 0.0
    return image


class FinalEvalDataset(Dataset):
    def __init__(self, stems, labels, channel_list, norm_stats, augment_fn=None):
        assert len(PRELOAD_CACHE) > 0

        self.augment_fn = augment_fn
        means = np.array([norm_stats[ch]["mean"] for ch in channel_list],
                         dtype=np.float32).reshape(-1, 1, 1)
        stds  = np.array([norm_stats[ch]["std"]  for ch in channel_list],
                         dtype=np.float32).reshape(-1, 1, 1)
        stds  = np.where(stds == 0, 1.0, stds)

        self.images, self.label_list = [], []
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
    def __init__(self, optimizer, lr_max, warmup_epochs, total_epochs):
        self.optimizer     = optimizer
        self.lr_max        = lr_max
        self.lr_start      = lr_max / 10.0
        self.warmup_epochs = warmup_epochs
        self.cosine_epochs = total_epochs - warmup_epochs
        self._set_lr(1)

    def step(self, epoch):
        self._set_lr(epoch)

    def _set_lr(self, epoch):
        if epoch <= self.warmup_epochs:
            frac = (epoch - 1) / max(self.warmup_epochs - 1, 1)
            lr   = self.lr_start + frac * (self.lr_max - self.lr_start)
        else:
            t    = epoch - self.warmup_epochs
            T    = self.cosine_epochs
            lr   = self.lr_max * 1e-3 + 0.5 * (self.lr_max - self.lr_max * 1e-3) * (
                1 + np.cos(np.pi * t / max(T, 1)))
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def get_last_lr(self):
        return [pg["lr"] for pg in self.optimizer.param_groups]


def train_one_fold(model, train_loader, val_loader, device,
                   max_epochs, lr, weight_decay, patience, clip_norm,
                   use_warmup=False, warmup_epochs=10,
                   verbose=True, fold_id=None):

    model     = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)

    if use_warmup:
        scheduler = WarmupCosineScheduler(optimizer, lr_max=lr,
                                          warmup_epochs=warmup_epochs,
                                          total_epochs=max_epochs)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs, eta_min=lr * 1e-3)

    criterion       = nn.MSELoss()
    best_r2         = -np.inf
    best_rmse       = np.inf
    best_epoch      = 0
    best_state_dict = None
    no_improve_cnt  = 0
    fold_str        = f"Fold {fold_id}" if fold_id else "Fold"

    for epoch in range(1, max_epochs + 1):
        if use_warmup:
            scheduler.step(epoch)

        model.train()
        train_losses = []
        for images, lbls in train_loader:
            images = images.to(device, non_blocking=True)
            lbls   = lbls.to(device, non_blocking=True)
            optimizer.zero_grad()
            preds  = model(images)
            loss   = criterion(preds, lbls)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            optimizer.step()
            train_losses.append(loss.item())

        if not use_warmup:
            scheduler.step()

        model.eval()
        preds_all, labels_all = [], []
        with torch.no_grad():
            for images, lbls in val_loader:
                images = images.to(device, non_blocking=True)
                preds_all.append(model(images).cpu().numpy())
                labels_all.append(lbls.numpy())

        preds_all  = np.concatenate(preds_all)
        labels_all = np.concatenate(labels_all)
        val_r2     = r2_score(labels_all, preds_all)
        val_rmse   = float(np.sqrt(np.mean((preds_all - labels_all) ** 2)))

        if val_r2 > best_r2:
            best_r2         = val_r2
            best_rmse       = val_rmse
            best_epoch      = epoch
            no_improve_cnt  = 0
            best_state_dict = copy.deepcopy(model.state_dict())
        else:
            no_improve_cnt += 1

        if verbose:
            wu_tag = (f" [WU{epoch}/{warmup_epochs}]"
                      if use_warmup and epoch <= warmup_epochs else "")
            mark   = "*" if no_improve_cnt == 0 else " "
            lr_now = scheduler.get_last_lr()[0]
            print(f"  [{fold_str}] Epoch {epoch:3d}/{max_epochs}{wu_tag} | "
                  f"Loss={np.mean(train_losses):8.1f} | "
                  f"ValR2={val_r2:.4f} {mark} | "
                  f"ValRMSE={val_rmse:6.2f} | "
                  f"LR={lr_now:.2e} | NoImp={no_improve_cnt}/{patience}")

        if no_improve_cnt >= patience:
            if verbose:
                print(f"  [{fold_str}] Early stop @ epoch {epoch}. "
                      f"BestValR2={best_r2:.4f} @ epoch {best_epoch}")
            break

    if verbose and no_improve_cnt < patience:
        print(f"  [{fold_str}] Done {max_epochs} epochs. "
              f"BestValR2={best_r2:.4f} @ epoch {best_epoch}")

    return best_r2, best_rmse, best_epoch, best_state_dict


def eval_on_test(builder_fn, model_key, in_channels, best_state_dict,
                 test_loader, device):
    model = builder_fn(model_key, in_channels=in_channels).to(device)
    model.load_state_dict(best_state_dict)
    model.eval()

    preds_all, labels_all = [], []
    with torch.no_grad():
        for images, lbls in test_loader:
            images = images.to(device, non_blocking=True)
            preds_all.append(model(images).cpu().numpy())
            labels_all.append(lbls.numpy())

    preds_all  = np.concatenate(preds_all)
    labels_all = np.concatenate(labels_all)
    test_r2    = r2_score(labels_all, preds_all)
    test_rmse  = float(np.sqrt(np.mean((preds_all - labels_all) ** 2)))

    del model
    return test_r2, test_rmse, preds_all, labels_all


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    start_total = time.time()

    print("=" * 70)
    print("FINAL EVALUATION - Test Set Unlock")
    print(f"Models    : {[m[0] for m in TARGET_MODELS]}")
    print(f"Input     : {G_BEST} (10 channels)  Aug: A3")
    print(f"Device    : {DEVICE}")
    print(f"CV        : {N_FOLDS}-Fold (seed={SEED}, same as ablation)")
    print(f"MaxEpoch  : {MAX_EPOCHS}  Patience: {PATIENCE}")
    print(f"LR/WD     : {LR}/{WEIGHT_DECAY}  Clip: {CLIP_NORM}")
    print(f"Test set is SEALED until after training completes each fold.")
    print("=" * 70)

    norm_stats                      = load_norm_stats(NORM_STATS)
    train_stems, test_stems, labels = load_split_and_labels(SPLIT_CSV, LABEL_CSV)
    all_stems                       = train_stems + test_stems
    preload_all_channels(all_stems, DOM_DIR, INDICES_DIR, verbose=True)

    train_stems = np.array(train_stems)
    ch_list     = G_CONFIGS[G_BEST]
    in_channels = len(ch_list)

    print(f"\nTrainVal: {len(train_stems)} | Test (sealed): {len(test_stems)} | "
          f"Channels: {in_channels}\n")

    test_ds = FinalEvalDataset(test_stems, labels, ch_list, norm_stats,
                               augment_fn=None)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=0, pin_memory=True)
    print(f"Test loader built: {len(test_stems)} samples, "
          f"{len(test_loader)} batches. SEALED.\n")

    kf           = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_indices = list(kf.split(train_stems))

    all_fold_results = []
    model_summaries  = []

    for display_name, builder_fn, model_key, params_M in TARGET_MODELS:
        wup_cfg     = WARMUP_CONFIG[model_key]
        use_warmup  = wup_cfg["use_warmup"]
        warmup_eps  = wup_cfg["warmup_epochs"]
        start_m     = time.time()

        print("-" * 70)
        print(f"[{display_name}]  key={model_key}  params~{params_M}M  "
              f"warmup={use_warmup}")
        print("-" * 70)

        fold_val_r2s,  fold_val_rmses  = [], []
        fold_test_r2s, fold_test_rmses = [], []
        fold_epochs                    = []
        test_preds_per_fold  = []
        test_labels_ref      = None

        for fold_id, (tr_idx, val_idx) in enumerate(fold_indices, start=1):
            fold_start = time.time()
            tr_stems   = train_stems[tr_idx].tolist()
            val_stems  = train_stems[val_idx].tolist()

            train_ds = FinalEvalDataset(tr_stems,  labels, ch_list, norm_stats,
                                        augment_fn=augment_a3)
            val_ds   = FinalEvalDataset(val_stems, labels, ch_list, norm_stats,
                                        augment_fn=None)

            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                                      shuffle=True,  num_workers=0, pin_memory=True)
            val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                                      shuffle=False, num_workers=0, pin_memory=True)

            model = builder_fn(model_key, in_channels=in_channels)

            best_val_r2, best_val_rmse, best_epoch, best_state_dict = \
                train_one_fold(
                    model         = model,
                    train_loader  = train_loader,
                    val_loader    = val_loader,
                    device        = DEVICE,
                    max_epochs    = MAX_EPOCHS,
                    lr            = LR,
                    weight_decay  = WEIGHT_DECAY,
                    patience      = PATIENCE,
                    clip_norm     = CLIP_NORM,
                    use_warmup    = use_warmup,
                    warmup_epochs = warmup_eps,
                    verbose       = True,
                    fold_id       = fold_id,
                )
            del model

            test_r2, test_rmse, test_preds, test_labels = eval_on_test(
                builder_fn      = builder_fn,
                model_key       = model_key,
                in_channels     = in_channels,
                best_state_dict = best_state_dict,
                test_loader     = test_loader,
                device          = DEVICE,
            )
            del best_state_dict

            fold_time = time.time() - fold_start

            fold_val_r2s.append(best_val_r2)
            fold_val_rmses.append(best_val_rmse)
            fold_test_r2s.append(test_r2)
            fold_test_rmses.append(test_rmse)
            fold_epochs.append(best_epoch)
            test_preds_per_fold.append(test_preds)
            if test_labels_ref is None:
                test_labels_ref = test_labels

            all_fold_results.append({
                "model":        display_name,
                "model_key":    model_key,
                "fold":         fold_id,
                "val_r2":       round(best_val_r2,  4),
                "val_rmse":     round(best_val_rmse, 2),
                "test_r2":      round(test_r2,       4),
                "test_rmse":    round(test_rmse,      2),
                "best_epoch":   best_epoch,
                "time_s":       round(fold_time,      1),
            })

            print(f"  --> [{display_name}] Fold {fold_id}: "
                  f"ValR2={best_val_r2:.4f} | "
                  f"TestR2={test_r2:.4f} | "
                  f"TestRMSE={test_rmse:.2f} | "
                  f"epoch={best_epoch} | {fold_time/60:.1f}min\n")

        ensemble_preds = np.mean(np.stack(test_preds_per_fold, axis=0), axis=0)
        ensemble_r2    = r2_score(test_labels_ref, ensemble_preds)
        ensemble_rmse  = float(np.sqrt(np.mean((ensemble_preds - test_labels_ref)**2)))

        m_time = time.time() - start_m
        model_summaries.append({
            "model":             display_name,
            "model_key":         model_key,
            "params_M":          params_M,
            "mean_val_r2":       round(float(np.mean(fold_val_r2s)),   4),
            "std_val_r2":        round(float(np.std(fold_val_r2s)),    4),
            "mean_val_rmse":     round(float(np.mean(fold_val_rmses)), 2),
            "mean_test_r2":      round(float(np.mean(fold_test_r2s)),   4),
            "std_test_r2":       round(float(np.std(fold_test_r2s)),    4),
            "mean_test_rmse":    round(float(np.mean(fold_test_rmses)), 2),
            "std_test_rmse":     round(float(np.std(fold_test_rmses)),  2),
            "ensemble_test_r2":  round(ensemble_r2,   4),
            "ensemble_test_rmse":round(ensemble_rmse,  2),
            "mean_best_epoch":   round(float(np.mean(fold_epochs)), 1),
            "time_min":          round(m_time / 60, 1),
        })

        mean_val_r2_disp = np.mean(fold_val_r2s)
        std_val_r2_disp = np.std(fold_val_r2s)
        mean_test_r2_disp = np.mean(fold_test_r2s)
        std_test_r2_disp = np.std(fold_test_r2s)
        mean_test_rmse_disp = np.mean(fold_test_rmses)

        print(f"  ==> [{display_name}] "
              f"ValR2={mean_val_r2_disp:.4f}+-{std_val_r2_disp:.4f} | "
              f"TestR2={mean_test_r2_disp:.4f}+-{std_test_r2_disp:.4f} | "
              f"TestRMSE={mean_test_rmse_disp:.2f} | "
              f"Ensemble={ensemble_r2:.4f} | "
              f"total={m_time/60:.1f}min\n")

    total_time = time.time() - start_total

    df_folds   = pd.DataFrame(all_fold_results)
    df_summary = pd.DataFrame(model_summaries)
    df_summary_sorted = df_summary.sort_values(
        "mean_test_r2", ascending=False).reset_index(drop=True)
    best_model = df_summary_sorted.iloc[0]["model"]

    df_folds.to_csv(os.path.join(OUT_DIR, "final_eval_all_folds.csv"),
                    index=False, encoding="utf-8-sig")
    df_summary.to_csv(os.path.join(OUT_DIR, "final_eval_summary.csv"),
                      index=False, encoding="utf-8-sig")

    _plot_val_vs_test(df_summary_sorted, best_model, OUT_DIR)
    _plot_fold_test_stability(df_folds, [m[0] for m in TARGET_MODELS], OUT_DIR)
    _write_report(df_summary_sorted, df_folds, best_model, total_time, OUT_DIR)

    best_mean_test_r2 = df_summary_sorted.iloc[0]["mean_test_r2"]
    best_std_test_r2 = df_summary_sorted.iloc[0]["std_test_r2"]

    print("=" * 70)
    print(f"All done. Total time: {total_time/3600:.2f}h")
    print(f"Best model (Test R2): {best_model}  "
          f"({best_mean_test_r2:.4f}+-{best_std_test_r2:.4f})")
    print(f"Results -> {OUT_DIR}")
    print("=" * 70)


def _plot_val_vs_test(df_summary, best_model, out_dir):
    models   = df_summary["model"].tolist()
    n        = len(models)
    x        = np.arange(n)
    width    = 0.32

    val_means   = df_summary["mean_val_r2"].tolist()
    val_stds    = df_summary["std_val_r2"].tolist()
    test_means  = df_summary["mean_test_r2"].tolist()
    test_stds   = df_summary["std_test_r2"].tolist()
    ens_means   = df_summary["ensemble_test_r2"].tolist()

    fig, ax = plt.subplots(figsize=(10, 6))

    bars_val  = ax.bar(x - width/2, val_means,  width, yerr=val_stds,
                       label="Val R2 (CV, selection only)",
                       color="#3498db", edgecolor="black", linewidth=0.7,
                       capsize=5, alpha=0.7, zorder=3)
    bars_test = ax.bar(x + width/2, test_means, width, yerr=test_stds,
                       label="Test R2 (sealed, paper reportable)",
                       color="#2ecc71", edgecolor="black", linewidth=0.7,
                       capsize=5, zorder=3)

    for i, (xi, ens) in enumerate(zip(x, ens_means)):
        ax.scatter(xi + width/2, ens, marker="D", s=60,
                   color="#e74c3c", zorder=5,
                   label="Test R2 ensemble" if i == 0 else "")

    for bar, mean, std in zip(bars_test, test_means, test_stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + 0.005,
                f"{mean:.4f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color="#1a5e38")

    best_idx = models.index(best_model)
    ax.text(x[best_idx] + width/2, 0.01, "BEST",
            ha="center", va="bottom", fontsize=8,
            fontweight="bold", color="white")

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel("R2", fontsize=12)
    ax.set_title(
        "Final Evaluation: Val R2 vs Test R2\n"
        "(Val = model selection only; Test = sealed, paper-reportable generalization)",
        fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(val_means + test_means) + max(val_stds + test_stds) + 0.12)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, loc="lower right")

    plt.tight_layout()
    out_path = os.path.join(out_dir, "final_val_vs_test.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Chart saved: {out_path}")


def _plot_fold_test_stability(df_folds, model_names, out_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    colors  = ["#2ecc71", "#3498db", "#e74c3c",
               "#f39c12", "#9b59b6"]

    for i, model_name in enumerate(model_names):
        rows = df_folds[df_folds["model"] == model_name].sort_values("fold")
        folds    = rows["fold"].tolist()
        test_r2s = rows["test_r2"].tolist()
        mean_r2  = np.mean(test_r2s)
        ax.plot(folds, test_r2s,
                color=colors[i % len(colors)], marker="o",
                linewidth=2, markersize=8,
                label=f"{model_name} (mean={mean_r2:.4f})", zorder=3)
        for f, r2 in zip(folds, test_r2s):
            ax.annotate(f"{r2:.3f}", xy=(f, r2),
                        xytext=(0, 8), textcoords="offset points",
                        ha="center", fontsize=7,
                        color=colors[i % len(colors)])

    ax.set_xlabel("Fold", fontsize=12)
    ax.set_ylabel("Test R2", fontsize=12)
    ax.set_xticks(list(range(1, 6)))
    ax.set_title(
        "Per-Fold Test R2 - Generalization Stability\n"
        "(Consistent across folds -> robust generalization)",
        fontsize=10, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "final_fold_test_stability.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Stability chart saved: {out_path}")


def _write_report(df_summary, df_folds, best_model, total_time, out_dir):
    report_path = os.path.join(out_dir, "final_eval_report.txt")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 75 + "\n")
        f.write("FINAL EVALUATION REPORT - SEALED TEST SET\n")
        f.write("=" * 75 + "\n\n")

        f.write("[ EXPERIMENT SETUP ]\n")
        f.write("-" * 45 + "\n")
        f.write(f"  Target models  : ResNet-50 / DenseNet-121 / ConvNeXt-Tiny\n")
        f.write(f"  Input          : G3 (10ch)  Aug: A3\n")
        f.write(f"  CV             : 5-Fold (seed={SEED}, same as all ablation)\n")
        test_count = len(df_folds[df_folds['fold']==1]['test_r2'])
        f.write(f"  Test set size  : {test_count} <- "
                f"per-fold test (same sealed set each time)\n")
        f.write(f"  MaxEpoch       : {MAX_EPOCHS}  Patience: {PATIENCE}\n")
        f.write(f"  DenseNet warmup: 10 epochs (Pre-activation BN stability)\n")
        f.write(f"  Total time     : {total_time/3600:.2f} hours\n\n")

        f.write("[ PAPER-REPORTABLE RESULTS (Test R2, sealed set) ]\n")
        f.write("-" * 75 + "\n")
        f.write(f"  These are the ONLY numbers you should report in the paper.\n")
        f.write(f"  Val R2 was used ONLY for model selection and must NOT be\n")
        f.write(f"    reported as final performance.\n\n")

        header_model = "Model"
        header_params = "Params"
        header_test_r2 = "TestR2(mean+-std)"
        header_test_rmse = "TestRMSE(mean+-std)"
        header_ensemble = "Ensemble"
        header_epavg = "EpAvg"
        f.write(f"  {header_model:<18} {header_params:>7}  {header_test_r2:>18}  "
                f"{header_test_rmse:>20}  {header_ensemble:>10}  {header_epavg:>7}\n")
        f.write("  " + "-" * 72 + "\n")
        for _, row in df_summary.iterrows():
            mark = " <- BEST" if row["model"] == best_model else ""
            model_val = row["model"]
            params_val = row["params_M"]
            mean_test_r2_val = row["mean_test_r2"]
            std_test_r2_val = row["std_test_r2"]
            mean_test_rmse_val = row["mean_test_rmse"]
            std_test_rmse_val = row["std_test_rmse"]
            ensemble_val = row["ensemble_test_r2"]
            epoch_val = row["mean_best_epoch"]
            f.write(
                f"  {model_val:<18} {params_val:>6.1f}M  "
                f"{mean_test_r2_val:.4f} +- {std_test_r2_val:.4f}  "
                f"    {mean_test_rmse_val:.2f} +- {std_test_rmse_val:.2f}  "
                f"     {ensemble_val:.4f}  "
                f"  {epoch_val:>6.1f}{mark}\n"
            )
        f.write("\n")

        f.write("[ VAL R2 REFERENCE (model selection only, NOT for paper) ]\n")
        f.write("-" * 75 + "\n")
        header_model2 = "Model"
        header_val_r2 = "ValR2(mean+-std)"
        header_val_rmse = "ValRMSE"
        f.write(f"  {header_model2:<18} {header_val_r2:>18}  {header_val_rmse:>9}\n")
        f.write("  " + "-" * 50 + "\n")
        for _, row in df_summary.iterrows():
            model_val2 = row["model"]
            mean_val_r2_val = row["mean_val_r2"]
            std_val_r2_val = row["std_val_r2"]
            mean_val_rmse_val = row["mean_val_rmse"]
            f.write(f"  {model_val2:<18} "
                    f"{mean_val_r2_val:.4f} +- {std_val_r2_val:.4f}  "
                    f"{mean_val_rmse_val:>9.2f}\n")
        f.write("\n")

        f.write("[ PER-FOLD DETAIL ]\n")
        f.write("-" * 75 + "\n")
        for model_name in [m[0] for m in TARGET_MODELS]:
            rows = df_folds[df_folds["model"] == model_name].copy()
            if rows.empty:
                continue
            f.write(f"\n  {model_name}:\n")
            header_fold = "Fold"
            header_vr2 = "ValR2"
            header_tr2 = "TestR2"
            header_trmse = "TestRMSE"
            header_bep = "BestEp"
            header_time = "Time(s)"
            f.write(f"    {header_fold:>5}  {header_vr2:>8}  {header_tr2:>8}  "
                    f"{header_trmse:>10}  {header_bep:>8}  {header_time:>8}\n")
            f.write("    " + "-" * 52 + "\n")
            for _, r in rows.iterrows():
                fold_val = int(r["fold"])
                val_r2_val = r["val_r2"]
                test_r2_val = r["test_r2"]
                test_rmse_val = r["test_rmse"]
                best_epoch_val = int(r["best_epoch"])
                time_s_val = r["time_s"]
                f.write(f"    {fold_val:>5}  {val_r2_val:>8.4f}  "
                        f"{test_r2_val:>8.4f}  {test_rmse_val:>10.2f}  "
                        f"{best_epoch_val:>8}  {time_s_val:>8.1f}\n")
            tv = rows["val_r2"].tolist()
            tt = rows["test_r2"].tolist()
            tr = rows["test_rmse"].tolist()
            te = rows["best_epoch"].tolist()
            mean_label = "Mean"
            std_label = "Std"
            f.write(f"    {mean_label:>5}  {np.mean(tv):>8.4f}  "
                    f"{np.mean(tt):>8.4f}  {np.mean(tr):>10.2f}  "
                    f"{np.mean(te):>8.1f}\n")
            f.write(f"    {std_label:>5}  {np.std(tv):>8.4f}  "
                    f"{np.std(tt):>8.4f}  {np.std(tr):>10.2f}\n")

        f.write("\n\n[ VAL -> TEST GENERALIZATION GAP ]\n")
        f.write("-" * 75 + "\n")
        f.write("  A large positive gap (Val > Test) indicates overfitting to\n"
                "  the validation folds via hyperparameter tuning.\n"
                "  A small or negative gap means generalization is robust.\n\n")
        for _, row in df_summary.iterrows():
            gap  = row["mean_val_r2"] - row["mean_test_r2"]
            sign = "+" if gap >= 0 else ""
            interp = ("-> small gap, robust" if abs(gap) < 0.02
                      else "-> moderate gap" if abs(gap) < 0.05
                      else "-> WARNING: large gap, possible overfit to CV")
            f.write(f"  {row['model']:<18}: Val-Test gap = {sign}{gap:.4f}  {interp}\n")

        best_row = df_summary[df_summary["model"] == best_model].iloc[0]
        best_params_val = best_row["params_M"]
        best_mean_test_r2_val = best_row["mean_test_r2"]
        best_std_test_r2_val = best_row["std_test_r2"]
        best_mean_test_rmse_val = best_row["mean_test_rmse"]
        best_std_test_rmse_val = best_row["std_test_rmse"]
        best_ensemble_val = best_row["ensemble_test_r2"]

        f.write(f"\n\n[ CONCLUSION ]\n")
        f.write("-" * 75 + "\n")
        f.write(
            f"  Best model (Test R2) : {best_model}\n"
            f"  Params               : {best_params_val:.1f}M\n"
            f"  Test R2  (paper)     : {best_mean_test_r2_val:.4f} +- "
            f"{best_std_test_r2_val:.4f}\n"
            f"  Test RMSE            : {best_mean_test_rmse_val:.2f} +- "
            f"{best_std_test_rmse_val:.2f}\n"
            f"  Ensemble Test R2     : {best_ensemble_val:.4f}\n\n"
            f"  These numbers are the final, citable performance metrics.\n"
            f"  Do NOT report Val R2 in the paper as the primary result.\n"
        )
        f.write("\n" + "=" * 75 + "\n")

    print(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()