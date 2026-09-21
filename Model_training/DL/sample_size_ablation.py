import os
import sys
import time
import traceback
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(THIS_DIR)
DATA_DIR = os.path.join(BASE_DIR, "0_Data")

DOM_DIR = os.path.join(DATA_DIR, "1_DOM")
INDICES_DIR = os.path.join(DATA_DIR, "indices")
NORM_STATS = os.path.join(DATA_DIR, "norm_stats.json")
SPLIT_CSV = os.path.join(DATA_DIR, "dataset_split_index.csv")
LABEL_CSV = os.path.join(DATA_DIR, "wheat_count_labels.csv")
OUT_DIR = os.path.join(THIS_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, os.path.join(BASE_DIR, "1_G_ablation"))
sys.path.insert(0, os.path.join(BASE_DIR, "3_arch_compare", "2_DenseNet-121"))
sys.path.insert(0, os.path.join(BASE_DIR, "5_final_eval"))

from dataset import (G_CONFIGS, load_norm_stats, load_split_and_labels,
                      preload_all_channels)
from densenet_model import build_densenet
from run_final_eval import (FinalEvalDataset, train_one_fold, eval_on_test,
                             augment_a3, WARMUP_CONFIG)

G_BEST = "G3"
N_FOLDS = 5
MAX_EPOCHS_FULL = 150
PATIENCE = 20
LR = 1e-4
WEIGHT_DECAY = 1e-4
CLIP_NORM = 5.0
BATCH_SIZE = 32

SAMPLE_FRACTIONS = [0.10, 0.25, 0.50, 0.75, 1.00]
N_REPEATS = 3
SEED_BASE = 42

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def stratified_sample(stems, labels, frac, seed, n_bins=5):
    rng = np.random.RandomState(seed)
    values = np.array([labels[s] for s in stems])
    bin_edges = np.quantile(values, np.linspace(0, 1, n_bins + 1))
    bin_ids = np.digitize(values, bin_edges[1:-1])

    selected = []
    for b in range(n_bins):
        bin_stems = [s for s, bid in zip(stems, bin_ids) if bid == b]
        n_take = max(1, round(len(bin_stems) * frac))
        n_take = min(n_take, len(bin_stems))
        chosen = rng.choice(bin_stems, size=n_take, replace=False)
        selected.extend(chosen.tolist())
    return selected


def run_one_fold(subset_arr, tr_idx, val_idx, labels, ch_list, norm_stats,
                  in_channels, test_loader, max_epochs, wup_cfg, fold_id, verbose):
    tr_stems = subset_arr[tr_idx].tolist()
    val_stems = subset_arr[val_idx].tolist()

    train_ds = FinalEvalDataset(tr_stems, labels, ch_list, norm_stats, augment_fn=augment_a3)
    val_ds = FinalEvalDataset(val_stems, labels, ch_list, norm_stats, augment_fn=None)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=0, pin_memory=True)

    model = build_densenet("densenet121", in_channels=in_channels)

    best_val_r2, best_val_rmse, best_epoch, best_state_dict = train_one_fold(
        model=model, train_loader=train_loader, val_loader=val_loader,
        device=DEVICE, max_epochs=max_epochs, lr=LR,
        weight_decay=WEIGHT_DECAY, patience=PATIENCE, clip_norm=CLIP_NORM,
        use_warmup=wup_cfg["use_warmup"], warmup_epochs=wup_cfg["warmup_epochs"],
        verbose=verbose, fold_id=fold_id,
    )
    del model

    test_r2, test_rmse, _, _ = eval_on_test(
        builder_fn=build_densenet, model_key="densenet121",
        in_channels=in_channels, best_state_dict=best_state_dict,
        test_loader=test_loader, device=DEVICE,
    )
    del best_state_dict
    return best_val_r2, test_r2, test_rmse


def sanity_check(train_stems, test_stems, labels, ch_list, norm_stats, in_channels):
    log_path = os.path.join(OUT_DIR, "sanity_check_log.txt")
    print("=" * 70)
    print("[Phase 0] Sanity check - frac=1.0, 1 fold, 3 epochs")
    print("=" * 70)

    try:
        test_ds = FinalEvalDataset(test_stems, labels, ch_list, norm_stats, augment_fn=None)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=0, pin_memory=True)

        subset_arr = np.array(train_stems)
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED_BASE)
        tr_idx, val_idx = next(iter(kf.split(subset_arr)))

        wup_cfg = WARMUP_CONFIG["densenet121"]
        val_r2, test_r2, test_rmse = run_one_fold(
            subset_arr, tr_idx, val_idx, labels, ch_list, norm_stats, in_channels,
            test_loader, max_epochs=3, wup_cfg=wup_cfg, fold_id=0, verbose=True,
        )

        msg = (f"PASS: val_r2={val_r2:.4f}, test_r2={test_r2:.4f}, "
               f"test_rmse={test_rmse:.2f}")
        print(f"\n[Sanity Check] {msg}\n")
        with open(log_path, "w") as f:
            f.write(msg + "\n")
        return True

    except Exception as e:
        err_msg = f"FAIL: {e}\n{traceback.format_exc()}"
        print(f"\n[Sanity Check] {err_msg}\n")
        with open(log_path, "w") as f:
            f.write(err_msg + "\n")
        return False


def main():
    start_total = time.time()

    norm_stats = load_norm_stats(NORM_STATS)
    train_stems, test_stems, labels = load_split_and_labels(SPLIT_CSV, LABEL_CSV)
    all_stems = train_stems + test_stems
    preload_all_channels(all_stems, DOM_DIR, INDICES_DIR, verbose=True)

    ch_list = G_CONFIGS[G_BEST]
    in_channels = len(ch_list)

    ok = sanity_check(train_stems, test_stems, labels, ch_list, norm_stats, in_channels)
    if not ok:
        print("\n" + "!" * 70)
        print("Sanity check failed, program terminated. Please check the error message in results/sanity_check_log.txt,")
        print("and rerun this script after fixing it (Phase 1 has not started yet, so nothing will be re-executed unnecessarily).")
        print("!" * 70)
        sys.exit(1)

    print("=" * 70)
    print("[Phase 1] Full run - 5 fractions x 3 repeats x 5 folds = 75 folds")
    print(f"Fractions : {SAMPLE_FRACTIONS}")
    print(f"Repeats   : {N_REPEATS}")
    print(f"MaxEpochs : {MAX_EPOCHS_FULL}  Patience: {PATIENCE}")
    print(f"Device    : {DEVICE}")
    print("=" * 70)

    test_ds = FinalEvalDataset(test_stems, labels, ch_list, norm_stats, augment_fn=None)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=True)

    wup_cfg = WARMUP_CONFIG["densenet121"]
    all_records = []

    for frac in SAMPLE_FRACTIONS:
        for repeat in range(N_REPEATS):
            seed = SEED_BASE + repeat
            subset_stems = stratified_sample(train_stems, labels, frac, seed)
            n_sub = len(subset_stems)
            subset_arr = np.array(subset_stems)

            print("-" * 70)
            print(f"[frac={frac:.2f}] repeat={repeat} seed={seed} n_samples={n_sub}")
            print("-" * 70)

            kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
            for fold_id, (tr_idx, val_idx) in enumerate(kf.split(subset_arr), start=1):
                fold_start = time.time()
                val_r2, test_r2, test_rmse = run_one_fold(
                    subset_arr, tr_idx, val_idx, labels, ch_list, norm_stats,
                    in_channels, test_loader, max_epochs=MAX_EPOCHS_FULL,
                    wup_cfg=wup_cfg, fold_id=fold_id, verbose=False,
                )
                fold_time = time.time() - fold_start

                all_records.append({
                    "frac": frac, "n_train_samples": n_sub, "repeat": repeat,
                    "fold": fold_id, "val_r2": round(val_r2, 4),
                    "test_r2": round(test_r2, 4), "test_rmse": round(test_rmse, 2),
                    "time_s": round(fold_time, 1),
                })
                print(f"  fold {fold_id}: ValR2={val_r2:.4f} TestR2={test_r2:.4f} "
                      f"TestRMSE={test_rmse:.2f} ({fold_time:.0f}s)")

                pd.DataFrame(all_records).to_csv(
                    os.path.join(OUT_DIR, "sample_size_ablation_all.csv"), index=False)

    df = pd.DataFrame(all_records)
    summary = df.groupby(["frac", "n_train_samples"]).agg(
        mean_val_r2=("val_r2", "mean"), std_val_r2=("val_r2", "std"),
        mean_test_r2=("test_r2", "mean"), std_test_r2=("test_r2", "std"),
        mean_test_rmse=("test_rmse", "mean"), std_test_rmse=("test_rmse", "std"),
    ).reset_index().sort_values("n_train_samples")
    summary.to_csv(os.path.join(OUT_DIR, "sample_size_ablation_summary.csv"), index=False)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.errorbar(summary["n_train_samples"], summary["mean_test_r2"],
                yerr=summary["std_test_r2"], fmt="o-", color="#2ecc71",
                ecolor="#2c3e50", elinewidth=1.5, capsize=5, markersize=8,
                linewidth=2, zorder=3)
    ax.axvline(414, color="#e74c3c", linestyle="--", linewidth=1.5,
               label="Full development pool (n=414, used in paper)")
    ax.set_xscale("log")
    ax.set_xlabel("Number of Training Samples (log scale)", fontsize=12)
    ax.set_ylabel("Sealed Test R2 (mean +/- std across repeats)", fontsize=12)
    ax.set_title("Sample-Size Sensitivity: DenseNet-121 + G3 + A3",
                  fontsize=12, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=10, loc="lower right")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "learning_curve.png"), dpi=150, bbox_inches="tight")
    plt.close()

    total_time = time.time() - start_total
    print("\n" + "=" * 70)
    print(summary.to_string(index=False))
    print(f"\nTotal time: {total_time/3600:.2f}h")
    print(f"Results saved to: {OUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()