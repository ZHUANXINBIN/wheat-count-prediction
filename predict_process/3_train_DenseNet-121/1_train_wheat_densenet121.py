import os
import re
import copy
import json
import random
import warnings
import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import densenet121
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import tkinter as tk
from tkinter import filedialog, messagebox
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

SEED          = 42
VAL_RATIO     = 0.15
MAX_EPOCHS    = 150
PATIENCE      = 20
LR            = 1e-4
WEIGHT_DECAY  = 1e-4
CLIP_NORM     = 5.0
BATCH_SIZE    = 32
USE_WARMUP    = True
WARMUP_EPOCHS = 10
DROPOUT       = 0.3

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

BAND_INDEX = {
    "B1_Blue":      1,
    "B2_Green_RGB": 2,
    "B3_Red_RGB":   3,
    "B4_Green560":  4,
    "B5_Red650":    5,
    "B6_RedEdge":   6,
    "B7_NIR":       7,
}

CHANNEL_LIST = [
    "B1_Blue", "B2_Green_RGB", "B3_Red_RGB", "B4_Green560",
    "B5_Red650", "B6_RedEdge", "B7_NIR",
    "NDVI", "GNDVI", "NDRE",
]
IN_CHANNELS = len(CHANNEL_LIST)


def select_label_file() -> str:
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Select the label file (contains DOM filename + wheat count, .csv or .xlsx)",
        filetypes=[("Excel/CSV files", "*.csv *.xlsx *.xls"),
                  ("CSV files", "*.csv"),
                  ("Excel files", "*.xlsx *.xls"),
                  ("All files", "*.*")]
    )
    root.destroy()
    return path


def select_folder(title: str) -> str:
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askdirectory(title=title)
    root.destroy()
    return path


def load_label_table(label_path: str) -> pd.DataFrame:
    ext = os.path.splitext(label_path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(label_path, encoding="utf-8-sig")
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(label_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    return df


def guess_column(df: pd.DataFrame, keywords: list) -> str:
    cols = list(df.columns)
    for kw in keywords:
        for c in cols:
            if kw.lower() in str(c).lower():
                return c
    return None


def tif_stem(filename: str) -> str:
    for ext in (".tiff", ".tif"):
        if filename.lower().endswith(ext):
            return filename[: len(filename) - len(ext)]
    return os.path.splitext(filename)[0]


def build_label_dict(label_path: str, dom_dir: str):
    df = load_label_table(label_path)
    print(f"\nLabel file columns: {list(df.columns)}")

    fname_col = guess_column(df, ["filename", "文件名", "file_name", "stem", "name", "image", "tif"])
    count_col = guess_column(df, ["wheat_count", "count", "数量", "穗数", "小麦"])

    if fname_col is None or count_col is None:
        print("\nCould not automatically identify column names, please specify manually.")
        print(f"  Auto-detected filename column: {fname_col}")
        print(f"  Auto-detected count column   : {count_col}")
        print(f"  Available columns: {list(df.columns)}")
        if fname_col is None:
            fname_col = input("Please enter the filename column name: ").strip()
        if count_col is None:
            count_col = input("Please enter the wheat count column name: ").strip()

    print(f"Using filename column: '{fname_col}'  count column: '{count_col}'")

    df = df[[fname_col, count_col]].dropna()
    df[fname_col] = df[fname_col].astype(str).str.strip()

    df["_stem"] = df[fname_col].apply(tif_stem)

    tif_files = [f for f in os.listdir(dom_dir)
                if f.lower().endswith(".tif") or f.lower().endswith(".tiff")]
    tif_stem_map = {tif_stem(f): f for f in tif_files}

    labels = {}
    unmatched_label_rows = []
    for _, row in df.iterrows():
        stem = row["_stem"]
        if stem in tif_stem_map:
            labels[stem] = float(row[count_col])
        else:
            unmatched_label_rows.append(row[fname_col])

    matched_stems = list(labels.keys())
    matched_stem_set = set(matched_stems)
    unmatched_tif_files = [f for f in tif_files if tif_stem(f) not in matched_stem_set]

    return matched_stems, labels, unmatched_label_rows, unmatched_tif_files, tif_stem_map


def safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(b != 0, a / b, np.nan)
    return result.astype(np.float32)


def compute_indices(G, R, RE, N) -> dict:
    raw = {
        "NDVI":  safe_div(N - R,  N + R),
        "GNDVI": safe_div(N - G,  N + G),
        "NDRE":  safe_div(N - RE, N + RE),
    }
    out = {}
    for name, arr in raw.items():
        arr = np.where(np.isinf(arr), np.nan, arr)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        out[name] = arr.astype(np.float32)
    return out


def load_all_channels_for_stem(stem: str, dom_dir: str, tif_stem_map: dict):
    fname = tif_stem_map[stem]
    tif_path = os.path.join(dom_dir, fname)
    try:
        with rasterio.open(tif_path) as src:
            if src.count < 7:
                print(f"\n  [{stem}] Insufficient band count (only {src.count})")
                return None
            bands = {}
            for name, num in BAND_INDEX.items():
                bands[name] = src.read(num).astype(np.float32)
    except Exception as e:
        print(f"\n  [{stem}] Read failed: {e}")
        return None

    G, R, RE, N = bands["B4_Green560"], bands["B5_Red650"], bands["B6_RedEdge"], bands["B7_NIR"]
    indices = compute_indices(G, R, RE, N)
    return {**bands, **indices}


def augment_a1(image: np.ndarray) -> np.ndarray:
    if random.random() < 0.5:
        image = np.flip(image, axis=2)
    if random.random() < 0.5:
        image = np.flip(image, axis=1)
    k = random.randint(0, 3)
    if k > 0:
        image = np.rot90(image, k=k, axes=(1, 2))
    return np.ascontiguousarray(image)


def augment_a3(image: np.ndarray) -> np.ndarray:
    image = augment_a1(image)
    if random.random() < 0.1:
        ch = random.randint(0, image.shape[0] - 1)
        image = image.copy()
        image[ch] = 0.0
    return image


class WheatDataset(Dataset):
    def __init__(self, stems, labels, raw_cache, channel_list, norm_stats, augment_fn=None):
        self.augment_fn = augment_fn
        means = np.array([norm_stats[ch]["mean"] for ch in channel_list],
                         dtype=np.float32).reshape(-1, 1, 1)
        stds  = np.array([norm_stats[ch]["std"]  for ch in channel_list],
                         dtype=np.float32).reshape(-1, 1, 1)
        stds  = np.where(stds == 0, 1.0, stds)

        self.images, self.label_list = [], []
        for stem in stems:
            ch_arrays = [raw_cache[stem][ch] for ch in channel_list]
            image = np.stack(ch_arrays, axis=0)
            image = (image - means) / stds
            self.images.append(image.astype(np.float32))
            self.label_list.append(float(labels[stem]))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].copy()
        if self.augment_fn is not None:
            image = self.augment_fn(image)
        return (torch.from_numpy(np.ascontiguousarray(image)),
                torch.tensor(self.label_list[idx], dtype=torch.float32))


def _init_weights(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class WheatDenseNet121(nn.Module):
    def __init__(self, in_channels: int, dropout: float = 0.3):
        super().__init__()
        backbone = densenet121(weights=None)
        backbone.features.conv0 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.features  = backbone.features
        self.regressor = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, 1),
        )
        _init_weights(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.flatten(1)
        return self.regressor(x).squeeze(1)


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
            lr = self.lr_start + frac * (self.lr_max - self.lr_start)
        else:
            t = epoch - self.warmup_epochs
            T = self.cosine_epochs
            lr = self.lr_max * 1e-3 + 0.5 * (self.lr_max - self.lr_max * 1e-3) * (
                1 + np.cos(np.pi * t / max(T, 1)))
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def get_last_lr(self):
        return [pg["lr"] for pg in self.optimizer.param_groups]


def train_model(model, train_loader, val_loader, device):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    if USE_WARMUP:
        scheduler = WarmupCosineScheduler(optimizer, lr_max=LR,
                                          warmup_epochs=WARMUP_EPOCHS,
                                          total_epochs=MAX_EPOCHS)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=MAX_EPOCHS, eta_min=LR * 1e-3)

    criterion = nn.MSELoss()
    best_r2, best_rmse, best_epoch = -np.inf, np.inf, 0
    best_state_dict = None
    no_improve_cnt = 0
    log_rows = []

    for epoch in range(1, MAX_EPOCHS + 1):
        if USE_WARMUP:
            scheduler.step(epoch)

        model.train()
        train_losses = []
        for images, lbls in train_loader:
            images = images.to(device, non_blocking=True)
            lbls   = lbls.to(device, non_blocking=True)
            optimizer.zero_grad()
            preds = model(images)
            loss  = criterion(preds, lbls)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=CLIP_NORM)
            optimizer.step()
            train_losses.append(loss.item())

        if not USE_WARMUP:
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
        val_r2   = r2_score(labels_all, preds_all)
        val_rmse = float(np.sqrt(np.mean((preds_all - labels_all) ** 2)))
        train_loss_mean = float(np.mean(train_losses))

        if val_r2 > best_r2:
            best_r2, best_rmse, best_epoch = val_r2, val_rmse, epoch
            no_improve_cnt = 0
            best_state_dict = copy.deepcopy(model.state_dict())
        else:
            no_improve_cnt += 1

        lr_now = scheduler.get_last_lr()[0]
        mark = "*" if no_improve_cnt == 0 else " "
        print(f"  Epoch {epoch:3d}/{MAX_EPOCHS} | Loss={train_loss_mean:9.1f} | "
              f"ValR2={val_r2:.4f} {mark} | ValRMSE={val_rmse:6.2f} | "
              f"LR={lr_now:.2e} | NoImp={no_improve_cnt}/{PATIENCE}")

        log_rows.append({
            "epoch": epoch, "train_loss": round(train_loss_mean, 2),
            "val_r2": round(val_r2, 4), "val_rmse": round(val_rmse, 2),
            "lr": lr_now, "is_best": (no_improve_cnt == 0),
        })

        if no_improve_cnt >= PATIENCE:
            print(f"\n  Early stopping triggered @ epoch {epoch}. Best: BestValR2={best_r2:.4f} @ epoch {best_epoch}")
            break

    return best_state_dict, best_r2, best_rmse, best_epoch, pd.DataFrame(log_rows)


def main():
    print("=" * 70)
    print("Wheat Count Prediction - DenseNet-121 Training from Scratch (Deployment Version)")
    print("=" * 70)

    print("\n[1/3] Please select the label file (.csv or .xlsx, containing DOM filename + wheat count)...")
    label_path = select_label_file()
    if not label_path:
        print("Cancelled."); return
    print(f"   Label file: {label_path}")

    print("\n[2/3] Please select the DOM folder (60m altitude, 1m2 plots, 7-band .tif)...")
    dom_dir = select_folder("Select DOM Folder")
    if not dom_dir:
        print("Cancelled."); return
    print(f"   DOM folder: {dom_dir}")

    print("\n[3/3] Please select the output folder (to save trained model and logs)...")
    output_dir = select_folder("Select Output Folder")
    if not output_dir:
        print("Cancelled."); return
    print(f"   Output folder: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    print("\nMatching label file with DOM files...")
    matched_stems, labels, unmatched_labels, unmatched_tifs, tif_stem_map = \
        build_label_dict(label_path, dom_dir)

    print(f"\nSuccessfully matched: {len(matched_stems)} samples")
    if unmatched_labels:
        print(f"{len(unmatched_labels)} records in the label table had no matching DOM file (skipped)")
        for x in unmatched_labels[:10]:
            print(f"    {x}")
    if unmatched_tifs:
        print(f"{len(unmatched_tifs)} files in the DOM folder had no label (skipped)")
        for x in unmatched_tifs[:10]:
            print(f"    {x}")

    if len(matched_stems) < 20:
        messagebox.showerror("Error", f"Only {len(matched_stems)} samples were successfully matched, "
                                    f"which is too few. Please check whether the label file and DOM folder correspond.")
        return

    print(f"\nReading full channel data for {len(matched_stems)} samples (including on-the-fly vegetation index computation)...")
    raw_cache = {}
    failed_stems = []
    for stem in tqdm(matched_stems, desc="Loading", ncols=70):
        data = load_all_channels_for_stem(stem, dom_dir, tif_stem_map)
        if data is None:
            failed_stems.append(stem)
            continue
        raw_cache[stem] = data

    for s in failed_stems:
        matched_stems.remove(s)
    print(f"Successfully loaded {len(matched_stems)} samples, {len(failed_stems)} failed")

    print("\nComputing normalization statistics (based on all available samples)...")
    norm_stats = {}
    for ch in CHANNEL_LIST:
        all_vals = np.concatenate([raw_cache[s][ch].flatten() for s in matched_stems])
        valid = all_vals[np.isfinite(all_vals)]
        norm_stats[ch] = {
            "mean": float(np.mean(valid)),
            "std":  float(np.std(valid)),
            "n_pixels": int(len(valid)),
        }
        print(f"  {ch:<15} mean={norm_stats[ch]['mean']:>12.4f}  std={norm_stats[ch]['std']:>12.4f}")

    train_stems, val_stems = train_test_split(
        matched_stems, test_size=VAL_RATIO, random_state=SEED
    )
    print(f"\nData split: train={len(train_stems)}  val={len(val_stems)}  "
          f"(val is only used for early-stopping monitoring to select the best epoch)")

    train_ds = WheatDataset(train_stems, labels, raw_cache, CHANNEL_LIST, norm_stats,
                            augment_fn=augment_a3)
    val_ds   = WheatDataset(val_stems,   labels, raw_cache, CHANNEL_LIST, norm_stats,
                            augment_fn=None)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=True)

    print("\n" + "=" * 70)
    print(f"Starting training DenseNet-121 (10 channels, trained from scratch, device: {DEVICE})")
    print("=" * 70)

    model = WheatDenseNet121(in_channels=IN_CHANNELS, dropout=DROPOUT)
    best_state_dict, best_r2, best_rmse, best_epoch, df_log = train_model(
        model, train_loader, val_loader, DEVICE
    )

    print("\nSaving results...")

    model_path = os.path.join(output_dir, "densenet121_wheat_final.pth")
    torch.save(best_state_dict, model_path)

    norm_stats_path = os.path.join(output_dir, "norm_stats.json")
    with open(norm_stats_path, "w", encoding="utf-8") as f:
        json.dump(norm_stats, f, indent=2, ensure_ascii=False)

    channel_config_path = os.path.join(output_dir, "channel_config.json")
    with open(channel_config_path, "w", encoding="utf-8") as f:
        json.dump({
            "channel_list": CHANNEL_LIST,
            "in_channels": IN_CHANNELS,
            "model_key": "densenet121",
            "note": "During deployment inference, input channels must be concatenated strictly in this order, and normalized using the norm_stats.json in the same directory"
        }, f, indent=2, ensure_ascii=False)

    log_path = os.path.join(output_dir, "training_log.csv")
    df_log.to_csv(log_path, index=False, encoding="utf-8-sig")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(df_log["epoch"], df_log["train_loss"], color="#3498db", label="Train Loss (MSE)")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss"); ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(df_log["epoch"], df_log["val_r2"], color="#2ecc71", label="Val R2")
    ax2.axvline(best_epoch, color="#e74c3c", linestyle="--", label=f"Best epoch={best_epoch}")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("R2")
    ax2.set_title("Validation R2"); ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    curve_path = os.path.join(output_dir, "training_curve.png")
    plt.savefig(curve_path, dpi=150, bbox_inches="tight")
    plt.close()

    report_path = os.path.join(output_dir, "training_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("Wheat Count Prediction Model - Training Report\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Model              : DenseNet-121 (trained from scratch)\n")
        f.write(f"Input channels(10ch): {CHANNEL_LIST}\n")
        f.write(f"Data augmentation  : A3 (HFlip+VFlip+Rot90+ChannelDropout p=0.1)\n")
        f.write(f"Total samples      : {len(matched_stems)}  "
                f"(train={len(train_stems)}, val={len(val_stems)})\n")
        f.write(f"Normalization stats: computed based on all {len(matched_stems)} samples\n")
        f.write(f"MaxEpoch/Patience  : {MAX_EPOCHS} / {PATIENCE}\n")
        f.write(f"LR / WeightDecay   : {LR} / {WEIGHT_DECAY}\n")
        f.write(f"DenseNet warmup    : {WARMUP_EPOCHS} epochs\n\n")
        f.write(f"[ Best Validation Result (selected via early stopping) ]\n")
        f.write(f"  Best epoch     : {best_epoch}\n")
        f.write(f"  Val R2         : {best_r2:.4f}\n")
        f.write(f"  Val RMSE       : {best_rmse:.2f}\n\n")
        f.write(f"[ Output File Description ]\n")
        f.write(f"  densenet121_wheat_final.pth : deployment model weights\n")
        f.write(f"  norm_stats.json             : normalization parameters required for deployment inference\n")
        f.write(f"  channel_config.json         : names and order of the 10 channels, must remain consistent at deployment\n")
        f.write(f"  training_log.csv            : per-epoch training/validation records\n")
        f.write(f"  training_curve.png          : Loss/R2 curve plot\n\n")
        if unmatched_labels or unmatched_tifs or failed_stems:
            f.write(f"[ Data Matching Issues ]\n")
            f.write(f"  Labels not matched to DOM : {len(unmatched_labels)}\n")
            f.write(f"  DOM not matched to label  : {len(unmatched_tifs)}\n")
            f.write(f"  Samples failed to load    : {len(failed_stems)}\n")
        f.write("\n" + "=" * 70 + "\n")

    print("\n" + "=" * 70)
    print(f"Training complete!")
    print(f"   Best epoch: {best_epoch}  |  Val R2: {best_r2:.4f}  |  Val RMSE: {best_rmse:.2f}")
    print(f"   Model saved to: {model_path}")
    print(f"   All results output to: {output_dir}")
    print("=" * 70)

    messagebox.showinfo(
        "Training Complete",
        f"Training complete!\n\n"
        f"Best epoch: {best_epoch}\n"
        f"Val R2: {best_r2:.4f}\n"
        f"Val RMSE: {best_rmse:.2f}\n\n"
        f"Results saved to:\n{output_dir}"
    )


if __name__ == "__main__":
    main()