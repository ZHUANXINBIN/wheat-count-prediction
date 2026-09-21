import os
import warnings
import numpy as np
import pandas as pd
import rasterio
from rasterio.errors import NotGeoreferencedWarning
from scipy.stats import skew, kurtosis
from tqdm import tqdm
import cv2

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

import config


BAND_B = config.BAND_BLUE_RGB
BAND_G = config.BAND_GREEN_RGB
BAND_R = config.BAND_RED_RGB

YELLOW_H_LOW  = 10
YELLOW_H_HIGH = 40
YELLOW_S_MIN  = 25
YELLOW_V_MIN  = 50


def tif_stem(filename: str) -> str:
    for ext in (".tiff", ".tif"):
        if filename.lower().endswith(ext):
            return filename[: len(filename) - len(ext)]
    return os.path.splitext(filename)[0]


def uint16_to_uint8_stretch(arr: np.ndarray) -> np.ndarray:
    p2  = float(np.percentile(arr, 2))
    p98 = float(np.percentile(arr, 98))
    if p98 > p2:
        scaled = (arr.astype(np.float32) - p2) / (p98 - p2) * 255.0
    else:
        scaled = np.full_like(arr, 128.0, dtype=np.float32)
    return np.clip(scaled, 0, 255).astype(np.uint8)


def extract_channel_stats(channel: np.ndarray, prefix: str) -> dict:
    flat = channel.flatten().astype(np.float64)
    return {
        f"{prefix}_mean": float(np.mean(flat)),
        f"{prefix}_std":  float(np.std(flat, ddof=0)),
        f"{prefix}_skew": float(skew(flat)),
        f"{prefix}_kurt": float(kurtosis(flat)),
    }


def extract_color_features(tif_path: str) -> dict | None:
    try:
        with rasterio.open(tif_path) as src:
            data = src.read(
                list(range(1, config.N_VALID_BANDS + 1))
            ).astype(np.float32)
    except Exception as e:
        print(f"\n   Read error [{os.path.basename(tif_path)}]: {e}")
        return None

    row = {}

    b_u8 = uint16_to_uint8_stretch(data[BAND_B])
    g_u8 = uint16_to_uint8_stretch(data[BAND_G])
    r_u8 = uint16_to_uint8_stretch(data[BAND_R])

    row.update(extract_channel_stats(r_u8, "RGB_R"))
    row.update(extract_channel_stats(g_u8, "RGB_G"))
    row.update(extract_channel_stats(b_u8, "RGB_B"))

    bgr = np.stack([b_u8, g_u8, r_u8], axis=-1)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    h_ch = hsv[:, :, 0]
    s_ch = hsv[:, :, 1]
    v_ch = hsv[:, :, 2]

    row["HSV_H_mean"] = float(np.mean(h_ch))
    row["HSV_H_std"]  = float(np.std(h_ch, ddof=0))
    row["HSV_S_mean"] = float(np.mean(s_ch))
    row["HSV_S_std"]  = float(np.std(s_ch, ddof=0))
    row["HSV_V_mean"] = float(np.mean(v_ch))
    row["HSV_V_std"]  = float(np.std(v_ch, ddof=0))

    yellow_mask = (
        (h_ch >= YELLOW_H_LOW)  &
        (h_ch <= YELLOW_H_HIGH) &
        (s_ch >= YELLOW_S_MIN)  &
        (v_ch >= YELLOW_V_MIN)
    )
    row["yellow_ratio"] = float(yellow_mask.sum()) / yellow_mask.size

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

    a_ch = lab[:, :, 1].astype(np.float32) - 128.0
    b_ch = lab[:, :, 2].astype(np.float32) - 128.0

    row["Lab_a_mean"] = float(np.mean(a_ch))
    row["Lab_a_std"]  = float(np.std(a_ch, ddof=0))
    row["Lab_b_mean"] = float(np.mean(b_ch))
    row["Lab_b_std"]  = float(np.std(b_ch, ddof=0))

    return row


def run():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    print("\n[1/2] Scanning DOM directory...")
    tif_files = sorted([
        f for f in os.listdir(config.DOM_DIR)
        if f.lower().endswith(".tif") or f.lower().endswith(".tiff")
    ])
    print(f"   Found TIF files: {len(tif_files)}")
    print(f"   Conversion method: percentile stretch (2%-98%) per band")
    print(f"   Yellow ratio thresholds: H=[{YELLOW_H_LOW},{YELLOW_H_HIGH}], "
          f"S>={YELLOW_S_MIN}, V>={YELLOW_V_MIN}")

    expected_features = 3*4 + 3*2 + 1 + 2*2
    print(f"   Expected features: {expected_features}")
    print(f"   Color space breakdown:")
    print(f"     RGB (R/G/B x mean/std/skew/kurt): {3*4}")
    print(f"     HSV (H/S/V x mean/std):           {3*2}")
    print(f"     Yellow ratio:                      1")
    print(f"     Lab (a*/b* x mean/std):            {2*2}")

    print(f"\n[2/2] Extracting color features...")

    records = []

    for fname in tqdm(tif_files, desc="   Processing", ncols=80):
        fpath = os.path.join(config.DOM_DIR, fname)
        stem  = tif_stem(fname)

        feats = extract_color_features(fpath)

        if feats is None:
            row = {"stem": stem}
        else:
            feats["stem"] = stem
            row = feats

        records.append(row)

    df = pd.DataFrame(records)

    feature_df  = df.drop(columns=["stem"])
    n_nan_cols  = int((feature_df.isna().all()).sum())
    n_nan_cells = int(feature_df.isna().sum().sum())
    total_cells = feature_df.size

    print(f"\n   Samples processed:   {len(df)}")
    print(f"   Features per sample: {feature_df.shape[1]}")
    print(f"   Columns all-NaN:     {n_nan_cols}")
    print(f"   NaN cells:           {n_nan_cells} / {total_cells} "
          f"({n_nan_cells / total_cells * 100:.2f}%)")

    if len(df) > 0:
        first = df.iloc[0]
        print(f"\n   Key feature preview (first sample):")
        print(f"   [Expected after fix: H~20-60, Lab_a>0, Lab_b>0, yellow_ratio>0]")
        for key in ["RGB_R_mean", "RGB_G_mean", "RGB_B_mean",
                    "HSV_H_mean", "HSV_S_mean", "HSV_V_mean",
                    "Lab_a_mean", "Lab_b_mean", "yellow_ratio"]:
            if key in first:
                val = first[key]
                print(f"     {key}: {val:.4f}")

        yr = df["yellow_ratio"]
        print(f"\n   yellow_ratio distribution:")
        print(f"     min={yr.min():.4f}  max={yr.max():.4f}  "
              f"mean={yr.mean():.4f}  zeros={int((yr==0).sum())}/{len(yr)}")

    out_path = os.path.join(config.OUTPUT_DIR, "features_color.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n   [Output] Color features saved: {out_path}")
    print(f"   Shape: {df.shape}")

    return df


if __name__ == "__main__":
    run()