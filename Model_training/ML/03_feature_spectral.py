import os
import warnings
import numpy as np
import pandas as pd
import rasterio
from rasterio.errors import NotGeoreferencedWarning
from scipy.stats import skew, kurtosis
from tqdm import tqdm

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

import config

BAND_CONFIG = [
    (config.BAND_GREEN_MS, "B4G560"),
    (config.BAND_RED_MS, "B5R650"),
    (config.BAND_RE, "B6RE735"),
    (config.BAND_NIR, "B7NIR860"),
]

STAT_NAMES = [
    "mean", "std", "median", "skewness", "kurtosis",
    "cv", "mad", "q25", "q75", "iqr", "energy"
]


def tif_stem(filename: str) -> str:
    for ext in (".tiff", ".tif"):
        if filename.lower().endswith(ext):
            return filename[: len(filename) - len(ext)]
    return os.path.splitext(filename)[0]


def extract_band_stats(band_arr: np.ndarray, band_name: str) -> dict:
    row = {}
    flat = band_arr.flatten().astype(np.float64)

    n = len(flat)
    if n == 0:
        return {f"{band_name}_{s}": np.nan for s in STAT_NAMES}

    mean_val = float(np.mean(flat))
    std_val = float(np.std(flat, ddof=0))
    median_val = float(np.median(flat))
    q25_val = float(np.percentile(flat, 25))
    q75_val = float(np.percentile(flat, 75))

    skew_val = float(skew(flat))
    kurt_val = float(kurtosis(flat))

    cv_val = (std_val / mean_val) if abs(mean_val) > 1e-6 else np.nan

    mad_val = float(np.median(np.abs(flat - median_val)))

    iqr_val = q75_val - q25_val

    flat_norm = flat / 65535.0
    energy_val = float(np.mean(flat_norm ** 2))

    row[f"{band_name}_mean"] = mean_val
    row[f"{band_name}_std"] = std_val
    row[f"{band_name}_median"] = median_val
    row[f"{band_name}_skewness"] = skew_val
    row[f"{band_name}_kurtosis"] = kurt_val
    row[f"{band_name}_cv"] = cv_val
    row[f"{band_name}_mad"] = mad_val
    row[f"{band_name}_q25"] = q25_val
    row[f"{band_name}_q75"] = q75_val
    row[f"{band_name}_iqr"] = iqr_val
    row[f"{band_name}_energy"] = energy_val

    return row


def run():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    print("\n[1/2] Scanning DOM directory...")
    tif_files = sorted([
        f for f in os.listdir(config.DOM_DIR)
        if f.lower().endswith(".tif") or f.lower().endswith(".tiff")
    ])
    print(f"   Found TIF files: {len(tif_files)}")

    feature_cols = [
        f"{band_name}_{stat}"
        for _, band_name in BAND_CONFIG
        for stat in STAT_NAMES
    ]
    print(f"   Bands: {[name for _, name in BAND_CONFIG]}")
    print(f"   Stats per band: {len(STAT_NAMES)}")
    print(f"   Total features: {len(feature_cols)}  ({len(BAND_CONFIG)} bands x {len(STAT_NAMES)} stats)")

    print(f"\n[2/2] Extracting spectral statistical features...")

    records = []

    for fname in tqdm(tif_files, desc="   Processing", ncols=80):
        fpath = os.path.join(config.DOM_DIR, fname)
        stem = tif_stem(fname)
        row = {"stem": stem}

        try:
            with rasterio.open(fpath) as src:
                data = src.read(
                    list(range(1, config.N_VALID_BANDS + 1))
                ).astype(np.float32)

            for band_idx, band_name in BAND_CONFIG:
                band_arr = data[band_idx]
                stats = extract_band_stats(band_arr, band_name)
                row.update(stats)

        except Exception as e:
            print(f"\n   Read error [{fname}]: {e}")
            for col in feature_cols:
                row[col] = np.nan

        records.append(row)

    df = pd.DataFrame(records)

    feature_df = df.drop(columns=["stem"])
    n_nan_cols = int((feature_df.isna().all()).sum())
    n_nan_cells = int(feature_df.isna().sum().sum())
    total_cells = feature_df.size

    print(f"\n   Samples processed:   {len(df)}")
    print(f"   Features per sample: {len(feature_cols)}")
    print(f"   Columns all-NaN:     {n_nan_cols}")
    print(f"   NaN cells:           {n_nan_cells} / {total_cells} "
          f"({n_nan_cells / total_cells * 100:.2f}%)")

    if len(df) > 0:
        first = df.iloc[0]
        print(f"\n   Sample preview (first row):")
        for col in feature_cols[:8]:
            print(f"     {col}: {first[col]:.4f}")
        print(f"     ...")

    out_path = os.path.join(config.OUTPUT_DIR, "features_spectral.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n   [Output] Spectral features saved: {out_path}")
    print(f"   Shape: {df.shape}  (rows=samples, cols=stem+{len(feature_cols)} features)")

    return df


if __name__ == "__main__":
    run()