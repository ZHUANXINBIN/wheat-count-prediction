import os
import warnings
import numpy as np
import pandas as pd
import rasterio
from rasterio.errors import NotGeoreferencedWarning
from tqdm import tqdm
import spyndex

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

import config


STAT_NAMES = ["mean", "std", "median", "q25", "q75"]


def tif_stem(filename: str) -> str:
    for ext in (".tiff", ".tif"):
        if filename.lower().endswith(ext):
            return filename[: len(filename) - len(ext)]
    return os.path.splitext(filename)[0]


def load_ms_bands(tif_path: str) -> dict | None:
    try:
        with rasterio.open(tif_path) as src:
            data = src.read(
                list(range(1, config.N_VALID_BANDS + 1))
            ).astype(np.float32)

        return {
            "G":  data[config.BAND_GREEN_MS],
            "R":  data[config.BAND_RED_MS],
            "RE": data[config.BAND_RE],
            "N":  data[config.BAND_NIR],
        }
    except Exception as e:
        print(f"\n   Read error [{os.path.basename(tif_path)}]: {e}")
        return None


def safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(b != 0, a / b, np.nan)
    return result.astype(np.float32)


def extract_stats(arr: np.ndarray) -> dict:
    flat  = arr.flatten()
    valid = flat[~np.isnan(flat)]

    if len(valid) < 10:
        return {s: np.nan for s in STAT_NAMES}

    return {
        "mean":   float(np.mean(valid)),
        "std":    float(np.std(valid)),
        "median": float(np.median(valid)),
        "q25":    float(np.percentile(valid, 25)),
        "q75":    float(np.percentile(valid, 75)),
    }


def calc_manual_indices(b: dict) -> dict:
    G, R, RE, N = b["G"], b["R"], b["RE"], b["N"]

    indices = {}

    indices["NDVI"]   = safe_div(N - R,  N + R)

    indices["GNDVI"]  = safe_div(N - G,  N + G)

    indices["NDRE"]   = safe_div(N - RE, N + RE)

    indices["CIre"]   = safe_div(N, RE) - 1.0

    indices["CIgreen"]= safe_div(N, G)  - 1.0

    indices["MTCI"]   = safe_div(N - RE, RE - R)

    L = 0.5
    indices["SAVI"]   = safe_div(N - R, N + R + L) * (1.0 + L)

    indices["OSAVI"]  = safe_div(N - R, N + R + 0.16)

    inner = (2.0 * N + 1.0) ** 2 - 8.0 * (N - R)
    inner = np.where(inner < 0, 0.0, inner)
    indices["MSAVI"]  = ((2.0 * N + 1.0) - np.sqrt(inner)) / 2.0

    indices["RVI"]    = safe_div(N, R)

    indices["GRVI"]   = safe_div(N, G)

    indices["RERVI"]  = safe_div(N, RE)

    indices["DVI"]    = (N - R).astype(np.float32)

    alpha = 0.1
    indices["WDRVI"]  = safe_div(alpha * N - R, alpha * N + R)

    indices["TVI"]    = 0.5 * (120.0 * (N - G) - 200.0 * (R - G))

    indices["MCARI"]  = ((RE - R) - 0.2 * (RE - G)) * safe_div(RE, R)

    indices["TCARI"]  = 3.0 * ((RE - R) - 0.2 * (RE - G) * safe_div(RE, R))

    indices["PSRI"]   = safe_div(R - G, RE)

    indices["NDGI"]   = safe_div(G - R, G + R)

    indices["GI"]     = safe_div(G, R)

    return indices


def run():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    print("\n[1/3] Scanning DOM directory...")
    tif_files = sorted([
        f for f in os.listdir(config.DOM_DIR)
        if f.lower().endswith(".tif") or f.lower().endswith(".tiff")
    ])
    print(f"   Found TIF files: {len(tif_files)}")

    dummy_band = np.ones((64, 64), dtype=np.float32)
    dummy_b    = {"G": dummy_band, "R": dummy_band,
                  "RE": dummy_band, "N": dummy_band}
    index_names = list(calc_manual_indices(dummy_b).keys())

    feature_cols = [
        f"{idx}_{stat}"
        for idx in index_names
        for stat in STAT_NAMES
    ]
    print(f"\n[2/3] Vegetation indices to compute: {len(index_names)}")
    print(f"   Index names: {index_names}")
    print(f"   Total features: {len(feature_cols)}")

    print(f"\n[3/3] Extracting vegetation index features...")
    print(f"   Strategy: all manual formula (no spyndex dependency)")

    records = []

    for fname in tqdm(tif_files, desc="   Processing", ncols=80):
        fpath = os.path.join(config.DOM_DIR, fname)
        stem  = tif_stem(fname)
        row   = {"stem": stem}

        bands = load_ms_bands(fpath)

        if bands is None:
            for col in feature_cols:
                row[col] = np.nan
            records.append(row)
            continue

        indices = calc_manual_indices(bands)

        for idx_name, arr in indices.items():
            arr = np.where(np.isinf(arr), np.nan, arr)
            stats = extract_stats(arr)
            for stat, val in stats.items():
                row[f"{idx_name}_{stat}"] = val

        records.append(row)

    df = pd.DataFrame(records)

    feature_df  = df.drop(columns=["stem"])
    n_nan_cols  = int((feature_df.isna().all()).sum())
    n_nan_cells = int(feature_df.isna().sum().sum())
    total_cells = feature_df.size

    print(f"\n   Samples processed:    {len(df)}")
    print(f"   Indices computed:     {len(index_names)}")
    print(f"   Features per sample:  {len(feature_cols)}")
    print(f"   Columns all-NaN:      {n_nan_cols}")
    print(f"   NaN cells:            {n_nan_cells} / {total_cells} "
          f"({n_nan_cells / total_cells * 100:.2f}%)")

    if n_nan_cols > 0:
        nan_cols = feature_df.columns[feature_df.isna().all()].tolist()
        print(f"   All-NaN columns: {nan_cols[:10]}{'...' if len(nan_cols)>10 else ''}")

    out_path = os.path.join(config.OUTPUT_DIR, "features_vegetation.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n   [Output] Vegetation features saved: {out_path}")
    print(f"   Shape: {df.shape}  (rows=samples, cols=stem+{len(feature_cols)} features)")

    return df


if __name__ == "__main__":
    run()