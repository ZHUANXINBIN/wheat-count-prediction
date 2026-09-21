import os
import warnings
import numpy as np
import pandas as pd
import rasterio
from rasterio.errors import NotGeoreferencedWarning
from sklearn.cluster import KMeans
from scipy.sparse import lil_matrix
from tqdm import tqdm
import cv2

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import config

KMEANS_K = 3
KMEANS_SEED = config.RANDOM_SEED
KMEANS_INIT = 10
BOX_SIZES = [2, 4, 8, 16, 32]
MORAN_DOWNSAMPLE = 4


def tif_stem(filename: str) -> str:
    for ext in (".tiff", ".tif"):
        if filename.lower().endswith(ext):
            return filename[: len(filename) - len(ext)]
    return os.path.splitext(filename)[0]


def box_count(binary_img: np.ndarray, box_size: int) -> int:
    h, w = binary_img.shape
    h_crop = (h // box_size) * box_size
    w_crop = (w // box_size) * box_size
    cropped = binary_img[:h_crop, :w_crop]
    blocks = cropped.reshape(h_crop // box_size, box_size,
                             w_crop // box_size, box_size)
    return int((blocks.max(axis=(1, 3)) > 0).sum())


def compute_fractal_dim(band: np.ndarray) -> float:
    band_u8 = (band.astype(np.float32) / 65535.0 * 255.0).clip(0, 255).astype(np.uint8)
    _, binary = cv2.threshold(band_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    white_ratio = float(binary.sum()) / (255.0 * binary.size)
    if white_ratio < 0.01 or white_ratio > 0.99:
        return np.nan
    counts = []
    for s in BOX_SIZES:
        n = box_count(binary, s)
        if n > 0:
            counts.append((s, n))
    if len(counts) < 3:
        return np.nan
    log_inv_s = np.log(np.array([1.0 / s for s, _ in counts]))
    log_n = np.log(np.array([float(n) for _, n in counts]))
    coeffs = np.polyfit(log_inv_s, log_n, 1)
    return float(coeffs[0])


def compute_kmeans_ratios(data: np.ndarray) -> tuple:
    ms = np.stack([
        data[config.BAND_GREEN_MS],
        data[config.BAND_RED_MS],
        data[config.BAND_RE],
        data[config.BAND_NIR],
    ], axis=0)
    ms_norm = ms.astype(np.float32) / 65535.0
    pixels = ms_norm.reshape(4, -1).T
    kmeans = KMeans(n_clusters=KMEANS_K, n_init=KMEANS_INIT,
                    random_state=KMEANS_SEED, max_iter=300)
    labels = kmeans.fit_predict(pixels)
    nir_means = [(k, float(pixels[labels == k, 3].mean()) if (labels == k).sum() > 0 else 0.0)
                 for k in range(KMEANS_K)]
    sorted_clusters = sorted(nir_means, key=lambda x: x[1])
    remap = {old_k: new_k for new_k, (old_k, _) in enumerate(sorted_clusters)}
    remapped = np.array([remap[l] for l in labels])
    n_total = len(remapped)
    return float((remapped == 0).sum()) / n_total, float((remapped == 1).sum()) / n_total


_W_CACHE = {}


def get_queen_weights(h: int, w: int):
    key = (h, w)
    if key not in _W_CACHE:
        n = h * w
        W = lil_matrix((n, n), dtype=np.float64)

        for r in range(h):
            for c in range(w):
                i = r * w + c
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w:
                            W[i, r * w + nc if dr == 0 else nr * w + nc] = 1.0

        W = W.tocsr()
        row_sums = np.array(W.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1.0
        from scipy.sparse import diags
        D_inv = diags(1.0 / row_sums)
        W = D_inv.dot(W)

        _W_CACHE[key] = W

    return _W_CACHE[key]


def compute_morans_i(band: np.ndarray, downsample: int = MORAN_DOWNSAMPLE) -> float:
    ds = band[::downsample, ::downsample].astype(np.float64)
    h, w = ds.shape
    n = h * w

    x = ds.flatten()
    z = x - x.mean()
    z_sum_sq = float(np.dot(z, z))
    if z_sum_sq < 1e-10:
        return np.nan

    W = get_queen_weights(h, w)
    Wz = np.array(W.dot(z)).flatten()

    I = float(np.dot(z, Wz) / z_sum_sq)

    if not (-2.0 < I < 2.0):
        return np.nan
    return I


def compute_gearys_c(band: np.ndarray, downsample: int = MORAN_DOWNSAMPLE) -> float:
    ds = band[::downsample, ::downsample].astype(np.float64)
    h, w_size = ds.shape
    n = h * w_size

    x = ds.flatten()
    z = x - x.mean()
    z_sum_sq = float(np.dot(z, z))
    if z_sum_sq < 1e-10:
        return np.nan

    W = get_queen_weights(h, w_size)

    from scipy.sparse import diags
    col_sums = np.array(W.sum(axis=0)).flatten()

    Wx = np.array(W.dot(x)).flatten()
    WTx = np.array(W.T.dot(x)).flatten()
    numerator = (
            float(np.dot(x ** 2, np.ones(n)))
            + float(np.dot(x ** 2, col_sums))
            - float(np.dot(x, Wx))
            - float(np.dot(x, WTx))
    )

    C = float((n - 1) / (2.0 * n) * numerator / z_sum_sq)
    return C


def run():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    print("\n[1/2] Scanning DOM directory...")
    tif_files = sorted([
        f for f in os.listdir(config.DOM_DIR)
        if f.lower().endswith(".tif") or f.lower().endswith(".tiff")
    ])
    print(f"   Found TIF files: {len(tif_files)}")
    print(f"   Features: fractal_dim(1) + kmeans_ratio(2) + morans_i(1) + gearys_c(1) = 5")
    print(f"   Moran's I: downsampled 64 -> {64 // MORAN_DOWNSAMPLE} for speed")
    print(f"   Fix: explicit row-normalization of weight matrix (morans_i now in [-1,1])")

    print(f"\n[2/2] Extracting spatial features...")
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

            nir = data[config.BAND_NIR]

            row["fractal_dim"] = compute_fractal_dim(nir)
            r0, r1 = compute_kmeans_ratios(data)
            row["kmeans_cluster0_ratio"] = r0
            row["kmeans_cluster1_ratio"] = r1
            row["morans_i"] = compute_morans_i(nir)
            row["gearys_c"] = compute_gearys_c(nir)

        except Exception as e:
            print(f"\n   Error [{fname}]: {e}")
            row["fractal_dim"] = np.nan
            row["kmeans_cluster0_ratio"] = np.nan
            row["kmeans_cluster1_ratio"] = np.nan
            row["morans_i"] = np.nan
            row["gearys_c"] = np.nan

        records.append(row)

    df = pd.DataFrame(records)
    feature_df = df.drop(columns=["stem"])
    n_nan_cells = int(feature_df.isna().sum().sum())
    total_cells = feature_df.size

    print(f"\n   Samples processed:   {len(df)}")
    print(f"   Features per sample: {feature_df.shape[1]}")
    print(f"   NaN cells:           {n_nan_cells} / {total_cells} "
          f"({n_nan_cells / total_cells * 100:.2f}%)")

    print(f"\n   Feature distribution summary:")
    for col in feature_df.columns:
        s = feature_df[col]
        print(f"     {col}: min={s.min():.4f}  max={s.max():.4f}  "
              f"mean={s.mean():.4f}  NaN={s.isna().sum()}")

    out_path = os.path.join(config.OUTPUT_DIR, "features_spatial.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n   [Output] Spatial features saved: {out_path}")
    print(f"   Shape: {df.shape}")

    return df


if __name__ == "__main__":
    run()