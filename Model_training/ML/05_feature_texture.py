import os
import warnings
import numpy as np
import pandas as pd
import rasterio
from rasterio.errors import NotGeoreferencedWarning
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
import pywt
from tqdm import tqdm

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

import config


GLCM_LEVELS    = 64
GLCM_DISTANCES = [1, 3]
GLCM_ANGLES    = [0, np.pi/4, np.pi/2, 3*np.pi/4]
GLCM_PROPS     = ["contrast", "homogeneity", "energy", "correlation"]

GLCM_BANDS = [
    (config.BAND_RED_MS,   "B5R650"),
    (config.BAND_RE,       "B6RE735"),
    (config.BAND_NIR,      "B7NIR860"),
]

LBP_RADIUS   = 1
LBP_N_POINTS = 8
LBP_METHOD   = "uniform"

WAVELET_NAME   = "haar"
WAVELET_LEVEL  = 2


def tif_stem(filename: str) -> str:
    for ext in (".tiff", ".tif"):
        if filename.lower().endswith(ext):
            return filename[: len(filename) - len(ext)]
    return os.path.splitext(filename)[0]


def quantize_band(band: np.ndarray, levels: int = GLCM_LEVELS) -> np.ndarray:
    quantized = band.astype(np.float32) / 65535.0 * (levels - 1)
    quantized = np.clip(quantized, 0, levels - 1)
    return quantized.astype(np.uint8)


def glcm_entropy(glcm_matrix: np.ndarray) -> float:
    eps = 1e-10
    glcm_sum = glcm_matrix.sum(axis=(0, 1), keepdims=True)
    p = glcm_matrix / (glcm_sum + eps)
    entropy = -(p * np.log2(p + eps)).sum(axis=(0, 1))
    return float(entropy.mean())


def extract_glcm_features(band: np.ndarray, band_name: str) -> dict:
    row = {}
    q_band = quantize_band(band)

    for dist in GLCM_DISTANCES:
        glcm = graycomatrix(
            q_band,
            distances=[dist],
            angles=GLCM_ANGLES,
            levels=GLCM_LEVELS,
            symmetric=True,
            normed=False,
        )

        for prop in GLCM_PROPS:
            values = graycoprops(glcm, prop)
            row[f"{band_name}_GLCM_d{dist}_{prop}"] = float(values.mean())

        row[f"{band_name}_GLCM_d{dist}_entropy"] = glcm_entropy(glcm)

    return row


def extract_lbp_features(band: np.ndarray) -> dict:
    band_u8 = (band.astype(np.float32) / 65535.0 * 255.0).clip(0, 255).astype(np.uint8)

    lbp = local_binary_pattern(band_u8, LBP_N_POINTS, LBP_RADIUS, method=LBP_METHOD)

    n_bins = LBP_N_POINTS + 2
    hist, _ = np.histogram(lbp.flatten(), bins=n_bins,
                           range=(0, n_bins), density=True)

    uniformity = float(np.sum(hist ** 2))

    eps = 1e-10
    entropy = float(-np.sum(hist * np.log2(hist + eps)))

    return {
        "LBP_NIR_uniformity": uniformity,
        "LBP_NIR_entropy":    entropy,
    }


def extract_wavelet_features(band: np.ndarray) -> dict:
    coeffs = pywt.wavedec2(band.astype(np.float32), WAVELET_NAME, level=WAVELET_LEVEL)

    cA2 = coeffs[0]
    detail_L2 = coeffs[1]
    detail_L1 = coeffs[2]

    energy_L1 = float(np.mean([np.mean(d ** 2) for d in detail_L1]))

    energy_L2 = float(np.mean([np.mean(d ** 2) for d in detail_L2]))

    return {
        "WAV_NIR_cA2_mean":     float(np.mean(cA2)),
        "WAV_NIR_cA2_std":      float(np.std(cA2)),
        "WAV_NIR_detail_L1_energy": energy_L1,
        "WAV_NIR_detail_L2_energy": energy_L2,
    }


def run():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    print("\n[1/2] Scanning DOM directory...")
    tif_files = sorted([
        f for f in os.listdir(config.DOM_DIR)
        if f.lower().endswith(".tif") or f.lower().endswith(".tiff")
    ])
    print(f"   Found TIF files: {len(tif_files)}")

    n_glcm = len(GLCM_BANDS) * len(GLCM_DISTANCES) * (len(GLCM_PROPS) + 1)
    n_lbp  = 2
    n_wav  = 4
    print(f"   Expected features breakdown:")
    print(f"     GLCM ({len(GLCM_BANDS)} bands x {len(GLCM_DISTANCES)} distances x "
          f"{len(GLCM_PROPS)+1} props): {n_glcm}")
    print(f"     LBP (NIR, uniformity+entropy): {n_lbp}")
    print(f"     Wavelet (NIR, 2-level Haar):   {n_wav}")
    print(f"     Total: {n_glcm + n_lbp + n_wav}")
    print(f"   GLCM: levels={GLCM_LEVELS}, distances={GLCM_DISTANCES}, "
          f"angles=4, symmetric=True")
    print(f"\n   GLCM is slow. Estimated time: 5-15 minutes for 518 files.")

    print(f"\n[2/2] Extracting texture features...")

    records = []

    for fname in tqdm(tif_files, desc="   Processing", ncols=80):
        fpath = os.path.join(config.DOM_DIR, fname)
        stem  = tif_stem(fname)
        row   = {"stem": stem}

        try:
            with rasterio.open(fpath) as src:
                data = src.read(
                    list(range(1, config.N_VALID_BANDS + 1))
                ).astype(np.float32)

            for band_idx, band_name in GLCM_BANDS:
                band = data[band_idx]
                row.update(extract_glcm_features(band, band_name))

            nir_band = data[config.BAND_NIR]
            row.update(extract_lbp_features(nir_band))

            row.update(extract_wavelet_features(nir_band))

        except Exception as e:
            print(f"\n   Error [{fname}]: {e}")

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
        print(f"\n   Feature preview (first sample):")
        preview_keys = [
            "B7NIR860_GLCM_d1_contrast",
            "B7NIR860_GLCM_d1_homogeneity",
            "B7NIR860_GLCM_d1_energy",
            "B7NIR860_GLCM_d1_entropy",
            "LBP_NIR_uniformity",
            "LBP_NIR_entropy",
            "WAV_NIR_cA2_mean",
            "WAV_NIR_detail_L1_energy",
        ]
        for key in preview_keys:
            if key in first:
                print(f"     {key}: {first[key]:.6f}")

    out_path = os.path.join(config.OUTPUT_DIR, "features_texture.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n   [Output] Texture features saved: {out_path}")
    print(f"   Shape: {df.shape}")

    return df


if __name__ == "__main__":
    run()