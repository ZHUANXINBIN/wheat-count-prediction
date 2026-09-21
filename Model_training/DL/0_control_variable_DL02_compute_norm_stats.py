import os
import json
import numpy as np
import rasterio
import pandas as pd
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DOM_DIR     = os.path.join(BASE_DIR, "0_Data", "1_DOM")
INDICES_DIR = os.path.join(BASE_DIR, "0_Data", "indices")
SPLIT_CSV   = os.path.join(BASE_DIR, "0_Data", "dataset_split_index.csv")
OUT_JSON    = os.path.join(BASE_DIR, "0_Data", "norm_stats.json")

ORIGINAL_BANDS = [
    "B1_Blue",
    "B2_Green_RGB",
    "B3_Red_RGB",
    "B4_Green560",
    "B5_Red650",
    "B6_RedEdge",
    "B7_NIR",
]

INDEX_NAMES = ["NDVI", "GNDVI", "NDRE", "CIre", "CIgreen", "GRVI"]

ALL_CHANNELS = ORIGINAL_BANDS + INDEX_NAMES


def tif_stem(filename: str) -> str:
    for ext in (".tiff", ".tif"):
        if filename.lower().endswith(ext):
            return filename[: len(filename) - len(ext)]
    return os.path.splitext(filename)[0]


def load_all_channels(stem: str) -> np.ndarray | None:
    channels = []

    dom_path = os.path.join(DOM_DIR, stem + ".tif")
    try:
        with rasterio.open(dom_path) as src:
            for b in range(1, 8):
                channels.append(src.read(b).astype(np.float32))
    except Exception as e:
        print(f"\n  Original band read failed [{stem}]: {e}")
        return None

    for idx_name in INDEX_NAMES:
        idx_path = os.path.join(INDICES_DIR, idx_name, stem + ".tif")
        try:
            with rasterio.open(idx_path) as src:
                channels.append(src.read(1).astype(np.float32))
        except Exception as e:
            print(f"\n  Index read failed [{stem}][{idx_name}]: {e}")
            return None

    return np.stack(channels, axis=0)


def main():
    print("=" * 60)
    print("Normalization Statistics Computation (Training Set Only)")
    print("=" * 60)

    df_split = pd.read_csv(SPLIT_CSV)
    train_stems = df_split[df_split["split"] == "train"]["filename"].astype(str).str.strip().tolist()
    test_stems  = df_split[df_split["split"] == "test"]["filename"].astype(str).str.strip().tolist()
    print(f"\nTraining set samples: {len(train_stems)}")
    print(f"Test set samples: {len(test_stems)} (not included in statistics)")

    n_channels = len(ALL_CHANNELS)
    count  = np.zeros(n_channels, dtype=np.float64)
    mean   = np.zeros(n_channels, dtype=np.float64)
    M2     = np.zeros(n_channels, dtype=np.float64)

    failed = []
    for stem in tqdm(train_stems, desc="Computing training set statistics", ncols=70):
        data = load_all_channels(stem)
        if data is None:
            failed.append(stem)
            continue

        for c in range(n_channels):
            flat  = data[c].flatten()
            valid = flat[np.isfinite(flat)]
            if len(valid) == 0:
                continue
            for x in valid:
                count[c]  += 1
                delta      = x - mean[c]
                mean[c]   += delta / count[c]
                M2[c]     += delta * (x - mean[c])

    std = np.where(count > 1, np.sqrt(M2 / (count - 1)), 0.0)

    stats = {}
    for i, ch_name in enumerate(ALL_CHANNELS):
        stats[ch_name] = {
            "mean": float(mean[i]),
            "std":  float(std[i]),
            "n_pixels": int(count[i])
        }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    separator_line = "=" * 60
    print(f"\n{separator_line}")
    print(f"{'Channel':<15} {'Mean':>12} {'Std':>12}")
    print("-" * 42)
    for ch_name, s in stats.items():
        print(f"{ch_name:<15} {s['mean']:>12.4f} {s['std']:>12.4f}")

    print(f"\n{separator_line}")
    if failed:
        print(f"The following samples failed to read and were not included in statistics ({len(failed)} total):")
        for f in failed:
            print(f"   {f}")
    else:
        print(f"All training samples read successfully")
    print(f"Normalization statistics saved: {OUT_JSON}")
    print("=" * 60)


if __name__ == "__main__":
    main()