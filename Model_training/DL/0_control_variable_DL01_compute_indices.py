import os
import numpy as np
import rasterio
from rasterio.transform import from_bounds
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DOM_DIR     = os.path.join(BASE_DIR, "0_Data", "1_DOM")
INDICES_DIR = os.path.join(BASE_DIR, "0_Data", "indices")

BAND_G   = 4
BAND_R   = 5
BAND_RE  = 6
BAND_NIR = 7

INDEX_DEFINITIONS = {
    "NDVI":    lambda G, R, RE, N: safe_div(N - R,  N + R),
    "GNDVI":   lambda G, R, RE, N: safe_div(N - G,  N + G),
    "NDRE":    lambda G, R, RE, N: safe_div(N - RE, N + RE),
    "CIre":    lambda G, R, RE, N: safe_div(N, RE) - 1.0,
    "CIgreen": lambda G, R, RE, N: safe_div(N, G)  - 1.0,
    "GRVI":    lambda G, R, RE, N: safe_div(N, G),
}


def safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(b != 0, a / b, np.nan)
    return result.astype(np.float32)


def tif_stem(filename: str) -> str:
    for ext in (".tiff", ".tif"):
        if filename.lower().endswith(ext):
            return filename[: len(filename) - len(ext)]
    return os.path.splitext(filename)[0]


def compute_and_save(tif_path: str, out_dirs: dict) -> bool:
    fname = os.path.basename(tif_path)
    stem  = tif_stem(fname)

    try:
        with rasterio.open(tif_path) as src:
            G   = src.read(BAND_G).astype(np.float32)
            R   = src.read(BAND_R).astype(np.float32)
            RE  = src.read(BAND_RE).astype(np.float32)
            N   = src.read(BAND_NIR).astype(np.float32)
            profile = src.profile.copy()
    except Exception as e:
        print(f"\n  Read failed [{fname}]: {e}")
        return False

    profile.update(
        count=1,
        dtype="float32",
        nodata=np.nan
    )

    for idx_name, func in INDEX_DEFINITIONS.items():
        arr = func(G, R, RE, N)
        arr = np.where(np.isinf(arr), np.nan, arr)

        out_path = os.path.join(out_dirs[idx_name], stem + ".tif")
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(arr[np.newaxis, :, :])

    return True


def main():
    print("=" * 60)
    print("Vegetation Index Image Computation")
    print("=" * 60)

    out_dirs = {}
    for idx_name in INDEX_DEFINITIONS:
        d = os.path.join(INDICES_DIR, idx_name)
        os.makedirs(d, exist_ok=True)
        out_dirs[idx_name] = d
    print(f"\nOutput indices: {list(INDEX_DEFINITIONS.keys())}")
    print(f"Output root directory: {INDICES_DIR}")

    tif_files = sorted([
        f for f in os.listdir(DOM_DIR)
        if f.lower().endswith(".tif") or f.lower().endswith(".tiff")
    ])
    print(f"TIF files to process: {len(tif_files)}\n")

    success, failed = 0, []
    for fname in tqdm(tif_files, desc="Computing vegetation indices", ncols=70):
        fpath = os.path.join(DOM_DIR, fname)
        ok = compute_and_save(fpath, out_dirs)
        if ok:
            success += 1
        else:
            failed.append(fname)

    print(f"\n{'='*60}")
    print(f"Success: {success}")
    if failed:
        print(f"Failed: {len(failed)}")
        for f in failed:
            print(f"   {f}")
    else:
        print(f"No failed files")

    for idx_name, d in out_dirs.items():
        n = len([f for f in os.listdir(d) if f.endswith(".tif")])
        status = "OK" if n == len(tif_files) else "WARNING"
        print(f"  {status} {idx_name}: {n} tif files")
    print("=" * 60)


if __name__ == "__main__":
    main()