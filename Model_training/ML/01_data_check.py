import os
import warnings
import numpy as np
import pandas as pd
import rasterio
from rasterio.errors import NotGeoreferencedWarning
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

import config


def tif_stem(filename: str) -> str:
    for ext in (".tiff", ".tif"):
        if filename.lower().endswith(ext):
            return filename[: len(filename) - len(ext)]
    return os.path.splitext(filename)[0]


def check_single_tif(tif_path: str) -> dict:
    result = {
        "file":             os.path.basename(tif_path),
        "readable":         False,
        "shape_ok":         False,
        "actual_shape":     None,
        "has_nan":          False,
        "has_inf":          False,
        "zero_bands":       [],
        "saturated_bands":  [],
        "veg_ratio_ndre":   None,
        "issues":           [],
    }

    try:
        with rasterio.open(tif_path) as src:
            result["readable"] = True

            actual   = (src.count, src.height, src.width)
            result["actual_shape"] = str(actual)
            expected = (config.N_BANDS, config.IMAGE_SIZE[0], config.IMAGE_SIZE[1])
            if actual == expected:
                result["shape_ok"] = True
            else:
                result["issues"].append(
                    f"shape mismatch: expected {expected}, got {actual}"
                )

            data = src.read(
                list(range(1, config.N_VALID_BANDS + 1))
            ).astype(np.float32)

            if np.any(np.isnan(data)):
                result["has_nan"] = True
                result["issues"].append(
                    f"NaN pixels: {int(np.sum(np.isnan(data)))}"
                )
            if np.any(np.isinf(data)):
                result["has_inf"] = True
                result["issues"].append(
                    f"inf pixels: {int(np.sum(np.isinf(data)))}"
                )

            for b in range(data.shape[0]):
                band    = data[b]
                band_no = b + 1
                valid   = band[np.isfinite(band)]
                if len(valid) == 0:
                    result["issues"].append(f"Band{band_no}: no valid pixels")
                    continue
                bmin, bmax = float(valid.min()), float(valid.max())
                if bmax == 0:
                    result["zero_bands"].append(band_no)
                    result["issues"].append(f"Band{band_no}: all zero (black)")
                elif bmin == bmax:
                    result["saturated_bands"].append(band_no)
                    result["issues"].append(
                        f"Band{band_no}: uniform saturation (min=max={bmin:.4f})"
                    )

            if result["shape_ok"]:
                nir   = data[config.BAND_NIR]
                re    = data[config.BAND_RE]
                denom = nir + re
                with np.errstate(invalid="ignore", divide="ignore"):
                    ndre = np.where(denom != 0, (nir - re) / denom, 0.0)
                veg_pixels = np.sum(ndre > config.NDRE_MASK_THRESHOLD)
                result["veg_ratio_ndre"] = round(
                    float(veg_pixels) / ndre.size, 4
                )
                if result["veg_ratio_ndre"] < config.MIN_VEGETATION_RATIO:
                    result["issues"].append(
                        f"low vegetation coverage: "
                        f"{result['veg_ratio_ndre']*100:.1f}% "
                        f"(threshold {config.MIN_VEGETATION_RATIO*100:.0f}%)"
                    )

    except Exception as e:
        result["issues"].append(f"read error: {str(e)}")

    result["zero_bands"]     = str(result["zero_bands"])
    result["saturated_bands"]= str(result["saturated_bands"])
    result["issues"]         = " | ".join(result["issues"]) if result["issues"] else "OK"
    return result


def plot_label_distribution(series: pd.Series, save_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Wheat Count Distribution", fontsize=13)

    axes[0].hist(series, bins=30, color="#4C9BE8", edgecolor="white", linewidth=0.5)
    axes[0].set_title("Original Distribution")
    axes[0].set_xlabel("Wheat Count")
    axes[0].set_ylabel("Frequency")
    axes[0].axvline(series.mean(),   color="red",    linestyle="--",
                    label=f"Mean {series.mean():.1f}")
    axes[0].axvline(series.median(), color="orange", linestyle="--",
                    label=f"Median {series.median():.1f}")
    axes[0].legend(fontsize=8)

    log_series = np.log1p(series)
    axes[1].hist(log_series, bins=30, color="#5DBB8A", edgecolor="white", linewidth=0.5)
    axes[1].set_title("log(1 + Wheat Count) Distribution")
    axes[1].set_xlabel("log(1 + Wheat Count)")
    axes[1].set_ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   [Figure] Label distribution saved: {save_path}")


def compute_skewness(series: pd.Series) -> float:
    from scipy.stats import skew
    return float(skew(series.dropna()))


def run():
    os.makedirs(config.REPORT_DIR, exist_ok=True)

    print("\n[1/4] Scanning DOM directory for TIF files...")
    if not os.path.isdir(config.DOM_DIR):
        raise FileNotFoundError(f"DOM directory not found: {config.DOM_DIR}")

    tif_files = sorted([
        f for f in os.listdir(config.DOM_DIR)
        if f.lower().endswith(".tif") or f.lower().endswith(".tiff")
    ])
    print(f"   Found TIF files: {len(tif_files)}")
    if len(tif_files) == 0:
        raise RuntimeError("No TIF files found. Check DOM_DIR in config.py.")

    print(f"\n[2/4] Checking files ({len(tif_files)} total)...")
    records = []
    for i, fname in enumerate(tif_files):
        if (i + 1) % 50 == 0 or (i + 1) == len(tif_files):
            print(f"   Progress: {i+1}/{len(tif_files)}")
        fpath = os.path.join(config.DOM_DIR, fname)
        rec   = check_single_tif(fpath)
        rec["stem"] = tif_stem(fname)
        records.append(rec)

    tif_df = pd.DataFrame(records)

    print(f"\n[3/4] Loading label CSV and matching filenames...")
    if not os.path.isfile(config.LABEL_CSV):
        raise FileNotFoundError(f"Label CSV not found: {config.LABEL_CSV}")

    label_df = pd.read_csv(config.LABEL_CSV)
    print(f"   CSV rows: {len(label_df)}, columns: {list(label_df.columns)}")

    for col in [config.CSV_NAME_COL, config.CSV_LABEL_COL]:
        if col not in label_df.columns:
            raise KeyError(
                f"Column '{col}' not found in CSV. "
                f"Check CSV_NAME_COL / CSV_LABEL_COL in config.py.\n"
                f"  Actual columns: {list(label_df.columns)}"
            )

    label_df[config.CSV_NAME_COL] = (
        label_df[config.CSV_NAME_COL].astype(str).str.strip()
    )
    tif_df["stem"] = tif_df["stem"].str.strip()

    tif_stems = set(tif_df["stem"])
    csv_names = set(label_df[config.CSV_NAME_COL])
    only_in_tif = tif_stems - csv_names
    only_in_csv = csv_names - tif_stems
    matched     = tif_stems & csv_names

    print(f"   Matched: {len(matched)}")
    print(f"   Only in TIF (missing label): {len(only_in_tif)}")
    print(f"   Only in CSV (missing file):  {len(only_in_csv)}")

    if only_in_tif:
        print(f"   TIF without label (first 5): {sorted(only_in_tif)[:5]}")
    if only_in_csv:
        print(f"   CSV without file  (first 5): {sorted(only_in_csv)[:5]}")

    tif_df["in_csv"] = tif_df["stem"].isin(csv_names)

    print(f"\n[4/4] Analyzing label distribution...")
    labels = label_df[config.CSV_LABEL_COL].dropna()

    skewness = round(compute_skewness(labels), 3)
    label_stats = {
        "Count":    len(labels),
        "Min":      labels.min(),
        "Max":      labels.max(),
        "Mean":     round(labels.mean(), 2),
        "Median":   labels.median(),
        "Std":      round(labels.std(), 2),
        "Skewness": skewness,
        "Zero count": int((labels == 0).sum()),
    }
    for k, v in label_stats.items():
        print(f"   {k}: {v}")

    if abs(skewness) > 1.0:
        print(f"\n   Skewness={skewness} (|skewness|>1): distribution is skewed.")
        print(f"     Recommend setting USE_LOG_TRANSFORM=True in config.py.")
    else:
        print(f"\n   Skewness={skewness}: distribution is approximately symmetric.")
        print(f"     Log transform not required.")

    plot_label_distribution(
        labels,
        save_path=os.path.join(config.REPORT_DIR, "label_distribution.png")
    )

    label_lookup = (
        label_df.set_index(config.CSV_NAME_COL)[config.CSV_LABEL_COL].to_dict()
    )
    tif_df["wheat_count"] = tif_df["stem"].map(label_lookup)

    tif_df["usable"] = (
        tif_df["readable"] &
        tif_df["shape_ok"] &
        tif_df["in_csv"] &
        (tif_df["zero_bands"] == "[]") &
        (tif_df["wheat_count"].notna())
    )

    report_path = os.path.join(config.REPORT_DIR, "data_quality_report.csv")
    tif_df.to_csv(report_path, index=False, encoding="utf-8-sig")
    print(f"\n   [Report] Quality report saved: {report_path}")

    n_usable   = int(tif_df["usable"].sum())
    n_unusable = len(tif_df) - n_usable
    n_issue    = int((tif_df["issues"] != "OK").sum())

    summary_lines = [
        "=" * 50,
        "Data Quality Check Summary",
        "=" * 50,
        f"Total TIF files:        {len(tif_files)}",
        f"Total CSV labels:       {len(label_df)}",
        f"Filename matched:       {len(matched)}",
        f"Usable samples:         {n_usable}",
        f"Samples with issues:    {n_issue}",
        f"Unusable samples:       {n_unusable}",
        "-" * 50,
        f"Label min:              {label_stats['Min']}",
        f"Label max:              {label_stats['Max']}",
        f"Label mean:             {label_stats['Mean']}",
        f"Label skewness:         {label_stats['Skewness']}",
        f"Log transform advised:  {'Yes' if abs(skewness) > 1.0 else 'No'}",
        "=" * 50,
    ]
    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)

    summary_path = os.path.join(config.REPORT_DIR, "data_check_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"   [Report] Summary saved: {summary_path}")

    return tif_df


if __name__ == "__main__":
    run()