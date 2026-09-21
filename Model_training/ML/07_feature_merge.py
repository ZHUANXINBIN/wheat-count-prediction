import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import skew

warnings.filterwarnings("ignore")

import config


SCALE_PREFIXES = [
    "B4G560_",
    "B5R650_",
    "B6RE735_",
    "B7NIR860_",
    "WAV_NIR_cA2_",
    "WAV_NIR_detail_",
    "DVI_",
    "TVI_",
    "MCARI_",
    "TCARI_",
]


def load_feature_csv(filename: str, tag: str) -> pd.DataFrame:
    path = os.path.join(config.OUTPUT_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature file not found: {path}\n"
                                f"Please run the corresponding feature extraction script first.")
    df = pd.read_csv(path, encoding="utf-8-sig")
    n_feat = df.shape[1] - 1
    print(f"   [{tag}] {filename}: {len(df)} rows x {n_feat} features")
    return df


def check_stem_format(dfs: dict, label_df: pd.DataFrame) -> None:
    print("\n   Stem format check (first 2 samples):")
    for tag, df in dfs.items():
        samples = df["stem"].head(2).tolist()
        print(f"     [feature/{tag}]: {samples}")
    label_samples = label_df[config.CSV_NAME_COL].head(2).tolist()
    print(f"     [label/{config.CSV_NAME_COL}]: {label_samples}")
    print(f"   Confirm both formats match before proceeding.")


def scale_columns_by_prefix(df: pd.DataFrame, prefixes: list) -> tuple:
    scaled_cols = []
    for col in df.columns:
        for prefix in prefixes:
            if col.startswith(prefix):
                df[col] = df[col] / 65535.0
                scaled_cols.append(col)
                break
    return df, scaled_cols


def get_feature_group(col: str) -> str:
    if any(col.startswith(p) for p in [
            "B5R650_GLCM_", "B6RE735_GLCM_", "B7NIR860_GLCM_",
            "LBP_", "WAV_"]):
        return "Texture(06)"
    if any(col.startswith(p) for p in [
            "B4G560_", "B5R650_", "B6RE735_", "B7NIR860_"]):
        return "Spectral(04)"
    if any(col.startswith(p) for p in ["RGB_", "HSV_", "Lab_", "yellow"]):
        return "Color(05)"
    if any(col.startswith(p) for p in [
            "fractal_", "kmeans_", "morans_", "gearys_"]):
        return "Spatial(07)"
    return "Vegetation(03)"


def print_feature_group_summary(feature_cols: list) -> None:
    from collections import Counter
    groups = Counter(get_feature_group(c) for c in feature_cols)
    total_counted = sum(groups.values())

    print(f"\n   Feature group summary:")
    for gname in ["Vegetation(03)", "Spectral(04)", "Color(05)",
                  "Texture(06)", "Spatial(07)"]:
        display = {
            "Vegetation(03)": "Vegetation index (03)",
            "Spectral(04)":   "Spectral stats  (04)",
            "Color(05)":      "Color           (05)",
            "Texture(06)":    "Texture         (06)",
            "Spatial(07)":    "Spatial         (07)",
        }[gname]
        print(f"     {display}: {groups.get(gname, 0)}")
    print(f"     -------------------------------------")
    print(f"     Total: {len(feature_cols)}  (grouped: {total_counted})")
    if total_counted != len(feature_cols):
        unclassified = [c for c in feature_cols
                        if get_feature_group(c) not in groups]
        print(f"     Unclassified columns: {unclassified[:5]}")


def run():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    print("\n[1/6] Loading feature CSVs...")

    feature_files = {
        "vegetation": "features_vegetation.csv",
        "spectral":   "features_spectral.csv",
        "color":      "features_color.csv",
        "texture":    "features_texture.csv",
        "spatial":    "features_spatial.csv",
    }

    dfs = {}
    for tag, fname in feature_files.items():
        dfs[tag] = load_feature_csv(fname, tag)

    print("\n[2/6] Loading labels...")

    if not os.path.exists(config.LABEL_CSV):
        raise FileNotFoundError(
            f"Label CSV not found: {config.LABEL_CSV}\n"
            f"Check config.LABEL_CSV path."
        )

    label_df = pd.read_csv(config.LABEL_CSV, encoding="utf-8-sig")
    print(f"   Label file: {config.LABEL_CSV}")
    print(f"   Shape: {label_df.shape}")
    print(f"   Columns: {label_df.columns.tolist()}")

    check_stem_format(dfs, label_df)

    print("\n[3/6] Merging feature tables...")

    base_tags = list(dfs.keys())
    merged = dfs[base_tags[0]]
    for tag in base_tags[1:]:
        merged = pd.merge(merged, dfs[tag], on="stem", how="left")
        print(f"   After join [{tag}]: {merged.shape}")

    print(f"   Merged shape (features only): {merged.shape}")

    print("\n[4/6] Joining labels...")

    label_sub = label_df[[config.CSV_NAME_COL, config.CSV_LABEL_COL]].copy()
    label_sub = label_sub.rename(columns={config.CSV_NAME_COL: "stem"})

    df = pd.merge(merged, label_sub, on="stem", how="left")
    print(f"   After label join: {df.shape}")

    n_label_nan = int(df[config.CSV_LABEL_COL].isna().sum())
    n_total     = len(df)
    if n_label_nan == n_total:
        raise RuntimeError(
            f"ALL labels are NaN after join! stem format mismatch.\n"
            f"Feature stem example: {merged['stem'].iloc[0]}\n"
            f"Label stem example:   {label_sub['stem'].iloc[0]}\n"
            f"Please check and align stem formats."
        )
    if n_label_nan > 0:
        print(f"   {n_label_nan}/{n_total} samples have no label, dropping them")
        df = df.dropna(subset=[config.CSV_LABEL_COL]).reset_index(drop=True)
    print(f"   Usable samples with labels: {len(df)}")

    print("\n[5/6] Analyzing label & cleaning features...")

    y      = df[config.CSV_LABEL_COL].astype(float)
    y_skew = float(skew(y.dropna()))
    print(f"   {config.CSV_LABEL_COL}: "
          f"min={y.min():.0f}  max={y.max():.0f}  "
          f"mean={y.mean():.1f}  std={y.std():.1f}  skewness={y_skew:.3f}")

    df["label_log"] = np.log1p(y)
    log_skew        = float(skew(df["label_log"].dropna()))
    print(f"   label_log: min={df['label_log'].min():.3f}  "
          f"max={df['label_log'].max():.3f}  skewness={log_skew:.3f}")
    if y_skew > 1.0:
        print(f"   skewness > 1.0: recommend label_log for modeling")
    else:
        print(f"   skewness <= 1.0: wheat_count distribution is acceptable")

    meta_cols    = ["stem", config.CSV_LABEL_COL, "label_log"]
    feature_cols = [c for c in df.columns if c not in meta_cols]
    print(f"\n   Total feature columns before cleaning: {len(feature_cols)}")

    n_inf = int(np.isinf(df[feature_cols].select_dtypes(include=np.number).values).sum())
    if n_inf > 0:
        print(f"   Replacing {n_inf} inf values...")
        for col in feature_cols:
            if df[col].dtype in [np.float32, np.float64] and np.isinf(df[col]).any():
                finite_vals = df[col][np.isfinite(df[col])]
                p99 = float(finite_vals.quantile(0.99))
                p01 = float(finite_vals.quantile(0.01))
                df[col] = df[col].replace([np.inf], p99).replace([-np.inf], p01)
    else:
        print(f"   No inf values")

    n_nan = int(df[feature_cols].isna().sum().sum())
    if n_nan > 0:
        print(f"   Filling {n_nan} NaN values with column median...")
        for col in feature_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())
    else:
        print(f"   No NaN values")

    stds       = df[feature_cols].std()
    const_cols = stds[stds == 0].index.tolist()
    if const_cols:
        print(f"   Dropping {len(const_cols)} constant columns: {const_cols}")
        df           = df.drop(columns=const_cols)
        feature_cols = [c for c in feature_cols if c not in const_cols]
    else:
        print(f"   No constant columns")

    glcm_cols   = {c for c in feature_cols if "_GLCM_" in c}
    df, scaled_cols = scale_columns_by_prefix(df, SCALE_PREFIXES)
    wrong_scaled = [c for c in scaled_cols if c in glcm_cols]
    if wrong_scaled:
        for col in wrong_scaled:
            df[col] = df[col] * 65535.0
        print(f"   Reverted {len(wrong_scaled)} GLCM columns incorrectly scaled")
    real_scaled = [c for c in scaled_cols if c not in glcm_cols]
    print(f"   Scaled {len(real_scaled)} DN-magnitude columns by /65535 "
          f"(GLCM columns excluded):")
    for prefix in SCALE_PREFIXES:
        hits = [c for c in real_scaled if c.startswith(prefix)]
        if hits:
            print(f"     {prefix}* : {len(hits)} columns")

    print("\n[6/6] Final check and saving...")

    feature_cols = [c for c in df.columns if c not in meta_cols]

    n_nan_final = int(df[feature_cols].isna().sum().sum())
    n_inf_final = int(np.isinf(
        df[feature_cols].select_dtypes(include=np.number).values).sum())
    print(f"   Final NaN: {n_nan_final}  |  Final inf: {n_inf_final}")

    print_feature_group_summary(feature_cols)

    final_cols = ["stem", config.CSV_LABEL_COL, "label_log"] + feature_cols
    df         = df[final_cols]

    out_path = os.path.join(config.OUTPUT_DIR, "features_all.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n   [Output] features_all.csv -> {out_path}")
    print(f"   Shape: {df.shape}  "
          f"(rows={len(df)} samples, cols=3 meta + {len(feature_cols)} features)")

    return df


if __name__ == "__main__":
    run()