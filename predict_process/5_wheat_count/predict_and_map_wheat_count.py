import os
import glob
import json
import warnings
import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet121
import tkinter as tk
from tkinter import filedialog, messagebox
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

try:
    import geopandas as gpd
    from shapely.affinity import rotate as shp_rotate
    from shapely.ops import unary_union
    from shapely.geometry import Polygon as ShpPolygon, MultiPolygon
except ImportError:
    raise ImportError(
        "Missing geopandas / shapely dependencies, please run:\n"
        "    pip install geopandas shapely pyogrio\n"
        "then re-run this script."
    )

warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BAND_INDEX = {
    "B1_Blue":      1,
    "B2_Green_RGB": 2,
    "B3_Red_RGB":   3,
    "B4_Green560":  4,
    "B5_Red650":    5,
    "B6_RedEdge":   6,
    "B7_NIR":       7,
}


def select_folder(title: str) -> str:
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askdirectory(title=title)
    root.destroy()
    return path


def select_shapefile(title: str) -> str:
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title=title,
        filetypes=[("Shapefile", "*.shp"), ("All files", "*.*")]
    )
    root.destroy()
    return path


def select_column_dialog(columns: list, title: str = "Please select the field column corresponding to the DOM filename") -> str:
    result = {"selected": None}

    win = tk.Tk()
    win.title(title)
    win.geometry("400x400")

    tk.Label(win, text=title, font=("Arial", 11)).pack(pady=10)

    listbox = tk.Listbox(win, font=("Arial", 11), selectmode="single")
    for col in columns:
        listbox.insert("end", col)
    listbox.pack(fill="both", expand=True, padx=20, pady=10)
    if len(columns) > 0:
        listbox.selection_set(0)

    def confirm():
        sel = listbox.curselection()
        if sel:
            result["selected"] = columns[sel[0]]
        win.destroy()

    listbox.bind("<Double-Button-1>", lambda e: confirm())

    btn = tk.Button(win, text="OK", command=confirm, font=("Arial", 11), width=12)
    btn.pack(pady=10)

    win.mainloop()
    return result["selected"]


def find_model_files(model_dir: str):
    pth_files  = glob.glob(os.path.join(model_dir, "**", "*.pth"), recursive=True)
    norm_files = glob.glob(os.path.join(model_dir, "**", "norm_stats.json"), recursive=True)
    cfg_files  = glob.glob(os.path.join(model_dir, "**", "channel_config.json"), recursive=True)

    assert len(pth_files)  > 0, f"[Error] No .pth weight file found in {model_dir}"
    assert len(norm_files) > 0, f"[Error] norm_stats.json not found in {model_dir}"
    assert len(cfg_files)  > 0, f"[Error] channel_config.json not found in {model_dir}"

    if len(pth_files) > 1:
        print(f"Multiple .pth files found, using the first one by default: {pth_files[0]}")
    if len(norm_files) > 1:
        print(f"Multiple norm_stats.json found, using the first one by default: {norm_files[0]}")
    if len(cfg_files) > 1:
        print(f"Multiple channel_config.json found, using the first one by default: {cfg_files[0]}")

    return pth_files[0], norm_files[0], cfg_files[0]


def normalize_stem(name: str) -> str:
    name = str(name).strip()
    for ext in (".tiff", ".tif"):
        if name.lower().endswith(ext):
            name = name[: len(name) - len(ext)]
            break
    return name.strip()


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.flatten(1)
        return self.regressor(x).squeeze(1)


def predict_one_tif(tif_path: str, channel_list: list, norm_stats: dict, model):
    try:
        with rasterio.open(tif_path) as src:
            if src.count < 7:
                return None, f"Insufficient band count (only {src.count})"
            bands = {}
            for name, num in BAND_INDEX.items():
                bands[name] = src.read(num).astype(np.float32)
    except Exception as e:
        return None, f"Read failed: {e}"

    G, R, RE, N = bands["B4_Green560"], bands["B5_Red650"], bands["B6_RedEdge"], bands["B7_NIR"]
    indices = compute_indices(G, R, RE, N)
    all_channels = {**bands, **indices}

    try:
        ch_arrays = [all_channels[ch] for ch in channel_list]
    except KeyError as e:
        return None, f"Missing channel: {e}"

    image = np.stack(ch_arrays, axis=0).astype(np.float32)
    means = np.array([norm_stats[ch]["mean"] for ch in channel_list],
                     dtype=np.float32).reshape(-1, 1, 1)
    stds  = np.array([norm_stats[ch]["std"]  for ch in channel_list],
                     dtype=np.float32).reshape(-1, 1, 1)
    stds  = np.where(stds == 0, 1.0, stds)
    image = (image - means) / stds

    image_tensor = torch.from_numpy(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred = model(image_tensor).cpu().item()

    return pred, None


def _iter_polygons(geom):
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    if isinstance(geom, ShpPolygon):
        return [geom]
    return []


def compute_rotation_angle(union_geom) -> float:
    mrr = union_geom.minimum_rotated_rectangle
    coords = list(mrr.exterior.coords)

    edges = []
    for i in range(4):
        x1, y1 = coords[i]
        x2, y2 = coords[i + 1]
        length = np.hypot(x2 - x1, y2 - y1)
        angle  = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        edges.append((length, angle))

    longest_edge = max(edges, key=lambda e: e[0])
    edge_angle   = longest_edge[1] % 180.0
    rotation_needed = 90.0 - edge_angle
    return rotation_needed


def plot_distribution(gdf_valid: "gpd.GeoDataFrame", value_col: str, out_path: str):
    geoms  = gdf_valid.geometry.tolist()
    values = gdf_valid[value_col].tolist()

    union_geom = unary_union(geoms)
    rotation_deg = compute_rotation_angle(union_geom)
    centroid = union_geom.centroid

    rotated_geoms = [shp_rotate(g, rotation_deg, origin=centroid, use_radians=False)
                     for g in geoms]

    def get_extent(glist):
        bounds = [g.bounds for g in glist]
        minx = min(b[0] for b in bounds)
        maxx = max(b[2] for b in bounds)
        miny = min(b[1] for b in bounds)
        maxy = max(b[3] for b in bounds)
        return minx, miny, maxx, maxy

    minx, miny, maxx, maxy = get_extent(rotated_geoms)
    width, height = (maxx - minx), (maxy - miny)
    if width > height:
        rotated_geoms = [shp_rotate(g, 90.0, origin=centroid, use_radians=False)
                         for g in rotated_geoms]
        minx, miny, maxx, maxy = get_extent(rotated_geoms)

    vmin, vmax = float(np.min(values)), float(np.max(values))
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("RdYlGn")

    patches, patch_values = [], []
    for g, v in zip(rotated_geoms, values):
        for poly in _iter_polygons(g):
            xs, ys = poly.exterior.xy
            pts = list(zip(xs, ys))
            patches.append(MplPolygon(pts, closed=True))
            patch_values.append(v)

    fig, ax = plt.subplots(figsize=(8, 12))
    collection = PatchCollection(patches, cmap=cmap, norm=norm, edgecolor="none")
    collection.set_array(np.array(patch_values))
    ax.add_collection(collection)

    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect("equal")
    ax.axis("off")

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02, shrink=0.8)
    cbar.set_ticks([vmin, vmax])
    cbar.set_ticklabels([f"{vmin:.0f}", f"{vmax:.0f}"])
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(size=0)

    plt.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.1,
               facecolor="white")
    plt.close()


def main():
    print("=" * 70)
    print("Wheat Count Prediction App - DOM Inference + Shapefile Association + Distribution Plot")
    print("=" * 70)

    print("\n[1/4] Please select the DOM input folder (1m2 plots, 7-band .tif)...")
    dom_dir = select_folder("Select DOM Input Folder")
    if not dom_dir:
        print("Cancelled."); return

    tif_files = sorted([
        f for f in os.listdir(dom_dir)
        if f.lower().endswith(".tif") or f.lower().endswith(".tiff")
    ])
    if len(tif_files) == 0:
        messagebox.showerror("Error", "No .tif files found in the input folder!")
        return
    print(f"   Found {len(tif_files)} .tif files")

    print("\n[2/4] Please select the model folder (the output folder of the training script)...")
    model_dir = select_folder("Select Model Folder")
    if not model_dir:
        print("Cancelled."); return

    pth_path, norm_path, cfg_path = find_model_files(model_dir)
    print(f"   Weight file: {pth_path}")
    print(f"   Normalization file: {norm_path}")
    print(f"   Channel config: {cfg_path}")

    with open(norm_path, "r", encoding="utf-8") as f:
        norm_stats = json.load(f)
    norm_stats.pop("_meta", None)

    with open(cfg_path, "r", encoding="utf-8") as f:
        channel_cfg = json.load(f)
    channel_list = channel_cfg["channel_list"]
    in_channels  = channel_cfg["in_channels"]

    model = WheatDenseNet121(in_channels=in_channels)
    state_dict = torch.load(pth_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    print(f"   Model loaded, input channels({in_channels}ch): {channel_list}")

    print("\n[3/4] Please select the Shapefile (.shp) representing the real locations of these plots...")
    shp_path_in = select_shapefile("Select Plot Location Shapefile")
    if not shp_path_in:
        print("Cancelled."); return
    print(f"   Shapefile: {shp_path_in}")

    gdf = gpd.read_file(shp_path_in)
    attr_cols = [c for c in gdf.columns if c != "geometry"]
    if len(attr_cols) == 0:
        messagebox.showerror("Error", "This Shapefile has no attribute fields, cannot match.")
        return

    print(f"   Shapefile fields: {attr_cols}")
    print("   Please select the field column corresponding to the DOM filename in the pop-up window...")
    key_col = select_column_dialog(attr_cols, "Please select the field column corresponding to the DOM filename")
    if not key_col:
        print("No field selected, cancelled."); return
    print(f"   Selected field: '{key_col}'")

    print("\n[4/4] Please select the output folder...")
    output_dir = select_folder("Select Output Folder")
    if not output_dir:
        print("Cancelled."); return
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nStarting prediction for {len(tif_files)} samples...")
    csv_records = []
    pred_by_stem = {}
    skipped = []

    for fname in tqdm(tif_files, desc="Predicting", ncols=70):
        fpath = os.path.join(dom_dir, fname)
        pred, err = predict_one_tif(fpath, channel_list, norm_stats, model)

        if pred is None:
            skipped.append((fname, err))
            continue

        stem = normalize_stem(fname)
        csv_records.append({
            "filename": fname,
            "wheat_count": round(pred, 2),
            "wheat_count_int": int(round(pred)),
        })
        pred_by_stem[stem] = pred

    if skipped:
        print(f"\nThe following {len(skipped)} files failed prediction and were skipped:")
        for fname, err in skipped:
            print(f"   {fname}: {err}")

    if len(csv_records) == 0:
        messagebox.showerror("Error", "No samples were successfully predicted, please check the input data.")
        return

    df_csv = pd.DataFrame(csv_records)
    csv_path = os.path.join(output_dir, "wheat_count_predictions.csv")
    df_csv.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\nCSV saved: {csv_path}")

    def match_value(key_val):
        key_stem = normalize_stem(key_val)
        return pred_by_stem.get(key_stem, np.nan)

    gdf["wheat_cnt"] = gdf[key_col].apply(match_value)

    n_matched   = gdf["wheat_cnt"].notna().sum()
    n_unmatched = gdf["wheat_cnt"].isna().sum()
    print(f"\nShapefile matching result: successfully matched {n_matched} plots, "
          f"unmatched {n_unmatched} plots")

    if n_unmatched > 0:
        unmatched_keys = gdf.loc[gdf["wheat_cnt"].isna(), key_col].astype(str).tolist()
        print(f"The following field values had no matching prediction result in the DOM files (first 10):")
        for k in unmatched_keys[:10]:
            print(f"   {k}")

    unused_tifs = set(pred_by_stem.keys()) - set(
        normalize_stem(v) for v in gdf[key_col].astype(str).tolist()
    )
    if unused_tifs:
        print(f"The following {len(unused_tifs)} DOM prediction results could not be matched to any plot in the shapefile (first 10):")
        for k in list(unused_tifs)[:10]:
            print(f"   {k}")

    if n_matched == 0:
        messagebox.showerror(
            "Match Failed",
            f"The selected field '{key_col}' could not be matched with any DOM filenames at all. Please re-run and\n"
            f"select the correct field column (the field values must match the DOM filenames with the .tif extension removed)."
        )
        return

    shp_out_path = os.path.join(output_dir, "wheat_count_plots.shp")
    gdf.to_file(shp_out_path, encoding="utf-8")
    print(f"Shapefile saved: {shp_out_path}")

    gdf_valid = gdf[gdf["wheat_cnt"].notna()].copy()
    png_path = os.path.join(output_dir, "wheat_count_distribution.png")
    plot_distribution(gdf_valid, "wheat_cnt", png_path)
    print(f"Distribution plot saved: {png_path}")

    print("\n" + "=" * 70)
    print(f"All complete! Predicted samples: {len(csv_records)}, "
          f"Shapefile matched successfully: {n_matched} plots")
    print(f"Output directory: {output_dir}")
    print("=" * 70)

    messagebox.showinfo(
        "Done",
        f"Processing complete!\n\n"
        f"Number of predicted samples: {len(csv_records)}\n"
        f"Shapefile matched successfully: {n_matched} plots\n"
        f"{f'Unmatched: {n_unmatched} plots' if n_unmatched else ''}\n\n"
        f"CSV: wheat_count_predictions.csv\n"
        f"Shapefile: wheat_count_plots.shp\n"
        f"Distribution plot: wheat_count_distribution.png\n\n"
        f"Output directory:\n{output_dir}"
    )


if __name__ == "__main__":
    main()