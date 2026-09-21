import os
import warnings
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox

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

N_OUTLIERS = 10


def select_shapefile(title: str) -> str:
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title=title,
        filetypes=[("Shapefile", "*.shp"), ("All files", "*.*")]
    )
    root.destroy()
    return path


def select_folder(title: str) -> str:
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askdirectory(title=title)
    root.destroy()
    return path


def select_column_dialog(columns: list, title: str) -> str:
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
    tk.Button(win, text="OK", command=confirm, font=("Arial", 11), width=12).pack(pady=10)

    win.mainloop()
    return result["selected"]


def detect_outliers(values: np.ndarray, n_outliers: int) -> np.ndarray:
    median = np.median(values)
    mad = np.median(np.abs(values - median))

    if mad == 0:
        std = np.std(values)
        if std == 0:
            return np.array([], dtype=int)
        robust_z = (values - median) / std
    else:
        robust_z = 0.6745 * (values - median) / mad

    abs_z = np.abs(robust_z)
    n = min(n_outliers, len(values))
    outlier_idx = np.argsort(abs_z)[::-1][:n]
    return outlier_idx


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
    print("Wheat Count Distribution Plot (Outlier-Removed Version)")
    print("=" * 70)

    print("\n[1/3] Please select the Shapefile (.shp) containing wheat count information...")
    shp_path = select_shapefile("Select the Shapefile containing wheat count information")
    if not shp_path:
        print("Cancelled."); return
    print(f"   Shapefile: {shp_path}")

    gdf = gpd.read_file(shp_path)
    attr_cols = [c for c in gdf.columns if c != "geometry"]
    if len(attr_cols) == 0:
        messagebox.showerror("Error", "This Shapefile has no attribute fields.")
        return

    print(f"   Shapefile fields: {attr_cols}")
    print("   Please select the wheat count field in the pop-up window...")
    value_col = select_column_dialog(attr_cols, "Please select the wheat count field")
    if not value_col:
        print("No field selected, cancelled."); return
    print(f"   Selected field: '{value_col}'")

    gdf[value_col] = pd.to_numeric(gdf[value_col], errors="coerce")
    n_before = len(gdf)
    gdf = gdf[gdf[value_col].notna()].reset_index(drop=True)
    n_dropped_na = n_before - len(gdf)
    if n_dropped_na > 0:
        print(f"{n_dropped_na} records could not be parsed as numeric in field '{value_col}', excluded")

    if len(gdf) < 20:
        messagebox.showerror("Error", f"Only {len(gdf)} valid plots available, too few.")
        return

    print(f"\n[2/3] Detecting and removing approximately {N_OUTLIERS} outlier plots...")
    values = gdf[value_col].to_numpy(dtype=float)
    outlier_idx = detect_outliers(values, N_OUTLIERS)

    if len(outlier_idx) > 0:
        removed_rows = gdf.iloc[outlier_idx].copy()
        removed_rows = removed_rows.drop(columns="geometry")
        print(f"   Identified {len(outlier_idx)} outlier plots, values as follows:")
        for _, row in removed_rows.sort_values(value_col).iterrows():
            print(f"     {dict(row)}")
    else:
        removed_rows = pd.DataFrame()
        print("   No significant outliers detected.")

    keep_mask = np.ones(len(gdf), dtype=bool)
    keep_mask[outlier_idx] = False
    gdf_clean = gdf[keep_mask].reset_index(drop=True)
    print(f"   {len(gdf_clean)} plots remain for plotting after removal "
          f"(original {len(gdf)} - removed {len(outlier_idx)})")

    print("\n[3/3] Please select the output folder...")
    output_dir = select_folder("Select Output Folder")
    if not output_dir:
        print("Cancelled."); return
    os.makedirs(output_dir, exist_ok=True)

    if not removed_rows.empty:
        removed_csv_path = os.path.join(output_dir, "removed_outliers.csv")
        removed_rows.to_csv(removed_csv_path, index=False, encoding="utf-8-sig")
        print(f"\nRemoved list saved: {removed_csv_path}")

    shp_clean_path = os.path.join(output_dir, "wheat_count_plots_clean.shp")
    gdf_clean.to_file(shp_clean_path, encoding="utf-8")
    print(f"Outlier-removed Shapefile saved: {shp_clean_path}")

    png_path = os.path.join(output_dir, "wheat_distribution_clean.png")
    plot_distribution(gdf_clean, value_col, png_path)
    print(f"Distribution plot saved: {png_path}")

    print("\n" + "=" * 70)
    print(f"All complete! Original plots: {len(gdf)}, outliers removed: {len(outlier_idx)}, "
          f"used for plotting: {len(gdf_clean)}")
    print(f"Output directory: {output_dir}")
    print("=" * 70)

    messagebox.showinfo(
        "Done",
        f"Processing complete!\n\n"
        f"Original plot count: {len(gdf)}\n"
        f"Outliers removed: {len(outlier_idx)}\n"
        f"Used for plotting: {len(gdf_clean)}\n\n"
        f"Distribution plot: wheat_distribution_clean.png\n"
        f"Removed list: removed_outliers.csv\n"
        f"Cleaned Shapefile: wheat_count_plots_clean.shp\n\n"
        f"Output directory:\n{output_dir}"
    )


if __name__ == "__main__":
    main()