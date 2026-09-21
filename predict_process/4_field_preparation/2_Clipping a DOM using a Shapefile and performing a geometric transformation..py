import os
import numpy as np
import cv2
import rasterio
from rasterio.windows import Window
from rasterio.warp import calculate_default_transform, reproject, Resampling
import geopandas as gpd
from tkinter import Tk, filedialog, messagebox, simpledialog
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


OUTPUT_SIZE = 64
MARGIN_RATIO = 0.3


def select_file(title, filetypes):
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    filepath = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.destroy()
    return filepath


def select_folder(title):
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    folder = filedialog.askdirectory(title=title)
    root.destroy()
    return folder


def select_column(gdf):
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    columns = [col for col in gdf.columns if col != 'geometry']
    columns_str = "\n".join([f"{i}: {col}" for i, col in enumerate(columns)])
    message = f"Available columns:\n{columns_str}\n\nPlease enter column name or index:"
    result = simpledialog.askstring("Select Area Name Column", message, parent=root)
    root.destroy()

    if result is None:
        return None
    try:
        idx = int(result)
        if 0 <= idx < len(columns):
            return columns[idx]
    except ValueError:
        pass
    if result in columns:
        return result
    return None


def check_and_align_crs(gdf, dom_path):
    with rasterio.open(dom_path) as src:
        dom_crs = src.crs

    if dom_crs is None:
        raise ValueError("DOM image has no CRS information (CRS is empty), cannot align coordinate systems, please check the DOM file.")

    print(f"  - Shapefile CRS: {gdf.crs}")
    print(f"  - DOM CRS:        {dom_crs}")

    if gdf.crs is None:
        raise ValueError("Shapefile has no CRS information (CRS is empty), cannot determine if it matches the DOM.")

    if str(gdf.crs) == str(dom_crs):
        print("  CRS matches, no reprojection needed")
        return gdf, dom_crs
    else:
        print(f"  CRS mismatch, reprojecting Shapefile to DOM CRS {dom_crs} ...")
        gdf_aligned = gdf.to_crs(dom_crs)
        print("  Reprojection complete")
        return gdf_aligned, dom_crs


def get_square_corners_consistent_winding(geometry):
    coords = list(geometry.exterior.coords)[:-1]
    if len(coords) != 4:
        raise ValueError(f"Polygon should have 4 corners, but has {len(coords)}")
    return np.array(coords)


def signed_area(points):
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def match_winding_order(src_points, dst_points):
    src_sign = signed_area(src_points)
    dst_sign = signed_area(dst_points)

    if (src_sign > 0) != (dst_sign > 0):
        src_points = src_points[::-1].copy()
    return src_points


def calculate_bounding_box(pixel_corners, margin_ratio=MARGIN_RATIO):
    col_min = int(np.floor(pixel_corners[:, 0].min()))
    col_max = int(np.ceil(pixel_corners[:, 0].max()))
    row_min = int(np.floor(pixel_corners[:, 1].min()))
    row_max = int(np.ceil(pixel_corners[:, 1].max()))

    width = col_max - col_min
    height = row_max - row_min

    margin_w = int(width * margin_ratio)
    margin_h = int(height * margin_ratio)

    col_min = max(0, col_min - margin_w)
    row_min = max(0, row_min - margin_h)
    width = width + 2 * margin_w
    height = height + 2 * margin_h

    return col_min, row_min, width, height


def read_window_from_raster(src, col_off, row_off, width, height):
    col_off = max(0, col_off)
    row_off = max(0, row_off)

    if col_off + width > src.width:
        width = src.width - col_off
    if row_off + height > src.height:
        height = src.height - row_off

    window = Window(col_off, row_off, width, height)
    return src.read(window=window)


def perspective_transform_multiband(window_data, src_points_relative, output_size=OUTPUT_SIZE):
    dst_points = np.array([
        [0, 0],
        [output_size - 1, 0],
        [output_size - 1, output_size - 1],
        [0, output_size - 1]
    ], dtype=np.float32)

    src_points_relative = src_points_relative.astype(np.float32)
    M = cv2.getPerspectiveTransform(src_points_relative, dst_points)

    n_bands = window_data.shape[0]
    transformed_bands = []
    for band_idx in range(n_bands):
        band = window_data[band_idx]
        transformed_band = cv2.warpPerspective(
            band, M, (output_size, output_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        transformed_bands.append(transformed_band)

    return np.stack(transformed_bands, axis=0)


def save_as_geotiff(data, output_path, dtype=None):
    if dtype is None:
        dtype = data.dtype
    n_bands, height, width = data.shape
    with rasterio.open(
        output_path, 'w', driver='GTiff',
        height=height, width=width, count=n_bands,
        dtype=dtype, compress='lzw'
    ) as dst:
        for i in range(n_bands):
            dst.write(data[i], i + 1)


def sanitize_filename(name):
    invalid_chars = '<>:"/\\|?*'
    for ch in invalid_chars:
        name = name.replace(ch, '_')
    return name.strip() or "unnamed"


def main():
    print("=" * 70)
    print("DOM Cropping Tool - Perspective Transform Correction Version (corner winding direction fix)")
    print("Workflow: Select Shapefile -> Select DOM -> Select output folder -> CRS check -> Select name column -> Crop")
    print("=" * 70)

    print("\n[1/3] Selecting Shapefile...")
    shp_path = select_file("Select Shapefile", [("Shapefile", "*.shp")])
    if not shp_path:
        print("No Shapefile selected")
        return
    print(f"{Path(shp_path).name}")

    print("\n[2/3] Selecting DOM image...")
    dom_path = select_file("Select DOM Image", [("GeoTIFF", "*.tif *.tiff")])
    if not dom_path:
        print("No DOM selected")
        return
    print(f"{Path(dom_path).name}")

    print("\n[3/3] Selecting output folder...")
    output_folder = select_folder("Select Output Folder")
    if not output_folder:
        print("No output folder selected")
        return
    print(f"{output_folder}")
    os.makedirs(output_folder, exist_ok=True)

    print("\n" + "=" * 70)
    print("Processing...")
    print("=" * 70)

    print("\nReading Shapefile...")
    gdf = gpd.read_file(shp_path)
    print(f"  - Number of polygons: {len(gdf)}")

    print("\nChecking CRS consistency...")
    try:
        gdf, target_crs = check_and_align_crs(gdf, dom_path)
    except ValueError as e:
        print(str(e))
        messagebox.showerror("CRS Error", str(e))
        return

    print("\nSelecting area name column...")
    name_column = select_column(gdf)
    if not name_column:
        print("No name column selected")
        return
    print(f"Using column: {name_column}")

    with rasterio.open(dom_path) as src:
        print(f"\nDOM info:")
        print(f"  - Band count: {src.count}")
        print(f"  - Size: {src.width}x{src.height}")
        print(f"  - Data type: {src.dtypes[0]}")
        print(f"  - CRS: {src.crs}")

        original_dtype = src.dtypes[0]

        print(f"\nStarting to crop {len(gdf)} areas (window read + perspective transform)...")
        print("-" * 70)

        success = 0
        failed = 0
        used_names = set()

        for idx, row in gdf.iterrows():
            try:
                area_name = sanitize_filename(str(row[name_column]))
                base_name = area_name
                suffix = 1
                while area_name in used_names:
                    area_name = f"{base_name}_{suffix}"
                    suffix += 1
                used_names.add(area_name)

                geom = row.geometry
                geo_corners = get_square_corners_consistent_winding(geom)

                pixel_corners = []
                for x, y in geo_corners:
                    row_idx, col_idx = src.index(x, y)
                    pixel_corners.append([col_idx, row_idx])
                pixel_corners = np.array(pixel_corners, dtype=np.float32)

                if (pixel_corners[:, 0] < 0).all() or (pixel_corners[:, 0] >= src.width).all() or (pixel_corners[:, 1] < 0).all() or (pixel_corners[:, 1] >= src.height).all():
                    print(f"  [{idx+1}/{len(gdf)}] {area_name}: completely outside image bounds")
                    failed += 1
                    continue

                col_off, row_off, width, height = calculate_bounding_box(pixel_corners)

                col_off = max(0, min(col_off, src.width - 1))
                row_off = max(0, min(row_off, src.height - 1))
                width = min(width, src.width - col_off)
                height = min(height, src.height - row_off)

                if width <= 0 or height <= 0:
                    print(f"  [{idx+1}/{len(gdf)}] {area_name}: invalid window size")
                    failed += 1
                    continue

                window_data = read_window_from_raster(src, col_off, row_off, width, height)

                src_points_relative = pixel_corners.copy()
                src_points_relative[:, 0] -= col_off
                src_points_relative[:, 1] -= row_off

                if (src_points_relative < 0).any() or (src_points_relative[:, 0] >= width).any() or (src_points_relative[:, 1] >= height).any():
                    print(f"  [{idx+1}/{len(gdf)}] {area_name}: corner points outside window bounds")
                    failed += 1
                    continue

                dst_points_ref = np.array([
                    [0, 0], [OUTPUT_SIZE - 1, 0],
                    [OUTPUT_SIZE - 1, OUTPUT_SIZE - 1], [0, OUTPUT_SIZE - 1]
                ], dtype=np.float32)
                src_points_relative = match_winding_order(src_points_relative, dst_points_ref)

                transformed_data = perspective_transform_multiband(
                    window_data, src_points_relative, output_size=OUTPUT_SIZE
                )

                if transformed_data.max() == 0:
                    print(f"  [{idx+1}/{len(gdf)}] {area_name}: transformed data is empty")
                    failed += 1
                    continue

                output_path = os.path.join(output_folder, f"{area_name}.tif")
                save_as_geotiff(transformed_data, output_path, dtype=original_dtype)

                success += 1
                print(f"  [{idx+1}/{len(gdf)}] {area_name}: {transformed_data.shape}, range[{transformed_data.min():.0f}, {transformed_data.max():.0f}]")

            except Exception as e:
                failed += 1
                area_name = str(row.get(name_column, f"Area{idx}"))
                print(f"  [{idx+1}/{len(gdf)}] {area_name}: {str(e)}")

        print("-" * 70)
        print(f"\nProcessing complete!")
        print(f"  Success: {success}")
        print(f"  Failed: {failed}")
        print(f"  Total: {len(gdf)}")
        print(f"\nOutput directory: {output_folder}")
        print(f"\nNote: all outputs are {OUTPUT_SIZE}x{OUTPUT_SIZE} pixel square TIFs")
        print("       Corner winding direction has been corrected against output direction, avoiding mirror/flip")
        print("=" * 70)

        root = Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        messagebox.showinfo(
            "Done",
            f"Processing complete!\n\nSuccess: {success}\nFailed: {failed}\n\n"
            f"Output directory:\n{output_folder}\n\n"
            f"All outputs are {OUTPUT_SIZE}x{OUTPUT_SIZE} pixels\n"
            f"Corner winding direction corrected, no mirror/flip"
        )
        root.destroy()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()

        root = Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        messagebox.showerror("Error", f"Program error:\n{str(e)}")
        root.destroy()