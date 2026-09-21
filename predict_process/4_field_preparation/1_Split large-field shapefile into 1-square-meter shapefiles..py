import geopandas as gpd
from shapely.geometry import box, LineString
from shapely.affinity import rotate
import numpy as np
import os
from tkinter import Tk, filedialog
import math

def select_file(title="Select File"):
    Tk().withdraw()
    file_path = filedialog.askopenfilename(title=title, filetypes=[("Shapefile", "*.shp")])
    return file_path

def select_folder(title="Select Output Folder"):
    Tk().withdraw()
    folder_path = filedialog.askdirectory(title=title)
    return folder_path

def get_longest_edge_angle(geom):
    max_len = 0
    angle = 0
    for poly in geom.geoms if geom.geom_type == 'MultiPolygon' else [geom]:
        coords = list(poly.exterior.coords)
        for i in range(len(coords) - 1):
            p1, p2 = coords[i], coords[i + 1]
            line = LineString([p1, p2])
            length = line.length
            if length > max_len:
                max_len = length
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                angle = math.degrees(math.atan2(dy, dx))
    return angle

def main():
    input_shp = select_file("Select the shapefile to rotate and grid-cut")
    if not input_shp:
        print("No file selected, operation terminated.")
        return

    output_folder = select_folder("Select Output Folder")
    if not output_folder:
        print("No output folder selected, operation terminated.")
        return

    gdf = gpd.read_file(input_shp)
    gdf = gdf.to_crs(epsg=3857)
    geom_union = gdf.union_all()

    angle = get_longest_edge_angle(geom_union)
    print(f"Rotation angle: {angle:.2f} deg")

    centroid = geom_union.centroid
    geom_rotated = rotate(geom_union, -angle, origin=centroid, use_radians=False)

    minx, miny, maxx, maxy = geom_rotated.bounds
    x_steps = np.arange(minx, maxx, 1)
    y_steps = np.arange(miny, maxy, 1)
    polygons = []
    ids = []
    counter = 0

    for x in x_steps:
        for y in y_steps:
            cell = box(x, y, x+1, y+1)
            if geom_rotated.intersects(cell):
                clipped_geom = cell.intersection(geom_rotated)
                if not clipped_geom.is_empty:
                    cell_final = rotate(clipped_geom, angle, origin=centroid, use_radians=False)
                    polygons.append(cell_final)
                    counter += 1
                    ids.append(counter)

    out_gdf = gpd.GeoDataFrame({'id': ids, 'geometry': polygons}, crs=gdf.crs)

    input_filename = os.path.splitext(os.path.basename(input_shp))[0]
    output_path = os.path.join(output_folder, f"{input_filename}.shp")

    out_gdf.to_file(output_path)
    print(f"Done! Output file: {output_path}. Principal direction rotation: {angle:.2f} deg")

if __name__ == "__main__":
    main()