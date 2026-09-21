import os
import json
import subprocess
import cv2
import math
import tkinter as tk
from tkinter import filedialog
import geopandas as gpd
from shapely.geometry import Polygon

EXIFTOOL_PATH = r"Z:\Zhuang_Pro\2025\3_Wheatcount\exiftool.exe"


class RGBDroneProcessor:
    def __init__(self):
        self.rgb_sensor_width_mm = 17.3
        self.rgb_focal_mm = 12.29

        self.fixed_flight_height = 3.0

        self.image_exts = {'.jpg', '.jpeg', '.png'}

    def batch_read_metadata(self, folder_path):
        cmd = [
            EXIFTOOL_PATH, '-r', '-json', '-n',
            '-GPSLatitude', '-GPSLongitude',
            '-ImageWidth', '-ImageHeight',
            '-FocalLength',
            '-FlightYawDegree', '-GimbalYawDegree',
            '-RelativeAltitude', '-AbsoluteAltitude',
            '-Model',
            '-ext', 'jpg', '-ext', 'jpeg', '-ext', 'png',
            folder_path
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True,
                                     encoding='utf-8', errors='ignore')
            if not result.stdout.strip():
                print(f"exiftool produced no output. stderr: {result.stderr}")
                return []
            records = json.loads(result.stdout)
            print(f"exiftool successfully parsed metadata for {len(records)} files")
            return records
        except Exception as e:
            print(f"exiftool batch read failed: {e}")
            return []

    def get_camera_params(self, meta):
        focal = meta.get('FocalLength')
        try:
            focal = float(focal) if focal is not None else self.rgb_focal_mm
        except (TypeError, ValueError):
            focal = self.rgb_focal_mm
        return self.rgb_sensor_width_mm, focal

    def get_yaw(self, meta):
        for key in ('FlightYawDegree', 'GimbalYawDegree'):
            val = meta.get(key)
            if val is not None:
                try:
                    return float(val)
                except (TypeError, ValueError):
                    continue
        return None

    def calc_crop_size(self, sensor_width_mm, focal_mm, image_width_px):
        gsd = (sensor_width_mm * self.fixed_flight_height) / (focal_mm * image_width_px)
        return max(1, int(round(1.0 / gsd)))

    def crop_center_1m2(self, image_path, output_folder, crop_size):
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        h, w = img.shape[:2]
        crop_size = min(crop_size, w, h)
        start_x = (w - crop_size) // 2
        start_y = (h - crop_size) // 2
        cropped = img[start_y:start_y + crop_size, start_x:start_x + crop_size]
        name, ext = os.path.splitext(os.path.basename(image_path))
        output_path = os.path.join(output_folder, f"cropped_1m2_{name}{ext}")
        cv2.imwrite(output_path, cropped)
        return output_path

    def rotate_point(self, ox, oy, px, py, angle_deg):
        rad = math.radians(angle_deg)
        qx = ox + math.cos(rad) * (px - ox) - math.sin(rad) * (py - oy)
        qy = oy + math.sin(rad) * (px - ox) + math.cos(rad) * (py - oy)
        return qx, qy

    def create_shapefile(self, photo_infos, shapefile_folder):
        if not photo_infos:
            return None

        pts = gpd.GeoDataFrame(
            photo_infos,
            geometry=gpd.points_from_xy(
                [p['lon'] for p in photo_infos],
                [p['lat'] for p in photo_infos]),
            crs='EPSG:4326'
        ).to_crs('EPSG:6681')

        geometries, attributes = [], []
        half = 0.5
        for i, info in enumerate(photo_infos):
            yaw = info['yaw']
            math_angle = 90 - yaw

            px, py = pts.geometry.iloc[i].x, pts.geometry.iloc[i].y
            corners = [(px - half, py - half), (px + half, py - half),
                       (px + half, py + half), (px - half, py + half)]
            rotated = [self.rotate_point(px, py, x, y, math_angle) for x, y in corners]

            poly_wgs84 = gpd.GeoDataFrame(
                [1], geometry=[Polygon(rotated)], crs='EPSG:6681'
            ).to_crs('EPSG:4326').geometry.iloc[0]

            geometries.append(poly_wgs84)
            attributes.append({
                'filename': info['filename'][:10],
                'latitude': info['lat'],
                'longitude': info['lon'],
                'yaw_deg': round(yaw, 2),
                'crop_px': info['crop_size'],
                'flt_h': self.fixed_flight_height,
            })

        gdf = gpd.GeoDataFrame(attributes, geometry=geometries, crs='EPSG:4326')
        shp_path = os.path.join(shapefile_folder, 'center_1m2_squares.shp')
        gdf.to_file(shp_path)
        print(f"Shapefile saved: {shp_path}")
        return shp_path

    def process_folder(self):
        root = tk.Tk()
        root.withdraw()

        folder_path = filedialog.askdirectory(title='Select the folder containing drone photos')
        if not folder_path:
            print("No input folder selected")
            return
        output_folder = filedialog.askdirectory(title='Select the folder to save output results')
        if not output_folder:
            return

        cropped_folder = os.path.join(output_folder, 'cropped_1m2_center')
        shapefile_folder = os.path.join(output_folder, 'shapefile')
        os.makedirs(cropped_folder, exist_ok=True)
        os.makedirs(shapefile_folder, exist_ok=True)

        records = self.batch_read_metadata(folder_path)
        if not records:
            print("No metadata retrieved, check exiftool path/folder path for special characters")
            return

        processed = []
        for idx, meta in enumerate(records, 1):
            filepath = meta.get('SourceFile')
            if not filepath:
                continue

            ext = os.path.splitext(filepath)[1].lower()
            if ext not in self.image_exts:
                continue

            fname = os.path.basename(filepath)
            print(f"Processing {idx}/{len(records)}: {fname}")

            lat, lon = meta.get('GPSLatitude'), meta.get('GPSLongitude')
            if lat is None or lon is None:
                print("  Skipped: no GPS")
                continue

            yaw = self.get_yaw(meta)
            if yaw is None:
                print("  Yaw angle not found, skipping this image (check whether it actually contains XMP attitude info)")
                continue

            width = meta.get('ImageWidth')
            if not width:
                img_probe = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
                if img_probe is None:
                    print("  Skipped: image could not be read")
                    continue
                width = img_probe.shape[1]

            sensor_width, focal = self.get_camera_params(meta)
            crop_size = self.calc_crop_size(sensor_width, focal, width)
            print(f"  Yaw={yaw:.1f} deg, Focal length={focal}mm, Crop size={crop_size}px")

            cropped_path = self.crop_center_1m2(filepath, cropped_folder, crop_size)
            if not cropped_path:
                print("  Crop failed")
                continue

            processed.append({
                'filename': fname, 'lat': lat, 'lon': lon, 'yaw': yaw,
                'crop_size': crop_size
            })
            print("  Done")

        if processed:
            print("\nGenerating Shapefile...")
            shp_path = self.create_shapefile(processed, shapefile_folder)
            print(f"\nDone! Success: {len(processed)} images, Shapefile: {shp_path}")
        else:
            print("No valid photos")


def main():
    RGBDroneProcessor().process_folder()


if __name__ == '__main__':
    main()