import os
import cv2
import numpy as np
from pathlib import Path
from tkinter import Tk, filedialog
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import pandas as pd
from tqdm import tqdm
from datetime import datetime

MODEL_PATH = "3m_more_data_yolo11s.pt"
CONFIDENCE = 0.35
SLICE_SIZE = 640
OVERLAP_RATIO = 0.1
BOX_COLOR = (0, 0, 255)
BOX_THICKNESS = 2


def select_folder():
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    folder_path = filedialog.askdirectory(title="Select the folder containing images")
    root.destroy()
    return folder_path


def get_image_files(folder_path):
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    image_files = []
    for file in Path(folder_path).iterdir():
        if file.suffix.lower() in image_extensions and file.is_file():
            image_files.append(file)
    return sorted(image_files)


def sahi_detect_on_image(model, image_path, slice_size, overlap_ratio):
    result = get_sliced_prediction(
        str(image_path),
        model,
        slice_height=slice_size,
        slice_width=slice_size,
        overlap_height_ratio=overlap_ratio,
        overlap_width_ratio=overlap_ratio,
        verbose=0
    )

    detections = []
    for pred in result.object_prediction_list:
        bbox = pred.bbox
        detections.append({
            'bbox': np.array([bbox.minx, bbox.miny, bbox.maxx, bbox.maxy]),
            'confidence': pred.score.value
        })

    return detections


def save_yolo_labels(detections, image_shape, output_path):
    h, w = image_shape[:2]

    with open(output_path, 'w') as f:
        for det in detections:
            bbox = det['bbox']
            x1, y1, x2, y2 = bbox

            x_center = ((x1 + x2) / 2) / w
            y_center = ((y1 + y2) / 2) / h
            width = (x2 - x1) / w
            height = (y2 - y1) / h

            f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


def draw_boxes(image, detections, color=BOX_COLOR, thickness=BOX_THICKNESS):
    image_with_boxes = image.copy()

    for det in detections:
        bbox = det['bbox'].astype(int)
        x1, y1, x2, y2 = bbox
        cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), color, thickness)

    return image_with_boxes


def process_images(folder_path):
    print("=" * 70)
    print("SAHI Wheat Detection System")
    print("=" * 70)

    image_files = get_image_files(folder_path)
    if not image_files:
        print("No image files found!")
        return

    print(f"Found {len(image_files)} images")

    output_base = Path(folder_path) / "wheat_detection_results"
    output_base.mkdir(exist_ok=True)

    images_dir = output_base / "detected_images"
    labels_dir = output_base / "yolo_labels"
    stats_dir = output_base / "statistics"

    images_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)
    stats_dir.mkdir(exist_ok=True)

    print(f"Output directory: {output_base}")

    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found: {MODEL_PATH}")
        return

    print("Loading model...")
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=MODEL_PATH,
        confidence_threshold=CONFIDENCE,
        device="cuda:0"
    )
    print("Model loaded")

    results_data = []
    start_time = datetime.now()

    print("\nStarting SAHI detection...")
    for img_file in tqdm(image_files, desc="Progress"):
        try:
            detections = sahi_detect_on_image(
                detection_model, img_file,
                SLICE_SIZE, OVERLAP_RATIO
            )

            wheat_count = len(detections)

            image = cv2.imread(str(img_file))

            image_with_boxes = draw_boxes(image, detections)
            output_img_path = images_dir / img_file.name
            cv2.imwrite(str(output_img_path), image_with_boxes)

            label_path = labels_dir / f"{img_file.stem}.txt"
            save_yolo_labels(detections, image.shape, label_path)

            results_data.append({
                'region_name': img_file.stem,
                'filename': img_file.name,
                'wheat_count': wheat_count,
                'image_width': image.shape[1],
                'image_height': image.shape[0]
            })

        except Exception as e:
            print(f"\nFailed to process {img_file.name}: {str(e)}")
            results_data.append({
                'region_name': img_file.stem,
                'filename': img_file.name,
                'wheat_count': 0,
                'image_width': 0,
                'image_height': 0
            })

    elapsed = (datetime.now() - start_time).total_seconds()

    save_statistics(stats_dir, results_data, elapsed)

    total_wheat = sum(r['wheat_count'] for r in results_data)
    avg_wheat = total_wheat / len(results_data) if results_data else 0

    print(f"\n{'=' * 70}")
    print(f"Detection complete!")
    print(f"   Total images: {len(results_data)}")
    print(f"   Total detections: {total_wheat}")
    print(f"   Average per image: {avg_wheat:.1f}")
    print(f"   Processing time: {elapsed / 60:.1f} minutes")
    print(f"\nResults saved to:")
    print(f"   Detected images: {images_dir}/")
    print(f"   YOLO labels: {labels_dir}/")
    print(f"   Statistics: {stats_dir}/")
    print("=" * 70)


def save_statistics(stats_dir, results_data, elapsed):
    df_detailed = pd.DataFrame(results_data)

    total_count = df_detailed['wheat_count'].sum()
    avg_count = df_detailed['wheat_count'].mean()

    summary = pd.DataFrame([
        {
            'region_name': 'TOTAL',
            'filename': '-',
            'wheat_count': total_count,
            'image_width': '-',
            'image_height': '-'
        },
        {
            'region_name': 'AVERAGE',
            'filename': '-',
            'wheat_count': round(avg_count, 1),
            'image_width': '-',
            'image_height': '-'
        }
    ])

    df_detailed = pd.concat([df_detailed, summary], ignore_index=True)

    excel_path = stats_dir / "detection_details.xlsx"
    df_detailed.to_excel(excel_path, index=False, engine='openpyxl')

    df_simple = pd.DataFrame(results_data)[['region_name', 'wheat_count']]
    csv_path = stats_dir / "wheat_count_labels.csv"
    df_simple.to_csv(csv_path, index=False)

    txt_path = stats_dir / "detection_summary.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("SAHI Wheat Detection Report\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Detection time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model file: {MODEL_PATH}\n")
        f.write(f"Confidence threshold: {CONFIDENCE}\n")
        f.write(f"SAHI slice size: {SLICE_SIZE}x{SLICE_SIZE}\n")
        f.write(f"Overlap ratio: {OVERLAP_RATIO}\n")
        f.write(f"Processing time: {elapsed / 60:.1f} minutes\n\n")

        f.write(f"Total images: {len(results_data)}\n")
        f.write(f"Total detections: {total_count}\n")
        f.write(f"Average detections: {avg_count:.1f}\n")
        f.write(f"Max: {max(r['wheat_count'] for r in results_data)}\n")
        f.write(f"Min: {min(r['wheat_count'] for r in results_data)}\n\n")

        f.write("=" * 70 + "\n")
        f.write("Detailed Results\n")
        f.write("=" * 70 + "\n")
        f.write(f"{'Filename':<50} {'Count':<10}\n")
        f.write("-" * 70 + "\n")

        for result in results_data:
            f.write(f"{result['filename']:<50} {result['wheat_count']:<10}\n")


def main():
    folder_path = select_folder()

    if not folder_path:
        print("No folder selected, exiting")
        return

    print(f"Selected folder: {folder_path}")

    process_images(folder_path)


if __name__ == "__main__":
    main()