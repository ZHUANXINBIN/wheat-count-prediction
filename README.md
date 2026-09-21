# Wheat Count Prediction — Source Code

This repository contains the source code used in the paper *[Your Paper Title]*.
It covers three stages: (1) deep-learning model development and ablation studies,
(2) machine-learning (AutoGluon) baseline pipeline, and (3) the deployment/prediction
pipeline used to generate field-scale wheat count maps from UAV imagery.

## Repository Structure

```
Paper_code/
├── Model_training/
│   ├── wheat_count_labels.csv
│   ├── DL/          # Deep-learning ablation pipeline
│   │   ├── Data/                 # split index, normalization stats
│   │   └── DenseNet-121/          # DenseNet size comparison (121/169/201)
│   └── ML/          # AutoGluon feature-based ML pipeline
│       └── 2_train_model/
└── predict_process/ # End-to-end deployment pipeline
```

## Model_training/DL/ — Deep Learning Ablation Pipeline

Run in this order:

1. `0_control_variable_DL00_check_consistency.py` — verifies that DOM images, labels,
   and train/test split index are fully aligned.
2. `0_control_variable_DL01_compute_indices.py` — computes vegetation indices
   (NDVI, GNDVI, NDRE, CIre, CIgreen, GRVI) from the 7-band multispectral DOM.
3. `0_control_variable_DL02_compute_norm_stats.py` — computes per-channel
   normalization statistics from the training set only (avoids data leakage).
4. `1_G_ablation_dataset.py`, `1_G_ablation_model.py`, `1_G_ablation_trainer.py`,
   `1_G_ablation_run_g_ablation.py` — input-channel-group ablation (G0–G6),
   ResNet-50 backbone, 5-fold cross-validation.
5. `2_aug_ablation_run_g_ablation.py` — data augmentation strategy ablation
   (A0–A6), fixed on the best input group from step 4.
6. `3_arch_compare_models.py`, `3_arch_compare_measure_inference.py`,
   `3_arch_compare_run_arch_compare.py` — architecture comparison across
   6 backbones (ResNet-50, DenseNet-121, EfficientNet-B0, MobileNetV3-Small,
   ConvNeXt-Tiny, EfficientViT-B0), all trained from scratch.
7. `DenseNet-121/densenet_model.py`, `DenseNet-121/run_densenet_compare.py` —
   follow-up size comparison within the DenseNet family (121/169/201).
   `densenet_model.py` also provides `build_densenet()`, imported by
   `run_final_eval.py` and `sample_size_ablation.py`.
8. `run_final_eval.py` — trains the top-3 architectures with 5-fold CV and
   evaluates once on the sealed test set (paper-reportable numbers).
9. `fix_model_selection.py` — post-hoc correction confirming model selection
   is based on development-pool validation R², not the sealed test set.
10. `sample_size_ablation.py` — sensitivity analysis on training-set size
    (10%/25%/50%/75%/100% of the development pool), DenseNet-121 + G3 + A3.

## Model_training/ML/ — AutoGluon Feature-Based Pipeline

Run `main.py` to execute the full feature-engineering pipeline in order:

1. `01_data_check.py` — data quality check (band values, label distribution).
2. `02_feature_vegetation.py` — vegetation index statistical features.
3. `03_feature_spectral.py` — spectral statistical features (mean/std/skewness/etc.).
4. `04_feature_color.py` — RGB/HSV/Lab color-space features.
5. `05_feature_texture.py` — GLCM, LBP, and wavelet texture features.
6. `06_feature_spatial.py` — spatial structure features (fractal dimension,
   K-means cluster ratios, Moran's I, Geary's C).
7. `07_feature_merge.py` — merges all feature tables, cleans NaN/inf, aligns
   with labels.

Then, inside `2_train_model/`:

8. `10_train_baseline.py` — AutoGluon baseline model with all features.
9. `11_feature_curve.py` — feature-count ablation to determine the optimal
   number of top-importance features.
10. `12_train_final.py` — final model trained on the selected feature subset,
    with 1 control + 5 random-seed stability runs.

`config.py` in each folder centralizes all paths, band indices, and
hyperparameters.

## predict_process/ — Deployment Pipeline

1. `1_3m_to_60m/` — crops 3 m-altitude RGB images into 1 m² patches and
   geometrically transforms/re-grids DOM orthomosaics.
2. `2_YOLO_detection/` — YOLOv11s-based wheat spike detection and
   SAHI-based pseudo-label generation (includes the trained weights
   `3m_more_data_yolo11s.pt`).
3. `3_train_DenseNet-121/` — trains the deployment DenseNet-121 regression
   model on 10-channel (7 raw bands + NDVI + GNDVI + NDRE) 1 m² patches
   (includes the trained weights `densenet121_wheat_final.pth`).
4. `4_field_preparation/` — splits large-field shapefiles into 1 m² grid
   cells and prepares DOM clips for inference.
5. `5_wheat_count/` — runs inference over an entire field, joins predictions
   back to the shapefile geometry, and generates the wheat-count
   distribution map.

## Environment

- Python 3.12
- Key dependencies: `torch`, `torchvision`, `timm`, `rasterio`, `geopandas`,
  `shapely`, `opencv-python`, `scikit-image`, `pywt`, `autogluon.tabular`,
  `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `ultralytics`

Install with:

```bash
pip install -r requirements.txt
```

## Data Availability

Raw UAV imagery and field shapefiles are not included in this repository
due to data-sharing restrictions from the commercial farm operators
involved in this study. `wheat_count_labels.csv` (YOLO-derived pseudo-labels
and manual image-reference counts) and the split/normalization files under
`Model_training/DL/Data/` are included and are sufficient to reproduce the
reported ablation and sealed-test-set results. Further methodological
details are provided in the paper.

## Citation

If you use this code, please cite:

```
Zhuang, X., Burridge, J., Wang, H., & Guo, W. (2026). Cross-Resolution
Teacher--Student Pseudo-Label Supervision for Wheat Spike-Density
Estimation from Dual-Altitude UAV Multispectral Imagery.
Manuscript submitted for publication.
```
