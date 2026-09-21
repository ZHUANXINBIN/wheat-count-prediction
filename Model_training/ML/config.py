import os

DOM_DIR = r"Z:\Zhuang_Pro\2025\3_Wheatcount\17_control_variable_AutoGluon\0_Data\1_DOM"
LABEL_CSV = r"Z:\Zhuang_Pro\2025\3_Wheatcount\17_control_variable_AutoGluon\0_Data\wheat_count_labels.csv"

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
MASK_DIR   = os.path.join(OUTPUT_DIR, "masks")
MODEL_DIR  = os.path.join(OUTPUT_DIR, "models")
REPORT_DIR = os.path.join(OUTPUT_DIR, "reports")

CSV_NAME_COL   = "region_name"
CSV_LABEL_COL  = "wheat_count"

BAND_BLUE_RGB = 0
BAND_GREEN_RGB = 1
BAND_RED_RGB  = 2
BAND_GREEN_MS = 3
BAND_RED_MS   = 4
BAND_RE       = 5
BAND_NIR      = 6

SPYNDEX_BANDS = {
    "G": BAND_GREEN_MS,
    "R": BAND_RED_MS,
    "RE": BAND_RE,
    "N": BAND_NIR,
}

NDRE_MASK_THRESHOLD = 0.15
MIN_VEGETATION_RATIO = 0.10

GLCM_BANDS = [BAND_RED_MS, BAND_RE, BAND_NIR]
GLCM_BAND_NAMES = ["Red650", "RE735", "NIR860"]
GLCM_ANGLES  = [0, 0.785398, 1.5708, 2.35619]
GLCM_DISTANCES = [1, 3]
GLCM_LEVELS  = 64
LBP_RADIUS   = 3
LBP_N_POINTS = 8 * LBP_RADIUS
WAVELET_NAME   = "haar"
WAVELET_LEVELS = 2

KMEANS_K       = 3
KMEANS_RANDOM_STATE = 42

SPEARMAN_THRESHOLD = 0.05
VIF_THRESHOLD      = 10.0

AG_LABEL        = "wheat_count"
AG_EVAL_METRIC  = "rmse"
AG_TIME_LIMIT   = 600
AG_NUM_FOLDS    = 8
AG_RANDOM_SEED  = 42
AG_PRESETS      = ["extreme_quality", "best_quality"]

RANDOM_SEED = 42
IMAGE_SIZE  = (64, 64)
N_BANDS     = 8
N_VALID_BANDS = 7
USE_VEGETATION_MASK = False
USE_LOG_TRANSFORM = False