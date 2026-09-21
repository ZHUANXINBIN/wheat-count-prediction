import os

os.environ["AUTOGLUON_MAX_CONCURRENT_TRIALS"] = "1"

os.environ["OMP_NUM_THREADS"]     = "8"
os.environ["MKL_NUM_THREADS"]     = "8"
os.environ["OPENBLAS_NUM_THREADS"]= "8"

os.environ["CUDA_VISIBLE_DEVICES"] = ""

os.environ["AG_DYNAMIC_STACKING"] = "false"
os.environ["AG_PARALLEL_FOLD_FITTING_STRATEGY"] = "sequential"

os.environ["RAY_DISABLE_IMPORT_WARNING"] = "1"
os.environ["RAY_IGNORE_UNHANDLED_ERRORS"]= "1"
os.environ["RAY_LOG_TO_STDERR"]          = "0"
os.environ["RAY_NUM_CPUS"] = "1"

os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]         = "3"


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FEATURES_CSV = os.path.join(BASE_DIR, "features_all.csv")

OUTPUT_DIR          = os.path.join(BASE_DIR, "outputs")
BASELINE_MODEL_DIR  = os.path.join(OUTPUT_DIR, "baseline_model")
FINAL_MODEL_DIR     = os.path.join(OUTPUT_DIR, "final_model")
REPORT_DIR          = os.path.join(OUTPUT_DIR, "reports")

FEATURE_IMPORTANCE_CSV = os.path.join(OUTPUT_DIR, "feature_importance.csv")
TRAIN_DATA_CSV         = os.path.join(OUTPUT_DIR, "train_data.csv")
TEST_DATA_CSV          = os.path.join(OUTPUT_DIR, "test_data.csv")
FINAL_FEATURES_TXT     = os.path.join(OUTPUT_DIR, "final_features.txt")


TARGET_COL = "wheat_count"
META_COLS  = ["stem", "wheat_count", "label_log"]


TEST_SIZE   = 0.2
RANDOM_SEED = 42


BASELINE_TIME_LIMIT  = 600
CURVE_TIME_LIMIT     = 180
FINAL_TIME_LIMIT     = 3600
STABILITY_TIME_LIMIT = 3600

NUM_BAG_FOLDS = 5

BASELINE_STACK_LEVELS = 0
FINAL_STACK_LEVELS    = 1

BASELINE_PRESET = "good_quality"
FINAL_PRESET    = "best_quality"
EVAL_METRIC     = "r2"


CURVE_N_FEATURES = [10, 20, 30, 40, 50, 60, 80, 100, -1]


FINAL_N_FEATURES = 60


STABILITY_SEEDS = [42, 123, 456, 789, 2024]


IMPORTANCE_SUBSAMPLE_SIZE   = None
IMPORTANCE_NUM_SHUFFLE_SETS = 5
DELETE_INTERMEDIATE_MODELS  = False


def validate():
    errors = []

    if not os.path.exists(FEATURES_CSV):
        errors.append(f"  FEATURES_CSV not found: {FEATURES_CSV}")
    else:
        print(f"  FEATURES_CSV found: {os.path.basename(FEATURES_CSV)}")

    if FINAL_N_FEATURES != -1 and FINAL_N_FEATURES < 1:
        errors.append(
            f"  FINAL_N_FEATURES must be -1 or >=1, got {FINAL_N_FEATURES}")

    if TEST_SIZE <= 0 or TEST_SIZE >= 1:
        errors.append(f"  TEST_SIZE must be 0-1, got {TEST_SIZE}")

    print(f"  CUDA_VISIBLE_DEVICES = "
          f"'{os.environ.get('CUDA_VISIBLE_DEVICES')}' (GPU disabled)")
    print(f"  AG_DYNAMIC_STACKING  = "
          f"'{os.environ.get('AG_DYNAMIC_STACKING')}' (DyStack disabled)")
    print(f"  OMP_NUM_THREADS      = {os.environ.get('OMP_NUM_THREADS')}")
    print(f"  MAX_CONCURRENT_TRIALS= "
          f"{os.environ.get('AUTOGLUON_MAX_CONCURRENT_TRIALS')}")

    if errors:
        print("\n[config] Validation errors:")
        for e in errors:
            print(e)
        return False

    print(f"  Config validation passed.")
    return True


if __name__ == "__main__":
    print("=" * 55)
    print("Config Summary")
    print("=" * 55)
    print(f"  Machine:          18C/36T + dual GTX1080Ti + Win10")
    print(f"  Input:            {os.path.basename(FEATURES_CSV)}")
    print(f"  Output dir:       {OUTPUT_DIR}")
    print(f"  Target:           {TARGET_COL}")
    print(f"  Test size:        {TEST_SIZE} (~{int(518*TEST_SIZE)} samples)")
    print(f"  Random seed:      {RANDOM_SEED}")
    print(f"  Baseline time:    {BASELINE_TIME_LIMIT}s "
          f"({BASELINE_TIME_LIMIT//60}min)")
    print(f"  Curve time/N:     {CURVE_TIME_LIMIT}s x "
          f"{len(CURVE_N_FEATURES)} = "
          f"~{CURVE_TIME_LIMIT*len(CURVE_N_FEATURES)//60}min")
    print(f"  Final time:       {FINAL_TIME_LIMIT}s "
          f"({FINAL_TIME_LIMIT//60}min)")
    print(f"  CV folds:         {NUM_BAG_FOLDS}")
    print(f"  Baseline stacking:{BASELINE_STACK_LEVELS} (disabled)")
    print(f"  Final stacking:   {FINAL_STACK_LEVELS} (enabled)")
    print(f"  Curve N list:     {CURVE_N_FEATURES}")
    print(f"  FINAL_N_FEATURES: {FINAL_N_FEATURES}  (update after running 11)")
    print(f"  Stability seeds:  {STABILITY_SEEDS}")
    print()
    validate()