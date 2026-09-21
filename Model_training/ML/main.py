import os
import time
import config

def ensure_dirs():
    for d in [config.OUTPUT_DIR, config.MASK_DIR, config.MODEL_DIR, config.REPORT_DIR]:
        os.makedirs(d, exist_ok=True)

def run_step(step_num, step_name, module_func):
    print(f"\n{'='*60}")
    print(f"  Step {step_num}: {step_name}")
    print(f"{'='*60}")
    t0 = time.time()
    try:
        module_func()
        elapsed = time.time() - t0
        print(f"  Done, elapsed {elapsed:.1f}s")
    except Exception as e:
        print(f"  Failed: {e}")
        raise

def main():
    ensure_dirs()
    print("\nWheat Count Prediction ML Pipeline Starting")
    print(f"   DOM directory: {config.DOM_DIR}")
    print(f"   Label file: {config.LABEL_CSV}")
    print(f"   Output directory: {config.OUTPUT_DIR}")

    from data_check import run as run_data_check
    run_step(1, "Data Quality Check", run_data_check)

    from feature_vegetation import run as run_veg
    run_step(3, "Vegetation Index Feature Extraction", run_veg)

    from feature_spectral import run as run_spec
    run_step(4, "Spectral Statistical Feature Extraction", run_spec)

    from feature_color import run as run_color
    run_step(5, "Color Feature Extraction (RGB)", run_color)

    from feature_texture import run as run_tex
    run_step(6, "Texture Feature Extraction (GLCM/LBP/Wavelet)", run_tex)

    from feature_spatial import run as run_spa
    run_step(7, "Spatial Structure Feature Extraction", run_spa)

    from feature_merge import run as run_merge
    run_step(8, "Feature Merging and Cleaning", run_merge)

    from feature_selection import run as run_select
    run_step(9, "Feature Selection (Spearman + VIF)", run_select)

    from modeling import run as run_model
    run_step(10, "AutoGluon Modeling and Ablation Study", run_model)

    print(f"\n{'='*60}")
    print("  Full pipeline complete! Results saved in:", config.OUTPUT_DIR)
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()