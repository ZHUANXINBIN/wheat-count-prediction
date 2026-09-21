import config

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from autogluon.tabular import TabularPredictor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from datetime import datetime
import shutil

warnings.filterwarnings("ignore")


def calc_metrics(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask   = y_true != 0
    mape   = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) \
             if mask.sum() > 0 else np.nan
    return {
        "R2":   float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE":  float(mean_absolute_error(y_true, y_pred)),
        "MAPE": mape,
    }


def init_ray_sequential():
    try:
        import ray
        if ray.is_initialized():
            ray.shutdown()
        ray.init(
            num_cpus=1,
            num_gpus=0,
            ignore_reinit_error=True,
            include_dashboard=False,
            log_to_driver=False,
        )
        print("   Ray initialized: num_cpus=1 (sequential mode)")
    except Exception as e:
        print(f"   Ray init skipped: {e}")


def plot_feature_curve(curve_df, out_path):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(curve_df["n_features"], curve_df["R2"],
             "o-", color="#2980B9", linewidth=2, markersize=7, label="R2")
    ax1.set_xlabel("Number of Features", fontsize=12)
    ax1.set_ylabel("R2", color="#2980B9", fontsize=12)
    ax1.tick_params(axis="y", labelcolor="#2980B9")

    ax2 = ax1.twinx()
    ax2.plot(curve_df["n_features"], curve_df["RMSE"],
             "s--", color="#E74C3C", linewidth=1.5, markersize=6, label="RMSE")
    ax2.set_ylabel("RMSE", color="#E74C3C", fontsize=12)
    ax2.tick_params(axis="y", labelcolor="#E74C3C")

    best_idx  = curve_df["R2"].idxmax()
    best_row  = curve_df.loc[best_idx]
    ax1.axvline(x=best_row["n_features"], color="green",
                linestyle=":", linewidth=1.5, alpha=0.7)
    ax1.annotate(
        f"Best N={int(best_row['n_features'])}\nR2={best_row['R2']:.4f}",
        xy=(best_row["n_features"], best_row["R2"]),
        xytext=(best_row["n_features"] + 3, best_row["R2"] - 0.01),
        fontsize=9, color="green",
        arrowprops=dict(arrowstyle="->", color="green", lw=1.2),
    )

    for _, row in curve_df.iterrows():
        ax1.text(row["n_features"], row["R2"] + 0.002,
                 f"{row['R2']:.4f}", ha="center", fontsize=7, color="#2980B9")

    ax1.set_title("Feature Curve: N Features vs R2 / RMSE", fontsize=13)
    ax1.grid(True, alpha=0.3)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc="lower right", fontsize=10)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   [Plot] Saved: {out_path}")


def plot_curve_detail(curve_df, out_path):
    df = curve_df.copy().sort_values("n_features").reset_index(drop=True)
    r2_diff = df["R2"].diff().fillna(0)

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["#2ECC71" if v >= 0 else "#E74C3C" for v in r2_diff]
    ax.bar(range(len(df)), r2_diff, color=colors, width=0.6)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([str(int(n)) for n in df["n_features"]], fontsize=9)
    ax.set_xlabel("Number of Features", fontsize=11)
    ax.set_ylabel("Delta R2 (marginal gain)", fontsize=11)
    ax.set_title("Marginal R2 Gain per Feature Batch", fontsize=12)
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.grid(True, alpha=0.3, axis="y")

    for i, v in enumerate(r2_diff):
        ax.text(i, v + (0.0002 if v >= 0 else -0.0005),
                f"{v:+.4f}", ha="center", fontsize=7)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   [Plot] Saved: {out_path}")


def run():
    os.makedirs(config.REPORT_DIR, exist_ok=True)

    start_time = datetime.now()
    print("\n" + "=" * 65)
    print("11_feature_curve.py - Feature Number Ablation")
    print("=" * 65)

    print("\n[0/4] Initializing Ray (sequential mode)...")
    init_ray_sequential()

    print("\n[1/4] Loading sealed train/test data & feature importance...")

    for path, name in [(config.TRAIN_DATA_CSV, "train_data.csv"),
                       (config.TEST_DATA_CSV,  "test_data.csv"),
                       (config.FEATURE_IMPORTANCE_CSV, "feature_importance.csv")]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{name} not found: {path}\n"
                f"Please run 10_train_baseline.py first."
            )

    train_df     = pd.read_csv(config.TRAIN_DATA_CSV, encoding="utf-8-sig")
    test_df      = pd.read_csv(config.TEST_DATA_CSV,  encoding="utf-8-sig")
    importance_df= pd.read_csv(config.FEATURE_IMPORTANCE_CSV, encoding="utf-8-sig")

    print(f"   Train: {len(train_df)} samples x {train_df.shape[1]} cols")
    print(f"   Test:  {len(test_df)} samples  x {test_df.shape[1]} cols")
    print(f"   Feature importance: {len(importance_df)} features loaded")

    positive_features = (
        importance_df[importance_df["importance"] > 0]
        .sort_values("importance", ascending=False)["feature"]
        .tolist()
    )
    n_positive = len(positive_features)
    print(f"   Positive importance features: {n_positive} "
          f"(negative {len(importance_df) - n_positive} features excluded)")

    n_list = []
    for n in config.CURVE_N_FEATURES:
        if n == -1:
            n_list.append(n_positive)
        elif n <= n_positive:
            n_list.append(n)
        else:
            print(f"   N={n} > positive features ({n_positive}), skipped.")
    n_list = sorted(set(n_list))
    print(f"   N values to test: {n_list}")

    print(f"\n[2/4] Running feature curve "
          f"({len(n_list)} experiments x {config.CURVE_TIME_LIMIT}s each)...")
    print(f"   Estimated total time: "
          f"~{len(n_list) * config.CURVE_TIME_LIMIT // 60} min\n")

    curve_records = []
    curve_model_dir = os.path.join(config.OUTPUT_DIR, "_curve_model_tmp")

    for i, n in enumerate(n_list):
        selected_features = positive_features[:n]
        exp_start = datetime.now()

        print(f"   [{i+1}/{len(n_list)}] N={n:3d} features  "
              f"started at {exp_start.strftime('%H:%M:%S')} ...")

        cols = selected_features + [config.TARGET_COL]
        train_n = train_df[cols].copy()
        test_n  = test_df[cols].copy()

        if os.path.exists(curve_model_dir):
            shutil.rmtree(curve_model_dir)
        os.makedirs(curve_model_dir)

        try:
            predictor = TabularPredictor(
                label=config.TARGET_COL,
                problem_type="regression",
                eval_metric=config.EVAL_METRIC,
                path=curve_model_dir,
                verbosity=0,
            ).fit(
                train_data=train_n,
                time_limit=config.CURVE_TIME_LIMIT,
                presets=config.BASELINE_PRESET,
                num_bag_folds=config.NUM_BAG_FOLDS,
                num_bag_sets=1,
                num_stack_levels=config.BASELINE_STACK_LEVELS,
                dynamic_stacking=False,
                ag_args_fit={"num_cpus": 1},
            )

            y_pred   = predictor.predict(test_n.drop(columns=[config.TARGET_COL])).values
            y_test   = test_n[config.TARGET_COL].values
            metrics  = calc_metrics(y_test, y_pred)
            exp_time = (datetime.now() - exp_start).total_seconds()

            curve_records.append({
                "n_features": n,
                "R2":   metrics["R2"],
                "RMSE": metrics["RMSE"],
                "MAE":  metrics["MAE"],
                "MAPE": metrics["MAPE"],
                "runtime_s": round(exp_time, 1),
            })

            print(f"          R2={metrics['R2']:.4f}  "
                  f"RMSE={metrics['RMSE']:.2f}  "
                  f"time={exp_time/60:.1f}min")

        except Exception as e:
            print(f"          Failed: {e}")
            curve_records.append({
                "n_features": n,
                "R2": np.nan, "RMSE": np.nan,
                "MAE": np.nan, "MAPE": np.nan,
                "runtime_s": np.nan,
            })

    if os.path.exists(curve_model_dir):
        shutil.rmtree(curve_model_dir)
        print(f"\n   Cleaned up temporary model directory.")

    print(f"\n[3/4] Analyzing results...")

    curve_df = pd.DataFrame(curve_records).dropna(subset=["R2"])
    curve_df = curve_df.sort_values("n_features").reset_index(drop=True)

    print(f"\n   Feature Curve Results")
    print(f"   {'N':>6}  {'R2':>8}  {'RMSE':>8}  {'MAPE':>7}  {'Time':>6}")
    print(f"   {'-'*6}  {'-'*8}  {'-'*8}  {'-'*7}  {'-'*6}")
    for _, row in curve_df.iterrows():
        print(f"   {int(row['n_features']):>6}  "
              f"{row['R2']:>8.4f}  "
              f"{row['RMSE']:>8.2f}  "
              f"{row['MAPE']:>6.2f}%  "
              f"{row['runtime_s']/60:>5.1f}m")

    best_idx  = curve_df["R2"].idxmax()
    best_n    = int(curve_df.loc[best_idx, "n_features"])
    best_r2   = curve_df.loc[best_idx, "R2"]

    r2_diffs   = curve_df["R2"].diff().fillna(0).values
    max_gain   = r2_diffs.max()
    threshold  = max_gain * 0.05
    elbow_n    = best_n
    for j in range(1, len(curve_df)):
        if r2_diffs[j] < threshold:
            elbow_n = int(curve_df.iloc[j - 1]["n_features"])
            break

    print(f"\n   Best R2 at N={best_n}: {best_r2:.4f}")
    print(f"   Suggested elbow point: N={elbow_n}  "
          f"(first N where marginal gain < {threshold:.5f})")
    print(f"\n   Recommendation: set config.FINAL_N_FEATURES = {elbow_n}")
    print(f"     (or review feature_curve.png and choose manually)")

    print(f"\n[4/4] Saving outputs...")

    curve_csv_path = os.path.join(config.REPORT_DIR, "feature_curve.csv")
    curve_df.to_csv(curve_csv_path, index=False, encoding="utf-8-sig")
    print(f"   [Output] feature_curve.csv -> {curve_csv_path}")

    plot_feature_curve(
        curve_df,
        out_path=os.path.join(config.REPORT_DIR, "feature_curve.png")
    )
    plot_curve_detail(
        curve_df,
        out_path=os.path.join(config.REPORT_DIR, "feature_curve_marginal.png")
    )

    report_path = os.path.join(config.REPORT_DIR, "feature_curve_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("Feature Curve Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Date:              {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Positive features: {n_positive}\n")
        f.write(f"N values tested:   {n_list}\n")
        f.write(f"Time per N:        {config.CURVE_TIME_LIMIT}s\n\n")
        f.write("Results:\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'N':>6}  {'R2':>8}  {'RMSE':>8}  {'MAPE':>7}\n")
        for _, row in curve_df.iterrows():
            f.write(f"{int(row['n_features']):>6}  "
                    f"{row['R2']:>8.4f}  "
                    f"{row['RMSE']:>8.2f}  "
                    f"{row['MAPE']:>6.2f}%\n")
        f.write("\n")
        f.write(f"Best N:            {best_n} (R2={best_r2:.4f})\n")
        f.write(f"Suggested elbow:   {elbow_n}\n\n")
        f.write("Next step:\n")
        f.write("-" * 60 + "\n")
        f.write(f"  1. Review outputs/reports/feature_curve.png\n")
        f.write(f"  2. Set config.FINAL_N_FEATURES = <your chosen N>\n")
        f.write(f"     (auto-suggested: {elbow_n})\n")
        f.write(f"  3. Run 12_train_final.py\n")
    print(f"   [Output] feature_curve_report.txt -> {report_path}")

    total_time = (datetime.now() - start_time).total_seconds()
    print(f"\n{'=' * 65}")
    print(f"Feature curve complete in {total_time / 60:.1f} min")
    print(f"  Best N={best_n}  R2={best_r2:.4f}")
    print(f"\n  Next:")
    print(f"    1. Open outputs/reports/feature_curve.png")
    print(f"    2. Set config.FINAL_N_FEATURES = {elbow_n}  (auto-suggested)")
    print(f"    3. Run 12_train_final.py")
    print(f"{'=' * 65}\n")

    return curve_df


if __name__ == "__main__":
    config.validate()
    run()