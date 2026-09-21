import config    # ← 必须第一个import

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from autogluon.tabular import TabularPredictor
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


def plot_actual_vs_pred(y_true, y_pred, metrics, out_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.6, s=25,
               edgecolors="steelblue", facecolors="lightblue", linewidths=0.5)
    lo = min(y_true.min(), y_pred.min()) * 0.95
    hi = max(y_true.max(), y_pred.max()) * 1.05
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.2, label="1:1 line")
    ax.set_xlabel("Actual Wheat Count", fontsize=11)
    ax.set_ylabel("Predicted Wheat Count", fontsize=11)
    ax.set_title("Baseline Model: Actual vs Predicted", fontsize=12)
    info = (f"R²   = {metrics['R2']:.4f}\n"
            f"RMSE = {metrics['RMSE']:.2f}\n"
            f"MAE  = {metrics['MAE']:.2f}\n"
            f"MAPE = {metrics['MAPE']:.2f}%")
    ax.text(0.05, 0.95, info, transform=ax.transAxes,
            verticalalignment="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   [Plot] Saved: {out_path}")


def plot_feature_importance(importance_df, top_n, out_path):
    df_plot = importance_df.head(top_n).copy()
    colors  = ["#2ECC71" if v >= 0 else "#E74C3C"
               for v in df_plot["importance"]]
    fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.25)))
    ax.barh(range(len(df_plot)), df_plot["importance"],
            color=colors, height=0.7)
    ax.set_yticks(range(len(df_plot)))
    ax.set_yticklabels(df_plot["feature"], fontsize=7)
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_xlabel("Permutation Importance (R² drop)")
    ax.set_title(f"Top-{top_n} Feature Importance (Baseline Model)", fontsize=11)
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   [Plot] Saved: {out_path}")


def init_ray_sequential():
    """
    强制Ray以单CPU顺序模式初始化。
    必须在TabularPredictor创建之前调用。
    num_cpus=1 → ParallelLocalFoldFittingStrategy自动降级为顺序执行。
    include_dashboard=False → 消除dashboard ERROR日志。
    """
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
        print("   ✓ Ray initialized: num_cpus=1 (forced sequential mode)")
    except Exception as e:
        print(f"   ⚠ Ray init skipped: {e}")


def run():
    for d in [config.OUTPUT_DIR, config.BASELINE_MODEL_DIR, config.REPORT_DIR]:
        os.makedirs(d, exist_ok=True)

    start_time = datetime.now()
    print("\n" + "=" * 65)
    print("10_train_baseline.py — Baseline Training")
    print("=" * 65)

    # ── Step 0: 强制Ray顺序模式 ───────────────────────────────────
    # 必须在任何AutoGluon操作之前执行
    print("\n[0/5] Initializing Ray (sequential mode)...")
    init_ray_sequential()

    # ── Step 1: 读取特征文件 ──────────────────────────────────────
    print("\n[1/5] Loading features_all.csv...")
    if not os.path.exists(config.FEATURES_CSV):
        raise FileNotFoundError(
            f"Input file not found: {config.FEATURES_CSV}\n"
            f"Please place features_all.csv in: {config.BASE_DIR}"
        )
    df = pd.read_csv(config.FEATURES_CSV, encoding="utf-8-sig")
    print(f"   Loaded: {df.shape[0]} samples × {df.shape[1]} columns")

    if config.TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{config.TARGET_COL}' not found.")

    feature_cols = [c for c in df.columns if c not in config.META_COLS]
    print(f"   Feature columns: {len(feature_cols)}")
    print(f"   Target: {config.TARGET_COL}  "
          f"(min={df[config.TARGET_COL].min():.0f}, "
          f"max={df[config.TARGET_COL].max():.0f}, "
          f"mean={df[config.TARGET_COL].mean():.1f})")

    # ── Step 2: 80/20 划分，封存测试集 ───────────────────────────
    print(f"\n[2/5] Splitting data (test_size={config.TEST_SIZE}, "
          f"seed={config.RANDOM_SEED})...")
    model_cols = feature_cols + [config.TARGET_COL]
    df_model   = df[model_cols].copy()

    train_df, test_df = train_test_split(
        df_model,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED,
        shuffle=True,
    )
    train_df = train_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)

    print(f"   Train: {len(train_df)} samples")
    print(f"   Test:  {len(test_df)} samples  ← 封存，仅12/13使用")

    train_df.to_csv(config.TRAIN_DATA_CSV, index=False, encoding="utf-8-sig")
    test_df.to_csv(config.TEST_DATA_CSV,   index=False, encoding="utf-8-sig")
    print(f"   Saved: train_data.csv / test_data.csv → {config.OUTPUT_DIR}")

    # ── Step 3: AutoGluon 基线训练 ────────────────────────────────
    print(f"\n[3/5] Training baseline model...")
    print(f"   Preset:        {config.BASELINE_PRESET}")
    print(f"   Time limit:    {config.BASELINE_TIME_LIMIT}s "
          f"({config.BASELINE_TIME_LIMIT // 60}min)")
    print(f"   CV folds:      {config.NUM_BAG_FOLDS}")
    print(f"   Stack levels:  {config.BASELINE_STACK_LEVELS}")
    print(f"   Ray mode:      sequential (num_cpus=1)")
    print(f"   DyStack:       disabled")
    print(f"   ⏱ Training started at {datetime.now().strftime('%H:%M:%S')}...")

    if os.path.exists(config.BASELINE_MODEL_DIR) and \
       os.listdir(config.BASELINE_MODEL_DIR):
        shutil.rmtree(config.BASELINE_MODEL_DIR)
        os.makedirs(config.BASELINE_MODEL_DIR)
        print(f"   ⚠ Removed existing baseline_model/ and recreated.")

    predictor = TabularPredictor(
        label=config.TARGET_COL,
        problem_type="regression",
        eval_metric=config.EVAL_METRIC,
        path=config.BASELINE_MODEL_DIR,
        verbosity=2,
    ).fit(
        train_data=train_df,
        time_limit=config.BASELINE_TIME_LIMIT,
        presets=config.BASELINE_PRESET,
        num_bag_folds=config.NUM_BAG_FOLDS,
        num_bag_sets=1,
        num_stack_levels=config.BASELINE_STACK_LEVELS,
        dynamic_stacking=False,
        ag_args_fit={"num_cpus": 1},   # 与ray.init(num_cpus=1)保持一致
    )

    train_duration = (datetime.now() - start_time).total_seconds()
    print(f"   ✓ Training completed in {train_duration / 60:.1f} min")

    # ── Step 4: 评估 & 特征重要性 ─────────────────────────────────
    print(f"\n[4/5] Evaluating & computing feature importance...")

    y_test  = test_df[config.TARGET_COL].values
    X_test  = test_df.drop(columns=[config.TARGET_COL])
    y_pred  = predictor.predict(X_test).values
    metrics = calc_metrics(y_test, y_pred)

    print(f"\n   ── Baseline Performance (test set) ──────────────────")
    print(f"   R²   = {metrics['R2']:.4f}")
    print(f"   RMSE = {metrics['RMSE']:.2f}")
    print(f"   MAE  = {metrics['MAE']:.2f}")
    print(f"   MAPE = {metrics['MAPE']:.2f}%")
    print(f"   ─────────────────────────────────────────────────────")

    print(f"\n   Computing feature_importance() "
          f"(num_shuffle_sets={config.IMPORTANCE_NUM_SHUFFLE_SETS})...")
    print(f"   ⏱ This may take 2~5 minutes...")

    importance_raw = predictor.feature_importance(
        data=test_df,
        subsample_size=config.IMPORTANCE_SUBSAMPLE_SIZE,
        num_shuffle_sets=config.IMPORTANCE_NUM_SHUFFLE_SETS,
        silent=True,
    )

    importance_df = importance_raw.reset_index()
    importance_df.columns = ["feature"] + list(importance_df.columns[1:])
    importance_df = importance_df.sort_values(
        "importance", ascending=False).reset_index(drop=True)
    importance_df["rank"] = importance_df.index + 1

    n_positive = int((importance_df["importance"] > 0).sum())
    n_negative = int((importance_df["importance"] < 0).sum())
    n_zero     = int((importance_df["importance"] == 0).sum())

    print(f"\n   Feature importance summary:")
    print(f"     Positive (useful):  {n_positive}")
    print(f"     Zero (neutral):     {n_zero}")
    print(f"     Negative (harmful): {n_negative}  ← will be excluded in 12")

    print(f"\n   Top 20 features:")
    for _, row in importance_df.head(20).iterrows():
        marker = " ⚠" if row["importance"] < 0 else ""
        print(f"     {int(row['rank']):3d}. {row['feature']:45s}  "
              f"{row['importance']:+.6f}{marker}")

    if n_negative > 0:
        print(f"\n   Negative importance features ({n_negative}):")
        neg_df = importance_df[importance_df["importance"] < 0]
        for _, row in neg_df.iterrows():
            print(f"     {row['feature']:45s}  {row['importance']:+.6f}")

    importance_df.to_csv(config.FEATURE_IMPORTANCE_CSV,
                         index=False, encoding="utf-8-sig")
    print(f"\n   [Output] feature_importance.csv → {config.FEATURE_IMPORTANCE_CSV}")

    # ── Step 5: 保存所有输出 ──────────────────────────────────────
    print(f"\n[5/5] Saving outputs...")

    plot_actual_vs_pred(
        y_true=y_test, y_pred=y_pred, metrics=metrics,
        out_path=os.path.join(config.REPORT_DIR, "baseline_actual_vs_pred.png")
    )

    top_n_plot = min(40, len(importance_df))
    plot_feature_importance(
        importance_df=importance_df, top_n=top_n_plot,
        out_path=os.path.join(config.REPORT_DIR, "baseline_feature_importance.png")
    )

    try:
        leaderboard = predictor.leaderboard(test_df, silent=True)
        lb_path = os.path.join(config.REPORT_DIR, "baseline_leaderboard.csv")
        leaderboard.to_csv(lb_path, index=False, encoding="utf-8-sig")
        print(f"   [Output] baseline_leaderboard.csv → {lb_path}")
        print(f"\n   Top 5 models (leaderboard):")
        for _, row in leaderboard.head(5).iterrows():
            print(f"     {row['model']:40s}  score={row['score_test']:.4f}")
    except Exception as e:
        print(f"   ⚠ Leaderboard failed: {e}")

    report_path = os.path.join(config.REPORT_DIR, "baseline_evaluation.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("Baseline Model Evaluation Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Date:          {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Input:         {config.FEATURES_CSV}\n")
        f.write(f"Features:      {len(feature_cols)}\n")
        f.write(f"Train/Test:    {len(train_df)} / {len(test_df)}\n")
        f.write(f"Preset:        {config.BASELINE_PRESET}\n")
        f.write(f"Time limit:    {config.BASELINE_TIME_LIMIT}s\n")
        f.write(f"CV folds:      {config.NUM_BAG_FOLDS}\n")
        f.write(f"Stack levels:  {config.BASELINE_STACK_LEVELS}\n")
        f.write(f"DyStack:       disabled\n")
        f.write(f"Ray mode:      sequential (num_cpus=1)\n\n")
        f.write("Performance (test set):\n")
        f.write("-" * 60 + "\n")
        f.write(f"  R²   = {metrics['R2']:.4f}\n")
        f.write(f"  RMSE = {metrics['RMSE']:.2f}\n")
        f.write(f"  MAE  = {metrics['MAE']:.2f}\n")
        f.write(f"  MAPE = {metrics['MAPE']:.2f}%\n\n")
        f.write("Feature Importance Summary:\n")
        f.write("-" * 60 + "\n")
        f.write(f"  Total features:     {len(importance_df)}\n")
        f.write(f"  Positive (useful):  {n_positive}\n")
        f.write(f"  Zero (neutral):     {n_zero}\n")
        f.write(f"  Negative (harmful): {n_negative}\n\n")
        f.write("Next step:\n")
        f.write("-" * 60 + "\n")
        f.write("  Run 11_feature_curve.py to find optimal N features.\n")
        f.write(f"  Then set config.FINAL_N_FEATURES = N* in config.py\n")
    print(f"   [Output] baseline_evaluation.txt → {report_path}")

    total_time = (datetime.now() - start_time).total_seconds()
    print(f"\n{'=' * 65}")
    print(f"✓ Baseline training complete in {total_time / 60:.1f} min")
    print(f"  R² = {metrics['R2']:.4f}  |  RMSE = {metrics['RMSE']:.2f}")
    print(f"  Positive features: {n_positive}  |  Harmful features: {n_negative}")
    print(f"\n  Next: run 11_feature_curve.py")
    print(f"{'=' * 65}\n")

    return predictor, importance_df, metrics


if __name__ == "__main__":
    config.validate()
    run()
