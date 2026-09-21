import config

import os
import shutil
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularPredictor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")


FINAL_OUT  = os.path.join(config.OUTPUT_DIR, "final")
MODEL_DIR  = os.path.join(FINAL_OUT, "0_model")
REPORT_DIR = os.path.join(FINAL_OUT, "1_final_reports")

CONTROL_SEED     = "ctrl42"
STABILITY_SEEDS  = [123, 456, 789, 2024, 999]


def init_ray_sequential():
    try:
        import ray
        if ray.is_initialized():
            ray.shutdown()
        ray.init(num_cpus=1, num_gpus=0, ignore_reinit_error=True,
                 include_dashboard=False, log_to_driver=False)
        print("   Ray initialized: num_cpus=1")
    except Exception as e:
        print(f"   Ray init skipped: {e}")


def calc_metrics(y_true, y_pred):
    y_true  = np.array(y_true, dtype=float)
    y_pred  = np.array(y_pred, dtype=float)
    mask    = y_true != 0
    mape    = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) \
              if mask.sum() > 0 else np.nan
    abs_err = np.abs(y_true - y_pred)
    return {
        "R2":           float(r2_score(y_true, y_pred)),
        "RMSE":         float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE":          float(mean_absolute_error(y_true, y_pred)),
        "MAPE":         mape,
        "Max_Error":    float(abs_err.max()),
        "Median_Error": float(np.median(abs_err)),
    }


def fmt_leaderboard(lb: pd.DataFrame) -> pd.DataFrame:
    lb = lb.copy()
    if "score_val" in lb.columns:
        lb = lb.rename(columns={"score_val": "val_score(r2)"})
    keep = ["model", "val_score(r2)", "pred_time_val", "fit_time",
            "stack_level", "can_infer", "fit_order"]
    keep = [c for c in keep if c in lb.columns]
    return lb[keep].reset_index(drop=True)


def plot_prediction_panel(y_true, y_pred, label, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(f"Prediction Analysis - {label}", fontsize=14, y=1.01)

    residuals  = y_true - y_pred
    abs_err    = np.abs(residuals)
    vmin, vmax = y_true.min(), y_true.max()

    ax = axes[0, 0]
    ax.scatter(y_true, y_pred, alpha=0.55, edgecolors="k", linewidth=0.4, s=35)
    ax.plot([vmin, vmax], [vmin, vmax], "r--", lw=1.8, label="1:1 line")
    ax.set_xlabel("Actual Wheat Count"); ax.set_ylabel("Predicted Wheat Count")
    ax.set_title("Actual vs Predicted")
    ax.text(0.05, 0.93, f"R2 = {r2_score(y_true, y_pred):.4f}",
            transform=ax.transAxes, va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6))
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.scatter(y_pred, residuals, alpha=0.55, edgecolors="k", linewidth=0.4, s=35)
    ax.axhline(0, color="r", linestyle="--", lw=1.8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Residual (Actual - Predicted)")
    ax.set_title("Residual Plot"); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.hist(abs_err, bins=22, edgecolor="black", alpha=0.72, color="#3498DB")
    ax.axvline(np.median(abs_err), color="r", linestyle="--",
               label=f"Median={np.median(abs_err):.1f}")
    ax.axvline(abs_err.mean(), color="orange", linestyle=":",
               label=f"Mean={abs_err.mean():.1f}")
    ax.set_xlabel("Absolute Error"); ax.set_ylabel("Frequency")
    ax.set_title("Error Distribution"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    idx = np.argsort(y_true)
    ax.plot(y_true[idx], label="Actual", lw=2, color="#2ECC71")
    ax.plot(y_pred[idx], label="Predicted", lw=1.8, alpha=0.8, color="#E74C3C")
    ax.fill_between(range(len(y_true)),
                    y_true[idx] - abs_err[idx], y_true[idx] + abs_err[idx],
                    alpha=0.15, color="#E74C3C", label="Error Band")
    ax.set_xlabel("Sample Index (sorted by Actual)"); ax.set_ylabel("Wheat Count")
    ax.set_title("Prediction Tracking"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance_bar(importance_df, label, out_path, top_n=25):
    df = importance_df.sort_values("importance", ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.32)))
    colors = ["#2ECC71" if v >= 0 else "#E74C3C" for v in df["importance"]]
    ax.barh(range(len(df)), df["importance"].values, color=colors, edgecolor="white")
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["feature"].values, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Importance Score (shuffle-based)")
    ax.set_title(f"Top-{top_n} Feature Importance - {label}")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.grid(True, alpha=0.25, axis="x")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_leaderboard_bar(lb_df, label, out_path):
    df = lb_df.dropna(subset=["val_score(r2)"]).copy()
    df = df.sort_values("val_score(r2)", ascending=True)
    colors = ["#F39C12" if "WeightedEnsemble" in str(df["model"].iloc[i])
              else "#2980B9" for i in range(len(df))]
    fig, ax = plt.subplots(figsize=(10, max(5, len(df) * 0.38)))
    ax.barh(range(len(df)), df["val_score(r2)"].values, color=colors, edgecolor="white")
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["model"].values, fontsize=9)
    ax.set_xlabel("Validation R2")
    ax.set_title(f"AutoGluon Sub-Model Leaderboard - {label}")
    ax.grid(True, alpha=0.25, axis="x")
    for i, v in enumerate(df["val_score(r2)"].values):
        ax.text(v + 0.001, i, f"{v:.4f}", va="center", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_stability_overview(all_six_results, ctrl_label, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Stability Test - 6-Run Comparison (1 Control + 5 Random Splits)",
                 fontsize=13)

    labels = [r["run_label"] for r in all_six_results]
    r2s    = [r["R2"]        for r in all_six_results]
    rmses  = [r["RMSE"]      for r in all_six_results]
    ctrl_r2   = next(r["R2"]   for r in all_six_results if r["run_label"] == ctrl_label)
    ctrl_rmse = next(r["RMSE"] for r in all_six_results if r["run_label"] == ctrl_label)
    stab_r2s  = [r["R2"]   for r in all_six_results if r["run_label"] != ctrl_label]
    stab_rmse = [r["RMSE"] for r in all_six_results if r["run_label"] != ctrl_label]

    bar_colors = ["#27AE60" if lb == ctrl_label else "#2980B9" for lb in labels]

    ax = axes[0, 0]
    ax.bar(range(len(labels)), r2s, color=bar_colors, alpha=0.85, edgecolor="k", width=0.6)
    ax.axhline(np.mean(r2s), color="red", linestyle="--", lw=1.5,
               label=f"6-run Mean={np.mean(r2s):.4f}")
    ax.fill_between([-0.5, len(labels) - 0.5],
                    np.mean(r2s) - np.std(r2s), np.mean(r2s) + np.std(r2s),
                    alpha=0.10, color="red", label=f"+/-1sigma={np.std(r2s):.4f}")
    for i, v in enumerate(r2s):
        ax.text(i, v + 0.001, f"{v:.4f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=15, fontsize=8)
    ax.set_ylabel("R2"); ax.set_title("R2 - All 6 Runs")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")

    ax = axes[0, 1]
    rmse_colors = ["#1A8C4E" if lb == ctrl_label else "#E74C3C" for lb in labels]
    ax.bar(range(len(labels)), rmses, color=rmse_colors, alpha=0.85, edgecolor="k", width=0.6)
    ax.axhline(np.mean(rmses), color="darkred", linestyle="--", lw=1.5,
               label=f"6-run Mean={np.mean(rmses):.2f}")
    ax.fill_between([-0.5, len(labels) - 0.5],
                    np.mean(rmses) - np.std(rmses), np.mean(rmses) + np.std(rmses),
                    alpha=0.10, color="darkred", label=f"+/-1sigma={np.std(rmses):.2f}")
    for i, v in enumerate(rmses):
        ax.text(i, v + 0.1, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=15, fontsize=8)
    ax.set_ylabel("RMSE"); ax.set_title("RMSE - All 6 Runs")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1, 0]
    cats   = ["Control\n(ctrl42)", "5-Seed\nMean", "6-Run\nMean"]
    vals   = [ctrl_r2, np.mean(stab_r2s), np.mean(r2s)]
    errs   = [0,       np.std(stab_r2s),  np.std(r2s)]
    colors = ["#27AE60", "#2980B9", "#8E44AD"]
    bars   = ax.bar(cats, vals, color=colors, alpha=0.85, edgecolor="k",
                    width=0.45, yerr=errs, capsize=6,
                    error_kw=dict(elinewidth=1.5, ecolor="black"))
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.002,
                f"{v:.4f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("R2"); ax.set_title("R2 Group Comparison (with +/-1sigma error bar)")
    ax.set_ylim(min(vals) - 0.025, max(vals) + 0.02)
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1, 1]
    vals_r = [ctrl_rmse, np.mean(stab_rmse), np.mean(rmses)]
    errs_r = [0,         np.std(stab_rmse),  np.std(rmses)]
    bars   = ax.bar(cats, vals_r, color=colors, alpha=0.85, edgecolor="k",
                    width=0.45, yerr=errs_r, capsize=6,
                    error_kw=dict(elinewidth=1.5, ecolor="black"))
    for bar, v in zip(bars, vals_r):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.1,
                f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("RMSE"); ax.set_title("RMSE Group Comparison (with +/-1sigma error bar)")
    ax.set_ylim(min(vals_r) - 3, max(vals_r) + 3)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"   [Plot] {os.path.basename(out_path)}")


def plot_all_seeds_scatter(all_preds, out_path):
    colors_stab = ["#2980B9", "#E74C3C", "#2ECC71", "#F39C12", "#9B59B6"]
    fig, ax = plt.subplots(figsize=(9, 8))
    vmin = min(d["y_true"].min() for d in all_preds)
    vmax = max(d["y_true"].max() for d in all_preds)
    ax.plot([vmin, vmax], [vmin, vmax], "k--", lw=1.5, label="1:1 line")
    ci = 0
    for d in all_preds:
        if d["group"] == "control":
            ax.scatter(d["y_true"], d["y_pred"], alpha=0.65, s=32,
                       color="#27AE60", marker="D",
                       label=f"ctrl42 (R2={d['R2']:.4f})")
        else:
            ax.scatter(d["y_true"], d["y_pred"], alpha=0.30, s=18,
                       color=colors_stab[ci],
                       label=f"Seed {d['seed']} (R2={d['R2']:.4f})")
            ci += 1
    ax.set_xlabel("Actual Wheat Count", fontsize=11)
    ax.set_ylabel("Predicted Wheat Count", fontsize=11)
    ax.set_title("All 6 Runs - Actual vs Predicted Overlay\n"
                 "(ctrl42 + 5 Random Seeds)", fontsize=11)
    ax.legend(fontsize=8.5, loc="upper left"); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"   [Plot] {os.path.basename(out_path)}")


def plot_baseline_vs_final(baseline_r2, baseline_rmse,
                           ctrl_r2, ctrl_rmse,
                           mean6_r2, std6_r2,
                           mean6_rmse, std6_rmse,
                           out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "Model Performance Comparison\n"
        "Baseline (208 feat, 10min)  vs  Final (60 feat, 60min)",
        fontsize=12)

    cats   = ["Baseline\n(208 feat, 10min)",
              "Final ctrl42\n(60 feat, same split)",
              "Final 6-Run Mean\n(60 feat, +/-1sigma)"]
    r2s    = [baseline_r2,  ctrl_r2,  mean6_r2]
    rmses  = [baseline_rmse, ctrl_rmse, mean6_rmse]
    r2_err  = [0, 0, std6_r2]
    rm_err  = [0, 0, std6_rmse]
    colors  = ["#95A5A6", "#27AE60", "#2980B9"]

    ax = axes[0]
    bars = ax.bar(cats, r2s, color=colors, alpha=0.88, edgecolor="k", width=0.45,
                  yerr=r2_err, capsize=7,
                  error_kw=dict(elinewidth=1.8, ecolor="black"))
    for bar, v in zip(bars, r2s):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.002,
                f"{v:.4f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("R2"); ax.set_title("R2 Comparison")
    ax.set_ylim(min(r2s) - 0.025, max(r2s) + 0.018)
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    bars = ax.bar(cats, rmses, color=colors, alpha=0.88, edgecolor="k", width=0.45,
                  yerr=rm_err, capsize=7,
                  error_kw=dict(elinewidth=1.8, ecolor="black"))
    for bar, v in zip(bars, rmses):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.15,
                f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("RMSE"); ax.set_title("RMSE Comparison")
    ax.set_ylim(min(rmses) - 3, max(rmses) + 3)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"   [Plot] {os.path.basename(out_path)}")


def train_one_run(run_label, train_df, test_df, features, run_dir):
    os.makedirs(run_dir, exist_ok=True)
    ag_path = os.path.join(run_dir, "autogluon_model")
    if os.path.exists(ag_path):
        shutil.rmtree(ag_path)

    cols      = features + [config.TARGET_COL]
    train_sub = train_df[cols].copy()
    test_sub  = test_df[cols].copy()

    t0 = datetime.now()
    predictor = TabularPredictor(
        label=config.TARGET_COL,
        problem_type="regression",
        eval_metric=config.EVAL_METRIC,
        path=ag_path,
        verbosity=1,
    ).fit(
        train_data=train_sub,
        time_limit=config.FINAL_TIME_LIMIT,
        presets=config.BASELINE_PRESET,
        num_bag_folds=config.NUM_BAG_FOLDS,
        num_bag_sets=1,
        num_stack_levels=config.BASELINE_STACK_LEVELS,
        dynamic_stacking=False,
        ag_args_fit={"num_cpus": 1},
        refit_full=True,
    )
    fit_time = (datetime.now() - t0).total_seconds()

    y_pred  = predictor.predict(test_sub.drop(columns=[config.TARGET_COL])).values
    y_true  = test_sub[config.TARGET_COL].values
    metrics = calc_metrics(y_true, y_pred)
    metrics["run_label"] = run_label
    metrics["fit_time"]  = round(fit_time, 1)
    metrics["n_train"]   = len(train_sub)
    metrics["n_test"]    = len(test_sub)

    try:
        lb = predictor.leaderboard(test_sub, silent=True)
        lb = fmt_leaderboard(lb)
    except Exception:
        lb = pd.DataFrame()

    try:
        fi = predictor.feature_importance(test_sub, num_shuffle_sets=3)
        fi = fi.reset_index().rename(columns={"index": "feature"})
        if "feature" not in fi.columns and fi.index.name == "feature":
            fi = fi.reset_index()
    except Exception:
        fi = pd.DataFrame(columns=["feature", "importance"])

    pd.DataFrame({
        "actual":         y_true,
        "predicted":      y_pred,
        "error":          y_true - y_pred,
        "absolute_error": np.abs(y_true - y_pred),
        "pct_error":      np.abs((y_true - y_pred) /
                          np.where(y_true == 0, 1, y_true)) * 100,
    }).to_csv(os.path.join(run_dir, "test_predictions.csv"),
              index=False, encoding="utf-8-sig")
    lb.to_csv(os.path.join(run_dir, "leaderboard.csv"),
              index=False, encoding="utf-8-sig")
    fi.to_csv(os.path.join(run_dir, "feature_importance.csv"),
              index=False, encoding="utf-8-sig")

    plot_prediction_panel(y_true, y_pred, run_label,
                          os.path.join(run_dir, "prediction_analysis.png"))
    if len(fi) > 0 and "feature" in fi.columns:
        plot_feature_importance_bar(fi, run_label,
                                    os.path.join(run_dir, "feature_importance.png"))
    if len(lb) > 0:
        plot_leaderboard_bar(lb, run_label,
                             os.path.join(run_dir, "leaderboard.png"))

    _write_run_summary(run_label, metrics, lb, fi, features, run_dir, fit_time)
    return metrics, lb, fi, y_true, y_pred


def _write_run_summary(run_label, metrics, lb, fi, features, out_dir, fit_time):
    with open(os.path.join(out_dir, "run_summary.txt"), "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write(f"Run Summary  -  {run_label}\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Training Date  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Run Label      : {run_label}\n")
        f.write(f"Fit Time       : {fit_time/60:.1f} min\n")
        f.write(f"Train Samples  : {metrics['n_train']}\n")
        f.write(f"Test Samples   : {metrics['n_test']}\n")
        f.write(f"Features Used  : {len(features)}  (Top-{config.FINAL_N_FEATURES})\n\n")

        f.write("Evaluation Metrics\n")
        f.write(f"  R2            : {metrics['R2']:.6f}\n")
        f.write(f"  RMSE          : {metrics['RMSE']:.4f}\n")
        f.write(f"  MAE           : {metrics['MAE']:.4f}\n")
        f.write(f"  MAPE          : {metrics['MAPE']:.4f}%\n")
        f.write(f"  Max Error     : {metrics['Max_Error']:.2f}\n")
        f.write(f"  Median Error  : {metrics['Median_Error']:.2f}\n\n")

        f.write("AutoGluon Sub-Model Leaderboard\n")
        if len(lb) > 0:
            f.write(f"  {'Model':<45} {'Val R2':>10}  {'Fit(s)':>8}\n")
            f.write("  " + "-" * 66 + "\n")
            for _, row in lb.iterrows():
                val_r2 = row.get("val_score(r2)", np.nan)
                ft     = row.get("fit_time", np.nan)
                mark   = " *" if "WeightedEnsemble" in str(row["model"]) else ""
                f.write(f"  {str(row['model']):<45} {val_r2:>10.4f}  {ft:>8.1f}{mark}\n")
        else:
            f.write("  (not available)\n")
        f.write("\n")

        f.write("Top-20 Feature Importance\n")
        if len(fi) > 0 and "feature" in fi.columns:
            for rank, (_, row) in enumerate(
                fi.sort_values("importance", ascending=False).head(20).iterrows(), 1
            ):
                f.write(f"  {rank:>3}. {row['feature']:<42} {row['importance']:>+.6f}\n")
        else:
            f.write("  (not available)\n")


def write_global_report(ctrl_metrics, stability_results, all_six_results,
                        best_label, best_metrics,
                        features, baseline_r2, baseline_rmse, out_dir):
    path = os.path.join(out_dir, "final_report.txt")

    r2_all   = [r["R2"]   for r in all_six_results]
    rmse_all = [r["RMSE"] for r in all_six_results]
    mae_all  = [r["MAE"]  for r in all_six_results]
    mape_all = [r["MAPE"] for r in all_six_results]
    mean6_r2   = np.mean(r2_all);   std6_r2   = np.std(r2_all)
    mean6_rmse = np.mean(rmse_all); std6_rmse = np.std(rmse_all)

    df5 = pd.DataFrame(stability_results)

    quote = chr(34)

    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 74 + "\n")
        f.write("FINAL MODEL REPORT - AutoGluon Wheat Counting\n")
        f.write(f"Top-{config.FINAL_N_FEATURES} Features  x  "
                f"1 Control + 5 Random Splits  =  6 Independent Runs\n")
        f.write("=" * 74 + "\n\n")
        f.write(f"Report Date    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Features Used  : {len(features)} (Top-{config.FINAL_N_FEATURES} positive)\n")
        f.write(f"Time per Run   : {config.FINAL_TIME_LIMIT}s "
                f"({config.FINAL_TIME_LIMIT/60:.0f} min)\n")
        f.write(f"Total Runs     : 6  (ctrl42 + seeds {STABILITY_SEEDS})\n\n")

        f.write("=" * 74 + "\n")
        f.write("Section 1 - Baseline vs Final  [Fair Comparison - Identical Split]\n")
        f.write("  Both use seed=42 split: 414 train / 104 test  (same samples)\n")
        f.write("  Difference: features (208 vs 60) and training time (10min vs 60min)\n")
        f.write("=" * 74 + "\n\n")

        d_r2   = ctrl_metrics["R2"]   - baseline_r2
        d_rmse = ctrl_metrics["RMSE"] - baseline_rmse
        feat_pct = (1 - config.FINAL_N_FEATURES / 208) * 100

        f.write(f"  {'Metric':<20} {'Baseline':>16} {'Final ctrl42':>16} {'Delta':>12}\n")
        f.write("  " + "-" * 66 + "\n")
        f.write(f"  {'Features':<20} {'208':>16} {config.FINAL_N_FEATURES:>16} "
                f"  {'-'+str(208-config.FINAL_N_FEATURES)+' (-'+f'{feat_pct:.0f}%)':>10}\n")
        f.write(f"  {'Train Time':<20} {'10 min':>16} "
                f"{config.FINAL_TIME_LIMIT//60:>15} min {'':>12}\n")
        f.write(f"  {'R2':<20} {baseline_r2:>16.4f} {ctrl_metrics['R2']:>16.4f} "
                f"  {d_r2:>+10.4f}  {'improved' if d_r2>0 else 'degraded'}\n")
        f.write(f"  {'RMSE':<20} {baseline_rmse:>16.2f} {ctrl_metrics['RMSE']:>16.2f} "
                f"  {d_rmse:>+10.2f}  {'better' if d_rmse<0 else 'worse'}\n")
        f.write(f"  {'MAE':<20} {'46.11':>16} {ctrl_metrics['MAE']:>16.2f}\n")
        f.write(f"  {'MAPE':<20} {'8.93%':>16} {ctrl_metrics['MAPE']:>15.2f}%\n\n")
        f.write(f"  Feature reduction : 208 -> {config.FINAL_N_FEATURES} "
                f"({feat_pct:.1f}% fewer)\n")
        f.write(f"  R2 change         : {d_r2:+.4f}  "
                f"({'significant gain' if d_r2 > 0.005 else 'marginal gain' if d_r2 > 0 else 'slight degradation'})\n\n")

        f.write("=" * 74 + "\n")
        f.write("Section 2 - 6-Run Summary Statistics  [Primary Reported Results]\n")
        f.write("  Includes ctrl42 (seed=42 split) + 5 random splits\n")
        f.write("  All runs: same Top-60 features, same 60min time limit\n")
        f.write("=" * 74 + "\n\n")

        f.write(f"  {'Run':<12} {'R2':>10} {'RMSE':>8} {'MAE':>8} "
                f"{'MAPE':>8} {'MaxErr':>8} {'MedErr':>8} {'Time':>7} {'Split':>12}\n")
        f.write("  " + "-" * 80 + "\n")

        for r in all_six_results:
            split_note = "seed=42 (fixed)" if r["run_label"] == CONTROL_SEED \
                         else f"seed={r.get('seed','?')} (random)"
            mark = " *" if r["run_label"] == best_label else "  "
            f.write(f"  {mark}{r['run_label']:<10} "
                    f"{r['R2']:>10.4f} {r['RMSE']:>8.2f} {r['MAE']:>8.2f} "
                    f"{r['MAPE']:>7.2f}% {r['Max_Error']:>8.1f} "
                    f"{r['Median_Error']:>8.1f} {r['fit_time']/60:>6.1f}m "
                    f"  {split_note}\n")

        f.write("  " + "-" * 80 + "\n")
        f.write(f"  {'Mean':<12} {mean6_r2:>10.4f} {mean6_rmse:>8.2f} "
                f"{np.mean(mae_all):>8.2f} {np.mean(mape_all):>7.2f}%\n")
        f.write(f"  {'Std':<12} {std6_r2:>10.4f} {std6_rmse:>8.2f} "
                f"{np.std(mae_all):>8.2f} {np.std(mape_all):>7.2f}%\n")
        f.write(f"  {'Min':<12} {min(r2_all):>10.4f} {min(rmse_all):>8.2f}\n")
        f.write(f"  {'Max':<12} {max(r2_all):>10.4f} {max(rmse_all):>8.2f}\n\n")

        f.write("=" * 74 + "\n")
        f.write("Section 3 - Stability Assessment\n")
        f.write("=" * 74 + "\n\n")
        cv_r2   = std6_r2   / mean6_r2   * 100
        cv_rmse = std6_rmse / mean6_rmse * 100
        stab    = "STABLE" if cv_r2 < 1.0 else ("ACCEPTABLE" if cv_r2 < 3.0 else "UNSTABLE")
        r2_range = max(r2_all) - min(r2_all)

        f.write(f"  Based on 6 independent runs (different train/test splits):\n\n")
        f.write(f"  R2   Mean +/- Std  : {mean6_r2:.4f} +/- {std6_r2:.4f}\n")
        f.write(f"  RMSE Mean +/- Std  : {mean6_rmse:.2f} +/- {std6_rmse:.2f}\n")
        f.write(f"  MAE  Mean +/- Std  : {np.mean(mae_all):.2f} +/- {np.std(mae_all):.2f}\n")
        f.write(f"  MAPE Mean +/- Std  : {np.mean(mape_all):.2f}% +/- {np.std(mape_all):.2f}%\n\n")
        f.write(f"  R2 CV            : {cv_r2:.2f}%  "
                f"(CV < 1% = Stable,  < 3% = Acceptable)\n")
        f.write(f"  RMSE CV          : {cv_rmse:.2f}%\n")
        f.write(f"  R2 Range         : {min(r2_all):.4f} ~ {max(r2_all):.4f}  "
                f"(Delta={r2_range:.4f})\n")
        f.write(f"  Assessment       : {stab}\n\n")
        f.write(f"  Recommended paper sentence:\n")
        f.write(f"    {quote}The model achieved R2 = {mean6_r2:.4f} +/- {std6_r2:.4f}, "
                f"RMSE = {mean6_rmse:.2f} +/- {std6_rmse:.2f},\n")
        f.write(f"     and MAPE = {np.mean(mape_all):.2f}% +/- {np.std(mape_all):.2f}% "
                f"across 6 independent train/test splits\n")
        f.write(f"     (1 fixed + 5 random), demonstrating {stab.lower()} "
                f"generalization (R2 CV = {cv_r2:.2f}%).{quote}\n\n")

        f.write("=" * 74 + "\n")
        f.write("Section 4 - Best Model  (highest R2 among all 6 runs)\n")
        f.write("=" * 74 + "\n\n")
        f.write(f"  Best Run       : {best_label}\n")
        f.write(f"  R2             : {best_metrics['R2']:.6f}\n")
        f.write(f"  RMSE           : {best_metrics['RMSE']:.4f}\n")
        f.write(f"  MAE            : {best_metrics['MAE']:.4f}\n")
        f.write(f"  MAPE           : {best_metrics['MAPE']:.4f}%\n")
        f.write(f"  Max Error      : {best_metrics['Max_Error']:.2f}\n")
        f.write(f"  Median Error   : {best_metrics['Median_Error']:.2f}\n")
        f.write(f"  Train Samples  : {best_metrics['n_train']}\n")
        f.write(f"  Test Samples   : {best_metrics['n_test']}\n")
        f.write(f"  Location       : 0_model/best_model/\n\n")

        f.write("=" * 74 + "\n")
        f.write(f"Section 5 - Top-{config.FINAL_N_FEATURES} Features Used\n")
        f.write("=" * 74 + "\n\n")
        for i, feat in enumerate(features, 1):
            f.write(f"  {i:>3}. {feat}\n")
        f.write("\n")

        f.write("=" * 74 + "\n")
        f.write("Section 6 - Output Structure\n")
        f.write("=" * 74 + "\n\n")
        f.write("  final/\n")
        f.write("  |-- 0_model/\n")
        f.write("  |   |-- best_model/        (highest R2 run, for deployment)\n")
        f.write("  |   |-- ctrl42/            (control group, same split as baseline)\n")
        for s in STABILITY_SEEDS:
            f.write(f"  |   |-- seed_{s}/\n")
        f.write("  |-- 1_final_reports/\n")
        f.write("      |-- final_report.txt         (this file)\n")
        f.write("      |-- all_runs_metrics.csv\n")
        f.write("      |-- stability_chart.png      (paper figure: 6-run stability)\n")
        f.write("      |-- all_seeds_scatter.png    (paper figure: prediction overlay)\n")
        f.write("      |-- baseline_vs_final.png    (paper core comparison figure)\n")
        f.write("      |-- per_run_leaderboard/\n")
        f.write("              combined_leaderboard.csv\n")
        f.write("              submodel_avg_performance.csv\n")
        f.write("              leaderboard_ctrl42.csv\n")
        for s in STABILITY_SEEDS:
            f.write(f"              leaderboard_seed_{s}.csv\n")

    print(f"   [Report] final_report.txt -> {path}")


def write_leaderboard_reports(all_lbs, out_dir):
    lb_dir = os.path.join(out_dir, "per_run_leaderboard")
    os.makedirs(lb_dir, exist_ok=True)

    combined_rows = []
    for label, lb in all_lbs.items():
        if len(lb) == 0:
            continue
        lb_out = lb.copy()
        lb_out.insert(0, "run", label)
        lb_out.to_csv(os.path.join(lb_dir, f"leaderboard_{label}.csv"),
                      index=False, encoding="utf-8-sig")
        combined_rows.append(lb_out)

    if combined_rows:
        combined = pd.concat(combined_rows, ignore_index=True)
        combined.to_csv(os.path.join(lb_dir, "combined_leaderboard.csv"),
                        index=False, encoding="utf-8-sig")

        pivot = combined.groupby("model")["val_score(r2)"].agg(
            ["mean", "std", "min", "max", "count"]
        ).round(4).sort_values("mean", ascending=False).reset_index()
        pivot.columns = ["model", "mean_r2", "std_r2", "min_r2", "max_r2", "n_runs"]
        pivot.to_csv(os.path.join(lb_dir, "submodel_avg_performance.csv"),
                     index=False, encoding="utf-8-sig")
        print(f"   [Report] per_run_leaderboard/ -> {lb_dir}")


def run():
    t_start = datetime.now()
    print("\n" + "=" * 68)
    print("12_train_final.py - 1 Control + 5 Random Seeds = 6 Independent Runs")
    print("=" * 68)

    print("\n[0/6] Initializing Ray...")
    init_ray_sequential()

    for d in [MODEL_DIR, REPORT_DIR]:
        os.makedirs(d, exist_ok=True)

    print("\n[1/6] Loading features...")
    if not os.path.exists(config.FEATURE_IMPORTANCE_CSV):
        raise FileNotFoundError("feature_importance.csv not found. Run 10 first.")

    imp_df = pd.read_csv(config.FEATURE_IMPORTANCE_CSV, encoding="utf-8-sig")
    positive_features = (
        imp_df[imp_df["importance"] > 0]
        .sort_values("importance", ascending=False)["feature"]
        .tolist()
    )
    N        = config.FINAL_N_FEATURES
    features = positive_features[:N]
    print(f"   Features: Top-{N}  |  Top-3: {features[:3]}")

    for p, n in [(config.FEATURES_CSV,    "features_all.csv"),
                 (config.TRAIN_DATA_CSV,  "train_data.csv"),
                 (config.TEST_DATA_CSV,   "test_data.csv")]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"{n} not found: {p}")

    full_df    = pd.read_csv(config.FEATURES_CSV,   encoding="utf-8-sig")
    ctrl_train = pd.read_csv(config.TRAIN_DATA_CSV, encoding="utf-8-sig")
    ctrl_test  = pd.read_csv(config.TEST_DATA_CSV,  encoding="utf-8-sig")

    baseline_r2, baseline_rmse = 0.9005, 56.77
    baseline_eval = os.path.join(config.REPORT_DIR, "baseline_evaluation.txt")
    if os.path.exists(baseline_eval):
        with open(baseline_eval, encoding="utf-8") as bf:
            for line in bf:
                if ("R2" in line or "R2" in line) and "=" in line:
                    try: baseline_r2   = float(line.strip().split("=")[-1])
                    except: pass
                if "RMSE" in line and "=" in line:
                    try: baseline_rmse = float(line.strip().split("=")[-1])
                    except: pass

    total_runs = 1 + len(STABILITY_SEEDS)
    print(f"\n[2/6] Training {total_runs} runs "
          f"(<={config.FINAL_TIME_LIMIT//60}min each, "
          f"~{total_runs * config.FINAL_TIME_LIMIT // 60}min total)\n")

    all_lbs          = {}
    all_preds        = []
    all_six_results  = []
    stability_results = []
    ctrl_metrics     = None

    print(f"   -- [1/{total_runs}] ctrl42  "
          f"started {datetime.now().strftime('%H:%M:%S')} --")
    ctrl_dir = os.path.join(MODEL_DIR, "ctrl42")
    try:
        metrics, lb, fi, y_true, y_pred = train_one_run(
            "ctrl42", ctrl_train, ctrl_test, features, ctrl_dir
        )
        ctrl_metrics = metrics
        all_six_results.append(metrics)
        all_lbs[CONTROL_SEED] = lb
        all_preds.append({"group": "control", "seed": "ctrl42",
                          "y_true": y_true, "y_pred": y_pred, "R2": metrics["R2"]})
        print(f"      R2={metrics['R2']:.4f}  RMSE={metrics['RMSE']:.2f}  "
              f"time={metrics['fit_time']/60:.1f}min")
    except Exception as e:
        print(f"      ctrl42 failed: {e}")
        ctrl_metrics = {"R2": np.nan, "RMSE": np.nan, "MAE": np.nan, "MAPE": np.nan,
                        "Max_Error": np.nan, "Median_Error": np.nan,
                        "fit_time": 0, "n_train": 0, "n_test": 0, "run_label": "ctrl42"}

    for i, seed in enumerate(STABILITY_SEEDS):
        print(f"   -- [{i+2}/{total_runs}] seed_{seed}  "
              f"started {datetime.now().strftime('%H:%M:%S')} --")
        seed_dir = os.path.join(MODEL_DIR, f"seed_{seed}")
        train_s, test_s = train_test_split(
            full_df, test_size=0.2, random_state=seed, shuffle=True
        )
        try:
            metrics, lb, fi, y_true, y_pred = train_one_run(
                f"seed_{seed}", train_s, test_s, features, seed_dir
            )
            metrics["seed"] = seed
            stability_results.append(metrics)
            all_six_results.append(metrics)
            all_lbs[seed] = lb
            all_preds.append({"group": "stability", "seed": seed,
                              "y_true": y_true, "y_pred": y_pred, "R2": metrics["R2"]})
            print(f"      R2={metrics['R2']:.4f}  RMSE={metrics['RMSE']:.2f}  "
                  f"time={metrics['fit_time']/60:.1f}min")
        except Exception as e:
            print(f"      seed_{seed} failed: {e}")

    if not all_six_results:
        raise RuntimeError("All runs failed.")

    print(f"\n[3/6] Selecting best model...")
    best_label   = max(all_six_results, key=lambda r: r["R2"])["run_label"]
    best_metrics = next(r for r in all_six_results if r["run_label"] == best_label)
    print(f"   All R2: " +
          "  ".join([f"{r['run_label']}={r['R2']:.4f}" for r in all_six_results]))
    print(f"   Best: {best_label}  R2={best_metrics['R2']:.4f}")

    print(f"\n[4/6] Organizing 0_model/ ...")
    best_src = os.path.join(MODEL_DIR,
                            "ctrl42" if best_label == "ctrl42"
                            else best_label)
    best_dst = os.path.join(MODEL_DIR, "best_model")
    if os.path.exists(best_dst):
        shutil.rmtree(best_dst)
    shutil.copytree(best_src, best_dst)
    print(f"   best_model/ <- {best_label}  (R2={best_metrics['R2']:.4f})")

    print(f"\n[5/6] Writing reports...")

    pd.DataFrame(all_six_results).to_csv(
        os.path.join(REPORT_DIR, "all_runs_metrics.csv"),
        index=False, encoding="utf-8-sig"
    )
    print(f"   [CSV] all_runs_metrics.csv")

    mean6_r2   = np.mean([r["R2"]   for r in all_six_results])
    std6_r2    = np.std( [r["R2"]   for r in all_six_results])
    mean6_rmse = np.mean([r["RMSE"] for r in all_six_results])
    std6_rmse  = np.std( [r["RMSE"] for r in all_six_results])

    write_global_report(
        ctrl_metrics, stability_results, all_six_results,
        best_label, best_metrics,
        features, baseline_r2, baseline_rmse, REPORT_DIR
    )
    write_leaderboard_reports(all_lbs, REPORT_DIR)
    plot_stability_overview(
        all_six_results, CONTROL_SEED,
        os.path.join(REPORT_DIR, "stability_chart.png")
    )
    plot_all_seeds_scatter(
        all_preds,
        os.path.join(REPORT_DIR, "all_seeds_scatter.png")
    )
    plot_baseline_vs_final(
        baseline_r2, baseline_rmse,
        ctrl_metrics["R2"], ctrl_metrics["RMSE"],
        mean6_r2, std6_r2, mean6_rmse, std6_rmse,
        os.path.join(REPORT_DIR, "baseline_vs_final.png")
    )

    total_min = (datetime.now() - t_start).total_seconds() / 60
    print(f"\n[6/6] Done.")
    print("\n" + "=" * 68)
    print(f"Complete in {total_min:.1f} min")
    print(f"\n  ctrl42           : R2={ctrl_metrics['R2']:.4f}  "
          f"RMSE={ctrl_metrics['RMSE']:.2f}")
    print(f"  6-Run Mean +/- Std : R2={mean6_r2:.4f} +/- {std6_r2:.4f}  "
          f"RMSE={mean6_rmse:.2f} +/- {std6_rmse:.2f}")
    print(f"  Best Model     : {best_label}  R2={best_metrics['R2']:.4f}")
    print(f"\n  Baseline (ref)   : R2={baseline_r2:.4f}  RMSE={baseline_rmse:.2f}")
    print(f"  Delta R2 ctrl42-base  : {ctrl_metrics['R2']-baseline_r2:+.4f}")
    print(f"\n  -> {REPORT_DIR}")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    config.validate()
    run()
