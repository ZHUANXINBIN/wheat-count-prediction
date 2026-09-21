import pandas as pd
import os

RESULTS_DIR = "results"

df_folds = pd.read_csv(os.path.join(RESULTS_DIR, "final_eval_all_folds.csv"))
df_summary = pd.read_csv(os.path.join(RESULTS_DIR, "final_eval_summary.csv"))

df_summary_by_val = df_summary.sort_values(
    "mean_val_r2", ascending=False).reset_index(drop=True)
selected_model = df_summary_by_val.iloc[0]["model"]

print("=" * 70)
print("Architecture Selection Verification: Sorted by Development Pool Val R2")
print("=" * 70)
for _, row in df_summary_by_val.iterrows():
    mark = " <-- Final model under the new selection criterion" if row["model"] == selected_model else ""
    print(f"  {row['model']:<15} ValR2={row['mean_val_r2']:.4f}+-{row['std_val_r2']:.4f}  "
          f"(TestR2={row['mean_test_r2']:.4f}+-{row['std_test_r2']:.4f}){mark}")

old_selected = df_summary.sort_values("mean_test_r2", ascending=False).iloc[0]["model"]
print(f"\nOriginal method (selected by Test R2): {old_selected}")
print(f"New method (selected by Val R2)      : {selected_model}")

if old_selected == selected_model:
    print("\nBoth selection criteria yield the same result. No need to change the numbers in the paper abstract, only the methodology wording needs correction.")
else:
    print("\nThe two selection criteria yield different results! The Results/Discussion/Conclusion need to be rewritten,")
    print(f"   and the sealed test results of {selected_model} must replace the original {old_selected} results in the paper.")

final_row = df_summary[df_summary["model"] == selected_model].iloc[0]
print(f"\n[The only test set result that should be written into the paper]")
print(f"  Model: {selected_model}")
print(f"  Test R2 (5-fold mean): {final_row['mean_test_r2']:.4f} +- {final_row['std_test_r2']:.4f}")
print(f"  Test RMSE            : {final_row['mean_test_rmse']:.2f} +- {final_row['std_test_rmse']:.2f}")
print(f"  Ensemble Test R2      : {final_row['ensemble_test_r2']:.4f}")

print(f"\n[Other architectures - development pool ablation evidence only, not compared alongside sealed test scores]")
for _, row in df_summary.iterrows():
    if row["model"] != selected_model:
        print(f"  {row['model']:<15} Val R2 (development pool) = "
              f"{row['mean_val_r2']:.4f} +- {row['std_val_r2']:.4f}")

out_path = os.path.join(RESULTS_DIR, "final_eval_summary_corrected.csv")
df_summary_by_val.to_csv(out_path, index=False)
print(f"\nCorrected sorted results saved: {out_path}")

print("\n" + "=" * 70)
print("Suggested table for the paper (Markdown, verify values manually before pasting)")
print("=" * 70)
print("| Architecture | Val R2 (dev. pool, 5-fold) | Test R2 (sealed, reported) |")
print("|---|---|---|")
for _, row in df_summary_by_val.iterrows():
    test_cell = (f"{row['mean_test_r2']:.4f} +/- {row['std_test_r2']:.4f}"
                 if row["model"] == selected_model else "not reported (not selected)")
    print(f"| {row['model']} | {row['mean_val_r2']:.4f} +/- {row['std_val_r2']:.4f} | {test_cell} |")