import os
import pandas as pd

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DOM_DIR     = os.path.join(BASE_DIR, "0_Data", "1_DOM")
SPLIT_CSV   = os.path.join(BASE_DIR, "0_Data", "dataset_split_index.csv")
LABEL_CSV   = os.path.join(BASE_DIR, "0_Data", "wheat_count_labels.csv")


def tif_stem(filename: str) -> str:
    for ext in (".tiff", ".tif"):
        if filename.lower().endswith(ext):
            return filename[: len(filename) - len(ext)]
    return os.path.splitext(filename)[0]


def main():
    print("=" * 60)
    print("Data Consistency Check")
    print("=" * 60)

    tif_files = sorted([
        f for f in os.listdir(DOM_DIR)
        if f.lower().endswith(".tif") or f.lower().endswith(".tiff")
    ])
    dom_stems = set(tif_stem(f) for f in tif_files)
    print(f"\n[DOM Folder]  Found TIF files: {len(tif_files)}")

    df_split = pd.read_csv(SPLIT_CSV)
    assert "filename" in df_split.columns, \
        f"[Error] 'filename' column not found in dataset_split_index.csv, actual columns: {df_split.columns.tolist()}"
    assert "split" in df_split.columns, \
        f"[Error] 'split' column not found in dataset_split_index.csv"
    split_stems = set(df_split["filename"].astype(str).str.strip())
    n_train = (df_split["split"] == "train").sum()
    n_test  = (df_split["split"] == "test").sum()
    print(f"\n[Split File]  Total rows: {len(df_split)}  (train={n_train}, test={n_test})")

    df_label = pd.read_csv(LABEL_CSV)
    name_col = df_label.columns[0]
    assert "wheat_count" in df_label.columns, \
        f"[Error] 'wheat_count' column not found in wheat_count_labels.csv, actual columns: {df_label.columns.tolist()}"
    label_stems = set(df_label[name_col].astype(str).str.strip())
    print(f"\n[Label File]   Total rows: {len(df_label)}  (column name: '{name_col}')")

    all_match = dom_stems & split_stems & label_stems
    print(f"\n[Three-way Alignment]   Common matched samples: {len(all_match)}")

    missing_in_split = dom_stems - split_stems
    missing_in_label = dom_stems - label_stems
    extra_in_split   = split_stems - dom_stems
    extra_in_label   = label_stems - dom_stems

    def report_diff(title, s):
        if s:
            print(f"\n  {title} ({len(s)}):")
            for x in sorted(s)[:10]:
                print(f"      {x}")
            if len(s) > 10:
                print(f"      ... total {len(s)}")
        else:
            print(f"  {title}: No difference")

    print()
    report_diff("In DOM but missing from Split", missing_in_split)
    report_diff("In DOM but missing from Label", missing_in_label)
    report_diff("In Split but missing from DOM", extra_in_split)
    report_diff("In Label but missing from DOM", extra_in_label)

    print("\n" + "=" * 60)
    if (len(missing_in_split) == 0 and len(missing_in_label) == 0 and
            len(extra_in_split) == 0 and len(extra_in_label) == 0 and
            len(all_match) == 518):
        print("Check passed: all three files are fully aligned, 518 samples total, you may proceed to the next step.")
    else:
        print("Check failed: inconsistencies exist, please fix the issues reported above before continuing.")
    print("=" * 60)


if __name__ == "__main__":
    main()