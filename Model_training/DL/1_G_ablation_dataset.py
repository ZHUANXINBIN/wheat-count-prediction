import os
import json
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import rasterio
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

G_CONFIGS = {
    "G0": ["B1_Blue", "B2_Green_RGB", "B3_Red_RGB", "B4_Green560",
           "B5_Red650", "B6_RedEdge", "B7_NIR"],

    "G1": ["B1_Blue", "B2_Green_RGB", "B3_Red_RGB", "B4_Green560",
           "B5_Red650", "B6_RedEdge", "B7_NIR",
           "NDRE", "CIre"],

    "G2": ["B1_Blue", "B2_Green_RGB", "B3_Red_RGB", "B4_Green560",
           "B5_Red650", "B6_RedEdge", "B7_NIR",
           "CIgreen", "GRVI"],

    "G3": ["B1_Blue", "B2_Green_RGB", "B3_Red_RGB", "B4_Green560",
           "B5_Red650", "B6_RedEdge", "B7_NIR",
           "NDVI", "GNDVI", "NDRE"],

    "G4": ["B1_Blue", "B2_Green_RGB", "B3_Red_RGB", "B4_Green560",
           "B5_Red650", "B6_RedEdge", "B7_NIR",
           "NDVI", "NDRE", "CIre"],

    "G5": ["B1_Blue", "B2_Green_RGB", "B3_Red_RGB", "B4_Green560",
           "B5_Red650", "B6_RedEdge", "B7_NIR",
           "GNDVI", "CIgreen", "GRVI"],

    "G6": ["B1_Blue", "B2_Green_RGB", "B3_Red_RGB", "B4_Green560",
           "B5_Red650", "B6_RedEdge", "B7_NIR",
           "NDVI", "GNDVI", "NDRE", "CIre", "CIgreen", "GRVI"],
}

BAND_INDEX = {
    "B1_Blue":      1,
    "B2_Green_RGB": 2,
    "B3_Red_RGB":   3,
    "B4_Green560":  4,
    "B5_Red650":    5,
    "B6_RedEdge":   6,
    "B7_NIR":       7,
}

INDEX_NAMES = {"NDVI", "GNDVI", "NDRE", "CIre", "CIgreen", "GRVI"}


PRELOAD_CACHE = {}


def preload_all_channels(stems, dom_dir, indices_dir, verbose=True):
    global PRELOAD_CACHE
    if len(PRELOAD_CACHE) > 0:
        print(f"[Preload] Cache already loaded ({len(PRELOAD_CACHE)} stems), skipping.")
        return

    all_channels = list(BAND_INDEX.keys()) + list(INDEX_NAMES)
    iter_stems = tqdm(stems, desc="[Preload] Loading all channels to RAM") if verbose else stems

    for stem in iter_stems:
        cache = {}
        for ch in all_channels:
            if ch in BAND_INDEX:
                tif_path = os.path.join(dom_dir, stem + ".tif")
                with rasterio.open(tif_path) as src:
                    arr = src.read(BAND_INDEX[ch]).astype(np.float32)
            else:
                tif_path = os.path.join(indices_dir, ch, stem + ".tif")
                with rasterio.open(tif_path) as src:
                    arr = src.read(1).astype(np.float32)
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            cache[ch] = arr
        PRELOAD_CACHE[stem] = cache

    print(f"[Preload] Done. {len(PRELOAD_CACHE)} stems cached in RAM.")


def augment_a1(image: np.ndarray) -> np.ndarray:
    if random.random() < 0.5:
        image = np.flip(image, axis=2)
    if random.random() < 0.5:
        image = np.flip(image, axis=1)
    k = random.randint(0, 3)
    if k > 0:
        image = np.rot90(image, k=k, axes=(1, 2))
    return np.ascontiguousarray(image)


class WheatDataset(Dataset):
    def __init__(
        self,
        stems: list,
        labels: dict,
        channel_list: list,
        norm_stats: dict,
        augment: bool = False,
    ):
        assert len(PRELOAD_CACHE) > 0, \
            "[Dataset Error] PRELOAD_CACHE is empty! Please call preload_all_channels() before creating a Dataset."

        self.augment = augment

        means, stds = [], []
        for ch in channel_list:
            assert ch in norm_stats, \
                f"[Dataset Error] Channel '{ch}' not found in norm_stats.json"
            means.append(norm_stats[ch]["mean"])
            stds.append(norm_stats[ch]["std"])
        means = np.array(means, dtype=np.float32).reshape(-1, 1, 1)
        stds  = np.array(stds,  dtype=np.float32).reshape(-1, 1, 1)
        stds  = np.where(stds == 0, 1.0, stds)

        self.images = []
        self.label_list = []

        for stem in stems:
            ch_arrays = [PRELOAD_CACHE[stem][ch] for ch in channel_list]
            image = np.stack(ch_arrays, axis=0)
            image = (image - means) / stds
            self.images.append(image)
            self.label_list.append(float(labels[stem]))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].copy()

        if self.augment:
            image = augment_a1(image)

        image_tensor = torch.from_numpy(image)
        label_tensor = torch.tensor(self.label_list[idx], dtype=torch.float32)
        return image_tensor, label_tensor


def load_norm_stats(norm_stats_path: str) -> dict:
    assert os.path.exists(norm_stats_path), \
        f"[Error] norm_stats.json does not exist: {norm_stats_path}"
    with open(norm_stats_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_split_and_labels(split_csv: str, label_csv: str):
    df_split = pd.read_csv(split_csv)
    df_label = pd.read_csv(label_csv)

    name_col = df_label.columns[0]
    labels   = dict(zip(
        df_label[name_col].astype(str).str.strip(),
        df_label["wheat_count"].astype(float)
    ))
    train_stems = df_split[df_split["split"] == "train"]["filename"] \
                      .astype(str).str.strip().tolist()
    test_stems  = df_split[df_split["split"] == "test"]["filename"] \
                      .astype(str).str.strip().tolist()

    return train_stems, test_stems, labels


if __name__ == "__main__":
    BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DOM_DIR     = os.path.join(BASE_DIR, "0_Data", "1_DOM")
    INDICES_DIR = os.path.join(BASE_DIR, "0_Data", "indices")
    NORM_STATS  = os.path.join(BASE_DIR, "0_Data", "norm_stats.json")
    SPLIT_CSV   = os.path.join(BASE_DIR, "0_Data", "dataset_split_index.csv")
    LABEL_CSV   = os.path.join(BASE_DIR, "0_Data", "wheat_count_labels.csv")

    print("=" * 55)
    print("dataset.py v2 self-check (memory preload version)")
    print("=" * 55)

    norm_stats                      = load_norm_stats(NORM_STATS)
    train_stems, test_stems, labels = load_split_and_labels(SPLIT_CSV, LABEL_CSV)
    all_stems = train_stems + test_stems

    print(f"Training set: {len(train_stems)}  Test set: {len(test_stems)}")

    preload_all_channels(all_stems, DOM_DIR, INDICES_DIR, verbose=True)

    for g_name, ch_list in G_CONFIGS.items():
        ds = WheatDataset(
            stems        = train_stems[:4],
            labels       = labels,
            channel_list = ch_list,
            norm_stats   = norm_stats,
            augment      = True,
        )
        img, lbl = ds[0]
        status = "OK" if img.shape == (len(ch_list), 64, 64) else "FAIL"
        print(f"  {status} {g_name}: shape={tuple(img.shape)}, "
              f"label={lbl.item():.0f}, "
              f"mean={img.mean():.3f}, std={img.std():.3f}")

    print("\nSelf-check complete.")