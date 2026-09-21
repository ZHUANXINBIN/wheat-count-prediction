import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import r2_score


def evaluate(model, loader, device):
    model.eval()
    preds_all  = []
    labels_all = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            preds  = model(images)
            preds_all.append(preds.cpu().numpy())
            labels_all.append(labels.numpy())

    preds_all  = np.concatenate(preds_all)
    labels_all = np.concatenate(labels_all)

    r2   = r2_score(labels_all, preds_all)
    rmse = float(np.sqrt(np.mean((preds_all - labels_all) ** 2)))

    return r2, rmse


def train_one_fold(
    model,
    train_loader,
    val_loader,
    device,
    max_epochs: int = 150,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    patience: int = 20,
    verbose: bool = True,
    fold_id: int = None,
):
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = lr,
        weight_decay = weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max = max_epochs,
        eta_min = lr * 1e-3,
    )
    criterion = nn.MSELoss()

    best_r2        = -np.inf
    best_rmse      = np.inf
    no_improve_cnt = 0
    fold_str       = f"Fold {fold_id}" if fold_id is not None else "Fold"

    for epoch in range(1, max_epochs + 1):

        model.train()
        train_losses = []

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            preds = model(images)
            loss  = criterion(preds, labels)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()

        val_r2, val_rmse = evaluate(model, val_loader, device)
        train_loss_avg   = float(np.mean(train_losses))

        if val_r2 > best_r2:
            best_r2        = val_r2
            best_rmse      = val_rmse
            no_improve_cnt = 0
        else:
            no_improve_cnt += 1

        if verbose:
            current_lr = scheduler.get_last_lr()[0]
            improve_mark = "*" if no_improve_cnt == 0 else " "
            print(
                f"  [{fold_str}] Epoch {epoch:3d}/{max_epochs} | "
                f"TrainLoss={train_loss_avg:8.1f} | "
                f"ValR2={val_r2:.4f} {improve_mark} | "
                f"ValRMSE={val_rmse:6.2f} | "
                f"LR={current_lr:.2e} | "
                f"NoImprove={no_improve_cnt}/{patience}"
            )

        if no_improve_cnt >= patience:
            if verbose:
                print(f"  [{fold_str}] Early stopping at epoch {epoch}. "
                      f"Best Val R2={best_r2:.4f}, RMSE={best_rmse:.2f}")
            break

    if verbose and no_improve_cnt < patience:
        print(f"  [{fold_str}] Training completed. "
              f"Best Val R2={best_r2:.4f}, RMSE={best_rmse:.2f}")

    return best_r2, best_rmse


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from torch.utils.data import DataLoader
    from dataset import WheatDataset, G_CONFIGS, load_norm_stats, load_split_and_labels
    from model import build_model

    BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DOM_DIR     = os.path.join(BASE_DIR, "0_Data", "1_DOM")
    INDICES_DIR = os.path.join(BASE_DIR, "0_Data", "indices")
    NORM_STATS  = os.path.join(BASE_DIR, "0_Data", "norm_stats.json")
    SPLIT_CSV   = os.path.join(BASE_DIR, "0_Data", "dataset_split_index.csv")
    LABEL_CSV   = os.path.join(BASE_DIR, "0_Data", "wheat_count_labels.csv")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 55)
    print("trainer.py self-check (using G3, first 60 samples, 5 epochs)")
    print(f"Device: {device}")
    print("=" * 55)

    norm_stats                      = load_norm_stats(NORM_STATS)
    train_stems, test_stems, labels = load_split_and_labels(SPLIT_CSV, LABEL_CSV)

    ch_list = G_CONFIGS["G3"]
    train_ds = WheatDataset(train_stems[:48], labels, ch_list,
                            DOM_DIR, INDICES_DIR, norm_stats, augment=True)
    val_ds   = WheatDataset(train_stems[48:60], labels, ch_list,
                            DOM_DIR, INDICES_DIR, norm_stats, augment=False)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False,
                              num_workers=0, pin_memory=True)

    model = build_model(in_channels=len(ch_list), pretrained=True)

    best_r2, best_rmse = train_one_fold(
        model        = model,
        train_loader = train_loader,
        val_loader   = val_loader,
        device       = device,
        max_epochs   = 5,
        patience     = 5,
        verbose      = True,
        fold_id      = 1,
    )

    status = "OK" if isinstance(best_r2, float) else "FAIL"
    print(f"\n{status} trainer.py self-check complete: best_r2={best_r2:.4f}, best_rmse={best_rmse:.2f}")