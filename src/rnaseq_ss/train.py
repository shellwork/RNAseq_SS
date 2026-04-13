"""
Training script for DeepMeripCNN on .npz dataset.

Dataset layout (0_5_ds.npz):
  X        : (N, 1, 400) float32  -- downsampled signal
  Y        : (N, 1, 400) float32  -- original (target) signal
  category : (N,) str             -- 'peak' or 'weak'
  chrom    : (N,) str             -- chromosome name

Chromosome split:
  test  : chr8, chr9
  val   : chr7, chrX
  train : all remaining chroms

Loss:
  regression     : MSELoss  (regression head)
  classification : BCEWithLogitsLoss  (classification head)
  total = mse + cls_weight * bce
"""

import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import time 

from CNN import DeepMeripCNN


def roc_auc_score(labels, probs):
    """Compute ROC-AUC using the trapezoidal rule (no sklearn required)."""
    labels = np.asarray(labels)
    probs  = np.asarray(probs)
    if len(np.unique(labels)) < 2:
        return float("nan")
    order = np.argsort(-probs)
    labels = labels[order]
    tp = np.cumsum(labels)
    fp = np.cumsum(1 - labels)
    tpr = tp / tp[-1]
    fpr = fp / fp[-1]
    # prepend origin
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])
    return float(np.trapz(tpr, fpr))

# ===========================================================================
# 1. Chromosome splits
# ===========================================================================
TEST_CHROMS = {"chr8", "chr9"}
VAL_CHROMS  = {"chr7", "chrX"}


# ===========================================================================
# 2. Dataset  (memory-mapped to avoid loading ~7 GB into RAM)
# ===========================================================================
class NpzDataset(Dataset):
    """
    Lazy-loading dataset backed by a memory-mapped .npz file.

    Each sample:
      X   : [400, 1]   float32  (model input: [L, feature_dim])
      Y   : [400]      float32  (regression target per position)
      lbl : scalar     float32  (1.0 = peak, 0.0 = weak)
    """

    def __init__(self, X_mmap, Y_mmap, labels: np.ndarray, indices: np.ndarray):
        self.X      = X_mmap    # (N_total, 1, 400)  mmap
        self.Y      = Y_mmap    # (N_total, 1, 400)  mmap
        self.labels = labels    # (N_total,)  float32
        self.indices = indices  # indices belonging to this split

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        i = self.indices[idx]
        # X: (1, 400) -> (400, 1)  so model sees [L, feature_dim]
        x   = self.X[i].transpose(1, 0).copy()   # (400, 1)
        y   = self.Y[i].squeeze(0).copy()         # (400,)
        lbl = self.labels[i]
        return (
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.tensor(lbl, dtype=torch.float32),
        )


def load_splits(npz_path: str):
    data = np.load(npz_path, allow_pickle=False)  # NO mmap_mode
    X      = data["X"]        # fully in RAM now
    Y      = data["Y"]
    chroms = data["chrom"]
    cats   = data["category"]
    labels = (cats == "peak").astype(np.float32)

    all_idx   = np.arange(len(chroms))
    test_idx  = all_idx[np.isin(chroms, list(TEST_CHROMS))]
    val_idx   = all_idx[np.isin(chroms, list(VAL_CHROMS))]
    train_idx = all_idx[~np.isin(chroms, list(TEST_CHROMS | VAL_CHROMS))]

    make = lambda idx: NpzDataset(X, Y, labels, idx)
    return make(train_idx), make(val_idx), make(test_idx)


# ===========================================================================
# 3. Evaluation helper
# ===========================================================================
def evaluate(model, loader, device, mse_crit, bce_crit, cls_weight: float):
    """Returns (total_loss, mse, auc)."""
    model.eval()
    total_loss = total_mse = 0.0
    n = 0
    all_probs, all_labels = [], []

    with torch.no_grad():
        for X_b, Y_b, lbl_b in loader:
            
            X_b   = X_b.to(device)    # [B, L, 1]
            Y_b   = Y_b.to(device)    # [B, L]
            lbl_b = lbl_b.to(device)  # [B]

            reg, cls_logits = model(X_b)  # [B, L], [B, L]

            loss_reg = mse_crit(reg, Y_b)
            cls_target = lbl_b.unsqueeze(1).expand_as(cls_logits)
            loss_cls   = bce_crit(cls_logits, cls_target)
            loss       = loss_reg + cls_weight * loss_cls

            bs = X_b.size(0)
            total_loss += loss.item()     * bs
            total_mse  += loss_reg.item() * bs
            n += bs

            # window-level probability: mean sigmoid over positions
            probs = torch.sigmoid(cls_logits.mean(dim=1)).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(lbl_b.cpu().numpy().tolist())

    avg_loss = total_loss / n
    avg_mse  = total_mse  / n
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float("nan")

    return avg_loss, avg_mse, auc


# ===========================================================================
# 4. Training loop
# ===========================================================================
def train(
    npz_path: str       = "scripts/0_5_ds.npz",
    # model hyper-params
    feature_dim: int    = 1,
    d_model: int        = 128,
    num_res_blocks: int = 6,
    dropout: float      = 0.1,
    # training hyper-params
    epochs: int         = 200,
    batch_size: int     = 64,
    lr: float           = 1e-3,
    weight_decay: float = 1e-4,
    cls_weight: float   = 0.5,   # weight for classification loss
    save_dir: str       = "checkpoints",
    num_workers: int    = 4,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── data ──
    print("Loading dataset splits...")
    train_ds, val_ds, test_ds = load_splits(npz_path)
    print(f"  Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    # ── model / optimizer / scheduler ──
    model = DeepMeripCNN(
        feature_dim=feature_dim,
        d_model=d_model,
        cnn_channels=d_model,
        num_res_blocks=num_res_blocks,
        dropout=dropout,
    ).to(device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    mse_crit  = nn.MSELoss()
    bce_crit  = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    os.makedirs(save_dir, exist_ok=True)
    best_val_loss = float("inf")

    # ── training ──
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        n = 0
        t0 = time.time() 

        for X_b, Y_b, lbl_b in train_loader:
            t_data = time.time()
            X_b   = X_b.to(device)    # [B, L, 1]
            Y_b   = Y_b.to(device)    # [B, L]
            lbl_b = lbl_b.to(device)  # [B]

            reg, cls_logits = model(X_b)

            loss_reg   = mse_crit(reg, Y_b)
            cls_target = lbl_b.unsqueeze(1).expand_as(cls_logits)
            loss_cls   = bce_crit(cls_logits, cls_target)
            loss       = loss_reg + cls_weight * loss_cls

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss.item() * X_b.size(0)
            n += X_b.size(0)
            torch.cuda.synchronize()
            t1 = time.time()
            print(f"Data wait: {t_data - t0:.4f}s | GPU compute: {t1 - t_data:.4f}s")

            t0 = t1
            if n > batch_size * 5:                 # ← only print first ~5 batches
                break

        scheduler.step()
        train_loss = train_loss_sum / n

        val_loss, val_mse, val_auc = evaluate(
            model, val_loader, device, mse_crit, bce_crit, cls_weight
        )

        print(
            f"Epoch [{epoch:03d}/{epochs}]  "
            f"Train: {train_loss:.5f}  "
            f"Val Loss: {val_loss:.5f}  MSE: {val_mse:.5f}  AUC: {val_auc:.4f}  "
            f"LR: {scheduler.get_last_lr()[0]:.2e}"
        )

        # save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_mse": val_mse,
                    "val_auc": val_auc,
                },
                os.path.join(save_dir, "best_model.pt"),
            )
            print(f"  Saved best model (val_loss={val_loss:.5f})")

        # periodic checkpoint
        if epoch % 10 == 0:
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "val_loss": val_loss},
                os.path.join(save_dir, f"model_epoch{epoch:03d}.pt"),
            )

    print(f"\nTraining complete. Best val loss: {best_val_loss:.5f}")

    # ── test evaluation ──
    print("\nEvaluating on test set (best checkpoint)...")
    ckpt = torch.load(os.path.join(save_dir, "best_model.pt"), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    test_loss, test_mse, test_auc = evaluate(
        model, test_loader, device, mse_crit, bce_crit, cls_weight
    )
    print(
        f"Test  Loss: {test_loss:.5f}  "
        f"Test MSE: {test_mse:.5f}  "
        f"Test AUC: {test_auc:.4f}"
    )
    return model


# ===========================================================================
# 5. Entry point
# ===========================================================================
if __name__ == "__main__":
    train(
        npz_path="scripts/0_5_ds.npz",
        feature_dim=1,
        d_model=128,
        num_res_blocks=6,
        dropout=0.1,
        epochs=50,
        batch_size=64,
        lr=1e-3,
        cls_weight=0.5,
        save_dir="checkpoints",
        num_workers=4,
    )
