#!/usr/bin/env python3
"""
Train multiple models in parallel across GPUs and compare regression performance.

Models:
  - Identity (Y=X)          : no training, sanity-check lower bound
  - Baseline (TransEnc+CNNDec): replicates DeepMerip/DeepTrans architecture
  - CNN (ResCNN+Dilated)     : our pure-CNN solution
  - GAN (ResCNN+Discriminator): adversarial training variant

GPU assignment: N models distributed round-robin across M GPUs.
  Models on the same GPU train sequentially; different GPUs run in parallel.

Usage:
    python scripts/compare_models.py --npz scripts/0_5_ds.npz --epochs 50
"""

import argparse
import os
import sys
import time
import json
import threading
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "rnaseq_ss"))

from model import DeepMeripBaseline
from CNN import DeepMeripCNN
from GAN import DeepMeripGAN

# ===========================================================================
# Constants
# ===========================================================================
TEST_CHROMS = {"chr8", "chr9"}
VAL_CHROMS = {"chr7", "chrX"}


# ===========================================================================
# Regression Metrics
# ===========================================================================
def regression_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> dict:
    mask = np.isfinite(y_pred) & np.isfinite(y_true)
    y_pred, y_true = y_pred[mask], y_true[mask]

    residuals = y_pred - y_true
    mse  = float(np.mean(residuals ** 2))
    rmse = float(np.sqrt(mse))
    mae  = float(np.mean(np.abs(residuals)))

    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    try:    pr, _ = pearsonr(y_pred, y_true)
    except: pr = float("nan")
    try:    sr, _ = spearmanr(y_pred, y_true)
    except: sr = float("nan")

    return {"mse": mse, "rmse": rmse, "mae": mae,
            "r2": r2, "pearson_r": float(pr), "spearman_r": float(sr)}


def normalized_gain(model_r2, identity_r2):
    denom = 1.0 - identity_r2
    return (model_r2 - identity_r2) / denom if denom > 0 else float("nan")


# ===========================================================================
# Dataset
# ===========================================================================
class NpzDataset(Dataset):
    def __init__(self, X, Y, depths, sample_ids_enc, indices):
        self.X, self.Y = X, Y
        self.depths, self.sample_ids_enc = depths, sample_ids_enc
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        x = self.X[i].transpose(1, 0).copy()
        y = self.Y[i].squeeze(0).copy()
        return (torch.from_numpy(x), torch.from_numpy(y),
                torch.tensor(self.depths[i], dtype=torch.float32),
                torch.tensor(self.sample_ids_enc[i], dtype=torch.long))


def load_splits(npz_path, clip_quantile=0.999):
    data = np.load(npz_path, allow_pickle=False)
    X, Y = data["X"].copy(), data["Y"].copy()
    chroms, depths = data["chrom"], data["depth"]
    sample_ids = data["sample_id"]

    unique_samples = np.unique(sample_ids)
    sid_map = {s: i for i, s in enumerate(unique_samples)}
    sid_enc = np.array([sid_map[s] for s in sample_ids], dtype=np.int64)

    all_idx = np.arange(len(chroms))
    test_idx  = all_idx[np.isin(chroms, list(TEST_CHROMS))]
    val_idx   = all_idx[np.isin(chroms, list(VAL_CHROMS))]
    train_idx = all_idx[~np.isin(chroms, list(TEST_CHROMS | VAL_CHROMS))]

    # Train-only normalization
    train_ref = np.concatenate([X[train_idx].ravel(), Y[train_idx].ravel()])
    upper = float(np.quantile(train_ref, clip_quantile))
    del train_ref
    X, Y = np.clip(X, 0, upper), np.clip(Y, 0, upper)
    print(f"  Clip upper (train-only, q={clip_quantile}): {upper:.4f}")

    make = lambda idx: NpzDataset(X, Y, depths, sid_enc, idx)
    return make(train_idx), make(val_idx), make(test_idx), unique_samples


class IdentityModel(nn.Module):
    def forward(self, x):
        return x.squeeze(-1)


# ===========================================================================
# Evaluation
# ===========================================================================
@torch.no_grad()
def evaluate_full(model, loader, device):
    model.eval()
    preds, targets, depths, sids = [], [], [], []
    for X_b, Y_b, d_b, s_b in loader:
        X_b = X_b.to(device)
        pred = model(X_b)
        preds.append(pred.cpu().numpy())
        targets.append(Y_b.numpy())
        depths.append(d_b.numpy())
        sids.append(s_b.numpy())
    return (np.concatenate(preds), np.concatenate(targets),
            np.concatenate(depths), np.concatenate(sids))


def evaluate_with_metrics(model, loader, device, label="",
                          sample_names=None, identity_r2=None):
    preds, targets, depths, sids = evaluate_full(model, loader, device)

    overall = regression_metrics(preds.flatten(), targets.flatten())
    gain_str = ""
    if identity_r2 is not None:
        overall["gain"] = normalized_gain(overall["r2"], identity_r2)
        gain_str = f"  Gain={overall['gain']:.4f}"
    print(f"\n  [{label}] Overall:  MSE={overall['mse']:.6f}  RMSE={overall['rmse']:.4f}  "
          f"MAE={overall['mae']:.4f}  R²={overall['r2']:.4f}  "
          f"Pearson={overall['pearson_r']:.4f}{gain_str}")

    per_depth = {}
    uds = sorted(np.unique(depths))
    if len(uds) > 1:
        for d in uds:
            m = regression_metrics(preds[depths == d].flatten(), targets[depths == d].flatten())
            per_depth[f"{d:.2f}"] = m
            print(f"  [{label}] depth={d:.2f}: R²={m['r2']:.4f}  Pearson={m['pearson_r']:.4f}")

    per_sample = {}
    if sample_names is not None:
        for sid_int, sname in enumerate(sample_names):
            mask = sids == sid_int
            if mask.sum() == 0: continue
            m = regression_metrics(preds[mask].flatten(), targets[mask].flatten())
            if identity_r2 is not None:
                m["gain"] = normalized_gain(m["r2"], identity_r2)
            per_sample[sname] = m
            print(f"  [{label}] {sname}: R²={m['r2']:.4f}  Pearson={m['pearson_r']:.4f}")

    return overall, per_depth, per_sample


@torch.no_grad()
def evaluate_quick(model, loader, device, criterion):
    model.eval()
    total_loss, n = 0.0, 0
    all_p, all_t = [], []
    for X_b, Y_b, _, _ in loader:
        X_b, Y_b = X_b.to(device), Y_b.to(device)
        pred = model(X_b)
        loss = criterion(pred, Y_b)
        bs = X_b.size(0)
        total_loss += loss.item() * bs; n += bs
        all_p.append(pred.cpu().numpy().flatten())
        all_t.append(Y_b.cpu().numpy().flatten())
    all_p, all_t = np.concatenate(all_p), np.concatenate(all_t)
    mask = np.isfinite(all_p) & np.isfinite(all_t)
    all_p, all_t = all_p[mask], all_t[mask]
    mse = total_loss / n
    mae = float(np.mean(np.abs(all_p - all_t)))
    ss_res = np.sum((all_p - all_t)**2)
    ss_tot = np.sum((all_t - all_t.mean())**2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    try:    pr, _ = pearsonr(all_p, all_t)
    except: pr = float("nan")
    return mse, r2, pr, mae


# ===========================================================================
# Training: standard supervised
# ===========================================================================
def train_standard(model, model_name, train_loader, val_loader, device, args,
                   log_every=500):
    tag = f"[{model_name}/{device}]"
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    total_batches = len(train_loader)
    print(f"\n{'='*60}\n{tag} Training  params={n_params:,}  "
          f"batches/epoch={total_batches:,}\n{'='*60}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = {"train_mse": [], "val_mse": [], "val_r2": [],
               "val_pearson": [], "val_mae": [], "epoch_time": []}
    best_val, save_path = float("inf"), os.path.join(args.save_dir, f"best_{model_name}.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, n, t0, t_log = 0.0, 0, time.time(), time.time()
        for bi, (X_b, Y_b, _, _) in enumerate(train_loader, 1):
            X_b = X_b.to(device, non_blocking=True)
            Y_b = Y_b.to(device, non_blocking=True)
            loss = criterion(model(X_b), Y_b)
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            bs = X_b.size(0); loss_sum += loss.item() * bs; n += bs
            if bi % log_every == 0:
                dt = time.time() - t_log; spd = log_every / dt
                print(f"  {tag} Ep {epoch:03d}  batch {bi:,}/{total_batches:,}  "
                      f"MSE={loss_sum/n:.6f}  {spd:.0f} b/s  ETA {(total_batches-bi)/spd:.0f}s")
                t_log = time.time()

        scheduler.step()
        train_mse = loss_sum / n
        val_mse, val_r2, val_pr, val_mae = evaluate_quick(model, val_loader, device, criterion)
        elapsed = time.time() - t0
        history["train_mse"].append(train_mse); history["val_mse"].append(val_mse)
        history["val_r2"].append(val_r2); history["val_pearson"].append(val_pr)
        history["val_mae"].append(val_mae); history["epoch_time"].append(elapsed)

        saved = ""
        if val_mse < best_val:
            best_val = val_mse
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_mse": val_mse, "val_r2": val_r2}, save_path)
            saved = "  *saved*"
        print(f"{tag} Epoch {epoch:03d}/{args.epochs}  Train={train_mse:.6f}  "
              f"Val={val_mse:.6f}  R²={val_r2:.4f}  Pearson={val_pr:.4f}  "
              f"MAE={val_mae:.4f}  LR={scheduler.get_last_lr()[0]:.2e}  "
              f"{elapsed:.0f}s{saved}")

    print(f"{tag} Done. Best val MSE={best_val:.6f}")
    return history, save_path


# ===========================================================================
# Training: GAN (adversarial)
# ===========================================================================
def train_gan(model, model_name, train_loader, val_loader, device, args,
              log_every=500, adv_weight=0.01, n_critic=1):
    """Alternating G/D training. G loss = MSE + adv_weight * adversarial."""
    tag = f"[{model_name}/{device}]"
    model = model.to(device)
    n_params_g = sum(p.numel() for p in model.generator.parameters())
    n_params_d = sum(p.numel() for p in model.discriminator.parameters())
    total_batches = len(train_loader)
    print(f"\n{'='*60}\n{tag} GAN Training  G={n_params_g:,}  D={n_params_d:,}  "
          f"batches/epoch={total_batches:,}\n{'='*60}")

    mse_crit = nn.MSELoss()
    bce_crit = nn.BCEWithLogitsLoss()

    opt_g = torch.optim.AdamW(model.generator.parameters(), lr=args.lr, weight_decay=1e-4)
    opt_d = torch.optim.AdamW(model.discriminator.parameters(), lr=args.lr * 0.5, weight_decay=1e-4)
    sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=args.epochs)
    sched_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=args.epochs)

    history = {"train_mse": [], "val_mse": [], "val_r2": [],
               "val_pearson": [], "val_mae": [], "epoch_time": []}
    best_val, save_path = float("inf"), os.path.join(args.save_dir, f"best_{model_name}.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        g_loss_sum, n, t0, t_log = 0.0, 0, time.time(), time.time()

        for bi, (X_b, Y_b, _, _) in enumerate(train_loader, 1):
            X_b = X_b.to(device, non_blocking=True)
            Y_b = Y_b.to(device, non_blocking=True)
            bs = X_b.size(0)

            # ── Discriminator step ──
            reg_fake, _ = model.generate(X_b)
            real_input = Y_b.unsqueeze(-1)       # [B, L, 1]
            fake_input = reg_fake.detach().unsqueeze(-1)

            d_real = model.discriminate(real_input)
            d_fake = model.discriminate(fake_input)
            d_loss = (bce_crit(d_real, torch.ones_like(d_real)) +
                      bce_crit(d_fake, torch.zeros_like(d_fake))) * 0.5

            opt_d.zero_grad(); d_loss.backward()
            nn.utils.clip_grad_norm_(model.discriminator.parameters(), 1.0)
            opt_d.step()

            # ── Generator step ──
            reg_fake, _ = model.generate(X_b)
            d_fake_for_g = model.discriminate(reg_fake.unsqueeze(-1))
            g_adv = bce_crit(d_fake_for_g, torch.ones_like(d_fake_for_g))
            g_mse = mse_crit(reg_fake, Y_b)
            g_loss = g_mse + adv_weight * g_adv

            opt_g.zero_grad(); g_loss.backward()
            nn.utils.clip_grad_norm_(model.generator.parameters(), 1.0)
            opt_g.step()

            g_loss_sum += g_mse.item() * bs; n += bs

            if bi % log_every == 0:
                dt = time.time() - t_log; spd = log_every / dt
                print(f"  {tag} Ep {epoch:03d}  batch {bi:,}/{total_batches:,}  "
                      f"G_mse={g_loss_sum/n:.6f}  D={d_loss.item():.4f}  "
                      f"{spd:.0f} b/s  ETA {(total_batches-bi)/spd:.0f}s")
                t_log = time.time()

        sched_g.step(); sched_d.step()
        train_mse = g_loss_sum / n

        # Eval uses forward() which calls generator only
        val_mse, val_r2, val_pr, val_mae = evaluate_quick(model, val_loader, device, mse_crit)
        elapsed = time.time() - t0
        history["train_mse"].append(train_mse); history["val_mse"].append(val_mse)
        history["val_r2"].append(val_r2); history["val_pearson"].append(val_pr)
        history["val_mae"].append(val_mae); history["epoch_time"].append(elapsed)

        saved = ""
        if val_mse < best_val:
            best_val = val_mse
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_mse": val_mse, "val_r2": val_r2}, save_path)
            saved = "  *saved*"
        print(f"{tag} Epoch {epoch:03d}/{args.epochs}  G_mse={train_mse:.6f}  "
              f"Val={val_mse:.6f}  R²={val_r2:.4f}  Pearson={val_pr:.4f}  "
              f"MAE={val_mae:.4f}  {elapsed:.0f}s{saved}")

    print(f"{tag} Done. Best val MSE={best_val:.6f}")
    return history, save_path


# ===========================================================================
# Model registry
# ===========================================================================
def build_model_registry():
    """Returns list of model specs. Add new models here."""
    return [
        {
            "name": "Baseline",
            "class": DeepMeripBaseline,
            "kwargs": dict(feature_dim=1, d_model=64, nhead=4,
                           num_encoder_layers=3, dim_feedforward=256, dropout=0.1),
            "trainer": "standard",
        },
        {
            "name": "CNN",
            "class": DeepMeripCNN,
            "kwargs": dict(feature_dim=1, d_model=128, cnn_channels=128,
                           num_res_blocks=6, dropout=0.1),
            "trainer": "standard",
        },
        {
            "name": "GAN",
            "class": DeepMeripGAN,
            "kwargs": dict(feature_dim=1, d_model=128, cnn_channels=128,
                           num_res_blocks=6, disc_channels=64, disc_layers=4, dropout=0.1),
            "trainer": "gan",
        },
    ]


TRAINERS = {
    "standard": train_standard,
    "gan":      train_gan,
}


# ===========================================================================
# Plotting
# ===========================================================================
def plot_comparison(histories, test_metrics, save_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    trained_names = list(histories.keys())
    all_names = ["Identity"] + trained_names
    cmap = plt.cm.tab10
    model_colors = {name: cmap(i) for i, name in enumerate(all_names)}
    model_colors["Identity"] = "#999999"

    epochs = range(1, len(next(iter(histories.values()))["train_mse"]) + 1)

    # ── Figure 1: Training curves (2x2) — all models overlaid ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, key, ylabel, title in [
        (axes[0, 0], "train_mse", "MSE", "Training MSE"),
        (axes[0, 1], "val_mse",   "MSE", "Validation MSE"),
        (axes[1, 0], "val_r2",    "$R^2$", "Validation $R^2$"),
        (axes[1, 1], "val_mae",   "MAE", "Validation MAE"),
    ]:
        for name in trained_names:
            ax.plot(epochs, histories[name][key], label=name,
                    color=model_colors[name], linewidth=2)
        ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel); ax.set_title(title)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "training_curves.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Figure 1b: Per-model detail curves ──
    for name in trained_names:
        hist = histories[name]
        color = model_colors[name]
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"{name} — Training Dynamics", fontsize=14, y=1.02)

        ax = axes[0]
        ax.plot(epochs, hist["train_mse"], label="Train", color=color, linewidth=2)
        ax.plot(epochs, hist["val_mse"], label="Val", color=color,
                linestyle="--", alpha=0.8, linewidth=2)
        ax.set_xlabel("Epoch"); ax.set_ylabel("MSE"); ax.set_title("MSE (Train vs Val)")
        ax.legend(); ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.plot(epochs, hist["val_r2"], color=color, linewidth=2)
        best_ep = int(np.argmax(hist["val_r2"])) + 1
        best_r2 = max(hist["val_r2"])
        ax.axhline(y=best_r2, color="gray", linestyle=":", alpha=0.5)
        ax.annotate(f"best={best_r2:.4f} (ep {best_ep})",
                    xy=(best_ep, best_r2), fontsize=9,
                    xytext=(best_ep + len(list(epochs)) * 0.05, best_r2 - 0.01),
                    arrowprops=dict(arrowstyle="->", color="gray"))
        ax.set_xlabel("Epoch"); ax.set_ylabel("$R^2$"); ax.set_title("Validation $R^2$")
        ax.grid(True, alpha=0.3)

        ax = axes[2]
        l1, = ax.plot(epochs, hist["val_pearson"], color=color, linewidth=2, label="Pearson r")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Pearson r", color=color)
        ax2 = ax.twinx()
        l2, = ax2.plot(epochs, hist["val_mae"], color="gray", linestyle="--",
                        linewidth=2, label="MAE")
        ax2.set_ylabel("MAE", color="gray")
        ax.set_title("Val Pearson r & MAE")
        ax.legend(handles=[l1, l2], fontsize=9, loc="center right"); ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, f"detail_{name.lower()}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ── Figure 2: Test bar chart ──
    fig, axes = plt.subplots(1, 5, figsize=(24, 5))
    bar_names = all_names
    colors = [model_colors[n] for n in bar_names]

    for ax, mk, ylabel, title in [
        (axes[0], "mse", "MSE", "Test MSE"),
        (axes[1], "mae", "MAE", "Test MAE"),
        (axes[2], "r2",  "$R^2$", "Test $R^2$"),
        (axes[3], "pearson_r", "Pearson r", "Test Pearson r"),
    ]:
        vals = [test_metrics[n]["overall"][mk] for n in bar_names]
        bars = ax.bar(bar_names, vals, color=colors, width=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f"{v:.4f}", ha="center", va="bottom", fontsize=9)
        ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(True, alpha=0.3, axis="y")

    ax = axes[4]
    gain_names = trained_names
    gain_vals = [test_metrics[n]["overall"].get("gain", 0) for n in gain_names]
    gain_colors = [model_colors[n] for n in gain_names]
    bars = ax.bar(gain_names, gain_vals, color=gain_colors, width=0.5)
    for bar, v in zip(bars, gain_vals):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Normalized Gain"); ax.set_title("Gain over Identity")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "test_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Figure 3: Per-sample comparison ──
    sample_keys = sorted(test_metrics[trained_names[0]].get("per_sample", {}).keys())
    if sample_keys:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        x_pos = np.arange(len(sample_keys))
        w = 0.8 / len(all_names)
        for ax, mk, ylabel, title in [
            (axes[0], "mse",  "MSE", "Test MSE by Sample"),
            (axes[1], "r2",   "$R^2$", "Test $R^2$ by Sample"),
            (axes[2], "gain", "Gain", "Gain over Identity by Sample"),
        ]:
            for i, name in enumerate(all_names):
                ps = test_metrics[name].get("per_sample", {})
                vals = [ps.get(sk, {}).get(mk, 0) for sk in sample_keys]
                ax.bar(x_pos + i * w, vals, w, label=name, color=model_colors[name])
            ax.set_xticks(x_pos + w * len(all_names) / 2)
            ax.set_xticklabels(sample_keys, rotation=30, ha="right", fontsize=8)
            ax.set_ylabel(ylabel); ax.set_title(title)
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, "per_sample_comparison.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"\nPlots saved to {save_dir}/")


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="Compare models")
    parser.add_argument("--npz", default="scripts/0_5_ds.npz")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-dir", default="results/comparison")
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # ── GPU setup ──
    n_gpus = torch.cuda.device_count()
    if n_gpus > 0:
        for i in range(n_gpus):
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {name} ({mem:.1f} GB)")
    else:
        print("  No GPU, using CPU")

    # ── Load data ──
    print("\nLoading dataset...")
    train_ds, val_ds, test_ds, sample_names = load_splits(args.npz)
    print(f"  Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}")
    print(f"  Samples: {list(sample_names)}")

    # ── Build model registry ──
    registry = build_model_registry()
    n_models = len(registry)

    # ── Assign GPUs: round-robin across available GPUs ──
    for i, spec in enumerate(registry):
        if n_gpus > 0:
            spec["device"] = torch.device(f"cuda:{i % n_gpus}")
        else:
            spec["device"] = torch.device("cpu")

    print(f"\n  {n_models} models, all training in parallel (one thread each):")
    for spec in registry:
        print(f"    {spec['device']}: {spec['name']}")

    # ── Each model gets its own DataLoaders (independent iterators) ──
    workers_per = max(1, args.num_workers // max(n_models, 1))
    model_loaders = {}
    for spec in registry:
        pin = spec["device"].type == "cuda"
        tl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=workers_per, pin_memory=pin, persistent_workers=True)
        vl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=workers_per, pin_memory=pin, persistent_workers=True)
        model_loaders[spec["name"]] = (tl, vl)

    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    # ── Identity baseline ──
    eval_device = torch.device("cuda:0" if n_gpus > 0 else "cpu")
    print(f"\n{'='*60}\nIdentity Baseline (Y_pred = X)\n{'='*60}")
    identity_model = IdentityModel().to(eval_device)
    id_overall, id_per_depth, id_per_sample = evaluate_with_metrics(
        identity_model, test_loader, eval_device, "Identity", sample_names)
    identity_r2 = id_overall["r2"]

    # ── Parallel training: one thread per model, all concurrent ──
    results = {}

    def _model_worker(spec):
        tl, vl = model_loaders[spec["name"]]
        model = spec["class"](**spec["kwargs"])
        trainer_fn = TRAINERS[spec["trainer"]]
        hist, ckpt = trainer_fn(model, spec["name"], tl, vl, spec["device"], args)
        results[spec["name"]] = (hist, ckpt)

    t0_all = time.time()
    print(f"\n>>> Launching {n_models} training threads...")
    threads = []
    for spec in registry:
        t = threading.Thread(target=_model_worker, args=(spec,))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

    print(f"\n>>> All models trained in {time.time() - t0_all:.0f}s")

    # ── Test evaluation ──
    print(f"\n{'='*60}\nTest Set Evaluation (best checkpoints)\n{'='*60}")
    test_metrics = {"Identity": {"overall": id_overall, "per_depth": id_per_depth,
                                 "per_sample": id_per_sample}}
    histories = {}

    for spec in registry:
        name = spec["name"]
        hist, ckpt_path = results[name]
        histories[name] = hist

        model = spec["class"](**spec["kwargs"]).to(eval_device)
        ckpt = torch.load(ckpt_path, map_location=eval_device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        overall, per_depth, per_sample = evaluate_with_metrics(
            model, test_loader, eval_device, name,
            sample_names=sample_names, identity_r2=identity_r2)
        test_metrics[name] = {"overall": overall, "per_depth": per_depth,
                              "per_sample": per_sample}

    # ── Save metrics ──
    save_data = {}
    for name in ["Identity"] + [s["name"] for s in registry]:
        entry = dict(test_metrics[name])
        if name != "Identity":
            h = histories[name]
            entry["best_val_mse"] = min(h["val_mse"])
            entry["best_val_r2"] = max(h["val_r2"])
            entry["total_time"] = sum(h["epoch_time"])
            spec = next(s for s in registry if s["name"] == name)
            entry["params"] = sum(p.numel() for p in spec["class"](**spec["kwargs"]).parameters())
        else:
            entry["params"] = 0
        save_data[name] = entry

    with open(os.path.join(args.save_dir, "metrics.json"), "w") as f:
        json.dump(save_data, f, indent=2)

    # ── Plot ──
    plot_comparison(histories, test_metrics, args.save_dir)

    # ── Summary table ──
    print(f"\n{'='*70}")
    print("SUMMARY (Test Set)")
    print(f"{'='*70}")
    hdr = f"{'Model':<20} {'MSE':>8} {'RMSE':>8} {'MAE':>8} {'R²':>8} {'Pearson':>8} {'Gain':>8}"
    print(hdr)
    print("-" * len(hdr))
    for name in ["Identity"] + [s["name"] for s in registry]:
        m = test_metrics[name]["overall"]
        g = m.get("gain", 0.0)
        print(f"{name:<20} {m['mse']:>8.5f} {m['rmse']:>8.4f} {m['mae']:>8.4f} "
              f"{m['r2']:>8.4f} {m['pearson_r']:>8.4f} {g:>8.4f}")
    print(f"{'='*70}\nDone!")


if __name__ == "__main__":
    main()
