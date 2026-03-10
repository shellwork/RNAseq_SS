from __future__ import annotations

import argparse
from dataclasses import asdict

import torch
import yaml
from torch.utils.data import DataLoader

from rnaseq_ss.config import DataConfig, ModelConfig, TrainingConfig
from rnaseq_ss.data import SyntheticCoverageDataset, collate_samples
from rnaseq_ss.losses import multitask_loss
from rnaseq_ss.model import DeepMeripBaseline


def load_config(path: str) -> tuple[DataConfig, ModelConfig, TrainingConfig]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return (
        DataConfig(**cfg.get("data", {})),
        ModelConfig(**cfg.get("model", {})),
        TrainingConfig(**cfg.get("training", {})),
    )


def run_epoch(model, loader, optimizer, train_cfg: TrainingConfig, device: torch.device, train: bool):
    model.train() if train else model.eval()
    total_loss = 0.0

    for batch in loader:
        x = batch.noisy_signal.to(device)
        y_reg = batch.clean_signal.to(device)
        y_cls = batch.peak_label.to(device)

        with torch.set_grad_enabled(train):
            pred_reg, pred_cls = model(x)
            loss = multitask_loss(
                pred_reg,
                y_reg,
                pred_cls,
                y_cls,
                regression_weight=train_cfg.regression_weight,
                classification_weight=train_cfg.classification_weight,
            )

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RNAseq_SS baseline model")
    parser.add_argument("--config", default="configs/baseline_m6a.yaml")
    args = parser.parse_args()

    data_cfg, model_cfg, train_cfg = load_config(args.config)
    device = torch.device(train_cfg.device if torch.cuda.is_available() or train_cfg.device == "cpu" else "cpu")

    train_ds = SyntheticCoverageDataset(
        size=data_cfg.train_size,
        window_size=data_cfg.window_size,
        feature_dim=data_cfg.feature_dim,
        depth_candidates=data_cfg.depth_candidates,
        seed=42,
    )
    val_ds = SyntheticCoverageDataset(
        size=data_cfg.val_size,
        window_size=data_cfg.window_size,
        feature_dim=data_cfg.feature_dim,
        depth_candidates=data_cfg.depth_candidates,
        seed=123,
    )

    train_loader = DataLoader(train_ds, batch_size=data_cfg.batch_size, shuffle=True, collate_fn=collate_samples)
    val_loader = DataLoader(val_ds, batch_size=data_cfg.batch_size, shuffle=False, collate_fn=collate_samples)

    model = DeepMeripBaseline(**asdict(model_cfg)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    for epoch in range(1, train_cfg.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, train_cfg, device, train=True)
        val_loss = run_epoch(model, val_loader, optimizer, train_cfg, device, train=False)
        print(f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")


if __name__ == "__main__":
    main()
