# RNAseq_SS

A baseline denoising framework for low-depth enrichment-based sequencing data, inspired by DeepMerip.

## Project Goal

This baseline focuses on:
- Modeling both local patterns and long-range dependencies in coverage profiles using a **CNN + Transformer** hybrid architecture.
- Using **multi-task learning** for:
  - Regression: predict cleaner continuous enrichment signal.
  - Classification: predict whether each position is a peak.
- Performing **random downsampling inside the DataLoader** so each batch contains varying sequencing depths, enabling the model to learn recovery of high-depth signal from low-depth inputs.

## Planned File Structure

```text
RNAseq_SS/
├── configs/
│   ├── baseline_m6a.yaml
│   └── baseline_domain_transfer.yaml
├── scripts/
│   └── train_baseline.py
├── src/
│   └── rnaseq_ss/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── losses.py
│       └── model.py
├── tests/
│   └── test_model_shapes.py
├── pyproject.toml
└── README.md
```

## Data Design (Key: Downsampling)

`src/rnaseq_ss/data.py` provides `downsample_coverage` and applies per-sample random `depth_ratio` selection in `SyntheticCoverageDataset.__getitem__` (for example, 0.1/0.2/0.5/1.0), then performs Poisson thinning on high-depth signal.

- Inputs: `noisy_signal` (channel 0 is downsampled noisy coverage; the last channel explicitly stores the sample-specific depth ratio).
- Targets:
  - `clean_signal`: original high-depth signal (regression target).
  - `peak_label`: peak/non-peak binary labels derived from clean signal.

This setup directly simulates how the same site behaves under different sequencing depths, which matches the core objective of training with random downsampling to reconstruct the original signal.

## Quick Start

1. Install dependencies (recommended in a virtual environment):

```bash
pip install -e .
```

2. Run baseline training:

```bash
python scripts/train_baseline.py --config configs/baseline_m6a.yaml
```

3. Run tests:

```bash
pytest -q
```

## How to Integrate Real Data

- Replace `SyntheticCoverageDataset` with a real dataset class that:
  - Reads original high-depth IP/Input coverage.
  - Randomly samples `depth_ratio` in `__getitem__` and dynamically creates low-depth inputs.
  - Keeps high-depth signal as supervision target.
- Configure domain-specific `depth_candidates` in `configs/*.yaml`.
- Add richer evaluation metrics later: AUROC/AUPRC, peak-level F1, Pearson/Spearman.
