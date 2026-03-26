#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
dataprocessing_fast.py

Optimized version: batch chromosome reads + vectorized windowing + multiprocessing.
Supports flexible downsampling depth specification.

Key optimizations vs. original:
  1. bw.values() reads entire chromosome at once (not per-window bw.stats())
  2. numpy sliding_window_view for zero-copy window slicing
  3. Vectorized near-zero / background filtering (no Python loops)
  4. multiprocessing.Pool parallelizes across (sample, depth) pairs

Usage examples:

  # Multiple ds groups → mixed into one dataset (default behavior)
  python dataprocessing_fast.py \
      --root-dir /path/to/data \
      --ds-levels 0.5 0.7 0.9 \
      --output mixed_depth_dataset.npz

  # Single ds group only
  python dataprocessing_fast.py \
      --root-dir /path/to/data \
      --ds-levels 0.5 \
      --output ds0.5_dataset.npz

  # Custom ds directory naming pattern (default: ds_{level}_bw)
  python dataprocessing_fast.py \
      --root-dir /path/to/data \
      --ds-levels 0.3 0.5 \
      --ds-dir-pattern "ds_{level}_bw" \
      --output custom_dataset.npz

Directory structure expected:
  {root-dir}/bw/*.bw              # full-depth
  {root-dir}/ds_0.5_bw/*.bw      # low-depth (naming follows --ds-dir-pattern)
  {root-dir}/ds_0.7_bw/*.bw
  ...

Outputs NPZ with:
  X            [N, 1, window_bins] float32
  Y            [N, 1, window_bins] float32
  depth        [N] float32
  category     [N] <U16
  sample_id    [N] <U128
  chrom        [N] <U64
  start_bp     [N] int64
  end_bp       [N] int64
"""

import argparse
import os
import re
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from multiprocessing import Pool, cpu_count
from functools import partial

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pyBigWig


# ──────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Build mixed-depth training pairs from bigWig tracks (optimized).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # I/O
    parser.add_argument("--root-dir", type=str, required=True,
                        help="Root directory containing bw/ and ds_*_bw/ subdirectories")
    parser.add_argument("--output", type=str, default="mixed_depth_dataset.npz")

    # Downsampling control — core new feature
    parser.add_argument("--ds-levels", type=float, nargs="+", required=True,
                        help="Downsampling levels to use, e.g. --ds-levels 0.5 0.7 0.9 "
                             "Multiple values → mixed dataset; single value → single-depth dataset")
    parser.add_argument("--ds-dir-pattern", type=str, default="ds_{level}_bw",
                        help="Directory name pattern. {level} is replaced by each ds level. "
                             "Default: ds_{level}_bw → ds_0.5_bw, ds_0.7_bw, ...")
    parser.add_argument("--full-dir-name", type=str, default="bw",
                        help="Name of subdirectory containing full-depth bigWigs (default: bw)")

    # Windowing
    parser.add_argument("--window-bins", type=int, default=400)
    parser.add_argument("--bin-size", type=int, default=25, help="bp per bin")
    parser.add_argument("--stride-bins", type=int, default=200)

    # Processing
    parser.add_argument("--summary-stat", type=str, default="mean", choices=["mean", "max"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0,
                        help="Number of parallel workers (0 = auto = cpu_count)")

    # Filtering
    parser.add_argument("--near-zero-max-thresh", type=float, default=0.02)
    parser.add_argument("--near-zero-mean-thresh", type=float, default=0.005)
    parser.add_argument("--background-max-thresh", type=float, default=0.10)
    parser.add_argument("--background-mean-thresh", type=float, default=0.02)
    parser.add_argument("--background-keep-ratio", type=float, default=0.20)

    # Normalization
    parser.add_argument("--peak-quantile", type=float, default=0.90)
    parser.add_argument("--clip-quantile", type=float, default=0.999)
    parser.add_argument("--log1p", action="store_true", default=True)
    parser.add_argument("--no-log1p", dest="log1p", action="store_false")

    # Chromosome filter
    parser.add_argument("--chrom-regex", type=str,
                        default=r"^(chr)?([1-9]|1[0-9]|2[0-2]|X|Y)$",
                        help="Regex for canonical chromosomes")

    return parser.parse_args()


# ──────────────────────────────────────────────
# File discovery
# ──────────────────────────────────────────────

def list_bw_files(folder: str) -> List[Path]:
    p = Path(folder)
    if not p.exists():
        return []
    return sorted(x for x in p.glob("*.bw") if x.is_file())


def sample_id_from_full_bw(path: Path) -> str:
    return path.stem


def sample_id_from_low_bw(path: Path) -> Tuple[str, float]:
    m = re.match(r"^(.*)\.ds_([\d.]+)$", path.stem)
    if not m:
        raise ValueError(f"Cannot parse low-depth filename: {path.name}")
    return m.group(1), float(m.group(2))


def build_pair_table(
    root_dir: Path,
    full_dir_name: str,
    ds_levels: List[float],
    ds_dir_pattern: str,
) -> List[Tuple[str, float, Path, Path]]:
    """Build list of (sample_id, depth, low_path, full_path) tuples."""
    full_dir = root_dir / full_dir_name
    full_files = list_bw_files(str(full_dir))
    full_map: Dict[str, Path] = {sample_id_from_full_bw(p): p for p in full_files}

    if not full_map:
        raise RuntimeError(f"No full-depth .bw files found in {full_dir}")

    print(f"[INFO] Found {len(full_map)} full-depth samples in {full_dir}")

    pairs = []
    for level in ds_levels:
        dir_name = ds_dir_pattern.replace("{level}", str(level))
        low_dir = root_dir / dir_name
        low_files = list_bw_files(str(low_dir))

        if not low_files:
            print(f"[WARN] No .bw files in {low_dir}, skipping ds level {level}")
            continue

        matched = 0
        for low_bw in low_files:
            sample_id, depth = sample_id_from_low_bw(low_bw)
            if sample_id in full_map:
                pairs.append((sample_id, depth, low_bw, full_map[sample_id]))
                matched += 1
            else:
                print(f"[WARN] No full-depth match for {low_bw.name}")

        print(f"[INFO] ds_level={level}: {matched}/{len(low_files)} samples matched")

    return pairs


def get_valid_chroms(full_bw, low_bw, chrom_regex: str) -> List[Tuple[str, int]]:
    """Return list of (chrom, length) tuples passing filters."""
    rgx = re.compile(chrom_regex)
    full_chroms = full_bw.chroms()
    low_chroms = low_bw.chroms()

    valid = []
    for chrom, full_len in full_chroms.items():
        if chrom not in low_chroms:
            continue
        if low_chroms[chrom] != full_len:
            continue
        if not rgx.match(chrom):
            continue
        valid.append((chrom, full_len))
    return valid


# ──────────────────────────────────────────────
# Core vectorized processing (per chromosome)
# ──────────────────────────────────────────────

def read_chrom_binned(bw, chrom: str, chrom_len: int,
                      bin_size: int, stat: str = "mean") -> np.ndarray:
    """
    Read entire chromosome signal at once, then bin in numpy.
    Replaces thousands of per-window bw.stats() calls.
    """
    raw = np.array(bw.values(chrom, 0, chrom_len), dtype=np.float32)
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)

    usable = (chrom_len // bin_size) * bin_size
    raw = raw[:usable]
    reshaped = raw.reshape(-1, bin_size)

    if stat == "mean":
        return reshaped.mean(axis=1)
    else:
        return reshaped.max(axis=1)


def extract_windows_vectorized(
    binned_low: np.ndarray,
    binned_full: np.ndarray,
    window_bins: int,
    stride_bins: int,
    nz_max_thresh: float,
    nz_mean_thresh: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized window extraction + near-zero filtering.
    Returns (X_windows, Y_windows, start_indices_in_bins).
    """
    n_bins = len(binned_low)
    if n_bins < window_bins:
        return np.empty((0, window_bins), np.float32), \
               np.empty((0, window_bins), np.float32), \
               np.empty((0,), np.int64)

    # sliding_window_view: zero-copy strided views
    x_views = sliding_window_view(binned_low, window_bins)[::stride_bins]   # [N, W]
    y_views = sliding_window_view(binned_full, window_bins)[::stride_bins]  # [N, W]

    # Vectorized near-zero filtering
    x_max = x_views.max(axis=1)
    y_max = y_views.max(axis=1)
    x_mean = x_views.mean(axis=1)
    y_mean = y_views.mean(axis=1)

    keep = ~(
        (x_max <= nz_max_thresh) &
        (y_max <= nz_max_thresh) &
        (x_mean <= nz_mean_thresh) &
        (y_mean <= nz_mean_thresh)
    )

    # Compute start indices (in bin units)
    starts = np.arange(0, n_bins - window_bins + 1, stride_bins, dtype=np.int64)

    # Make contiguous copies only for kept windows
    return (
        np.ascontiguousarray(x_views[keep]),
        np.ascontiguousarray(y_views[keep]),
        starts[keep],
    )


# ──────────────────────────────────────────────
# Worker function for multiprocessing
# ──────────────────────────────────────────────

def process_one_pair(args_tuple):
    """
    Process a single (low_bw, full_bw) pair.
    Called by multiprocessing pool.
    Returns dict with arrays for all windows from this pair.
    """
    (sample_id, depth, low_path, full_path,
     window_bins, bin_size, stride_bins, summary_stat,
     chrom_regex, nz_max_thresh, nz_mean_thresh) = args_tuple

    t0 = time.time()

    all_x, all_y = [], []
    all_starts, all_ends = [], []
    all_chroms = []
    all_full_max, all_full_mean = [], []
    all_low_max, all_low_mean = [], []

    with pyBigWig.open(str(low_path)) as low_bw, \
         pyBigWig.open(str(full_path)) as full_bw:

        valid_chroms = get_valid_chroms(full_bw, low_bw, chrom_regex)

        for chrom, chrom_len in valid_chroms:
            window_bp = window_bins * bin_size
            if chrom_len < window_bp:
                continue

            # ── Key optimization: one read per chromosome ──
            binned_low = read_chrom_binned(low_bw, chrom, chrom_len,
                                           bin_size, summary_stat)
            binned_full = read_chrom_binned(full_bw, chrom, chrom_len,
                                            bin_size, summary_stat)

            # ── Vectorized windowing + filtering ──
            x_wins, y_wins, starts_bins = extract_windows_vectorized(
                binned_low, binned_full,
                window_bins, stride_bins,
                nz_max_thresh, nz_mean_thresh,
            )

            n = x_wins.shape[0]
            if n == 0:
                continue

            starts_bp = starts_bins * bin_size
            ends_bp = starts_bp + window_bp

            all_x.append(x_wins)
            all_y.append(y_wins)
            all_starts.append(starts_bp)
            all_ends.append(ends_bp)
            all_chroms.append(np.full(n, chrom, dtype="<U64"))
            all_full_max.append(y_wins.max(axis=1))
            all_full_mean.append(y_wins.mean(axis=1))
            all_low_max.append(x_wins.max(axis=1))
            all_low_mean.append(x_wins.mean(axis=1))

    elapsed = time.time() - t0

    if len(all_x) == 0:
        print(f"  [PAIR] {sample_id} depth={depth} → 0 windows ({elapsed:.1f}s)")
        return None

    result = {
        "x": np.concatenate(all_x, axis=0),
        "y": np.concatenate(all_y, axis=0),
        "start_bp": np.concatenate(all_starts),
        "end_bp": np.concatenate(all_ends),
        "chrom": np.concatenate(all_chroms),
        "full_max": np.concatenate(all_full_max),
        "full_mean": np.concatenate(all_full_mean),
        "low_max": np.concatenate(all_low_max),
        "low_mean": np.concatenate(all_low_mean),
        "sample_id": sample_id,
        "depth": np.float32(depth),
        "n_windows": int(np.concatenate(all_x, axis=0).shape[0]),
    }

    print(f"  [PAIR] {sample_id} depth={depth} → {result['n_windows']} windows ({elapsed:.1f}s)")
    return result


# ──────────────────────────────────────────────
# Category assignment (vectorized)
# ──────────────────────────────────────────────

def assign_categories_vectorized(
    full_max: np.ndarray,
    full_mean: np.ndarray,
    low_max: np.ndarray,
    low_mean: np.ndarray,
    peak_quantile: float,
    bg_max_thresh: float,
    bg_mean_thresh: float,
) -> np.ndarray:
    """Vectorized category assignment, returns string array."""
    n = len(full_max)
    categories = np.full(n, "weak", dtype="<U16")

    # Background
    is_bg = (
        (full_max <= bg_max_thresh) &
        (low_max <= bg_max_thresh) &
        (full_mean <= bg_mean_thresh) &
        (low_mean <= bg_mean_thresh)
    )
    categories[is_bg] = "background"

    # Peak
    peak_thresh = float(np.quantile(full_max, peak_quantile))
    is_peak = full_max >= peak_thresh
    categories[is_peak] = "peak"  # peak overrides background if both match

    print(f"[INFO] Peak threshold (q={peak_quantile:.2f}): {peak_thresh:.6f}")
    return categories


def subsample_categories_vectorized(
    categories: np.ndarray,
    background_keep_ratio: float,
    seed: int,
) -> np.ndarray:
    """Return boolean mask of kept indices."""
    rng = np.random.RandomState(seed)

    mask = np.ones(len(categories), dtype=bool)

    bg_idx = np.where(categories == "background")[0]
    peak_n = int(np.sum(categories == "peak"))
    weak_n = int(np.sum(categories == "weak"))

    if len(bg_idx) > 0 and background_keep_ratio < 1.0:
        keep_n = max(1, int(len(bg_idx) * background_keep_ratio))
        drop_idx = rng.choice(bg_idx, size=len(bg_idx) - keep_n, replace=False)
        mask[drop_idx] = False

    kept_bg = int(mask[bg_idx].sum()) if len(bg_idx) > 0 else 0

    print(f"[INFO] Categories before subsampling: peak={peak_n} weak={weak_n} background={len(bg_idx)}")
    print(f"[INFO] Categories after  subsampling: peak={peak_n} weak={weak_n} background={kept_bg} total={mask.sum()}")

    return mask


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    t_start = time.time()
    root_dir = Path(args.root_dir)

    print("=" * 60)
    print("dataprocessing_fast.py — Optimized bigWig processor")
    print("=" * 60)
    print(f"[CONFIG] root_dir       : {root_dir}")
    print(f"[CONFIG] ds_levels      : {args.ds_levels}")
    print(f"[CONFIG] ds_dir_pattern : {args.ds_dir_pattern}")
    print(f"[CONFIG] window         : {args.window_bins} bins × {args.bin_size} bp = {args.window_bins * args.bin_size} bp")
    print(f"[CONFIG] stride         : {args.stride_bins} bins = {args.stride_bins * args.bin_size} bp")
    print(f"[CONFIG] output         : {args.output}")

    if len(args.ds_levels) == 1:
        print(f"[MODE]  Single-depth mode: only ds_{args.ds_levels[0]}")
    else:
        print(f"[MODE]  Mixed-depth mode: {len(args.ds_levels)} levels → merged dataset")
    print()

    # ── Build pair table ──
    pairs = build_pair_table(root_dir, args.full_dir_name, args.ds_levels, args.ds_dir_pattern)
    if not pairs:
        raise RuntimeError("No matched (low-depth, full-depth) bigWig pairs found.")
    print(f"\n[INFO] Total matched pairs: {len(pairs)}\n")

    # ── Parallel processing ──
    n_workers = args.workers if args.workers > 0 else min(len(pairs), cpu_count())
    print(f"[INFO] Using {n_workers} worker processes\n")

    worker_args = [
        (sid, depth, lp, fp,
         args.window_bins, args.bin_size, args.stride_bins, args.summary_stat,
         args.chrom_regex, args.near_zero_max_thresh, args.near_zero_mean_thresh)
        for sid, depth, lp, fp in pairs
    ]

    if n_workers == 1:
        results = [process_one_pair(a) for a in worker_args]
    else:
        with Pool(processes=n_workers) as pool:
            results = pool.map(process_one_pair, worker_args)

    results = [r for r in results if r is not None]
    if not results:
        raise RuntimeError("No windows produced from any pair.")

    # ── Merge results from all pairs ──
    print(f"\n[INFO] Merging results from {len(results)} pairs...")

    X = np.concatenate([r["x"] for r in results], axis=0)
    Y = np.concatenate([r["y"] for r in results], axis=0)
    start_bp = np.concatenate([r["start_bp"] for r in results])
    end_bp = np.concatenate([r["end_bp"] for r in results])
    chrom = np.concatenate([r["chrom"] for r in results])
    full_max = np.concatenate([r["full_max"] for r in results])
    full_mean = np.concatenate([r["full_mean"] for r in results])
    low_max = np.concatenate([r["low_max"] for r in results])
    low_mean = np.concatenate([r["low_mean"] for r in results])
    depth = np.concatenate([
        np.full(r["n_windows"], r["depth"], dtype=np.float32) for r in results
    ])
    sample_id = np.concatenate([
        np.full(r["n_windows"], r["sample_id"], dtype="<U128") for r in results
    ])

    total_windows = X.shape[0]
    print(f"[INFO] Total windows after near-zero drop: {total_windows}")

    # ── Category assignment ──
    categories = assign_categories_vectorized(
        full_max, full_mean, low_max, low_mean,
        args.peak_quantile, args.background_max_thresh, args.background_mean_thresh,
    )

    # ── Subsampling ──
    keep_mask = subsample_categories_vectorized(
        categories, args.background_keep_ratio, args.seed,
    )

    X = X[keep_mask]
    Y = Y[keep_mask]
    depth = depth[keep_mask]
    categories = categories[keep_mask]
    sample_id = sample_id[keep_mask]
    chrom = chrom[keep_mask]
    start_bp = start_bp[keep_mask]
    end_bp = end_bp[keep_mask]

    # ── Transform ──
    if args.log1p:
        X = np.log1p(X)
        Y = np.log1p(Y)

    clip_ref = np.concatenate([X.ravel(), Y.ravel()])
    upper = float(np.quantile(clip_ref, args.clip_quantile))
    X = np.clip(X, 0.0, upper)
    Y = np.clip(Y, 0.0, upper)
    print(f"[INFO] Clip upper bound (q={args.clip_quantile}): {upper:.6f}")

    # Add channel dim: [N, W] → [N, 1, W]
    X = X[:, None, :]
    Y = Y[:, None, :]

    # ── Save ──
    np.savez_compressed(
        args.output,
        X=X.astype(np.float32),
        Y=Y.astype(np.float32),
        depth=depth,
        category=categories,
        sample_id=sample_id,
        chrom=chrom,
        start_bp=start_bp,
        end_bp=end_bp,
    )

    elapsed = time.time() - t_start

    print()
    print("=" * 60)
    print(f"[DONE] Saved to: {args.output}")
    print(f"       X shape : {X.shape}")
    print(f"       Y shape : {Y.shape}")
    print(f"       Total   : {X.shape[0]} windows")
    print(f"       Time    : {elapsed:.1f}s")
    print()

    unique_depths, depth_counts = np.unique(depth, return_counts=True)
    print("  Depth distribution:")
    for d, n in zip(unique_depths, depth_counts):
        print(f"    ds_{d}: {n} windows")

    unique_cat, cat_counts = np.unique(categories, return_counts=True)
    print("  Category distribution:")
    for c, n in zip(unique_cat, cat_counts):
        print(f"    {c}: {n}")

    unique_samples = np.unique(sample_id)
    print(f"  Samples: {len(unique_samples)} ({', '.join(unique_samples[:10])}{'...' if len(unique_samples) > 10 else ''})")
    print("=" * 60)


if __name__ == "__main__":
    main()