#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
dataprocessing.py

Build mixed-depth training pairs from bigWig tracks:
(sample, depth_level) -> full_depth

Expected directory structure under current working directory:
./bw/*.bw                 # full-depth bigWigs
./ds_0.5_bw/*.bw          # low-depth bigWigs
./ds_0.7_bw/*.bw
./ds_0.9_bw/*.bw

Expected naming convention:
full-depth:   SAMPLE.bw
low-depth:    SAMPLE.ds_0.5.bw
              SAMPLE.ds_0.7.bw
              SAMPLE.ds_0.9.bw

Outputs:
NPZ file containing:
    X            [N, 1, window_bins] float32
    Y            [N, 1, window_bins] float32
    depth        [N] float32
    category     [N] <U16
    sample_id    [N] <U128
    chrom        [N] <U64
    start_bp     [N] int64
    end_bp       [N] int64

Usage example:
python dataprocessing.py \
    --window-bins 400 \
    --bin-size 25 \
    --stride-bins 200 \
    --output mixed_depth_dataset.npz
"""

import argparse
import os
import re
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pyBigWig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root-dir",
        type=str,
        required=True,
        help="root directory containing bw/, ds_0.5_bw/, ds_0.7_bw/, ds_0.9_bw/",
    )
    parser.add_argument("--output", type=str, default="mixed_depth_dataset.npz")

    parser.add_argument("--window-bins", type=int, default=400)
    parser.add_argument("--bin-size", type=int, default=25, help="bp per bin")
    parser.add_argument("--stride-bins", type=int, default=200)

    parser.add_argument("--summary-stat", type=str, default="mean", choices=["mean", "max"])
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--near-zero-max-thresh", type=float, default=0.02)
    parser.add_argument("--near-zero-mean-thresh", type=float, default=0.005)

    parser.add_argument("--background-max-thresh", type=float, default=0.10)
    parser.add_argument("--background-mean-thresh", type=float, default=0.02)
    parser.add_argument("--background-keep-ratio", type=float, default=0.20)

    parser.add_argument("--peak-quantile", type=float, default=0.90)
    parser.add_argument("--clip-quantile", type=float, default=0.999)

    parser.add_argument("--log1p", action="store_true", default=True)
    parser.add_argument("--no-log1p", dest="log1p", action="store_false")

    parser.add_argument(
        "--chrom-regex",
        type=str,
        default=r"^(chr)?([1-9]|1[0-9]|2[0-2]|X|Y)$",
        help="only use canonical chromosomes by default",
    )

    return parser.parse_args()


def list_bw_files(folder: str) -> List[Path]:
    p = Path(folder)
    if not p.exists():
        return []
    return sorted([x for x in p.glob("*.bw") if x.is_file()])


def sample_id_from_full_bw(path: Path) -> str:
    # SAMPLE.bw -> SAMPLE
    return path.stem


def sample_id_from_low_bw(path: Path) -> Tuple[str, float]:
    # SAMPLE.ds_0.5.bw -> (SAMPLE, 0.5)
    m = re.match(r"^(.*)\.ds_(0\.\d+)$", path.stem)
    if not m:
        raise ValueError(f"Cannot parse low-depth filename: {path.name}")
    return m.group(1), float(m.group(2))


def build_pair_table(full_dir: str, low_dirs: List[str]) -> List[Tuple[str, float, Path, Path]]:
    full_files = list_bw_files(full_dir)
    full_map: Dict[str, Path] = {sample_id_from_full_bw(p): p for p in full_files}

    pairs = []
    for low_dir in low_dirs:
        for low_bw in list_bw_files(low_dir):
            sample_id, depth = sample_id_from_low_bw(low_bw)
            if sample_id in full_map:
                pairs.append((sample_id, depth, low_bw, full_map[sample_id]))
            else:
                print(f"[WARN] No full-depth match for {low_bw.name}")

    return pairs


def get_valid_chroms(full_bw: pyBigWig.pyBigWig,
                     low_bw: pyBigWig.pyBigWig,
                     chrom_regex: str) -> List[str]:
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
        valid.append(chrom)
    return valid


def read_window(bw: pyBigWig.pyBigWig,
                chrom: str,
                start_bp: int,
                end_bp: int,
                n_bins: int,
                stat: str = "mean") -> np.ndarray:
    vals = bw.stats(chrom, start_bp, end_bp, nBins=n_bins, type=stat)
    arr = np.array(vals, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def near_zero_window(x: np.ndarray,
                     y: np.ndarray,
                     max_thresh: float,
                     mean_thresh: float) -> bool:
    x_max, y_max = float(np.max(x)), float(np.max(y))
    x_mean, y_mean = float(np.mean(x)), float(np.mean(y))
    return (
        x_max <= max_thresh and
        y_max <= max_thresh and
        x_mean <= mean_thresh and
        y_mean <= mean_thresh
    )


def background_window(x: np.ndarray,
                      y: np.ndarray,
                      max_thresh: float,
                      mean_thresh: float) -> bool:
    x_max, y_max = float(np.max(x)), float(np.max(y))
    x_mean, y_mean = float(np.mean(x)), float(np.mean(y))
    return (
        x_max <= max_thresh and
        y_max <= max_thresh and
        x_mean <= mean_thresh and
        y_mean <= mean_thresh
    )


def collect_candidates(
    pairs: List[Tuple[str, float, Path, Path]],
    window_bins: int,
    bin_size: int,
    stride_bins: int,
    summary_stat: str,
    chrom_regex: str,
    near_zero_max_thresh: float,
    near_zero_mean_thresh: float,
) -> List[dict]:
    candidates = []
    window_bp = window_bins * bin_size
    stride_bp = stride_bins * bin_size

    for sample_id, depth, low_path, full_path in pairs:
        print(f"[PAIR] sample={sample_id} depth={depth} low={low_path.name} full={full_path.name}")

        with pyBigWig.open(str(low_path)) as low_bw, pyBigWig.open(str(full_path)) as full_bw:
            valid_chroms = get_valid_chroms(full_bw, low_bw, chrom_regex)

            for chrom in valid_chroms:
                chrom_len = full_bw.chroms()[chrom]
                if chrom_len < window_bp:
                    continue

                for start_bp in range(0, chrom_len - window_bp + 1, stride_bp):
                    end_bp = start_bp + window_bp

                    x = read_window(low_bw, chrom, start_bp, end_bp, window_bins, summary_stat)
                    y = read_window(full_bw, chrom, start_bp, end_bp, window_bins, summary_stat)

                    if near_zero_window(
                        x, y,
                        max_thresh=near_zero_max_thresh,
                        mean_thresh=near_zero_mean_thresh,
                    ):
                        continue

                    candidates.append(
                        {
                            "sample_id": sample_id,
                            "depth": np.float32(depth),
                            "chrom": chrom,
                            "start_bp": np.int64(start_bp),
                            "end_bp": np.int64(end_bp),
                            "x": x,
                            "y": y,
                            "full_max": float(np.max(y)),
                            "full_mean": float(np.mean(y)),
                            "low_max": float(np.max(x)),
                            "low_mean": float(np.mean(x)),
                        }
                    )

    return candidates


def assign_categories(
    candidates: List[dict],
    peak_quantile: float,
    background_max_thresh: float,
    background_mean_thresh: float,
):
    if len(candidates) == 0:
        return

    full_max_values = np.array([c["full_max"] for c in candidates], dtype=np.float32)
    peak_thresh = float(np.quantile(full_max_values, peak_quantile))

    print(f"[INFO] peak threshold (full_max q={peak_quantile:.2f}): {peak_thresh:.6f}")

    for c in candidates:
        is_bg = (
            c["full_max"] <= background_max_thresh and
            c["low_max"] <= background_max_thresh and
            c["full_mean"] <= background_mean_thresh and
            c["low_mean"] <= background_mean_thresh
        )

        if is_bg:
            c["category"] = "background"
        elif c["full_max"] >= peak_thresh:
            c["category"] = "peak"
        else:
            c["category"] = "weak"


def subsample_categories(
    candidates: List[dict],
    background_keep_ratio: float,
    seed: int,
) -> List[dict]:
    rng = random.Random(seed)

    peak = [c for c in candidates if c["category"] == "peak"]
    weak = [c for c in candidates if c["category"] == "weak"]
    background = [c for c in candidates if c["category"] == "background"]

    rng.shuffle(peak)
    rng.shuffle(weak)
    rng.shuffle(background)

    keep_bg_n = int(len(background) * background_keep_ratio)
    keep_bg_n = max(keep_bg_n, 1) if len(background) > 0 and background_keep_ratio > 0 else 0
    background_kept = background[:keep_bg_n]

    kept = peak + weak + background_kept
    rng.shuffle(kept)

    print("[INFO] category counts before filtering:")
    print(f"       peak={len(peak)} weak={len(weak)} background={len(background)}")
    print("[INFO] category counts after filtering:")
    print(f"       peak={len(peak)} weak={len(weak)} background={len(background_kept)} total={len(kept)}")

    return kept


def transform_and_pack(
    candidates: List[dict],
    log1p: bool,
    clip_quantile: float,
) -> Dict[str, np.ndarray]:
    if len(candidates) == 0:
        raise RuntimeError("No windows left after filtering.")

    X = np.stack([c["x"] for c in candidates], axis=0).astype(np.float32)
    Y = np.stack([c["y"] for c in candidates], axis=0).astype(np.float32)

    if log1p:
        X = np.log1p(X)
        Y = np.log1p(Y)

    clip_ref = np.concatenate([X.reshape(-1), Y.reshape(-1)], axis=0)
    upper = float(np.quantile(clip_ref, clip_quantile))
    X = np.clip(X, 0.0, upper)
    Y = np.clip(Y, 0.0, upper)

    # Add channel dim for 1D CNN: [N, 1, L]
    X = X[:, None, :]
    Y = Y[:, None, :]

    data = {
        "X": X.astype(np.float32),
        "Y": Y.astype(np.float32),
        "depth": np.array([c["depth"] for c in candidates], dtype=np.float32),
        "category": np.array([c["category"] for c in candidates], dtype="<U16"),
        "sample_id": np.array([c["sample_id"] for c in candidates], dtype="<U128"),
        "chrom": np.array([c["chrom"] for c in candidates], dtype="<U64"),
        "start_bp": np.array([c["start_bp"] for c in candidates], dtype=np.int64),
        "end_bp": np.array([c["end_bp"] for c in candidates], dtype=np.int64),
    }
    return data


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    root_dir = Path(args.root_dir)
    full_dir = root_dir / "bw"
    low_dirs = [
        root_dir / "ds_0.5_bw",
        root_dir / "ds_0.7_bw",
        root_dir / "ds_0.9_bw",
    ]

    print(f"[INFO] root_dir: {root_dir}")
    print(f"[INFO] full_dir: {full_dir}")
    print(f"[INFO] low_dirs: {low_dirs}")

    pairs = build_pair_table(str(full_dir), [str(x) for x in low_dirs])
    if len(pairs) == 0:
        raise RuntimeError("No matched (low-depth, full-depth) bigWig pairs found.")

    print(f"[INFO] matched pairs: {len(pairs)}")

    candidates = collect_candidates(
        pairs=pairs,
        window_bins=args.window_bins,
        bin_size=args.bin_size,
        stride_bins=args.stride_bins,
        summary_stat=args.summary_stat,
        chrom_regex=args.chrom_regex,
        near_zero_max_thresh=args.near_zero_max_thresh,
        near_zero_mean_thresh=args.near_zero_mean_thresh,
    )

    print(f"[INFO] candidate windows after near-zero drop: {len(candidates)}")
    if len(candidates) == 0:
        raise RuntimeError("No candidate windows after near-zero filtering.")

    assign_categories(
        candidates=candidates,
        peak_quantile=args.peak_quantile,
        background_max_thresh=args.background_max_thresh,
        background_mean_thresh=args.background_mean_thresh,
    )

    kept = subsample_categories(
        candidates=candidates,
        background_keep_ratio=args.background_keep_ratio,
        seed=args.seed,
    )

    packed = transform_and_pack(
        candidates=kept,
        log1p=args.log1p,
        clip_quantile=args.clip_quantile,
    )

    np.savez_compressed(args.output, **packed)

    print(f"[DONE] saved to: {args.output}")
    print(f"       X shape: {packed['X'].shape}")
    print(f"       Y shape: {packed['Y'].shape}")
    unique_cat, counts = np.unique(packed["category"], return_counts=True)
    print("       category distribution:")
    for c, n in zip(unique_cat, counts):
        print(f"         {c}: {n}")

if __name__ == "__main__":
    main()