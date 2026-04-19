"""
Peak-calling comparison: original vs. recovered (model-enhanced) ChIP-seq signal.

Pipeline:
  1. Convert raw BAM → .npz (same windowed format as training data)
  2. Load downsampled .npz and run trained model to produce recovered signal
  3. Export original & recovered signals as bedGraph files
  4. Call peaks on both using MACS2
  5. Compare peaks: overlap count + Jaccard index

Requirements:
  pip install pysam numpy torch macs2  (or have macs2/macs3 on PATH)
  bedtools must be on PATH for overlap analysis

Usage:
  python peak_calling_comparison.py \
      --bam         ENCFF882YNM.bam \
      --downsampled paired_ratio_0.5.npz \
      --checkpoint  checkpoints/best_model.pt \
      --outdir      peak_comparison_results
"""

import argparse
import os
import subprocess
import numpy as np
import torch
from torch.utils.data import DataLoader

from CNN import DeepMeripCNN
from train import NpzDataset
from dataprocess import (
    sort_and_index_bam,
    extract_all_windows,
)
# ── Test chromosomes (must match training split) ──
TEST_CHROMS = ["chr8", "chr9"]

# ═══════════════════════════════════════════════════════════════════════════
# 1. BAM → .npz conversion
# ═══════════════════════════════════════════════════════════════════════════
def bam_to_npz(bam_path: str, out_npz: str, chroms: list = None):
    """
    Convert a BAM file into the windowed .npz format using the same
    pipeline as bam_windowed_coverage_pysam.py.
    """
    if chroms is None:
        chroms = TEST_CHROMS

    # Sort and index if needed
    work_dir = os.path.dirname(out_npz) or "."
    sorted_bam = sort_and_index_bam(bam_path, work_dir)

    # Extract windows using the same method as training data
    signals, chroms_list, starts_list = extract_all_windows(
        sorted_bam, chroms=list(chroms), fast=True
    )

    np.savez(out_npz,
             input=signals,
             target=signals,
             chroms=np.array(chroms_list),
             starts=np.array(starts_list, dtype=np.int64),
             depth_ratio=np.float32(1.0))

    print(f"  Saved {len(signals):,} windows to {out_npz}")
    return out_npz


# ═══════════════════════════════════════════════════════════════════════════
# 2. Model inference → recovered signal
# ═══════════════════════════════════════════════════════════════════════════
def run_inference(npz_path: str, checkpoint_path: str, batch_size: int = 64,
                  num_workers: int = 4):
    """
    Run the trained DeepMeripCNN on the downsampled test data.

    Returns
    -------
    recovered : np.ndarray (N, 400)  -- model output
    chroms    : np.ndarray (N,)
    starts    : np.ndarray (N,)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Load data (test chroms only)
    data = np.load(npz_path, allow_pickle=False)
    # X = data["input"]           # (N_total, 400)
    # Y = data["target"]          # (N_total, 400)  -- not used here
    # chroms = data["chroms"]     # (N_total,)
    # starts = data["starts"]     # (N_total,)
    X = data["X"].squeeze(1)
    Y = data["Y"].squeeze(1)
    chroms = data["chrom"]
    starts = data["start_bp"]
    # Filter to test chromosomes
    mask = np.isin(chroms, TEST_CHROMS)
    test_idx = np.where(mask)[0]
    print(f"  Test windows: {len(test_idx):,}")

    test_ds = NpzDataset(X, Y, test_idx)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    # Load model
    model = DeepMeripCNN(
        feature_dim=1, d_model=128, cnn_channels=128,
        num_res_blocks=6, dropout=0.1,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Inference
    all_preds = []
    with torch.no_grad():
        for X_b, _ in test_loader:
            X_b = X_b.to(device)
            pred = model(X_b)           # [B, 400]
            all_preds.append(pred.cpu().numpy())

    recovered = np.concatenate(all_preds, axis=0)  # (N_test, 400)
    return recovered, chroms[mask], starts[mask]


# ═══════════════════════════════════════════════════════════════════════════
# 3. Signal → bedGraph
# ═══════════════════════════════════════════════════════════════════════════
def signal_to_bedgraph(signals: np.ndarray, chroms: np.ndarray,
                       starts: np.ndarray, out_path: str,
                       window_size: int = 400):
    """
    Write per-base signals as a sorted bedGraph file.

    Parameters
    ----------
    signals : (N, 400) array
    chroms  : (N,) chromosome names
    starts  : (N,) window start positions (bp)
    out_path: output .bedGraph file
    """
    # Sort by chrom, then start
    order = np.lexsort((starts, chroms))

    with open(out_path, "w") as f:
        for idx in order:
            chrom = chroms[idx]
            start = int(starts[idx])
            sig = signals[idx]
            for j, val in enumerate(sig):
                if val > 0:
                    f.write(f"{chrom}\t{start + j}\t{start + j + 1}\t{val:.4f}\n")

    print(f"  Written {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
# 4. MACS2 peak calling on bedGraph
# ═══════════════════════════════════════════════════════════════════════════
def call_peaks_macs2(bedgraph_path: str, outdir: str, name: str,
                     genome_size: str = "hs", qvalue: float = 0.05):
    """
    Call peaks from a bedGraph using MACS2 bdgpeakcall.

    For calling peaks directly from signal (no BAM), we use:
      macs2 bdgpeakcall -i <bedgraph> -o <output> --cutoff-analysis
    """
    os.makedirs(outdir, exist_ok=True)
    peak_file = os.path.join(outdir, f"{name}_peaks.narrowPeak")

    # Use bdgpeakcall for bedGraph input
    # First estimate a cutoff from the signal
    cmd = [
        "macs2", "bdgpeakcall",
        "-i", bedgraph_path,
        "-o", peak_file,
        "--no-trackline",
    ]

    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  MACS2 stderr: {result.stderr}")
        # Fallback: try with a manual cutoff
        cmd_fallback = [
            "macs2", "bdgpeakcall",
            "-i", bedgraph_path,
            "-o", peak_file,
            "--cutoff", "2.0",
            "--min-length", "200",
            "--max-gap", "50",
            "--no-trackline",
        ]
        print(f"  Retrying with cutoff: {' '.join(cmd_fallback)}")
        result = subprocess.run(cmd_fallback, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  MACS2 FAILED: {result.stderr}")
            return None

    print(f"  Peaks written to {peak_file}")
    return peak_file


# ═══════════════════════════════════════════════════════════════════════════
# 5. Peak comparison: overlap + count
# ═══════════════════════════════════════════════════════════════════════════
def load_peaks(peak_file: str):
    """Load a peak file into a list of (chrom, start, end) tuples."""
    peaks = []
    with open(peak_file) as f:
        for line in f:
            if line.startswith("#") or line.strip() == "":
                continue
            parts = line.strip().split("\t")
            peaks.append((parts[0], int(parts[1]), int(parts[2])))
    return peaks


def peaks_to_bed(peaks: list, out_path: str):
    """Write peaks as a sorted BED file for bedtools."""
    peaks_sorted = sorted(peaks, key=lambda x: (x[0], x[1]))
    with open(out_path, "w") as f:
        for chrom, start, end in peaks_sorted:
            f.write(f"{chrom}\t{start}\t{end}\n")
    return out_path


def compare_peaks(original_peak_file: str, recovered_peak_file: str,
                  outdir: str):
    """
    Compare two peak sets:
      - Total peak counts
      - Overlapping peak count (bedtools intersect)
      - Jaccard index (bedtools jaccard)
    """
    orig_peaks = load_peaks(original_peak_file)
    rec_peaks = load_peaks(recovered_peak_file)

    print("\n" + "=" * 60)
    print("PEAK COMPARISON RESULTS")
    print("=" * 60)
    print(f"  Original peaks : {len(orig_peaks):,}")
    print(f"  Recovered peaks: {len(rec_peaks):,}")

    # Write sorted BED files for bedtools
    orig_bed = peaks_to_bed(orig_peaks, os.path.join(outdir, "original.bed"))
    rec_bed = peaks_to_bed(rec_peaks, os.path.join(outdir, "recovered.bed"))

    # --- Overlap analysis with bedtools ---
    # Peaks in original that overlap with recovered
    try:
        result = subprocess.run(
            ["bedtools", "intersect", "-a", orig_bed, "-b", rec_bed, "-u"],
            capture_output=True, text=True, check=True
        )
        orig_overlapping = len([l for l in result.stdout.strip().split("\n")
                                if l.strip()])

        # Peaks in recovered that overlap with original
        result2 = subprocess.run(
            ["bedtools", "intersect", "-a", rec_bed, "-b", orig_bed, "-u"],
            capture_output=True, text=True, check=True
        )
        rec_overlapping = len([l for l in result2.stdout.strip().split("\n")
                               if l.strip()])

        print(f"\n  Original peaks overlapping recovered: "
              f"{orig_overlapping}/{len(orig_peaks)} "
              f"({100 * orig_overlapping / max(len(orig_peaks), 1):.1f}%)")
        print(f"  Recovered peaks overlapping original: "
              f"{rec_overlapping}/{len(rec_peaks)} "
              f"({100 * rec_overlapping / max(len(rec_peaks), 1):.1f}%)")

        # --- Jaccard index ---
        result_j = subprocess.run(
            ["bedtools", "jaccard", "-a", orig_bed, "-b", rec_bed],
            capture_output=True, text=True, check=True
        )
        jaccard_lines = result_j.stdout.strip().split("\n")
        if len(jaccard_lines) >= 2:
            header = jaccard_lines[0].split("\t")
            values = jaccard_lines[1].split("\t")
            jaccard_dict = dict(zip(header, values))
            jaccard_idx = float(jaccard_dict.get("jaccard", 0))
            print(f"\n  Jaccard index: {jaccard_idx:.4f}")
            print(f"  Intersection (bp): {jaccard_dict.get('intersection', 'N/A')}")
            print(f"  Union (bp):        {jaccard_dict.get('union-intersection', 'N/A')}")

    except FileNotFoundError:
        print("\n  WARNING: bedtools not found. Falling back to Python overlap.")
        _python_overlap(orig_peaks, rec_peaks)
    except subprocess.CalledProcessError as e:
        print(f"\n  bedtools error: {e.stderr}")
        _python_overlap(orig_peaks, rec_peaks)

    print("=" * 60)


def _python_overlap(peaks_a: list, peaks_b: list):
    """
    Fallback overlap calculation in pure Python (no bedtools).
    Counts how many peaks in A overlap at least one peak in B and vice versa.
    """
    def overlaps(a, b):
        return a[0] == b[0] and a[1] < b[2] and b[1] < a[2]

    # Group peaks by chromosome for efficiency
    from collections import defaultdict
    b_by_chrom = defaultdict(list)
    for p in peaks_b:
        b_by_chrom[p[0]].append(p)

    a_by_chrom = defaultdict(list)
    for p in peaks_a:
        a_by_chrom[p[0]].append(p)

    # Count A peaks overlapping B
    a_overlap = 0
    for p_a in peaks_a:
        for p_b in b_by_chrom.get(p_a[0], []):
            if overlaps(p_a, p_b):
                a_overlap += 1
                break

    # Count B peaks overlapping A
    b_overlap = 0
    for p_b in peaks_b:
        for p_a in a_by_chrom.get(p_b[0], []):
            if overlaps(p_b, p_a):
                b_overlap += 1
                break

    print(f"  Original peaks overlapping recovered: "
          f"{a_overlap}/{len(peaks_a)} "
          f"({100 * a_overlap / max(len(peaks_a), 1):.1f}%)")
    print(f"  Recovered peaks overlapping original: "
          f"{b_overlap}/{len(peaks_b)} "
          f"({100 * b_overlap / max(len(peaks_b), 1):.1f}%)")

    # Simple Jaccard on base pairs
    from functools import reduce
    def total_bp(peaks):
        return sum(p[2] - p[1] for p in peaks)

    def intersect_bp(pa, pb):
        total = 0
        for a in pa:
            for b in pb:
                if a[0] == b[0]:
                    ov = min(a[2], b[2]) - max(a[1], b[1])
                    if ov > 0:
                        total += ov
        return total

    bp_a = total_bp(peaks_a)
    bp_b = total_bp(peaks_b)
    bp_int = intersect_bp(peaks_a, peaks_b)
    bp_union = bp_a + bp_b - bp_int
    jaccard = bp_int / bp_union if bp_union > 0 else 0.0
    print(f"\n  Jaccard index (bp): {jaccard:.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Peak-calling comparison: original vs recovered signal")
    parser.add_argument("--bam", required=True,
                        help="Raw BAM file (e.g. ENCFF882YNM.bam)")
    parser.add_argument("--downsampled", required=True,
                        help="Downsampled .npz (e.g. paired_ratio_0.5.npz)")
    parser.add_argument("--checkpoint", required=True,
                        help="Model checkpoint .pt file")
    parser.add_argument("--outdir", default="peak_comparison_results",
                        help="Output directory")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # ── Step 1: BAM → .npz (original signal, test chroms only) ──
    print("\n[Step 1] Converting BAM to .npz (test chroms only) ...")
    original_npz = os.path.join(args.outdir, "original_test.npz")
    bam_to_npz(args.bam, original_npz, chroms=TEST_CHROMS)

    # ── Step 2: Model inference on downsampled data ──
    print("\n[Step 2] Running model inference on downsampled data ...")
    recovered, rec_chroms, rec_starts = run_inference(
        args.downsampled, args.checkpoint,
        batch_size=args.batch_size, num_workers=args.num_workers
    )

    # ── Step 3: Load original signal for test chroms ──
    print("\n[Step 3] Loading original signal ...")
    orig_data = np.load(original_npz, allow_pickle=False)
    orig_signals = orig_data["input"]
    orig_chroms = orig_data["chroms"]
    orig_starts = orig_data["starts"]

    # ── Step 4: Export bedGraphs ──
    print("\n[Step 4] Writing bedGraph files ...")
    orig_bg = os.path.join(args.outdir, "original.bedGraph")
    rec_bg = os.path.join(args.outdir, "recovered.bedGraph")

    signal_to_bedgraph(orig_signals, orig_chroms, orig_starts, orig_bg)
    signal_to_bedgraph(recovered, rec_chroms, rec_starts, rec_bg)

    # ── Step 5: MACS2 peak calling ──
    print("\n[Step 5] Calling peaks with MACS2 ...")
    orig_peaks = call_peaks_macs2(orig_bg, args.outdir, name="original")
    rec_peaks = call_peaks_macs2(rec_bg, args.outdir, name="recovered")

    if orig_peaks is None or rec_peaks is None:
        print("ERROR: Peak calling failed for one or both signals.")
        return

    # ── Step 6: Compare peaks ──
    print("\n[Step 6] Comparing peaks ...")
    compare_peaks(orig_peaks, rec_peaks, args.outdir)

    # ── Step 7: Save all tracks for IGV/UCSC visualization ──
    print("\n[Step 7] Saving visualization tracks ...")

    # Also export the downsampled signal as bedGraph for comparison
    ds_data = np.load(args.downsampled, allow_pickle=False)
    ds_mask = np.isin(ds_data["chrom"], TEST_CHROMS)
    ds_bg = os.path.join(args.outdir, "downsampled.bedGraph")
    #signal_to_bedgraph(
    #    ds_data["input"][ds_mask],
    #    ds_data["chroms"][ds_mask],
    #    ds_data["starts"][ds_mask],
    #    ds_bg
    #)
    signal_to_bedgraph(
        ds_data["X"].squeeze(1)[ds_mask],
        ds_data["chrom"][ds_mask],
        ds_data["start_bp"][ds_mask],
        ds_bg
    )

    # Sort all bedGraph files (required by IGV and UCSC)
    for bg_name in ["original.bedGraph", "recovered.bedGraph", "downsampled.bedGraph"]:
        bg_path = os.path.join(args.outdir, bg_name)
        sorted_path = bg_path + ".sorted"
        os.system(f"sort -k1,1 -k2,2n {bg_path} > {sorted_path}")
        os.replace(sorted_path, bg_path)

    # Sort BED files
    for bed_name in ["original.bed", "recovered.bed"]:
        bed_path = os.path.join(args.outdir, bed_name)
        sorted_path = bed_path + ".sorted"
        os.system(f"sort -k1,1 -k2,2n {bed_path} > {sorted_path}")
        os.replace(sorted_path, bed_path)

    print(f"\n  All tracks saved to {args.outdir}/:")
    print(f"    Signal tracks (bedGraph):")
    print(f"      original.bedGraph     — raw BAM coverage")
    print(f"      downsampled.bedGraph  — downsampled input")
    print(f"      recovered.bedGraph    — model-recovered signal")
    print(f"    Peak tracks (BED):")
    print(f"      original.bed          — peaks from original")
    print(f"      recovered.bed         — peaks from recovered")
    print(f"\n  Load these into IGV: File → Load from File → select all 5 files")

if __name__ == "__main__":
    main()