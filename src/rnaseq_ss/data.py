from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass
class SampleBatch:
    noisy_signal: torch.Tensor
    clean_signal: torch.Tensor
    peak_label: torch.Tensor
    depth_ratio: torch.Tensor


def downsample_coverage(signal: torch.Tensor, depth_ratio: float, generator: torch.Generator | None = None) -> torch.Tensor:
    """Randomly downsample a high-depth signal with Poisson thinning.

    The sampled counts are rescaled by `depth_ratio` so the expected value
    stays aligned with the original signal magnitude.
    """
    if not (0.0 < depth_ratio <= 1.0):
        raise ValueError(f"depth_ratio must be in (0, 1], got {depth_ratio}")
    counts = torch.clamp(signal, min=0.0) * depth_ratio
    sampled = torch.poisson(counts, generator=generator)
    return sampled / depth_ratio


class SyntheticCoverageDataset(Dataset):
    """Synthetic dataset with on-the-fly random downsampling."""

    def __init__(
        self,
        size: int,
        window_size: int,
        feature_dim: int = 5,
        depth_candidates: list[float] | None = None,
        seed: int = 42,
    ):
        super().__init__()
        if feature_dim < 2:
            raise ValueError("feature_dim must be >= 2 so one channel can encode depth ratio")

        self.size = size
        self.window_size = window_size
        self.feature_dim = feature_dim
        self.depth_candidates = depth_candidates or [0.1, 0.2, 0.5, 0.8, 1.0]

        base_gen = torch.Generator().manual_seed(seed)
        peak_center = torch.randint(low=32, high=window_size - 32, size=(size,), generator=base_gen)

        clean = torch.zeros(size, window_size)
        peaks = torch.zeros(size, window_size)
        background = torch.randn(size, window_size, feature_dim - 1, generator=base_gen) * 0.05

        x = torch.arange(window_size).float()
        for idx, center in enumerate(peak_center):
            width = torch.randint(low=5, high=20, size=(1,), generator=base_gen).item()
            amp = torch.rand(1, generator=base_gen).item() * 2.0 + 1.0
            profile = amp * torch.exp(-((x - center) ** 2) / (2 * width**2))
            clean[idx] = profile + 0.1  # ensure positive baseline counts
            peaks[idx] = (profile > profile.mean()).float()

        self.clean = clean
        self.peaks = peaks
        self.background = background
        self.sample_seed = seed + 10000

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> SampleBatch:
        sample_gen = torch.Generator().manual_seed(self.sample_seed + idx + torch.randint(0, 1_000_000, (1,)).item())
        depth_idx = torch.randint(0, len(self.depth_candidates), (1,), generator=sample_gen).item()
        depth_ratio = float(self.depth_candidates[depth_idx])

        clean_signal = self.clean[idx]
        downsampled = downsample_coverage(clean_signal, depth_ratio=depth_ratio, generator=sample_gen)
        noisy_signal = torch.zeros(self.window_size, self.feature_dim)
        noisy_signal[:, 0] = downsampled + torch.randn(self.window_size, generator=sample_gen) * 0.05
        noisy_signal[:, 1:-1] = self.background[idx][:, : max(self.feature_dim - 2, 0)]
        noisy_signal[:, -1] = depth_ratio

        return SampleBatch(
            noisy_signal=noisy_signal,
            clean_signal=clean_signal,
            peak_label=self.peaks[idx],
            depth_ratio=torch.tensor(depth_ratio, dtype=torch.float32),
        )


def collate_samples(batch: list[SampleBatch]) -> SampleBatch:
    return SampleBatch(
        noisy_signal=torch.stack([item.noisy_signal for item in batch], dim=0),
        clean_signal=torch.stack([item.clean_signal for item in batch], dim=0),
        peak_label=torch.stack([item.peak_label for item in batch], dim=0),
        depth_ratio=torch.stack([item.depth_ratio for item in batch], dim=0),
    )
