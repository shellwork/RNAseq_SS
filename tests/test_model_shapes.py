import torch

from rnaseq_ss.data import SyntheticCoverageDataset, collate_samples, downsample_coverage
from rnaseq_ss.model import DeepMeripBaseline


def test_forward_shapes():
    model = DeepMeripBaseline(feature_dim=5, d_model=64, cnn_channels=64, num_heads=8, num_transformer_layers=1)
    x = torch.randn(3, 128, 5)
    reg, cls = model(x)
    assert reg.shape == (3, 128)
    assert cls.shape == (3, 128)


def test_downsample_coverage_preserves_shape():
    signal = torch.ones(128) * 10.0
    sampled = downsample_coverage(signal, depth_ratio=0.2, generator=torch.Generator().manual_seed(0))
    assert sampled.shape == signal.shape


def test_dataset_batch_contains_depth_ratio():
    ds = SyntheticCoverageDataset(size=4, window_size=64, feature_dim=5, depth_candidates=[0.1, 0.5, 1.0], seed=7)
    batch = collate_samples([ds[0], ds[1]])
    assert batch.noisy_signal.shape == (2, 64, 5)
    assert batch.depth_ratio.shape == (2,)
