from dataclasses import dataclass, field


@dataclass
class DataConfig:
    feature_dim: int = 5
    window_size: int = 256
    train_size: int = 4096
    val_size: int = 1024
    batch_size: int = 32
    depth_candidates: list[float] = field(default_factory=lambda: [0.1, 0.2, 0.5, 0.8, 1.0])


@dataclass
class ModelConfig:
    feature_dim: int = 5
    d_model: int = 128
    cnn_channels: int = 128
    num_heads: int = 8
    num_transformer_layers: int = 2
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 10
    regression_weight: float = 1.0
    classification_weight: float = 1.0
    device: str = "cpu"
