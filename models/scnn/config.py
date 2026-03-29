from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

EncodingMode = Literal["rate", "latency", "delta"]
SurrogateType = Literal[
    "fast_sigmoid",
    "atan",
    "sigmoid",
    "triangular",
    "spike_rate_escape",
    "straight_through_estimator",
]
ArchitectureType = Literal["base", "deep"]
BackpropMode = Literal["manual", "backprop_bptt", "backprop_tbptt", "backprop_rtrl"]
LossType = Literal[
    "ce_rate_loss",
    "ce_count_loss",
    "ce_max_membrane_loss",
    "mse_count_loss",
    "mse_membrane_loss",
    "ce_temporal_loss",
    "mse_temporal_loss",
]
RegularizerType = Literal["none", "l1_rate_sparsity"]


@dataclass
class DataConfig:
    data_root: str = "data/ASLDVS"
    encoding: EncodingMode = "rate"
    sensor_size: tuple[int, int] = (180, 240)
    num_steps: int = 20
    batch_size: int = 32
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    seed: int = 42
    delta_threshold: float = 0.1
    num_workers: int = 0
    pin_memory: bool = False


@dataclass
class ModelConfig:
    architecture: ArchitectureType = "base"
    input_channels: int = 2
    num_classes: int = 24
    conv_channels: tuple[int, int] = (32, 64)
    deep_conv_channels: tuple[int, int, int] = (32, 64, 128)
    kernel_size: int = 3
    pool_kernel: int = 2
    hidden_size: int = 256
    beta: float = 0.9
    threshold: float = 1.0
    surrogate: SurrogateType = "fast_sigmoid"
    surrogate_kwargs: dict[str, Any] = field(default_factory=lambda: {"slope": 25.0})
    dropout: float = 0.2


@dataclass
class TrainConfig:
    mode: BackpropMode = "manual"
    epochs: int = 5
    lr: float = 1e-3
    weight_decay: float = 1e-5
    optimizer: Literal["adam", "adamw", "sgd"] = "adam"
    loss: LossType = "ce_rate_loss"
    loss_kwargs: dict[str, Any] = field(default_factory=dict)
    regularizer: RegularizerType = "none"
    regularizer_kwargs: dict[str, Any] = field(default_factory=dict)
    regularizer_weight: float = 1e-4
    accuracy_fn: Literal["accuracy_rate", "accuracy_temporal"] = "accuracy_rate"
    tbptt_k: int = 5
    grad_clip: float | None = 1.0
    max_train_batches: int | None = None
    max_val_batches: int | None = None
    max_test_batches: int | None = None
    log_every: int = 25
    device: str = "auto"


@dataclass
class ResultConfig:
    out_dir: str = "runs/scnn"
    run_name: str = "default"
    save_plots: bool = True
    save_checkpoint: bool = True


@dataclass
class SNNConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    result: ResultConfig = field(default_factory=ResultConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_json(cls, path: str | Path) -> "SNNConfig":
        payload = json.loads(Path(path).read_text())
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SNNConfig":
        data = DataConfig(**payload.get("data", {}))
        model = ModelConfig(**payload.get("model", {}))
        train = TrainConfig(**payload.get("train", {}))
        result = ResultConfig(**payload.get("result", {}))
        return cls(data=data, model=model, train=train, result=result)


def default_config() -> SNNConfig:
    return SNNConfig()


__all__ = [
    "DataConfig",
    "ModelConfig",
    "TrainConfig",
    "ResultConfig",
    "SNNConfig",
    "default_config",
]
