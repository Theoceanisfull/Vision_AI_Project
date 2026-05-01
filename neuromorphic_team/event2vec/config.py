from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

EncodingMode = Literal["rate", "latency", "delta"]

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


@dataclass
class DataConfig:
    data_root: str = "data/ASLDVS"
    encoding: EncodingMode = "rate"
    sensor_size: tuple[int, int] = (180, 240)
    pool_kernel: tuple[int, int] = (6, 8)
    num_steps: int = 20
    max_tokens: int = 1024
    batch_size: int = 32
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    seed: int = 42
    delta_threshold: float = 0.1
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True


@dataclass
class ModelConfig:
    num_classes: int = 24
    d_model: int = 64
    depth: int = 2
    num_heads: int = 2
    ffn_dim: int = 128
    dropout: float = 0.1
    pool_after_each_block: tuple[bool, ...] = field(default_factory=lambda: (False, False))


@dataclass
class TrainConfig:
    epochs: int = 5
    lr: float = 3e-4
    weight_decay: float = 1e-4
    optimizer: Literal["adam", "adamw"] = "adamw"
    label_smoothing: float = 0.0
    grad_clip: float | None = 1.0
    max_train_batches: int | None = None
    max_val_batches: int | None = None
    max_test_batches: int | None = None
    log_every: int = 50
    device: str = "auto"
    amp: bool = True
    amp_dtype: Literal["bfloat16", "float16"] = "bfloat16"


@dataclass
class ResultConfig:
    out_dir: str = "runs/event2vec"
    run_name: str = "rate"
    save_plots: bool = True
    save_checkpoint: bool = True


@dataclass
class Event2VecConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    result: ResultConfig = field(default_factory=ResultConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_json(self, path: str | Path) -> None:
        target = resolve_project_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_json(cls, path: str | Path) -> "Event2VecConfig":
        payload = json.loads(resolve_project_path(path).read_text())
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Event2VecConfig":
        data = DataConfig(**payload.get("data", {}))
        model = ModelConfig(**payload.get("model", {}))
        train = TrainConfig(**payload.get("train", {}))
        result = ResultConfig(**payload.get("result", {}))
        return cls(data=data, model=model, train=train, result=result)


def default_config() -> Event2VecConfig:
    return Event2VecConfig()


__all__ = [
    "DataConfig",
    "EncodingMode",
    "Event2VecConfig",
    "ModelConfig",
    "PROJECT_ROOT",
    "ResultConfig",
    "TrainConfig",
    "default_config",
    "resolve_project_path",
]
