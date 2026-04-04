from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import torch
from scipy.io import loadmat
from snntorch import spikegen
from torch.utils.data import DataLoader, Dataset

from .config import resolve_project_path

EncodingMode = Literal["rate", "latency", "delta"]


@dataclass(frozen=True)
class SplitDatasets:
    train: Dataset
    val: Dataset
    test: Dataset


class ASLDVSSpikeDataset(Dataset):
    """ASL-DVS dataset with selectable snntorch spike encodings.

    This loader follows the same raw event interpretation used in
    `notebooks/asldvs_viewer.ipynb`:
    - load MATLAB fields: x, y, ts, pol
    - apply tonic-style vertical flip: y -> (height - 1 - y)

    Output shape per item:
    - rate/latency: [num_steps, 2, height, width]
    - delta:        [num_steps, 2, height, width]

    Representation notes:
    - `rate` now preserves the temporal event bins and applies a per-bin
      Poisson spike conversion.
    - `latency` still encodes a single normalized activity map into
      time-to-first-spike, because snntorch latency coding is defined for a
      static feature map rather than a time-varying sequence.
    - `delta` operates on the same temporal event bins as `rate`.
    """

    def __init__(
        self,
        sample_paths: Sequence[Path],
        class_to_idx: dict[str, int],
        *,
        encoding: EncodingMode = "rate",
        sensor_size: tuple[int, int] = (180, 240),
        num_steps: int = 20,
        delta_threshold: float = 0.1,
    ) -> None:
        if not sample_paths:
            raise ValueError("sample_paths is empty")
        if encoding not in ("rate", "latency", "delta"):
            raise ValueError(f"Unsupported encoding: {encoding}")
        if num_steps < 2:
            raise ValueError("num_steps must be >= 2")

        self.sample_paths = [Path(p) for p in sample_paths]
        self.class_to_idx = class_to_idx
        self.encoding = encoding
        self.height, self.width = sensor_size
        self.num_steps = int(num_steps)
        self.delta_threshold = float(delta_threshold)

    def __len__(self) -> int:
        return len(self.sample_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path = self.sample_paths[index]
        label = self.class_to_idx[path.parent.name]

        x, y, ts, pol = self._load_events(path)
        temporal = self._events_to_temporal_channels(x, y, ts, pol)

        if self.encoding == "rate":
            spikes = spikegen.rate(temporal, time_var_input=True)
        elif self.encoding == "delta":
            spikes = spikegen.delta(
                temporal,
                threshold=self.delta_threshold,
                padding=True,
                off_spike=True,
            )
        else:
            static_image = self._temporal_to_static_image(temporal)
            spikes = spikegen.latency(
                static_image,
                num_steps=self.num_steps,
                normalize=True,
                clip=True,
            )

        return spikes.to(torch.float32), label

    def _temporal_to_static_image(self, temporal: torch.Tensor) -> torch.Tensor:
        static_image = temporal.sum(dim=0)
        return static_image / (static_image.max() + 1e-8)

    def _load_events(self, path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        mat = loadmat(path)
        x = mat["x"].reshape(-1).astype(np.int64)
        y = mat["y"].reshape(-1).astype(np.int64)
        ts = mat["ts"].reshape(-1).astype(np.int64)
        pol = mat["pol"].reshape(-1).astype(np.int8)

        # Same orientation adjustment as notebooks/asldvs_viewer.ipynb
        y = self.height - 1 - y

        valid = (
            (x >= 0)
            & (x < self.width)
            & (y >= 0)
            & (y < self.height)
        )
        if not np.any(valid):
            return (
                np.zeros(1, dtype=np.int64),
                np.zeros(1, dtype=np.int64),
                np.zeros(1, dtype=np.int64),
                np.zeros(1, dtype=np.int8),
            )

        return x[valid], y[valid], ts[valid], pol[valid]

    def _events_to_temporal_channels(
        self,
        x: np.ndarray,
        y: np.ndarray,
        ts: np.ndarray,
        pol: np.ndarray,
    ) -> torch.Tensor:
        """Bin events to [T, C=2, H, W] where C=(ON, OFF)."""
        frames = np.zeros((self.num_steps, 2, self.height, self.width), dtype=np.float32)

        t_min = int(ts.min())
        t_max = int(ts.max())
        if t_max <= t_min:
            bins = np.zeros_like(ts, dtype=np.int64)
        else:
            edges = np.linspace(t_min, t_max + 1, self.num_steps + 1, dtype=np.int64)
            bins = np.searchsorted(edges, ts, side="right") - 1
            bins = np.clip(bins, 0, self.num_steps - 1)

        on_mask = pol > 0
        off_mask = ~on_mask

        np.add.at(frames, (bins[on_mask], 0, y[on_mask], x[on_mask]), 1.0)
        np.add.at(frames, (bins[off_mask], 1, y[off_mask], x[off_mask]), 1.0)

        max_val = float(frames.max())
        if max_val > 0:
            frames /= max_val

        return torch.from_numpy(frames)


def _collect_samples(data_root: Path) -> tuple[list[Path], dict[str, int]]:
    class_dirs = sorted([p for p in data_root.iterdir() if p.is_dir()])
    if not class_dirs:
        raise FileNotFoundError(f"No class folders found under {data_root}")

    class_to_idx = {class_dir.name: idx for idx, class_dir in enumerate(class_dirs)}
    sample_paths: list[Path] = []

    for class_dir in class_dirs:
        mats = sorted(class_dir.glob("*.mat"))
        if not mats:
            continue
        sample_paths.extend(mats)

    if not sample_paths:
        raise FileNotFoundError(f"No .mat files found under {data_root}")

    return sample_paths, class_to_idx


def _stratified_split(
    sample_paths: Sequence[Path],
    *,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[Path], list[Path], list[Path]]:
    if not (0 < train_ratio < 1):
        raise ValueError("train_ratio must be in (0, 1)")
    if not (0 <= val_ratio < 1):
        raise ValueError("val_ratio must be in [0, 1)")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be < 1")

    rng = np.random.default_rng(seed)
    by_class: dict[str, list[Path]] = {}
    for p in sample_paths:
        by_class.setdefault(p.parent.name, []).append(p)

    train: list[Path] = []
    val: list[Path] = []
    test: list[Path] = []

    for cls_name, cls_paths in sorted(by_class.items()):
        cls_paths = list(cls_paths)
        rng.shuffle(cls_paths)
        n = len(cls_paths)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        # Keep non-empty val/test whenever possible.
        if n >= 3:
            n_train = max(1, min(n_train, n - 2))
            n_val = max(1, min(n_val, n - n_train - 1))

        train.extend(cls_paths[:n_train])
        val.extend(cls_paths[n_train : n_train + n_val])
        test.extend(cls_paths[n_train + n_val :])

    return train, val, test


def build_asldvs_splits(
    data_root: str | Path = "data/ASLDVS",
    *,
    encoding: EncodingMode = "rate",
    sensor_size: tuple[int, int] = (180, 240),
    num_steps: int = 20,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    delta_threshold: float = 0.1,
) -> SplitDatasets:
    data_root = resolve_project_path(data_root).resolve()
    sample_paths, class_to_idx = _collect_samples(data_root)
    train_paths, val_paths, test_paths = _stratified_split(
        sample_paths,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )

    ds_kwargs = {
        "class_to_idx": class_to_idx,
        "encoding": encoding,
        "sensor_size": sensor_size,
        "num_steps": num_steps,
        "delta_threshold": delta_threshold,
    }

    return SplitDatasets(
        train=ASLDVSSpikeDataset(train_paths, **ds_kwargs),
        val=ASLDVSSpikeDataset(val_paths, **ds_kwargs),
        test=ASLDVSSpikeDataset(test_paths, **ds_kwargs),
    )


def build_asldvs_dataloaders(
    data_root: str | Path = "data/ASLDVS",
    *,
    encoding: EncodingMode = "rate",
    sensor_size: tuple[int, int] = (180, 240),
    num_steps: int = 20,
    batch_size: int = 16,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    delta_threshold: float = 0.1,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> dict[str, DataLoader]:
    splits = build_asldvs_splits(
        data_root=data_root,
        encoding=encoding,
        sensor_size=sensor_size,
        num_steps=num_steps,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        delta_threshold=delta_threshold,
    )

    return {
        "train": DataLoader(
            splits.train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "val": DataLoader(
            splits.val,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "test": DataLoader(
            splits.test,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }


__all__ = [
    "ASLDVSSpikeDataset",
    "SplitDatasets",
    "build_asldvs_splits",
    "build_asldvs_dataloaders",
]
